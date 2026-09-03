"""Retriever sobre `samples/`, em pgvector (ADR 005).

O índice vive no mesmo Postgres do checkpointer. O modelo de embedding vem de
`EMBEDDING_MODEL` e o chunking de `CHUNK_SIZE` e `CHUNK_OVERLAP`, contados em
tokens e não em caracteres, que é o que a configuração diz.
"""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Settings, get_settings
from ..models import MissingGatewayKey
from ..observability import emit

#: Extensões que o corpus aceita. PDF exigiria mais uma dependência e o corpus é
#: documentação aberta, então texto basta.
SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})

#: Arquivos que descrevem o corpus, e por isso não fazem parte dele. Indexar o
#: SOURCES.md entregaria ao agente as respostas das tarefas-armadilha, que é o
#: contrário do que elas medem.
CORPUS_METADATA_FILES = frozenset({"SOURCES.md", "README.md"})

#: Separa os campos que compõem o id de um chunk. Um caractere que não aparece
#: em texto comum, para dois campos diferentes nunca formarem o mesmo material.
ID_SEPARATOR = chr(31)

#: Nome da coleção no pgvector. Trocar isto invalida o índice existente.
COLLECTION_NAME = "pauta_samples"

#: O tokenizer usado para medir o chunk. Só conta tokens, não chama API.
TOKEN_ENCODING = "cl100k_base"


class EmptyCorpus(RuntimeError):
    """`samples/` não tem nenhum documento indexável."""


@dataclass(frozen=True)
class IndexReport:
    """O que a indexação produziu. Vai para o README, medido e não estimado."""

    documents: int
    chunks: int
    skipped: int

    def __str__(self) -> str:
        return f"{self.documents} documentos, {self.chunks} chunks, {self.skipped} ignorados"


def samples_dir(settings: Settings | None = None) -> Path:
    del settings
    return Path(__file__).resolve().parents[3] / "samples"


def sqlalchemy_url(database_url: str) -> str:
    """`langchain-postgres` fala SQLAlchemy, que exige o driver no esquema."""
    if database_url.startswith("postgresql+"):
        return database_url
    return database_url.replace("postgresql://", "postgresql+psycopg://", 1)


def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """Embeddings pelo mesmo gateway dos modelos de chat."""
    resolved = settings or get_settings()
    if not resolved.OPENROUTER_API_KEY:
        raise MissingGatewayKey("OPENROUTER_API_KEY não definida; sem embeddings não há índice")
    return OpenAIEmbeddings(
        model=resolved.EMBEDDING_MODEL,
        base_url=resolved.OPENROUTER_BASE_URL,
        api_key=resolved.OPENROUTER_API_KEY,  # type: ignore[arg-type]  # aceita str
        check_embedding_ctx_length=False,
    )


def load_documents(directory: Path) -> tuple[list[Document], int]:
    """Lê o corpus. Devolve os documentos e quantos arquivos foram ignorados."""
    if not directory.is_dir():
        return [], 0
    documents: list[Document] = []
    skipped = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped += 1
            continue
        if path.name in CORPUS_METADATA_FILES:
            skipped += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            skipped += 1
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.relative_to(directory).as_posix()},
            )
        )
    return documents, skipped


def make_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    resolved = settings or get_settings()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING,
        chunk_size=resolved.CHUNK_SIZE,
        chunk_overlap=resolved.CHUNK_OVERLAP,
    )


def chunk_id(chunk: Document, collection_name: str = COLLECTION_NAME) -> str:
    """Id determinístico: reindexar o mesmo texto não duplica linha no banco.

    A coleção entra no hash porque a chave primária de `langchain_pg_embedding`
    é só o `id`, sem o `collection_id` junto. Sem isso, indexar o mesmo texto em
    duas coleções faria a segunda roubar a linha da primeira por upsert, em vez
    de guardar as duas.
    """
    source = str(chunk.metadata.get("source", ""))
    material = ID_SEPARATOR.join([collection_name, source, chunk.page_content])
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def split_documents(
    documents: Iterable[Document],
    settings: Settings | None = None,
) -> list[Document]:
    """Quebra em chunks e numera cada um dentro do seu documento."""
    splitter = make_splitter(settings)
    chunks: list[Document] = []
    for document in documents:
        pieces = splitter.split_documents([document])
        for position, piece in enumerate(pieces):
            piece.metadata = {**piece.metadata, "chunk": position}
            chunks.append(piece)
    return chunks


def get_store(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    collection_name: str = COLLECTION_NAME,
) -> PGVector:
    """Abre a coleção no pgvector, criando a extensão e as tabelas se faltarem.

    Os embeddings podem ser injetados, que é como o teste exercita o banco de
    verdade sem gastar chamada de API.
    """
    resolved = settings or get_settings()
    return PGVector(
        embeddings=embeddings or get_embeddings(resolved),
        connection=sqlalchemy_url(resolved.DATABASE_URL),
        collection_name=collection_name,
        use_jsonb=True,
        create_extension=True,
    )


def index_samples(
    settings: Settings | None = None,
    *,
    directory: Path | None = None,
    embeddings: Embeddings | None = None,
    collection_name: str = COLLECTION_NAME,
) -> IndexReport:
    """Indexa `samples/`. Rodar duas vezes não duplica chunk."""
    resolved = settings or get_settings()
    source_dir = directory or samples_dir(resolved)
    documents, skipped = load_documents(source_dir)
    if not documents:
        raise EmptyCorpus(f"nenhum documento indexável em {source_dir}")

    chunks = split_documents(documents, resolved)
    store = get_store(resolved, embeddings=embeddings, collection_name=collection_name)
    store.add_documents(chunks, ids=[chunk_id(chunk, collection_name) for chunk in chunks])

    report = IndexReport(documents=len(documents), chunks=len(chunks), skipped=skipped)
    emit(
        "node_end",
        node="retriever",
        message="corpus indexado",
        documents=report.documents,
        chunks=report.chunks,
        skipped=report.skipped,
    )
    return report


def search(
    query: str,
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    collection_name: str = COLLECTION_NAME,
) -> Sequence[Document]:
    resolved = settings or get_settings()
    store = get_store(resolved, embeddings=embeddings, collection_name=collection_name)
    return store.similarity_search(query, k=resolved.RETRIEVER_TOP_K)


def format_results(results: Sequence[Document]) -> str:
    """Cada trecho volta com a origem colada, para o Finding poder citar a fonte."""
    if not results:
        return "nenhum trecho encontrado no corpus local"
    blocks = []
    for document in results:
        source = document.metadata.get("source", "desconhecida")
        chunk = document.metadata.get("chunk", 0)
        blocks.append(f"[fonte: samples/{source} chunk {chunk}]\n{document.page_content}")
    return "\n\n".join(blocks)


@tool
def retriever(query: str) -> str:
    """Busca trechos nos documentos locais de samples/.

    Devolve os trechos mais próximos da pergunta, cada um com o arquivo de origem.
    Use quando a tarefa citar os documentos locais, o corpus ou samples/.
    """
    try:
        return format_results(search(query))
    except Exception as exc:
        return f"não foi possível consultar o corpus local: {type(exc).__name__}: {exc}"


@lru_cache
def get_retriever_tool() -> BaseTool:
    return retriever

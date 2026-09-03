"""Retriever em pgvector. O que fala com banco é marcado e pulado sem ele."""

import os
import uuid
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from pauta.config import Settings, get_settings
from pauta.tools.retriever import (
    COLLECTION_NAME,
    SUPPORTED_SUFFIXES,
    EmptyCorpus,
    chunk_id,
    format_results,
    index_samples,
    load_documents,
    split_documents,
    sqlalchemy_url,
)

requires_postgres = pytest.mark.skipif(
    os.environ.get("PAUTA_TEST_POSTGRES") != "1",
    reason="precisa de Postgres; rode docker compose up e exporte PAUTA_TEST_POSTGRES=1",
)


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def fake_embeddings() -> DeterministicFakeEmbedding:
    """Embedding determinístico: exercita o pgvector sem chamar API nenhuma."""
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def collection() -> str:
    """Coleção própria por teste, para um não enxergar o índice do outro."""
    return f"pauta_test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "relatorio.md").write_text(
        "# Relatório\n\n" + ("A latência média caiu depois da mudança. " * 60),
        encoding="utf-8",
    )
    (tmp_path / "notas.txt").write_text("Nota curta sobre custo por requisição.", encoding="utf-8")
    (tmp_path / "ignorado.pdf").write_bytes(b"%PDF-1.4 nao suportado")
    (tmp_path / "vazio.md").write_text("   \n", encoding="utf-8")
    subdir = tmp_path / "anexos"
    subdir.mkdir()
    (subdir / "anexo.md").write_text("Anexo com detalhe de metodologia.", encoding="utf-8")
    return tmp_path


def test_only_text_formats_are_read(corpus: Path) -> None:
    documents, skipped = load_documents(corpus)
    sources = sorted(str(d.metadata["source"]) for d in documents)
    assert sources == ["anexos/anexo.md", "notas.txt", "relatorio.md"]
    assert skipped == 2  # o pdf e o arquivo vazio
    assert ".pdf" not in SUPPORTED_SUFFIXES


def test_a_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    documents, skipped = load_documents(tmp_path / "nao-existe")
    assert documents == []
    assert skipped == 0


def test_source_metadata_is_relative_to_the_corpus(corpus: Path) -> None:
    documents, _ = load_documents(corpus)
    assert all(not str(d.metadata["source"]).startswith("/") for d in documents)


def test_chunking_uses_the_configured_size(corpus: Path, settings: Settings) -> None:
    documents, _ = load_documents(corpus)
    small = settings.model_copy(update={"CHUNK_SIZE": 50, "CHUNK_OVERLAP": 10})
    chunks = split_documents(documents, small)
    assert len(chunks) > len(documents)
    assert all("chunk" in chunk.metadata for chunk in chunks)


def test_chunks_are_numbered_within_each_document(corpus: Path, settings: Settings) -> None:
    documents, _ = load_documents(corpus)
    small = settings.model_copy(update={"CHUNK_SIZE": 40, "CHUNK_OVERLAP": 0})
    chunks = split_documents(documents, small)
    by_source: dict[str, list[int]] = {}
    for chunk in chunks:
        by_source.setdefault(str(chunk.metadata["source"]), []).append(int(chunk.metadata["chunk"]))
    for positions in by_source.values():
        assert positions == list(range(len(positions)))


def test_the_chunk_id_is_stable_and_content_addressed() -> None:
    """Reindexar o mesmo texto não pode duplicar linha no banco."""
    one = Document(page_content="mesmo texto", metadata={"source": "a.md"})
    two = Document(page_content="mesmo texto", metadata={"source": "a.md"})
    other = Document(page_content="mesmo texto", metadata={"source": "b.md"})
    assert chunk_id(one) == chunk_id(two)
    assert chunk_id(one) != chunk_id(other)


def test_the_chunk_id_separates_collections() -> None:
    """A primary key de langchain_pg_embedding é só o id, sem a coleção junto.

    Sem a coleção no hash, indexar o mesmo texto numa segunda coleção faz upsert
    e move a linha da primeira, em vez de guardar as duas.
    """
    chunk = Document(page_content="mesmo texto", metadata={"source": "a.md"})
    assert chunk_id(chunk, "colecao_um") != chunk_id(chunk, "colecao_dois")


def test_an_empty_corpus_says_so(tmp_path: Path) -> None:
    with pytest.raises(EmptyCorpus):
        index_samples(directory=tmp_path)


def test_results_carry_the_source() -> None:
    results = [
        Document(page_content="trecho um", metadata={"source": "relatorio.md", "chunk": 0}),
        Document(page_content="trecho dois", metadata={"source": "notas.txt", "chunk": 3}),
    ]
    rendered = format_results(results)
    assert "[fonte: samples/relatorio.md chunk 0]" in rendered
    assert "[fonte: samples/notas.txt chunk 3]" in rendered


def test_no_results_is_stated_not_hidden() -> None:
    assert "nenhum trecho encontrado" in format_results([])


def test_the_driver_is_added_to_the_url() -> None:
    assert sqlalchemy_url("postgresql://u:p@h:5432/d") == "postgresql+psycopg://u:p@h:5432/d"
    already = "postgresql+psycopg://u:p@h:5432/d"
    assert sqlalchemy_url(already) == already


@requires_postgres
def test_indexing_is_idempotent(corpus: Path, settings: Settings, collection: str) -> None:
    """Indexar duas vezes o mesmo corpus não cria chunk repetido."""
    from pauta.tools.retriever import get_store

    common = {"embeddings": fake_embeddings(), "collection_name": collection}
    first = index_samples(settings, directory=corpus, **common)  # type: ignore[arg-type]
    second = index_samples(settings, directory=corpus, **common)  # type: ignore[arg-type]
    assert first.chunks == second.chunks

    store = get_store(settings, **common)  # type: ignore[arg-type]
    found = store.similarity_search("latência", k=50)
    assert len(found) == first.chunks
    ids = [
        chunk_id(Document(page_content=d.page_content, metadata=d.metadata), collection)
        for d in found
    ]
    assert len(ids) == len(set(ids))


@requires_postgres
def test_a_stored_chunk_keeps_its_source(corpus: Path, settings: Settings, collection: str) -> None:
    from pauta.tools.retriever import get_store

    common = {"embeddings": fake_embeddings(), "collection_name": collection}
    index_samples(settings, directory=corpus, **common)  # type: ignore[arg-type]
    found = get_store(settings, **common).similarity_search("qualquer", k=50)  # type: ignore[arg-type]
    sources = {str(d.metadata["source"]) for d in found}
    assert sources == {"anexos/anexo.md", "notas.txt", "relatorio.md"}
    assert COLLECTION_NAME == "pauta_samples"

"""Compara modelos de embedding na recuperação do corpus real, antes de trocar.

Indexa `samples/` em memória com o chunking de produção (`load_documents` +
`split_documents`) e mede recall@k e MRR de cada candidato sobre um conjunto
de perguntas com documento-ouro conhecido. Não toca no pgvector nem no índice
de produção.

Uso: uv run python eval/benchmark_embeddings.py
"""

from dataclasses import dataclass
from sys import stdout

import numpy as np
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from pauta.config import get_settings
from pauta.tools.retriever import load_documents, samples_dir, split_documents

#: Perguntas em linguagem natural, cada uma com os documentos que a respondem.
#: A última é armadilha: não tem resposta no corpus, e o que se mede é a
#: margem entre o melhor chunk e o resto — modelo bom não tem confiança aqui.
QUERIES: list[tuple[str, frozenset[str]]] = [
    (
        "quanto custa por mês uma GPU dedicada para rodar modelo de linguagem",
        frozenset({"custos-plataforma-2026.md"}),
    ),
    ("qual é o custo por requisição da plataforma", frozenset({"custos-plataforma-2026.md"})),
    (
        "qual foi o p95 de latência medido no semestre",
        frozenset({"relatorio-latencia-2026.md", "latencia.md"}),
    ),
    ("como a equipe mediu a latência", frozenset({"relatorio-latencia-2026.md"})),
    (
        "quais os próximos passos depois do relatório de desempenho",
        frozenset({"relatorio-latencia-2026.md"}),
    ),
    ("qual o tamanho ideal de chunk para recuperação em RAG", frozenset({"notas-chunking-rag.md"})),
    ("vantagens da janela fixa com sobreposição no chunking", frozenset({"notas-chunking-rag.md"})),
    ("o que é recuperação de informação", frozenset({"recuperacao-de-informacao.md"})),
    (
        "como funciona tokenização em processamento de linguagem natural",
        frozenset({"processamento-de-linguagem-natural.md"}),
    ),
    ("política de retenção de dados da seção de compliance", frozenset()),
]

TOP_K = 5  # o RETRIEVER_TOP_K de produção


@dataclass(frozen=True)
class Candidate:
    name: str
    embeddings: OpenAIEmbeddings


def cosine_rank(query: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    """Índices dos chunks em ordem de similaridade decrescente."""
    scores = chunks @ query / (np.linalg.norm(chunks, axis=1) * np.linalg.norm(query))
    return np.argsort(-scores)


def main() -> None:
    settings = get_settings()
    documents, _ = load_documents(samples_dir())
    chunks = split_documents(documents)
    sources = [str(c.metadata["source"]) for c in chunks]
    stdout.write(f"corpus: {len(documents)} documentos, {len(chunks)} chunks\n\n")

    candidatos = [
        Candidate(
            "nomic local (LM Studio)",
            OpenAIEmbeddings(
                model="text-embedding-nomic-embed-text-v1.5",
                base_url="http://127.0.0.1:1234/v1",
                api_key=SecretStr("lm-studio"),  # o servidor local não valida
                # O LM Studio não aceita input como array de tokens; desligar o
                # truncamento por contexto faz o cliente mandar a string crua.
                check_embedding_ctx_length=False,
            ),
        ),
        Candidate(
            "text-embedding-3-small (OpenRouter, atual)",
            OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.EMBEDDING_BASE_URL,
                api_key=settings.EMBEDDING_API_KEY,  # type: ignore[arg-type]
            ),
        ),
    ]

    for cand in candidatos:
        matriz = np.array(cand.embeddings.embed_documents([c.page_content for c in chunks]))
        hits = 0
        rr_total = 0.0
        armadilha_margem = 0.0
        stdout.write(f"== {cand.name}\n")
        for pergunta, ouro in QUERIES:
            q = np.array(cand.embeddings.embed_query(pergunta))
            ordem = cosine_rank(q, matriz)
            top = [sources[i] for i in ordem[:TOP_K]]
            if not ouro:
                # Armadilha: margem entre o 1º e o 5º colocado. Margem alta em
                # pergunta sem resposta é sinal de confiança espúria.
                scores = sorted(
                    (matriz @ q / (np.linalg.norm(matriz, axis=1) * np.linalg.norm(q))),
                    reverse=True,
                )
                armadilha_margem = float(scores[0] - scores[4])
                stdout.write(f"   [armadilha] {pergunta[:60]}  margem={armadilha_margem:.3f}\n")
                continue
            acertou = bool(ouro & set(top))
            hits += acertou
            primeiro = next((n for n, s in enumerate(top, 1) if s in ouro), None)
            rr_total += 1.0 / primeiro if primeiro else 0.0
            stdout.write(f"   {'ok ' if acertou else 'FALHOU'} {pergunta[:60]}  top1={top[0]}\n")
        respondidas = len(QUERIES) - 1
        stdout.write(
            f"   recall@{TOP_K}: {hits}/{respondidas}   MRR: {rr_total / respondidas:.2f}\n"
        )
        stdout.write(f"   margem na armadilha (menor é melhor): {armadilha_margem:.3f}\n\n")


if __name__ == "__main__":
    main()

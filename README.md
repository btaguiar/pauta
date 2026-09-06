# Pauta

[![CI](https://github.com/btaguiar/pauta/actions/workflows/ci.yml/badge.svg)](https://github.com/btaguiar/pauta/actions/workflows/ci.yml)

```
Multi-agent analytical briefings with LangGraph: dynamic supervision,
a bounded critic loop, human-in-the-loop interrupts and a token budget.
```

*Pauta* is the Brazilian newsroom word for an assignment brief: the question, the
sources to check, and who reviews it. That is what this system produces.

## Status

Em construção. O grafo roda de ponta a ponta, a persistência está provada e
existe um comando para pedir um briefing. O que falta é número medido.

O que já está de pé:

- 5 nós, um nível de supervisão. O supervisor decide por output estruturado, o
  código corrige a decisão dele, e uma rota determinística assume quando o parse
  falha.
- Crítico com limite de refações, que aponta lacunas e não reescreve a resposta.
- Retomada por `thread_id` sobre checkpointer de Postgres. Um teste mata o
  processo no meio da run e prova que ela continua de onde parou. Esse teste roda
  no CI, contra um Postgres de verdade, em todo push.
- Interrupt antes da redação com `HITL_MODE=interrupt`, e retomada com feedback
  do revisor.
- Orçamento de tokens em duas camadas, entre nós e dentro do nó.
- Avaliação que lê os rótulos das 26 tarefas do golden set, repete cada tarefa,
  reporta o desvio entre repetições e grava um JSON por rodada com o commit no
  nome. Método e limitações em [EVALUATION.md](EVALUATION.md).
- 284 testes. `ruff` e `mypy --strict` limpos sobre `src`, `tests` e `eval`.

## Rodar

```
cp .env.example .env      # preencha MODEL_WORKER, MODEL_ROUTER, MODEL_CRITIC
docker compose up -d
uv run python -m pauta "vale a pena migrar de API por token para GPU dedicada?"
```

A run é durável por padrão. Se o processo cair, `python -m pauta --list` mostra
o `thread_id` e `python -m pauta --resume THREAD_ID` continua de onde parou. Com
`HITL_MODE=interrupt` o grafo congela antes da redação e espera aprovação, que
chega pelo mesmo `--resume`, com `--feedback` opcional.

## O que ainda não existe

Dito aqui antes que você procure:

- Nenhuma combinação de modelos foi medida, então o repositório não publica
  número de qualidade nenhum. `MODEL_WORKER`, `MODEL_ROUTER` e `MODEL_CRITIC`
  não têm valor padrão no código, de propósito.
- O juiz de fidelidade existe e está calibrado apenas contra casos construídos.
  Enquanto isso, nenhum número dele vale.
- Sem API HTTP e sem Dockerfile da aplicação. O compose sobe o Postgres.

As decisões de arquitetura e as sete ADRs estão em
[ARCHITECTURE.md](ARCHITECTURE.md).

## Corpus de exemplo

O retriever indexa apenas o que está em [samples/](samples/). Hoje são 6
documentos, que geram 22 chunks de até 800 tokens com 100 de sobreposição,
medidos e não estimados. Três vêm da Wikipédia em português, sob CC BY-SA 4.0.
Três foram escritos para o projeto, com números fictícios, porque o conjunto de
avaliação precisa de casos que documentação pública não oferece sob medida.
A origem de cada um está em [samples/SOURCES.md](samples/SOURCES.md).

# Pauta

```
Multi-agent analytical briefings with LangGraph: dynamic supervision,
a bounded critic loop, human-in-the-loop interrupts and a token budget.
```

*Pauta* is the Brazilian newsroom word for an assignment brief: the question, the
sources to check, and who reviews it. That is what this system produces.

## Status

Em construção. O grafo roda de ponta a ponta e a persistência está provada.
Ainda não existe um comando para pedir um briefing: o único executável do repo é
o harness de avaliação.

O que já está de pé:

- 5 nós, um nível de supervisão. O supervisor decide por output estruturado, o
  código corrige a decisão dele, e uma rota determinística assume quando o parse
  falha.
- Crítico com limite de refações, que aponta lacunas e não reescreve a resposta.
- Retomada por `thread_id` sobre checkpointer de Postgres. Um teste mata o
  processo no meio da run e prova que ela continua de onde parou.
- Interrupt antes da redação com `HITL_MODE=interrupt`, e retomada com feedback
  do revisor.
- Orçamento de tokens em duas camadas, entre nós e dentro do nó.
- 181 testes, 4 deles pulados por exigirem Postgres local. `ruff` e `mypy
  --strict` limpos sobre `src`, `tests` e `eval`.

O que ainda não existe, dito aqui antes que você procure:

- Nenhum comando para rodar um briefing. Sem API, sem Dockerfile, sem CI.
- A avaliação mede custo e latência. Qualidade ainda não: o golden set tem 26
  tarefas rotuladas e o harness ainda não lê os rótulos.
- Nenhuma combinação de modelos foi medida. `MODEL_WORKER`, `MODEL_ROUTER` e
  `MODEL_CRITIC` estão vazios de propósito, sem default no código.

Quando houver número de qualidade, ele vem de execução real e o `EVALUATION.md`
diz como foi medido.

As decisões de arquitetura e as sete ADRs estão em
[ARCHITECTURE.md](ARCHITECTURE.md).

## Corpus de exemplo

O retriever indexa apenas o que está em [samples/](samples/). Hoje são 6
documentos, que geram 22 chunks de até 800 tokens com 100 de sobreposição,
medidos e não estimados. Três vêm da Wikipédia em português, sob CC BY-SA 4.0.
Três foram escritos para o projeto, com números fictícios, porque o conjunto de
avaliação precisa de casos que documentação pública não oferece sob medida.
A origem de cada um está em [samples/SOURCES.md](samples/SOURCES.md).

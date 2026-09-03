# Pauta

```
Multi-agent analytical briefings with LangGraph: dynamic supervision,
a bounded critic loop, human-in-the-loop interrupts and a token budget.
```

*Pauta* is the Brazilian newsroom word for an assignment brief: the question, the
sources to check, and who reviews it. That is what this system produces.

## Status

Em construção. Semana 1 do roadmap: grafo mínimo de ponta a ponta.
Nada aqui foi medido ainda. Quando houver número, ele vem de execução real e
o `EVALUATION.md` diz como foi medido.

As decisões de arquitetura e as sete ADRs estão em
[ARCHITECTURE.md](ARCHITECTURE.md).

## Corpus de exemplo

O retriever indexa apenas o que está em [samples/](samples/). Hoje são 6
documentos, que geram 22 chunks de até 800 tokens com 100 de sobreposição,
medidos e não estimados. Três vêm da Wikipédia em português, sob CC BY-SA 4.0.
Três foram escritos para o projeto, com números fictícios, porque o conjunto de
avaliação precisa de casos que documentação pública não oferece sob medida.
A origem de cada um está em [samples/SOURCES.md](samples/SOURCES.md).

# Avaliação

Como o pauta é medido, o que cada número significa, e o que ainda não foi medido.

Este documento descreve o método. Ele não traz resultados: nenhuma rodada
completa foi executada até aqui, porque a combinação de modelos ainda não foi
escolhida. Quando houver número, ele entra aqui com a data, o commit e o `n`.

## O golden set

`eval/tasks.jsonl`, 26 tarefas escritas à mão. Contagem medida, não estimada:

| rótulo | tarefas |
|---|---|
| `must_contain` preenchido | 19 |
| `should_flag_uncertainty` | 11 |
| `needs_research` | 11 |
| `needs_calculus` | 10 |
| `requires_corpus` | 7 |
| `max_steps` | 26 |

As 11 tarefas com `should_flag_uncertainty` são armadilhas. Elas pedem algo que
não é conhecível, como o preço da saca de café arábica em 2030. O sucesso nelas
é recusar ou marcar a incerteza. Responder um número é o erro.

As 7 tarefas com `requires_corpus` dependem do índice sobre `samples/`. Sem
corpus indexado elas são puladas e contadas como puladas, nunca como falha.

## Métricas determinísticas

Calculadas em `eval/scoring.py`. Nenhuma delas chama modelo.

**`cobertura`.** Fração das substrings de `must_contain` que aparecem no
briefing. Caixa e acento são normalizados, então `GPU` e `gpu` são a mesma
exigência. Tarefa sem `must_contain` não entra na média, em vez de entrar
como zero.

**`incerteza_sinalizada`.** Se o briefing admite não saber, nas tarefas
marcadas como armadilha. É uma aproximação por marcador de texto, não
compreensão. Os marcadores saem do prompt do writer, que manda escrever "Não foi
possível validar X" e abrir uma seção de ressalvas. Mudar aquele prompt exige
revisar esta lista.

**`calculadora_usada`.** Se a tool `calculator` foi de fato chamada, nas tarefas
que exigem conta. O dado vem dos eventos `tool_call` que o grafo emite, não do
estado final. Um finding do analyst prova que ele escreveu algo, não que ele
calculou.

**`pesquisa_feita`.** Se a run registrou ao menos uma descoberta, nas tarefas
que exigem pesquisa.

**`dentro_do_teto`.** Se a run terminou dentro do `max_steps` da tarefa.

Toda média vem com o `_n` que a sustenta. Média sem denominador esconde que ela
veio de duas tarefas.

## Repetições e desvio

Cada tarefa roda três vezes por padrão. Uma execução só não separa melhora de
ruído.

As repetições de uma tarefa viram um score só antes da média geral. Uma tarefa
rodada três vezes não pode pesar o triplo de uma rodada uma vez.

O desvio reportado tem uma definição específica, e ela precisa estar escrita:
para cada tarefa calcula-se o desvio das suas repetições, e o número publicado é
a média desses desvios entre as tarefas. Lê-se como "quanto o score de uma
tarefa costuma variar de uma execução para outra". Tarefa rodada uma vez só não
entra nesse cálculo.

## O juiz

O que não dá para decidir por regra vai para um juiz de LLM, em `eval/judge.py`.
Ele responde uma pergunta só: toda afirmação factual do briefing está apoiada no
material que os agentes reuniram?

A escala é binária, sim ou não. Escala de 1 a 5 tem âncora instável entre
execuções e entre modelos, e a média de notas instáveis parece precisa sem ser.

O juiz vem de `JUDGE_MODEL`, que precisa apontar para um provider diferente do
executor. Modelo que julga a própria saída se prefere.

Juiz que falha devolve "não julgado", que não é o mesmo que "não sustentado".
Um juiz fora do ar não é evidência de briefing ruim.

### Calibração, e por que ela decide se o número pode ser publicado

`python eval/calibrate_judge.py` roda o juiz sobre casos rotulados à mão e
calcula o Kappa de Cohen contra esses rótulos.

Kappa, e não acurácia. Um juiz que responde sim para tudo acerta 90% de um
conjunto 90% sim sem julgar nada. Kappa desconta essa concordância esperada por
acaso, e nesse caso devolve zero. Existe teste para exatamente esse cenário.

O piso é 0,60, o "substancial" de Landis e Koch. Abaixo dele o juiz discorda
demais do humano para o número dele significar alguma coisa, e o relatório diz
isso em vez de publicar o número.

## Artefatos por rodada

Cada execução grava três arquivos em `eval/results/`, nomeados com timestamp UTC
e o SHA curto do commit:

- `bruto_*.json`: o que cada execução produziu, briefing incluído.
- `eval_*.json`: cabeçalho, métricas e itens.
- `metricas_*.json`: só os números, para plotar evolução por commit sem
  carregar texto junto.

Todos carregam um bloco `config` com os modelos, a temperatura e os limites do
grafo. Sem ele dois arquivos de commits diferentes não são comparáveis, porque
não dá para saber se a diferença veio do código ou de outro modelo. A lista do
bloco é explícita, nunca um despejo da configuração, para nenhuma chave de API
chegar a um arquivo que pode ser comitado. Existe teste que verifica isso.

Rodadas locais ficam fora do git. Só o artefato do CI é publicado.

## A matriz de modelos

Uma configuração por vez responde "como está". A pergunta cara é outra: vale
pagar modelo melhor em qual papel?

`eval/matrix.py` roda o mesmo golden set em várias configurações e põe os
resultados lado a lado, com o delta contra a linha de base. As configurações
ficam em `eval/matrix.json`, uma por linha, variando um papel de cada vez a
partir de um piso barato.

Duas hipóteses que a matriz existe para testar:

- Router barato não custa qualidade, porque o código já corrige a decisão do
  supervisor em `enforce_rules`. Se o delta de `router-melhor` for zero, essa
  correção se pagou.
- Crítico barato custa caro, porque crítico fraco tende a carimbar `verdict:
  ok`. É o modo de falha que a ADR 002 existe para pegar.

O embedding não entra na matriz. Trocá-lo no meio compararia corpus diferentes,
não modelos diferentes.

O script não escolhe nada. Ele produz o número, e a escolha vira ADR no
`ARCHITECTURE.md` mais um comentário no `.env.example` dizendo o que a
justificou.

## Como rodar

```
docker compose up -d
uv run python eval/run_eval.py --limit 5 --repeats 1     # rodada rápida
uv run python eval/run_eval.py                           # 26 tarefas, 3 repetições
uv run python eval/run_eval.py --judge                   # com o juiz
uv run python eval/calibrate_judge.py                    # calibra o juiz
uv run python eval/matrix.py --limit 5                   # compara configurações
```

Exige `.env` preenchido. `MODEL_WORKER`, `MODEL_ROUTER`, `MODEL_CRITIC` e
`EMBEDDING_MODEL` não têm valor padrão no código, de propósito.

## O que a primeira execução real mostrou

Em 2026-09-08, commit `23ce942`, o sistema falou com um provider pela primeira
vez. Foram cinco runs, todas com `MODEL_WORKER=openai/gpt-5.6-luna`,
`MODEL_ROUTER=openai/gpt-5.4-mini` e `MODEL_CRITIC=openai/gpt-5.6-terra`. O `n`
é 1 por tarefa, então nada aqui é conclusão sobre qualidade. São defeitos, e
defeito aparece com n=1.

**O orçamento por run é um teto mole.** Uma tarefa gastou 80.728 tokens contra
`BUDGET_TOKENS_PER_RUN=60000`, 35% acima. O guardrail impede novas rodadas de
tool, e não interrompe a chamada em curso nem a extração final nem o writer.
O número serve para conter, não para garantir.

**Nada limita quantas vezes o research é reconvocado.** Numa tarefa cuja
resposta não está no corpus, o ciclo research com zero descobertas seguido de
supervisor se repetiu quatro vezes, até o orçamento acabar, sem nunca chegar ao
writer. `MAX_CRITIC_LOOPS` limita o crítico; não há equivalente para a pesquisa
infrutífera, e o backstop é o orçamento, que é o recurso mais caro.

**O timeout do research está calibrado para outro modelo.** O `.env.example`
registra "research leva ~29s com 3 buscas". Com os modelos acima, uma tarefa de
busca web estourou os 90s. Recalibrar faz parte de escolher os tiers.

**O supervisor pulou o analyst numa tarefa que exige conta.** `t10` pede
confirmar um percentual e está rotulada com `needs_calculus`. A rota foi
research, critic, writer, e a aritmética saiu de cabeça, contra a regra 1 do
prompt do analista. Foi a métrica `calculadora_usada` que pegou isso, no
primeiro uso do harness.

## Limitações declaradas

O que segue não foi resolvido. Está escrito aqui porque um documento de
avaliação que esconde os próprios limites não serve para nada.

**Nenhum resultado foi produzido ainda.** A combinação de modelos não foi
escolhida, então não existe rodada completa. Todas as métricas descritas acima
têm código e teste, e valor medido nenhum.

**O conjunto de calibração do juiz é construído, não amostrado.** São 8 casos,
4 sustentados e 4 não, escritos para serem inequívocos. Um conjunto sem caso
difícil superestima a concordância. Antes de publicar qualquer número de
fidelidade, é preciso acrescentar casos reais tirados de rodadas do eval e
rotulados à mão.

**`incerteza_sinalizada` é casamento de marcador.** Um briefing que recusa com
palavras fora da lista conta como erro, e um que usa a palavra "estimativa" no
sentido errado conta como acerto. O número é um piso, não uma medida de
compreensão.

**`pesquisa_feita` mede quantidade, não qualidade.** Uma descoberta irrelevante
com fonte válida passa.

**O corpus é pequeno.** Seis documentos, 22 chunks. As 7 tarefas que dependem
dele medem recuperação num universo pequeno demais para o número generalizar.

**A matriz de modelos nunca foi executada.** O instrumento existe e está testado,
o `eval/matrix.json` está versionado em branco, e nenhuma linha dele foi rodada.
Enquanto isso, a escolha dos tiers continua em aberto e o repositório não afirma
nada sobre qual modelo vale em qual papel.

# Avaliação

Como o pauta é medido, o que cada número significa, e o que ainda não foi medido.

Este documento descreve o método e o que as execuções reais mediram até agora.
A combinação de modelos não foi escolhida e o golden set nunca rodou inteiro.
Todo número aqui vem com data, commit e `n`.

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
executor. Modelo que julga a própria saída se prefere. O gateway de chat serve
Qwen, GLM e DeepSeek, então essa exigência é uma troca de id na mesma chave.

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

Calibração rodada em 2026-09-26 com `JUDGE_MODEL=qwen3.8-max` sobre os 8 casos:
kappa 0,75, concordância crua 0,875, zero sem resposta. Acima do piso, então o
juiz pode entrar no relatório. Artefato em
`eval/results/judge_calibration_20260926T030514Z.json`.

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

Uma linha também pode fixar `MAX_CRITIC_LOOPS`, e zero desliga o crítico. Linha
que não fixa roda com o valor do `.env`. O artefato grava o valor efetivo de
cada linha em `limites`, porque o bloco `config` comum sai da última
configuração que rodou.

Três hipóteses que a matriz existe para testar:

- Router barato não custa qualidade, porque o código já corrige a decisão do
  supervisor em `enforce_rules`. Se o delta de `router-melhor` for zero, essa
  correção se pagou.
- Crítico barato custa caro, porque crítico fraco tende a carimbar `verdict:
  ok`. É o modo de falha que a ADR 002 existe para pegar.
- O crítico se paga. `sem-critico` usa os modelos da linha de base, com o
  crítico desligado. Com `tudo-barato` e `critico-melhor`, dá as três condições:
  sem crítico, crítico barato, crítico melhor. Se a diferença entre elas ficar
  abaixo do desvio, a conclusão é que a diferença não foi medida, e ela vai
  para este documento do mesmo jeito.

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

Em 2026-09-08 o sistema falou com um provider pela primeira vez. Foram seis
runs, todas com `MODEL_WORKER=openai/gpt-5.6-luna`,
`MODEL_ROUTER=openai/gpt-5.4-mini` e `MODEL_CRITIC=openai/gpt-5.6-terra`. O `n`
é 1 por tarefa, então nada aqui é conclusão sobre qualidade. São defeitos, e
defeito aparece com n=1.

O artefato salvo em `eval/results/` cobre uma das seis runs, a `t10`. Os outros
números vêm da mesma sessão, sem artefato em `eval/results/`.

Latência medida por nó, em segundos: supervisor 1,4 a 2,0; research 10 a 17
sobre o corpus e 61 com busca web; critic 1,5 a 18; writer 6 a 10. O analyst
nunca foi roteado, então não tem medida.

**O orçamento por run é um teto mole.** Duas tarefas passaram do teto: 80.728 e
64.775 tokens contra `BUDGET_TOKENS_PER_RUN=60000`, 35% e 8% acima. O guardrail
impede novas rodadas de tool, e não interrompe a chamada em curso, nem a
extração final, nem o writer. O número serve para conter, não para garantir.

**Nada limitava quantas vezes o research era reconvocado, e agora limita.**
Numa tarefa cuja resposta não está no corpus, o ciclo research com zero
descobertas seguido de supervisor se repetiu quatro vezes, até o orçamento
acabar, sem nunca chegar ao writer. `MAX_EMPTY_RESEARCH` passou a existir,
simétrico ao `MAX_CRITIC_LOOPS`: esgotado o teto, a rota vai para o writer, que
sabe dizer o que não encontrou.

**O timeout do research estava calibrado para outro modelo, e foi corrigido.**
O `.env.example` registrava "~29s com 3 buscas". Medido de novo em execução
real: 10 a 17s sobre o corpus, 61s com busca web, e uma run estourou os 90s que
eram o teto. Subiu para 180s. O do analyst continua herdado do research: o
supervisor nunca roteou para ele em nenhuma das medições, então ele nunca foi
medido.

**O supervisor pulou o analyst numa tarefa que exige conta.** `t10` pede
confirmar um percentual e está rotulada com `needs_calculus`. A rota foi
research, critic, writer, e a aritmética saiu de cabeça, contra a regra 1 do
prompt do analista. Foi a métrica `calculadora_usada` que pegou isso, no
primeiro uso do harness.

## O que as execuções de 2026-09-26 mostraram

Nove runs com `glm-5.3` nos três papéis, contra o gateway de chat atual
(Model Studio da Alibaba). O `n` é 1 por tarefa.

**As nove morreram em timeout.** Cinco numa rodada de cinco tarefas do golden
set, quatro em rodadas de tarefa única. Oito no research, com o teto de 90s, e
uma no supervisor, com o teto de 20s. Nenhuma produziu briefing. Os artefatos
são `bruto_20260926T031219Z_891c6c5` e as quatro rodadas de tarefa única em
`f9eb101`.

**As medições de latência deste gateway mudaram os tetos de timeout.** Medido
em execução real: supervisor 2,8 a 10,7s; research completou em torno de 130s
com busca web, depois de estourar os 90s herdados do gateway anterior; analyst
estourou 90s; critic estourou 45s. Os tetos passaram a supervisor 30s, research
240s, analyst 240s, critic 150s e writer 120s. O do writer é estimativa com
folga sobre a medição antiga de 6 a 10s, não medida neste gateway. Commit
`6969a47`.

## O que a matriz mostrou

A matriz rodou duas vezes em 2026-09-26.

A primeira tentativa (`matriz_20260926T113415Z_891c6c5.json`, limit=1) perdeu
as cinco linhas para `NodeTimeoutError` antes de qualquer medição. Foi o que
levantou a recalibração de timeout acima.

A segunda (`matriz_20260926T143350Z_f9eb101.json`, limit=5, 1 repetição por
linha) produziu números, com taxa de falha alta e denominadores pequenos:

| configuração | falhas | cobertura (n) | incerteza (n) | tokens média (n) | latência média |
|---|---|---|---|---|---|
| base-barata | 2 de 5 | 1,00 (2) | 1,00 (1) | 65.883 (3) | 177,0s |
| sem-critico | 1 de 5 | 0,75 (2) | 1,00 (2) | 41.051 (4) | 102,3s |
| critico-melhor | 3 de 5 | 1,00 (2) | 0,00 (0) | 48.936 (2) | 114,8s |
| worker-melhor | 4 de 5 | 0,00 (0) | 1,00 (1) | 76.502 (1) | 126,8s |
| tudo-melhor | 4 de 5 | 0,00 (0) | 1,00 (1) | 79.028 (1) | 166,4s |

O denominador de cada média está entre parênteses. Em três linhas ele é 0 ou 1.
As falhas vão de 20% a 80% das tarefas. Com esse `n` e essa taxa de falha, a
diferença entre configurações não é mensurável. O delta contra a base mede o
que sobreviveu à execução, não o efeito do modelo. `custo_usd` saiu nulo em
todas as linhas porque `COST_PER_MTOK_USD` não estava preenchido.

As três hipóteses que a matriz existe para testar continuam sem resposta. O que
a segunda rodada diz sobre o método: enquanto a taxa de falha não cair para
perto de zero e cada linha não completar as 3 repetições, qualquer comparação
de modelo sai enviesada pelo que sobreviveu.

## Limitações declaradas

O que segue não foi resolvido. Está escrito aqui porque um documento de
avaliação que esconde os próprios limites não serve para nada.

**A combinação de modelos não foi escolhida.** O que está medido está acima. A
ADR que registra a escolha não existe porque a evidência para escolher ainda
não saiu.

**Os números da matriz não sustentam comparação de modelo.** A rodada de
2026-09-26 tem 1 repetição por linha, denominadores de 0 a 4 e taxa de falha
de 20% a 80%. Está descrita acima porque esconder não ajuda. As 3 repetições
completas sobre o golden set inteiro ainda não aconteceram.

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

**A matriz mede o que o gateway deixa passar.** As configurações comparadas
usam ids do Model Studio da Alibaba. Quem reproduzir com outro provider mede
outra coisa, e o bloco `config` de cada artefato é o que torna a diferença
visível.

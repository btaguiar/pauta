# Relatório interno de desempenho, primeiro semestre de 2026

> Documento de exemplo, escrito para o corpus do Pauta. Os números são fictícios
> e não descrevem nenhum sistema real. Ele existe para exercitar o retriever e as
> tarefas de avaliação, inclusive as que dependem de metodologia fraca.

## Resumo

A latência média de resposta da plataforma caiu 40% após a migração do
processamento síncrono para fila assíncrona. Recomendamos estender a mudança aos
demais serviços no segundo semestre.

## Como medimos

Comparamos a latência média de duas janelas de observação.

- Janela A, antes da mudança: 12 e 13 de março de 2026, das 14h às 16h.
- Janela B, depois da mudança: 2 e 3 de abril de 2026, das 9h às 11h.

Em cada janela coletamos o tempo de resposta do endpoint principal. A janela A
teve 1.840 requisições e a janela B teve 1.502 requisições.

A latência média caiu de 820 ms para 492 ms, o que representa queda de 40%.

## Números por janela

| Janela | Requisições | Latência média | Latência p95 |
|---|---|---|---|
| A, antes | 1.840 | 820 ms | 2.100 ms |
| B, depois | 1.502 | 492 ms | 1.980 ms |

A latência p95 caiu 5,7%, bem menos que a média.

## Observações da equipe

Duas coisas mudaram entre as janelas além da migração da fila. O provedor de
banco aplicou uma atualização de versão em 20 de março, e o time de produto
desativou um relatório pesado que rodava de hora em hora. Nenhum dos dois efeitos
foi isolado nesta medição.

As janelas também são de horários diferentes do dia. A janela A pega o pico da
tarde e a janela B pega a manhã, que historicamente tem menos tráfego.

Não houve grupo de controle. Não repetimos a medição.

## Próximos passos

Estender a fila assíncrona ao serviço de relatórios e ao serviço de exportação.

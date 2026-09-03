# Origem dos documentos deste corpus

O retriever do Pauta indexa apenas o que está nesta pasta. Nada aqui é dado
privado, e nada aqui descreve empresa ou sistema real.

São dois tipos de documento, e a diferença importa na hora de ler um resultado de
avaliação.

## Documentos de origem externa

Texto de terceiros, redistribuído sob a licença de cada um, sem alteração de
conteúdo. O cabeçalho de cada arquivo repete a atribuição.

| Arquivo | Origem | Licença |
|---|---|---|
| `recuperacao-de-informacao.md` | Wikipédia em português, artigo "Recuperação de informação" | CC BY-SA 4.0 |
| `latencia.md` | Wikipédia em português, artigo "Latência" | CC BY-SA 4.0 |
| `processamento-de-linguagem-natural.md` | Wikipédia em português, artigo "Processamento de linguagem natural" | CC BY-SA 4.0 |

O texto foi extraído em 2026-09-02 pela API do MediaWiki, em formato puro, sem
formatação e sem edição. A licença CC BY-SA 4.0 está em
<https://creativecommons.org/licenses/by-sa/4.0/deed.pt-br>.

## Documentos escritos para o corpus

Escritos para este projeto, com números fictícios. Existem porque o conjunto de
avaliação precisa de casos que documentação pública não oferece sob medida: um
relatório cuja metodologia não sustenta a própria conclusão, dois números que não
se reconciliam, e uma afirmação que contradiz outro documento da pasta.

| Arquivo | Para que serve na avaliação |
|---|---|
| `relatorio-latencia-2026.md` | Afirma queda de 40% na latência, com janelas de horários diferentes, duas mudanças não isoladas e nenhuma repetição. Serve para checar se o agente valida a alegação em vez de repeti-la. |
| `custos-plataforma-2026.md` | Traz custo por requisição e total anual apurados por processos diferentes, que não batem entre si. Serve para checar cruzamento de números. |
| `notas-chunking-rag.md` | Compara duas abordagens de chunking e afirma que a latência de recuperação não mudou depois da migração de fila. Contradiz o relatório de latência. |

Nenhum documento desta pasta tem seção de compliance, política de retenção de
dados ou prazo de guarda. Isso é proposital: uma das tarefas de avaliação
pergunta exatamente por isso, e a resposta certa é dizer que não existe no
corpus.

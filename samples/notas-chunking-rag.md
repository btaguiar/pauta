# Notas de arquitetura: chunking para recuperação

> Documento de exemplo, escrito para o corpus do Pauta. As observações de
> desempenho são fictícias.

## Por que o tamanho do chunk importa

O chunk é a unidade que o índice recupera. Chunk grande demais traz contexto
irrelevante junto com o trecho útil e gasta tokens à toa. Chunk pequeno demais
perde a referência que dava sentido à frase e força o recuperador a trazer mais
trechos para compensar.

## Abordagem 1: janela fixa com sobreposição

Divide o texto a cada N tokens, com sobreposição de M tokens entre chunks
vizinhos. É previsível, barato de calcular e independe da estrutura do documento.

Vantagens: implementação simples, custo de indexação estável, funciona em
qualquer formato de texto.

Desvantagens: corta no meio de frase e de tabela. Uma definição que começa no fim
de um chunk e termina no início do próximo aparece partida nos dois.

## Abordagem 2: quebra por estrutura do documento

Usa os títulos, seções e parágrafos como fronteira, e só aplica corte por tamanho
dentro de uma seção que ficou grande demais.

Vantagens: o chunk tende a ser uma unidade de sentido completa, e o título vira
metadado útil para filtrar a busca.

Desvantagens: depende de o documento ter estrutura. Documentação bem formatada se
beneficia; ata de reunião e transcrição, não. O tamanho do chunk fica irregular,
o que dificulta prever o custo de contexto.

## O que observamos no corpus de teste

Em um corpus de 180 páginas de documentação técnica, a quebra por estrutura
reduziu em 18% o número de trechos irrelevantes recuperados, medido por inspeção
manual de 50 consultas. O custo de indexação subiu 4%.

A latência de recuperação ficou praticamente igual nas duas abordagens, em torno
de 900 ms por consulta, incluindo a chamada de embedding. Esse número não mudou
depois da migração para fila assíncrona, porque a recuperação não passa pela fila.

## Recomendação

Para corpus de documentação técnica com títulos consistentes, a quebra por
estrutura compensa. Para corpus heterogêneo, a janela fixa com sobreposição é
mais previsível e mais fácil de manter.

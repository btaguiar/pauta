# Recuperação de informação

> Documento de origem externa, incluído no corpus de exemplo do Pauta.
> Fonte: Wikipédia em português, artigo "Recuperação de informação".
> URL: https://pt.wikipedia.org/wiki/Recupera%C3%A7%C3%A3o_de_informa%C3%A7%C3%A3o
> Licença: Creative Commons Atribuição-CompartilhaIgual 4.0 (CC BY-SA 4.0).
> Capturado em: 2026-09-02. Texto extraído sem formatação, sem outras alterações.

Recuperação de informação (RI) é uma área de pesquisa que lida com o armazenamento de documentos e, principalmente, a recuperação da informação associada a eles a partir de uma necessidade de informação do usuário, por meio de um grupo de tarefas como a busca ad-hoc, a filtragem, o roteamento e possivelmente outras. 
Atualmente, a maior parte da pesquisa está relacionada ao uso do computador para realizar essas tarefas, sendo importantes as contribuições de pesquisadores da Computação, porém suas origens estão na área de Ciência da Informação e Biblioteconomia, onde também são feitas contribuições de relevo. Seus efeitos, antes restritos a um público mais restrito, os usuários de bibliotecas ou pesquisadores em coleções de documentos, mas seu impacto se tornou essencial ao dia a dia com o aparecimento da World Wide Web e a internet comercial e a necessidade de mecanismos de busca.
Grande parte da pesquisa é sobre como retornar a melhor resposta para uma busca por informações em documentos, busca pelos documentos propriamente ditos ou busca por metadados que descrevam documentos. A mídia pode estar disponível sob forma de textos, de sons, de imagens, vídeos ou filmes ou de dados.  
O maior desafio está na questão de bases muito grandes e infinitas na prática, como a World Wide Web. 
Um das característica que definem a recuperação da informação e a separa do acesso tradicional aos bancos de dados é que não há uma resposta definitivamente correta a uma consulta, pois tudo depende da necessidade de informação do usuário naquele instante, assim a teoria básica de Bando de Dados e  linguagens como SQL não atendem seus requisitos. 


== Histórico acadêmico ==
O termo foi criado por Calvin Mooers entre 1948 e 1950, e o campo de pesquisa é interdisciplinar, baseado em muitas áreas. Por sua abrangência ele não é muito bem compreendido, sendo abordado tipicamente sob uma ou outra perspectiva. Ele está posicionado na junção de muitos campos já estabelecidos, tais como psicologia cognitiva, arquitetura da informação, projeto da informação, comportamento da informação humana, linguística, semiótica, ciência da informação, ciência da computação, biblioteconomia e estatística.
A partir da década de 1950, alguns pesquisadores, principalmente ligados a bibliotecas e gestão de documentos, perceberam que as capacidades dos computadores poderiam ser utilizadas para gerar sistemas de índices para consultas semelhantes aos catálogos das bibliotecas. Isso depois evoluiu para sistemas que poderiam acessar os resumos e finalmente o texto completo dos documentos. 
Entre os marcos da área estão a proposta do WRU Searching Selector, talvez o primeiro sistema que tenha funcionado na prática, e o sistema SMART, apresentado por Gerard Salton, considerado o pai da Recuperação da Informação. Os experimentos Cranfield, liderados por Cyril W. Cleverdon também tiveram grande impacto na área, praticamente definindo a forma como sistemas de RI são comparados.  
Em 1992 o Departamento de Defesa dos Estados Unidos, em conjunto com o Instituto Nacional de Padrões e Tecnologia (NIST), do mesmo país, patrocinou a Text Retrieval Conference (TREC, Conferência de Recuperação de Textos) como parte do programa TIPSTER. O objetivo disto foi observar a transformação da comunidade de recuperação de informações a partir do provimento de uma infraestrutura de suporte que era necessária para tal gigantesca avaliação das metodologias de recuperação de textos.


== Objetivo ==
Os documentos são geralmente textos ou partes do texto de documentos e o principal objetivo de um sistema de RI é recuperar informação (contida nos documentos) que possa ser útil ou relevante para o usuário. Tal informação (de interesse do usuário) é normalmente chamada de necessidade de informação do usuário. Infelizmente, caracterizar a necessidade de informação do usuário não é uma tarefa simples. Considere, por um momento, a seguinte necessidade de informação de um usuário no contexto da World Wide Web (ou simplesmente Web):
"Encontre todos os documentos contendo informações sobre a doença Neoplasma Benigno de forma que: (1) O paciente com a doença possua idade inferior a 50 anos e (2) seja diabético."


=== Consultas ===
Para obter documentos de seu interesse, o usuário deverá traduzir uma necessidade de informação em uma consulta. 
Apesar de haver uma forte área de pesquisa para responder perguntas em linguagem natural, muita da pesquisa feita em Recuperação da Inforamação é baseada em uma consulta formada por uma lista de palavras-chave fornecida pelo usuário, sendo essa a forma padrão de busca, circa 2023, das interfaces do utilizador das máquinas de busca na Web .  
Uma inconveniência imediata dessa abordagem é que o uso de palavras-chave usualmente introduz uma diferença de semântica entre a intenção do usuário e o conjunto de documentos retornados. Além disso, essa diferença de semântica pode ser ampliada devido à dificuldade adicional em se lidar com textos em linguagem natural, que nem sempre são bem estruturados e podem ser semanticamente ambíguos.


=== Resultados ===
O objetivo geral de um sistema de recuperação de informação é retornar os documentos mais relevantes para o usuário naquele instante. Relevância, porém, é um termo que pode ser definido de acordo com várias formas, incluindo as relevâncias afetiva, situacional, cognitiva, tópica e algoritmica. Além disso é um conceito que muda no tempo, pois após ler um documento encontrado por um sistema de recuperação de informação, o documento seguinte pode deixar de ser relevante.
Devido a ambiguidade da língua e as necessidades específicas de diferentes usuários, a presença de documentos (textos) pouco ou não relevantes entre os documentos retornados por uma consulta é praticamente certa. Nesse cenário, o principal objetivo dos sistemas de RI é recuperar o maior número possível de documentos relevantes e o menor número possível de documentos não relevantes.
Uma forma simples de obter um conjunto de respostas para uma consulta de usuário é determinar quais documentos em uma coleção contém as palavras da consulta, no que é conhecido como full-text retrieval. Todavia, isto não é o suficiente para satisfazer ao usuário em um sistema de RI. Técnicas tradicionais para resolver esse problema incluem a classificação do documento dentro de um conjunto de tópicos pré-determinados, usando técnicas típicas da Ciência da Informação como a classificação facetada; o uso de thesaurus ou técnicas de expansão de consulta, uso de informação da rede, como o algoritmo PageRank, técnicas de sistemas de recomendação,  etc.


== As Tarefas da Recuperação da Informação ==
Entre as tarefas da recuperação de informação estão:

Busca Ad-Hoc, buscar documentos em uma coleção fixa a partir de uma consulta gerada pelo usuário
Filtragem, verificar se documentos que chegam em uma coleção atendem a uma consulta previamente cadastrada por um usuário
Roteamento, fazer a tarefa de filtragem e ordenando os documentos por relevância
Browsing ou navegação, navegação entre documentos como feito na Web.


== Relevância ==
Relevância é o "A de um B existindo entre C e D como determinado por E", sendo que:

A pode ser medida, grau, estimativa, ...;
B pode ser correspondência, utilidade, ...;
C pode ser documento, texto , informação, ...;
D pode ser consulta, pedido, ..., e
E pode ser usuário, especialista, juiz, ....
Uma definição comum de relevância para sistemas de recuperação de informação foi dada por van Rijsbergen: a medida ou grau de correspondência ou utilidade existente entre um texto ou documento e uma consulta ou requisito de infomação para uma determinada pessoa. 
Para ser eficaz na tarefa de satisfazer a necessidade de informação do usuário, os sistemas de RI devem ordenar os documentos de uma coleção de acordo com o seu grau de relevância com a consulta do usuário. 
A noção de relevância é um conceito fundamental em recuperação de informação e é um componente chave para calcular a classificação (ordenação) de documentos em um conjunto de respostas a uma consulta do usuário.a


== Principais passos ==
Operação de Consulta - envolve a especificação de um conjunto de termos, associados ou não por operadores booleanos, que representa a necessidade de informação do usuário.
Operação de Indexação - envolve a criação de estruturas de dados associados aos documentos de uma coleção. Uma estrutura de dados bastante utilizada são as listas invertidas de termos/documentos.
Pesquisa e Ordenação - envolve o processo de recuperação de documentos de acordo com a consulta do usuário e sua ordenação através de um grau de similaridade entre o documento e a consulta.


== Modelos de Recuperação de Informação ==
Para calcular uma classificação, o sistema de RI usualmente adota um modelo para representar os documentos e a consulta do usuário. Muitos modelos ou abordagens para a computação da classificação tem sido propostos ao longo dos anos, sendo três modelos considerados clássicos:

o modelo booleano,
o modelo vetorial e
o modelo probabilístico.
Esses modelos servem de base para construção de muitos outros modelos, como o modelo booleano fuzzy, a Indexação por Semântica Latente, Modelos de Linguagem e modelos baseados em redes neurais.
Um modelo de recuperação de informação necessita de:

Um conjunto de representações de documentos
Um conjunto de representações de consulta
Um arcabouço que modela documentos, consultas e seus relacionamentos, e
Uma função de ordenação que associa um número real para cada documento dada uma consulta.


=== Termos de Indexação ===
Os modelos clássicos de recuperação de informação consideram que cada documento é representado por um conjunto de palavras-chave representativas, ou termos de indexação, que são consideradas como mutuamente independentes, o que é uma simplificação. 
Como um mesmo termo pode aparecer em diferentes documentos, é necessário distinguir a ocorrência de um termo 
  
    
      
        
          k
          
            i
          
        
      
    
    {\displaystyle k_{i}}
  
 em um documento 
  
    
      
        
          d
          
            j
          
        
      
    
    {\displaystyle d_{j}}
  
 da ocorrência deste mesmo termo em outro documento 
  
    
      
        
          d
          
            l
          
        
      
    
    {\displaystyle d_{l}}
  
. Para isso, a cada par termo-documento 
  
    
      
        [
        
          k
          
            i
          
        
        ,
        
          d
          
            j
          
        
        ]
      
    
    {\displaystyle [k_{i},d_{j}]}
  
 associa-se um peso 
  
    
      
        
          w
          
            i
            j
          
        
      
    
    {\displaystyle w_{ij}}
  
. A fórmula de calcular esse peso é uma dos principais áreas de estudo dos modelos derivados dos modelos vetorial e probabilístico. Já no modelo booleano, esse valor é 0 ou 1. 
Este peso deve ser utilizado para refletir a importância do termo 
  
    
      
        
          k
          
            i
          
        
      
    
    {\displaystyle k_{i}}
  
 no documento 
  
    
      
        
          d
          
            j
          
        
      
    
    {\displaystyle d_{j}}
  
, como discutido adiante. Analogamente, a cada par termo-consulta 
  
    
      
        [
        
          k
          
            i
          
        
        ,
        q
        ]
      
    
    {\displaystyle [k_{i},q]}
  
 associa-se um peso 
  
    
      
        
          w
          
            i
          
        
        ,
        q
      
    
    {\displaystyle w_{i},q}
  
. Esses pesos quantificam a importância da palavra chave em relação as outras palavras chaves em um mesmo documento ou consulta e em relação a outras palavras chaves em outros documentos de uma coleção.


== Exemplos de sistemas de recuperação da informação ==
Biblioteca virtual de saúde - Recupera a informação de diversos periódicos, e alguns são disponibilizados online, sendo que todos estes são voltados para a area de saúde.
Domínio público - Reúne livros que já podem ser disponibilizados online, ou seja, são de domínio público
Portal Capes - Disponibiliza artigos de periódicos de varias revistas nacionais e internacionais.
Google, Bing e todos os mecanismos de busca na Web.


== Ver também ==
Data Mining
Sistemas de recomendação


== Referências ==


== Bibliografia ==
Preservação Digital e suas facetas. São Carlos: Pedro & João. 2021. ISBN 978-65-5869-327-7

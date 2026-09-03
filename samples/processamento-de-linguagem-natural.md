# Processamento de linguagem natural

> Documento de origem externa, incluído no corpus de exemplo do Pauta.
> Fonte: Wikipédia em português, artigo "Processamento de linguagem natural".
> URL: https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural
> Licença: Creative Commons Atribuição-CompartilhaIgual 4.0 (CC BY-SA 4.0).
> Capturado em: 2026-09-02. Texto extraído sem formatação, sem outras alterações.

Processamento de língua natural (PLN) é uma subárea da ciência da computação, inteligência artificial e da linguística que estuda os problemas da geração e compreensão automática de línguas humanas naturais. Sistemas de geração de língua natural convertem informação de bancos de dados de computadores em linguagem compreensível ao ser humano e sistemas de compreensão de língua natural convertem ocorrências de linguagem humana em representações mais formais, mais facilmente manipuláveis por programas de computador. Alguns desafios do PLN são compreensão de língua natural, fazer com que computadores extraiam sentido de linguagem humana ou natural e geração de língua natural.


== História ==
A história do PLN começou na década de 1950, quando Alan Turing publicou o artigo "Computing Machinery and Intelligence", que propunha o que agora é chamado de teste de Turing como critério de inteligência.
Em 1954, a experiência de Georgetown envolveu a tradução automática de mais de sessenta frases russas para o inglês. Os autores afirmaram que dentro de três ou cinco anos a tradução automática seria um problema resolvido. No entanto, os avanços reais foram muito mais lentos do que o previsto e, após o relatório ALPAC em 1966, que constatou que a pesquisa de dez anos não conseguiu satisfazer as expectativas, o financiamento para este estudo em tradução automática foi reduzido drasticamente. Poucas pesquisas em tradução automática foram conduzidas até o final dos anos 80, quando os primeiros sistemas estatísticos de tradução foram desenvolvidos.
Alguns sistemas de PLN bem sucedidos desenvolvidos nos anos 60 foram SHRDLU, um sistema de língua natural que trabalhava em "blocks worlds" com vocabulário restrito e ELIZA, uma simulação de um psicoterapeuta escrita por Joseph Weizenbaum entre 1964 e 1966. Usando pouca informação sobre o pensamento ou a emoção humana, ELIZA criava, em alguns casos, interações surpreendentemente humanas. Quando o "paciente" excedia a base de conhecimento do programa, ELIZA fornecia uma resposta genérica, por exemplo, respondendo a "Minha cabeça dói" com "Por que você diz que sua cabeça dói?".
Durante a década de 1970, muitos programadores começaram a escrever "ontologias conceituais", que estruturaram a informação do mundo real em dados compreensíveis por computadores. Exemplos são MARGIE (SCHANK, 1975), SAM (CULLINGFORD, 1978), PAM (WILENSKY, 1978), TaleSpin (MEEHAN, 1976), QUALM (LEHNERT, 1977), Politics (CARBONELL, 1979) e Plot Units (LEHNERT, 1981 ). Neste período, muitos chatterbots foram escritos, como PARRY, Racter e Jabberwacky.
Até a década de 1980, a maioria dos sistemas de PLN se baseava em conjuntos complexos de regras manuscritas. A partir do final dos anos 1980, no entanto, houve uma revolução no PLN com a introdução de algoritmos de aprendizagem automática (aprendizado de máquina) para o processamento de linguagem. Isto foi devido tanto ao aumento constante do poder computacional (ver Lei de Moore) quanto à diminuição gradual da dominância das teorias da linguística chomskyanas (como a gramática gerativa), cujos fundamentos teóricos desestimularam o tipo de corpus linguístico que está subjacente à abordagem da aprendizagem automática ao processamento da linguagem.
Alguns dos algoritmos de aprendizado de máquinas mais antigos, como as árvores de decisão, produziam sistemas de regras rígidas então semelhantes às regras existentes na escritas à mão. No entanto, a marcação de partes da fala (part-of-speech tagging) introduziu o uso de modelos ocultos de Markov para o PLN e, cada vez mais, a pesquisa se concentrava em modelos estatísticos, que tomam decisões suaves e probabilísticas baseadas na atribuição de pesos reais aos recursos que compõem dados de entrada. Os modelos de linguagem de cache, sobre os quais muitos sistemas de reconhecimento de fala agora dependem, são exemplos de tais modelos estatísticos. Esses modelos são geralmente mais robustos quando dados informações desconhecidas, especialmente entrada que contém erros (como é muito comum para dados do mundo real) e produzem resultados mais confiáveis quando integrados em sistemas maiores que compreendem múltiplas tarefas.
Muitos dos sucessos iniciais notáveis ocorreram no campo da tradução automática, devido especialmente ao trabalho de pesquisa da IBM, que desenvolveu modelos estatísticos mais elaborados. Estes sistemas foram capazes de tirar proveito de corpora textuais multilíngues existentes  produzidos pelo Parlamento do Canadá e a União Europeia como resultado de leis que exigem a tradução de todos os processos governamentais em todas as línguas oficiais dos países. No entanto, a maioria dos sistemas dependia de corpora desenvolvido especificamente para tarefas implementadas por esses sistemas, o que era (e muitas vezes continua sendo) uma grande limitação no sucesso dos mesmo. Como resultado, uma grande quantidade de pesquisa passou de quantidades de dados limitadas a métodos de aprendizagem mais eficazes.
Pesquisas recentes têm se concentrado cada vez mais em algoritmos de aprendizagem semi-supervisionados e sem supervisão. Esses algoritmos são capazes de aprender com dados que não foram anotados manualmente com as respostas desejadas ou usando uma combinação de dados anotados e não anotados. Geralmente, esta tarefa é muito mais trabalhosa do que a aprendizagem supervisionada e normalmente produz resultados menos precisos para uma quantidade específica de dados de entrada. No entanto, há uma enorme quantidade de dados não anotados disponíveis (incluindo, entre outras coisas, todo o conteúdo da World Wide Web), que muitas vezes pode compensar os resultados inferiores.


== Usando a aprendizagem automática (aprendizado de máquina) ==
Os algoritmos modernos de PLN baseiam-se na aprendizagem mecânica, especialmente na aprendizagem de máquinas estatísticas. O paradigma da aprendizagem mecânica é diferente do da maioria das tentativas anteriores de processamento da linguagem. Anteriormente, implementações de tarefas de processamento de linguagem envolviam a codificação direta de grandes conjuntos de regras. O paradigma da aprendizagem automática (ou aprendizagem automática) induz a aprendizagem automática de regras através de análises de corpora de exemplos típicos do mundo real ao invés de usar algoritmos gerais de aprendizagem (muitas vezes, embora nem sempre, baseados em inferência estatística). Um corpus (plural "corpora") é um conjunto de documentos (ou frases individuais) que foram anotados à mão com os valores corretos a serem aprendidos.
Muitas classes diferentes de algoritmos de aprendizado de máquina foram aplicadas a tarefas de PLN. Esses algoritmos tomam como entrada um grande conjunto de "recursos" que são gerados a partir de dados de entrada.
Alguns dos algoritmos mais usados, como árvores de decisão, produziam sistemas de regras rígidas semelhantes aos sistemas de regras manuscritas mais comuns. No entanto, cada vez mais, a pesquisa tem se concentrado em modelos estatísticos, que tomam decisões flexíveis e probabilísticas baseadas em agregar pesos reais a cada característica de entrada. Tais modelos têm a vantagem de poder expressar a certeza relativa de muitas respostas possíveis diferentes em vez de apenas uma, produzindo resultados mais confiáveis quando esse modelo é incluído como um componente de um sistema maior.
Os sistemas baseados em algoritmos de aprendizagem mecânica têm muitas vantagens em relação às regras produzidas manualmente:

Os procedimentos de aprendizagem usados durante a aprendizagem da máquina focam-se automaticamente nos casos mais comuns, ao passo que quando se escrevem regras à mão, não é óbvio em que sentido o esforço deve ser dirigido.
Os procedimentos de aprendizagem automática podem fazer uso de algoritmos de inferência estatística para produzir modelos que são robustos a entradas não familiares (por exemplo, contendo palavras ou estruturas que não foram vistas antes) e a entradas errôneas (por exemplo, com palavras ou palavras incorretamente omitidas). Geralmente, lidar com essas entradas de forma com regras manuscritas ou sistemas de regras manuscritas que tomam decisões suaves é extremamente trabalhoso, propenso a erros e demorado.
Sistemas baseados em aprender automaticamente as regras podem ser mais precisos simplesmente fornecendo mais dados de entrada. No entanto, os sistemas baseados em regras escritas à mão só podem ser tornados mais precisos aumentando a complexidade das regras, o que é uma tarefa muito mais difícil. Em particular, há um limite para a complexidade de sistemas baseados em regras artesanais, para além dos quais os sistemas se tornam cada vez mais incontroláveis. No entanto, a criação de mais dados para entrada em sistemas de aprendizado de máquina requer simplesmente um aumento correspondente no número de horas trabalhadas por humanos, geralmente sem aumentos significativos na complexidade do processo de anotação.
O subcampo de PLN dedicado a abordagens de aprendizagem é conhecido como aprendizagem de língua natural (NLL) e sua conferência, a CoNLL, e orgão central, o SIGNLL, são patrocinados pela ACL, reconhecendo também as suas ligações com linguística computacional e aquisição de linguagem. Quando o objetivo da pesquisa de aprendizagem de linguagem computacional é entender mais sobre aquisição de linguagem humana, ou psicolinguística, a NLL sobrepõe-se no campo relacionado de psicolinguística computacional.


== Aplicações principais ==
A listagem a seguir traz alguns dos trabalhos mais pesquisados em PLN. Note que alguns deles têm aplicações no mundo real, enquanto outras servem mais frequentemente como tarefas secundárias que são usadas para auxiliar na resolução de tarefas maiores. O que distingue essas tarefas de outras tarefas potenciais e reais de PLN não é apenas o volume de pesquisa dedicado a elas, mas o fato de que para cada uma há tipicamente uma definição de problema bem especificada, uma métrica padrão para avaliar a tarefa, corpora padrão em que a tarefa pode ser avaliada e as competições dedicadas à tarefa específica.
Sumarização automática

Produz um resumo legível de uma parte do texto. Muitas vezes usado para fornecer resumos de texto de um tipo conhecido, como artigos na seção financeira de um jornal.
Resolução de correferência

Dada uma frase ou um pedaço maior de texto, determina quais palavras ("menções") se referem aos mesmos objetos ("entidades"). A resolução da anáfora é um exemplo específico dessa tarefa e está especificamente preocupada em combinar os pronomes com os substantivos ou nomes aos quais eles se referem. A tarefa mais geral da resolução de correferência também inclui a identificação dos chamados "relacionamentos de ponte" envolvendo expressões de referência. Por exemplo, numa frase como "Ele entrou na casa de João pela porta da frente", "a porta da frente" é uma expressão de referência e a relação da ponte a ser identificada é o fato de que a porta a ser referida é a porta da frente de John's (ao invés de alguma outra estrutura que também pode ser referida).
Análise do Discurso

Esta rubrica inclui uma série de tarefas relacionadas. Uma tarefa é identificar a estrutura discursiva do texto conectado, isto é, a natureza das relações discursivas entre sentenças (por exemplo, elaboração, explicação, contraste). Outra possível tarefa é reconhecer e classificar os atos de fala em um pedaço de texto (por exemplo, pergunta sim-não, pergunta de conteúdo, frase, afirmação, etc).
Máquina de tradução

Traduzir automaticamente texto de uma linguagem humana para outra. Este é uma das tarefas mais difíceis e faz parte de um tipo de problemas conhecidos como "AI-complete", ou seja, exigindo todos os diferentes tipos de conhecimento que os humanos possuem (gramática, semântica, fatos sobre o mundo real, etc) para resolvê-lo adequadamente.
Segmentação morfológica

Separa palavras em morfemas individuais e identifica classes de morfemas. A dificuldade desta tarefa depende muito da complexidade da morfologia (isto é, da estrutura das palavras) da linguagem que está sendo considerada. O inglês possui uma morfologia bastante simples, especialmente a morfologia flexional, e portanto é possível ignorar esta tarefa inteiramente e simplesmente modelar todas as formas possíveis de uma palavra (por exemplo, "open, opens, opened, opening") como palavras separadas. Em línguas como o turco ou o Meitei, uma língua indiana fortemente aglutinada, no entanto, tal abordagem não é possível, uma vez que cada entrada do dicionário tem milhares de formas de palavras possíveis.
Reconhecimento de entidade nomeada (NER)

Dado um fluxo de texto, determina quais são itens no mapa de texto para nomes próprios, como pessoas ou locais e qual é o tipo de cada nome (por exemplo, pessoa, local, organização). Embora a capitalização possa ajudar a reconhecer entidades nomeadas em idiomas como o inglês, essas informações podem não ajudar a determinar o tipo de entidade nomeada e, em alguns casos, são imprecisas ou insuficientes. Por exemplo, a primeira palavra de uma frase também é capitalizada e as entidades nomeadas muitas vezes abrangem várias palavras, com apenas algumas delas são capitalizadas. Além disso, muitas outras linguagens em scripts não-ocidentais (por exemplo, chinês ou árabe) não têm nenhuma capitalização e mesmo as línguas com maiúsculas podem não distinguir nomes. Por exemplo, o alemão capitaliza todos os substantivos, independentemente de se referirem a nomes, e o francês e o espanhol não capitalizam nomes que servem como adjetivos.
Geração de língua natural

Converte informações de bancos de dados de computador ou intenções semânticas em linguagem humana legível.
Compreensão da língua natural

Converte pedaços de texto em representações mais formais, como estruturas de lógica de primeira ordem, que são mais fáceis de manipular pelos programas de computador. A compreensão da língua natural envolve a identificação da semântica pretendida a partir da múltipla semântica possível que pode ser derivada de uma expressão de língua natural que geralmente toma a forma de notação organizada de conceitos de linguagens naturais. Entretanto, introdução e criação de linguagem metamodelo e ontologia são eficientes soluções empíricas. Uma formalização explícita da semântica de línguas naturais sem confusões com suposições implícitas como closed-world assumption (CWA) versus open-world assumption ou subjetiva Sim / Não versus objetivo Verdadeiro / Falso é esperada para a construção de uma base de formalização semântica.
Reconhecimento óptico de caracteres (OCR)

Dada uma imagem que representa o texto impresso, determina o texto correspondente.
Marcação de classe gramatical

Dada uma sentença, determina a classe gramatical de cada palavra. Muitas palavras, especialmente as comuns, podem servir como múltiplas partes do discurso. Em inglês, por exemplo, "book" pode ser um substantivo ("the book on the table") ou verbo ("book a flight"); "Set" pode ser um substantivo, verbo ou adjetivo; E "out" pode ser qualquer um de pelo menos cinco diferentes partes da fala. Algumas línguas têm mais ambiguidade do que outras. As línguas com pouca morfologia flexional, como o inglês, são particularmente propensas a tal ambiguidade. O chinês é propenso a tal ambiguidade porque é uma língua tonal durante a verbalização. Tal inflexão não é facilmente transmitida através das entidades empregadas dentro da ortografia para transmitir o significado pretendido.
Análise sintática (Parsing)

Determina a árvore de análise (análise gramatical) de uma frase. A gramática para as linguagens naturais é ambígua e frases típicas têm múltiplas análises possíveis. Na verdade, surpreendentemente, para uma frase típica pode haver milhares de análises em potencial (a maioria dos quais parecerá completamente absurda para um ser humano).
Respostas a perguntas

Dada uma questão de linguagem humana, determina sua resposta. As perguntas típicas têm uma resposta correta específica (como "Qual é o capital do Canadá?"), mas às vezes perguntas abertas também são consideradas (como "Qual é o significado da vida?"). Trabalhos recentes têm analisado questões ainda mais complexas.
Extração de relacionamento

Identifica as relações entre entidades nomeadas (por exemplo, quem é casado com quem) com base em textos.
Quebra de frases (sentence boundary disambiguation)

Encontra os limites da frase em um pedaço de texto. Os limites de frases são normalmente marcadas por pontos ou outros sinais de pontuação, mas esses mesmos caracteres podem servir outros propósitos.
Análise de subjetividade (sentiment analysis ou opinion mining)

Extrai informações subjetivas geralmente de um conjunto de documentos, muitas vezes usando revisões online para determinar a "polaridade" sobre objetos específicos. É especialmente útil para identificar tendências da opinião pública nas mídias sociais, para fins de marketing.
Reconhecimento de fala

Dado um clipe de som de uma pessoa ou pessoas falando, determina a representação textual do discurso. É o oposto da síntese de fala e é uma das áreas mais difíceis, conhecida como "AI-complete". Na fala natural quase não há pausas entre palavras sucessivas, por isso a segmentação de fala é uma subtarefa necessária de reconhecimento de fala. Nota-se também que, na maioria das linguagens faladas, os sons que representam letras sucessivas se misturam entre si em um processo denominado coarticulação, de modo que a conversão do sinal analógico em caracteres discretos pode ser um processo muito difícil de ser realizado.
Segmentação de fala

Dado um clipe de som de uma pessoa ou pessoas falando, separa-o em palavras. Uma subaplicação de reconhecimento de fala e normalmente agrupada com ele.
Análise morfológica e reconhecimento de tópicos

Dado um pedaço de texto, separa-o em segmentos cada um dos quais é dedicado a um tópico e identifica o tópico do segmento.
Análise morfológica e segmentação de palavras

Separa um pedaço de texto contínuo em palavras separadas. Para uma língua como o inglês, isso é bastante trivial, uma vez que as palavras são normalmente separadas por espaços. No entanto, algumas línguas escritas como chinês, japonês e tailandês não marcam limites de palavras de tal forma, e nessas línguas segmentação de texto é uma tarefa significativa que requer conhecimento do vocabulário e morfologia das palavras na língua.
Desambiguação

Muitas palavras têm mais de um significado, assim temos que selecionar o significado que faz mais sentido no contexto. Para este problema, em geral é dada uma lista de palavras e sentidos de palavras associadas de um dicionário ou recurso online, como o WordNet.
Em alguns casos, conjuntos de tarefas relacionadas são agrupados em subcampos de PLN que são frequentemente considerados separadamente da PLN como um todo, como os exemplos a seguir:
Recuperação de informação (IR)

Trata-se de armazenar, pesquisar e recuperar informações. É um campo separado dentro da ciência da computação (mais perto de bancos de dados), mas a IR se baseia em alguns métodos PLN (por exemplo, stemming). Algumas pesquisas e aplicações atuais procuram preencher a lacuna entre IR e PLN.
Extração de informação (IE)

Trata-se, em geral, da extração de informação semântica a partir do texto. Abrange tarefas como reconhecimento de entidade mencionada, resolução de correferência e de relacionamento de extração, etc.
Processamento de voz  Abrange reconhecimento de fala, síntese de fala e tarefas relacionadas.
Outras tarefas incluem:
Identificação na língua materna
Stemização
Simplificação do texto
Síntese de fala
Revisão de texto
Pesquisa em língua natural
Expansão da consulta
Pontuação de ensaio automatizado
Truecasing


== Estatística ==
Artigo principal: Gramática estocástica
O processamento estatístico em língua natural utiliza métodos estocásticos, probabilísticos e estatísticos para resolver algumas das dificuldades discutidas acima, especialmente aquelas que surgem porque frases mais longas são muito ambíguas quando processadas com gramáticas realistas, produzindo milhares ou milhões de análises possíveis. Métodos de desambiguação envolvem muitas vezes o uso de corpora e modelos de Markov. O projeto ESPRIT P26 (1984-1988), liderado pelo CSELT, explorou o problema do reconhecimento de fala comparando abordagem baseada em conhecimento e estatística: o resultado escolhido foi um modelo completamente estatístico. Um dos primeiros modelos de compreensão estatística da língua natural foi introduzido em 1991 por Roberto Pieraccini, Esther Levin e Chin-Hui Lee, da Bell Laboratories. O PLN compreende todas as abordagens quantitativas para processamento automatizado de linguagem, incluindo modelagem probabilística, teoria da informação e álgebra linear. A tecnologia para o PLN estatístico vem principalmente da aprendizagem automática e da mineração de dados, que são campos de inteligência artificial que envolvem o aprendizado a partir de dados.


== Avaliação ==
O objetivo da avaliação do PLN é uma medida de uma ou mais qualidades de um algoritmo ou de um sistema a fim de determinar se o algoritmo atende às metas dos projetistas ou o sistema de atendimento às necessidades de seus usuários. Investigação na avaliação PLN  tem ganhado atenção, porque a definição de critérios de avaliação é uma forma de especificar precisamente problemas do PLN. Uma métrica de avaliação de PLN em um sistema algorítmico permite a integração da compreensão de linguagem e geração de linguagem. Um conjunto preciso de critérios de avaliação, que pode ser aplicado principalmente a avaliações métricas, podendo permitir que várias equipes comparem suas soluções para um determinado problema do PLN.


== Cronologia da avaliação ==
Em 1983, iniciou-se o Projecto Esprit P26, que avaliou as Tecnologias da Fala (incluindo tópicos gerais como Sintaxe e Semântica) comparando as abordagens baseadas em regras com as estatísticas.
Em 1987, a primeira campanha de avaliação de textos escritos parece ser uma campanha dedicada à compreensão da mensagem (Pallet, 1998).
O projeto Parseval / GEIG comparou gramáticas de frase-estrutura (Black 1991).
Houve uma série de campanhas no projeto Tipster sobre tarefas como resumo, tradução e pesquisa (Hirschman 1998).
Em 1994, na Alemanha, o Morpholympics comparou marcadores morfológicos alemães.
As campanhas de Senseval & Romanseval foram realizadas com os objetivos de desambiguação semântica.
Em 1996, a campanha Sparkle comparou os analisadores sintáticos em quatro idiomas diferentes (inglês, francês, alemão e italiano).
Na França, o projeto Grace comparou um conjunto de 21 marcadores para o francês em 1997 (Adda 1999).
Em 2004, durante o projeto Technolangue / Easy, foram comparados 13 analisadores para o francês.
A avaliação em larga escala dos analisadores de dependência foi realizada no contexto das tarefas compartilhadas do CoNLL em 2006 e 2007.
Na França, no âmbito do projecto ANR-Passage (final de 2007), foram comparados 10 analisadores para o francês.
Em Itália, a campanha EVALITA foi realizada em 2007, 2009, 2011 e 2014 para comparar várias ferramentas de PLN e de voz para o site italiano - EVALITA.


== Diferentes tipos de avaliação ==
Avaliação intrínseca vs. extrínseca
A avaliação intrínseca considera um sistema PNL isolado e caracteriza seu desempenho em relação a um resultado padrão-excelência, conforme definido pelos avaliadores. A avaliação extrínseca, também chamada de avaliação em uso, considera o sistema PLN em um cenário mais complexo como um sistema embutido ou uma função precisa para um usuário humano. O desempenho extrínseco do sistema é então caracterizado em termos de utilidade em relação à tarefa global do sistema estranho ou do utilizador humano. Por exemplo, considere um analisador sintático que é baseado na saída de alguma parte do tagger de fala (POS). Uma avaliação intrínseca executaria o marcador POS em dados estruturados e compararia a saída do sistema do marcador POS com a saída padrão ouro. Uma avaliação extrínseca executaria o analisador com algum outro marcador POS e, em seguida, com o marcador POS novo e compara a precisão de análise.
Caixa preta vs. Avaliação da caixa de vidro
A avaliação em caixa preta requer que alguém execute um sistema PLN em um conjunto de dados de amostra e para medir uma série de parâmetros relacionados com a qualidade do processo, como velocidade, confiabilidade, consumo de recursos e, principalmente, a qualidade do resultado, como a precisão da anotação de dados ou a fidelidade de uma tradução. A avaliação da caixa de vidro examina a concepção do sistema; Os algoritmos que são implementados, os recursos linguísticos que utiliza, como o tamanho do vocabulário ou a expressão definida de cardinalidade. Dada a complexidade dos problemas da PLN, muitas vezes é difícil prever o desempenho apenas com base na avaliação da caixa de vidro; Mas este tipo de avaliação é mais informativo no que diz respeito à análise de erros ou desenvolvimentos futuros de um sistema.
Automática vs. avaliação manual
Em muitos casos, procedimentos automáticos podem ser definidos para avaliar um sistema de PLN, comparando sua saída com o padrão de excelência. Embora o custo de reproduzir o padrão de excelência possa ser bastante elevado, avaliação automática de bootstrapping sobre os mesmos dados de entrada pode ser repetida quantas vezes for necessário sem custos adicionais desordenados. No entanto, para muitos problemas de PLN a definição precisa de um padrão de excelência é uma tarefa complexa e pode se revelar impossível quando o acordo inter-anotador é insuficiente. A avaliação manual é melhor realizada por juízes humanos instruídos para estimar a qualidade de um sistema, ou mais frequentemente de uma amostra de sua produção, com base em uma série de critérios. Embora, graças à sua competência linguística, os juízes humanos possam ser considerados como a referência para uma série de tarefas de processamento de linguagem, há também uma variação considerável em suas classificações. É por isso que a avaliação automática é, por vezes, referida como avaliação objetiva enquanto a avaliação humana é perspectiva.


== Padronização ==
Um subcomitê ISO está trabalhando para facilitar a interoperabilidade entre recursos lexicais e programas PLN. O subcomitê faz parte do ISO / TC37 e é chamado ISO / TC37 / SC4. Alguns padrões ISO já estão publicados, mas a maioria deles está em construção, principalmente na representação de léxico (ver LMF), anotação e registro de categoria de dados.


== Ferramentas ==
Expert System S.p.A.
General Architecture for Text Engineering
Modular Audio Recognition Framework
Natural Language Toolkit (NLTK): uma biblioteca em Python
OpenNLP


== Veja também ==
Mineração de texto biomédica
Processo de processamento composto
Revisão assistida por computador
Linguagem natural controlada
Processamento linguístico profundo
Auxílio à leitura de línguas estrangeiras
Auxílio à escrita em língua estrangeira
Tecnologia da linguagem
A alocação de Dirichlet Latente (LDA)
Indexação semântica latente
Lista de ferramentas de processamento de língua natural
Mapa de LRE
Programação em língua natural
Reificação (linguística)
Dobradura semântica
Sistema de diálogo falado
Vetor do Pensamento
Pesquisa Transderivacional
Word2vec


== Referências ==


== Leitura adicional ==
Steven Bird, Ewan Klein, and Edward Loper (2009). Natural Language Processing with Python. O'Reilly Media. ISBN 978-0-596-51649-9.
Daniel Jurafsky and James H. Martin (2008). Speech and Language Processing, 2nd edition. Pearson Prentice Hall. ISBN 978-0-13-187321-6.
Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008). Introduction to Information Retrieval. Cambridge University Press. ISBN 978-0-521-86571-5. Official html and pdf versions available without charge.
Christopher D. Manning and Hinrich Schütze (1999). Foundations of Statistical Natural Language Processing. The MIT Press. ISBN 978-0-262-13360-9.
David M. W. Powers and Christopher C. R. Turk (1989). Machine Learning of Natural Language. Springer-Verlag. ISBN 978-0-387-19557-5.

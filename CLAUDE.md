# CLAUDE.md — regras deste repositório

Projeto: **Pauta**. Briefings analíticos multi-agente com LangGraph.
O *o quê* e o *por quê* estão em `DOSSIE.md`. Leia o dossiê inteiro antes da primeira tarefa de cada sessão. Este arquivo define *como se comportar aqui*.

---

## Regras duras (nunca quebre sem me perguntar)

1. **Nenhum modelo hardcoded.** Todo LLM vem de `get_model(role)` em `src/pauta/models.py`, que lê `MODEL_ROUTER`, `MODEL_CRITIC` e `MODEL_WORKER` do ambiente. Se você escrever o nome de um modelo dentro de um agente, está errado.
2. **Nenhuma credencial real, chave de API ou URL de produção no código ou nos testes.** Valores de desenvolvimento local que espelham o `docker-compose.yml` podem constar do `.env.example`, desde que não sirvam fora da máquina. Em produção, tudo vem do ambiente. O valor no `.env.example` e no `docker-compose.yml` são idênticos, sempre: se um mudar, o outro muda no mesmo commit. `.gitignore` com `.env` antes de qualquer código de configuração.
3. **Não invente assinatura de API.** LangGraph 1.2 mudou coisas (timeout e retry em `add_node`, drain cooperativo, `stream(version="v2")`). Antes de usar qualquer uma dessas, confira na documentação da versão pinada. Se não conseguir confirmar, pare e me pergunte em vez de chutar o nome do parâmetro.
4. **Teste junto com o código, no mesmo commit.** Não existe "escrevo os testes depois".
5. **Testes não chamam LLM real.** Use o `FakeChatModel` de `tests/fakes.py`. Se um teste precisa de chave, ele está no lugar errado e vai para `eval/`.
6. **Nada de `unittest.mock` para simular modelo.** O fake é explícito e determinístico.
7. **Não expanda o escopo.** São 5 nós no grafo. Subgrafo novo, agente novo ou tool nova só entram com justificativa medida no eval, e depois de me perguntar.
8. **Não instale dependência sem me avisar** qual e por quê. Pin da minor no `pyproject.toml`, lock comitado.

---

## Decisões abertas (não escolha sozinho)

Estas três estão pendentes no dossiê. Se o trabalho esbarrar em alguma, **pare e me pergunte**, não decida:

- **ADR 005:** Chroma embedded ou pgvector. Precisa estar decidida e escrita em `ARCHITECTURE.md` antes de codar o retriever.
- **Tiers de modelo:** qual modelo em `MODEL_WORKER`, `MODEL_ROUTER`, `MODEL_CRITIC`.
- **Embedding:** qual modelo em `EMBEDDING_MODEL`.

---

## Ordem de trabalho

Siga o roadmap da seção 10 do `DOSSIE.md`, semana por semana, item por item. Não pule para a semana 2 com itens da 1 em aberto. Ao terminar cada semana, rode o DoD descrito lá e me mostre o resultado antes de seguir.

---

## Convenções de código

- Python 3.11+, `uv` para dependências, `ruff` para lint e format, `mypy` para tipos.
- `mypy` limpo é requisito, não meta. O estado do grafo é `TypedDict` com reducers: é a única parte que o checker consegue proteger, então não a desperdice com `Any`.
- Type hints em toda função pública. Docstring curta só onde o nome não basta.
- Pydantic para tudo que cruza fronteira (API, output estruturado de LLM, config).
- Nada de `print`. Log estruturado em JSON via `src/pauta/observability.py`, com `run_id`, `thread_id`, `node`, `iteration`, `tokens_used`, `latency_ms`.
- Constante mágica não fica solta no meio do código: vai para `config.py`, como na seção 8.3 do dossiê.

## Estrutura

Respeite a árvore da seção 7 do dossiê. Arquivo novo fora dela precisa de justificativa. Um módulo, uma responsabilidade: o `builder.py` monta o grafo e não define prompt; o agente define comportamento e não instancia cliente.

## Erros

Todo caminho de falha tem tratamento explícito e emite evento `error` no stream. Falha visível é sempre melhor que run travada em silêncio. Retry só para erro de rede e rate limit, nunca para erro de validação de schema.

## Commits

Pequenos, um assunto por commit, mensagem no imperativo e em inglês (`add supervisor router fallback`). Não junte refactor com feature. Não comite `eval/results/` gerado em teste local, só o do nightly.

---

## Como falar comigo

- Antes de uma tarefa grande, me mostre o plano em até dez linhas e espere confirmação.
- Se encontrar contradição entre este arquivo, o `DOSSIE.md` e o que eu pedi no chat, **aponte a contradição** em vez de escolher em silêncio.
- Se algo do dossiê estiver tecnicamente errado na versão pinada, diga. O dossiê não é sagrado, mas mudança nele é decisão minha.
- Não me elogie por perguntas. Vá direto ao ponto.
- Não anuncie o que vai fazer e depois faça. Faça, e me diga o que mudou.

---

## Texto voltado ao público

README, `ARCHITECTURE.md`, `EVALUATION.md`, mensagens de erro da API e legendas do demo seguem estas regras:

- **Sem travessão.** Use vírgula, dois pontos ou frase nova.
- Frases curtas e diretas. Sem "vale notar", "é importante ressaltar", sem "não é X, é Y".
- Números medidos, nunca estimados sem dizer que são estimativa.
- Limitação declarada é feature do documento, não fraqueza. O que não foi validado aparece escrito.

---

## Definição de pronto, por tarefa

Uma tarefa só está pronta quando: o código roda, os testes passam, `ruff` e `mypy` estão limpos, o comportamento novo tem teste, e a documentação afetada (README ou ADR) foi atualizada no mesmo commit.

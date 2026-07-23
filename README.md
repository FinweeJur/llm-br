# llm-br

Camada LLM plugável **local-first** dos projetos FinweeJur. Ollama por padrão, nuvem quando precisa: Claude, DeepSeek e Maritaca (Sabiá).

Extraída de três implementações que já existiam e faziam a mesma coisa de jeitos diferentes — [YGG](https://github.com/FinweeJur/ygg) (`core/llm`), [Controle Popular — Congresso](https://github.com/FinweeJur/controle-popular-congresso) (`etl/llm`) e Vaire (`vaire/llm.py`). A lib é a **união** delas, não uma cópia de nenhuma.

## Instalação

```bash
pip install git+https://github.com/FinweeJur/llm-br@main
```

Para usar Claude, o SDK é opcional: `pip install "llm-br[anthropic] @ git+..."`.

## Uso

```python
from llm_br import get_llm

llm = get_llm()                       # sem config nenhuma: Ollama local
print(llm.gerar_texto("Resuma em uma frase: ..."))
```

Extração estruturada, que é o caso de ETL e classificador:

```python
dados = llm.gerar_json('Extraia {"nome":..., "uf":...} de: "Fulano, MG"')
```

Conversa com streaming, que é o caso de tela de chat:

```python
from llm_br import get_llm, Mensagem

for pedaco in get_llm().stream([Mensagem("user", "Olá")]):
    print(pedaco, end="")
```

## Configuração

| Variável | Default | Para quê |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` · `anthropic` · `deepseek` · `maritaca` · `openai_compat` · `auto` |
| `LLM_MODEL` | por provedor | nome do modelo |
| `LLM_BASE_URL` | por provedor | endpoint (só para provedor customizado) |
| `LLM_API_KEY` | vazio | chave da nuvem |
| `LLM_EMBED_MODEL` | `nomic-embed-text` | modelo de embeddings (Ollama) |

`LLM_PROVIDER=auto` sobe para a nuvem **só se houver chave**; senão fica no local.

Quem não usa variável de ambiente passa tudo explicitamente — é assim que o YGG liga a lib às settings do Django:

```python
get_llm("deepseek", api_key=settings.DEEPSEEK_API_KEY)
```

## O contrato

Dois níveis. Um provedor precisa implementar só o primeiro:

| Nível | Métodos | Para quê |
|---|---|---|
| **Tarefa** | `gerar_texto`, `gerar_json` | ETL, classificação, extração |
| **Conversa** | `chat`, `stream` | telas de chat |
| Opcionais | `chat_com_tools`, `embeddings` | agentes, RAG |

Quem implementa só `gerar_texto` **continua utilizável numa tela de chat**: `chat` e `stream` têm implementação padrão que achata o histórico. Consulte capacidades reais com `llm.suporta("stream" | "tools" | "embeddings" | "json_nativo")`.

Todo provedor expõe `identificacao` (`"ollama:llama3.1"`). Grave isso junto de qualquer análise: **comparar pontuações produzidas por modelos diferentes não é válido**, e sem o registro não dá para saber quais reprocessar.

## Decisões que valem explicar

**Não usa LiteLLM.** DeepSeek, Maritaca e praticamente toda nuvem falam `/v1/chat/completions` — o `OpenAICompatProvider` cobre todas em ~70 linhas, que é quase todo o ganho que o LiteLLM traria. Se um dia aparecer um provedor exótico, ele entra como mais um adapter atrás do mesmo contrato, inclusive um que delegue ao LiteLLM, sem que nada mude para quem consome.

**Ollama por HTTP, sem o SDK.** Uma dependência a menos, e é a implementação que rodou o benchmark de 30 proposições do Congresso.

**`gerar_json` usa decodificação restrita quando existe** (`format: "json"` no Ollama, `response_format` nas APIs OpenAI): o modelo fica *impedido* de emitir token que quebre o JSON, em vez de só ser instruído a se comportar. `extrair_json` continua como rede de segurança, porque modelo menor ainda embrulha resposta em ```json``` ou emenda uma frase antes.

**`embeddings` devolve `None` em vez de levantar.** RAG é recurso opcional; o chamador precisa poder cair na busca por palavra-chave se o modelo não estiver baixado.

**401 não é retentado.** Chave inválida não melhora esperando — falha em 1 chamada, não em 3.

**A lib não conhece suas ferramentas.** `chat_com_tools` recebe o executor injetado; o registro (`llm_br.tools`) é só o mecanismo.

## Desenvolvimento

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

52 testes, sem rede — tudo simulado.

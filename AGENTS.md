# AGENTS.md — llm-br

Orientação para uma IA (ou pessoa) que chega a este código para debugar ou expandir. Lê isto antes de mexer.

## O que é

Camada LLM plugável **local-first** dos projetos FinweeJur. Um contrato único, vários provedores: Ollama (local, padrão), Claude, DeepSeek, Maritaca. Pública, MIT.

**É dependência de três projetos ao mesmo tempo** — `ygg`, `controle-popular-congresso`, `Vaire`. Um bug aqui quebra os três. Por isso a CI roda em 3.10 e 3.12, e os consumidores fixam a **tag `v0.1.0`**, não `@main` — ver "Versionamento" abaixo.

## De onde veio (importa para não "simplificar" errado)

É a **união** de três implementações que já existiam e faziam a mesma coisa de jeitos diferentes: `core/llm` do Ygg (tipos, streaming, tool-calling), `etl/llm` do Congresso (config por ambiente, provider OpenAI-compat, `gerar_json` endurecido) e `vaire/llm.py` (embeddings, fallback `auto`). Cada uma tinha algo que as outras não tinham. **Se você olhar um método e pensar "isso é redundante", provavelmente é a peça que só um dos três tinha.** Não remova sem checar os três consumidores.

## Mapa dos arquivos

```
llm_br/
  base.py                  Contrato (LLMProvider, Mensagem, RespostaLLM) +
                           extrair_json() + com_retentativa(). O coração.
  __init__.py              get_llm() — a fábrica. Resolve provider por
                           argumento > ambiente > default local-first.
  tools.py                 Registro de ferramentas (tool-calling). A lib
                           conhece o PROTOCOLO; as ferramentas são do app.
  providers/
    ollama.py              HTTP direto (sem SDK). /api/generate e /api/chat.
    openai_compat.py       DeepSeek + Maritaca + qualquer /v1/chat/completions.
    anthropic.py           SDK oficial da Anthropic.
tests/                     60 testes, sem rede (tudo simulado com mock).
```

## O contrato, em dois níveis

Um provedor implementa ao menos o nível **tarefa**; o nível **conversa** tem default na base.

- **Tarefa** — `gerar_texto(prompt, system=, temperatura=)`, `gerar_json(...)`. Uso de ETL/classificação.
- **Conversa** — `chat(mensagens)`, `stream(mensagens)`. Uso de tela de chat. **Default na base**: achata o histórico e chama `gerar_texto`, então um provedor só-de-tarefa continua utilizável numa tela.
- **Opcionais** — `chat_com_tools(...)` (recebe o executor injetado — a lib não conhece as ferramentas), `embeddings(textos)` (devolve `None` em vez de levantar; RAG é opcional).

Cheque capacidade real com `provider.suporta("stream" | "tools" | "embeddings" | "json_nativo")`, não com `isinstance`.

`provider.identificacao` (`"ollama:llama3.1"`) — **grave isto junto de qualquer análise.** Comparar pontuação entre modelos diferentes não é válido; sem o registro não dá para saber o que reprocessar.

## Decisões que têm motivo (não desfaça sem novo argumento)

- **Sem LiteLLM.** DeepSeek/Maritaca/quase toda nuvem falam `/v1/chat/completions`; `openai_compat.py` cobre todas em ~70 linhas. Um adapter sobre LiteLLM cabe depois atrás do mesmo contrato, sem tocar em quem consome.
- **Ollama por HTTP, sem o SDK `ollama`.** Uma dependência a menos; é a implementação que rodou o benchmark do Congresso.
- **`gerar_json` usa decodificação restrita quando existe** (`format:"json"` no Ollama, `response_format` no OpenAI-compat) — o modelo fica *impedido* de quebrar o JSON, não só instruído. `extrair_json` é a rede de segurança para modelo menor que embrulha em ```` ```json ````.
- **401 não é retentado** (`erro.reententavel = False`) — chave inválida não melhora esperando.
- **404 do Ollama vira `rode: ollama pull <modelo>` e não é retentado** — baixar modelo é ação humana.

## Armadilhas conhecidas

- **`com_retentativa` NÃO usa `dormir=time.sleep` como default de parâmetro.** Default de parâmetro é avaliado na importação do módulo, o que congela a referência e faz `patch("llm_br.base.time.sleep")` não ter efeito — a retentativa ficaria não-testada. É resolvido dentro da função (`dormir = dormir or time.sleep`). Já mordeu uma vez.
- **`base_url` com `/v1`.** A lib monta `/v1/chat/completions` a partir da **raiz**. Se o config do consumidor já grava `https://api.deepseek.com/v1`, some `/v1/v1/` e toda chamada falha. O Vaire normaliza isso em `_base_url_para_lib()`; replicar em qualquer consumidor novo que leia URL de config.

## Como rodar e testar

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,anthropic]"
.venv/Scripts/python -m pytest        # 60 testes, sem rede, < 1s
```

Os testes simulam `requests.post` e o SDK — nenhum toca a rede. Ao adicionar comportamento HTTP, simule por `patch("requests.post", ...)`; ao testar retentativa, `patch("llm_br.base.time.sleep")`.

## Versionamento

Consumidores fixam a **tag** (`llm-br @ git+https://github.com/FinweeJur/llm-br@v0.1.0`), não `@main`. Ao evoluir a lib: mudar código → testar → **nova tag** (`v0.1.1`) → bumpar a tag no `requirements.txt` dos três consumidores de propósito. Nunca deixe consumidor em `@main`: um push aqui mudaria o que os três instalam na próxima CI, sem revisão.

Guia visual (para humano): `Obsidian Vault/guias/llm-br-guia-completo.html`.

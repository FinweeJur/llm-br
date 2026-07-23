"""llm-br — camada LLM plugável dos projetos FinweeJur.

Uso mínimo (lê o ambiente):

    from llm_br import get_llm
    llm = get_llm()
    print(llm.gerar_texto("Resuma em uma frase: ..."))

Uso em conversa:

    from llm_br import get_llm, Mensagem
    for pedaco in get_llm().stream([Mensagem("user", "Olá")]):
        print(pedaco, end="")

**Local-first por padrão.** Sem configuração nenhuma, cai no Ollama em
localhost. `LLM_PROVIDER=auto` usa a nuvem só se houver chave, senão o
local — que é a regra que os apps já seguiam cada um do seu jeito.

Variáveis de ambiente:

| Variável | Default | Para quê |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | ollama · anthropic · deepseek · maritaca · openai_compat · auto |
| `LLM_MODEL` | por provedor | nome do modelo |
| `LLM_BASE_URL` | por provedor | endpoint (só para provedor customizado) |
| `LLM_API_KEY` | vazio | chave da nuvem |
| `LLM_EMBED_MODEL` | `nomic-embed-text` | modelo de embeddings (Ollama) |

Quem não usa variável de ambiente (um app Django, por exemplo) passa tudo
explicitamente: `get_llm("deepseek", api_key=settings.DEEPSEEK_API_KEY)`.
"""
from __future__ import annotations

import os

from .base import (
    LLMError,
    LLMProvider,
    Mensagem,
    RespostaLLM,
    extrair_json,
    separar_system,
)
from .providers.anthropic import AnthropicProvider
from .providers.ollama import OllamaProvider
from .providers.openai_compat import BASES_CONHECIDAS, OpenAICompatProvider

__version__ = "0.1.0"

__all__ = [
    "get_llm", "provedores_disponiveis",
    "Mensagem", "RespostaLLM", "LLMProvider", "LLMError",
    "extrair_json", "separar_system",
    "OllamaProvider", "AnthropicProvider", "OpenAICompatProvider",
]

# Modelos default por provedor: o app não precisa saber o nome de cada um
# para dar o primeiro passo.
MODELOS_PADRAO = {
    "ollama": "llama3.1:8b-instruct-q4_K_M",
    "anthropic": "claude-sonnet-5",
    "deepseek": "deepseek-chat",
    "maritaca": "sabia-3",
    "openai_compat": "",
}

_ALIASES_OPENAI = {"deepseek", "maritaca", "openai_compat", "openai"}


def provedores_disponiveis() -> list[str]:
    return ["ollama", "anthropic", "deepseek", "maritaca", "openai_compat", "auto"]


def get_llm(
    provider: str | None = None,
    *,
    modelo: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
    modelo_embeddings: str | None = None,
    env: dict | None = None,
) -> LLMProvider:
    """Devolve o provedor configurado.

    Argumentos explícitos vencem o ambiente; o ambiente vence o default.
    `env` permite injetar um dicionário no lugar de `os.environ` — os testes
    usam isso, e um app com config própria (YAML, settings do Django) também.
    """
    amb = env if env is not None else os.environ

    nome = (provider or amb.get("LLM_PROVIDER") or "ollama").strip().lower()
    chave = api_key if api_key is not None else (amb.get("LLM_API_KEY") or "")

    if nome == "auto":
        # Local-first: só sobe para a nuvem se houver chave.
        nome = "deepseek" if chave else "ollama"

    modelo = modelo or amb.get("LLM_MODEL") or MODELOS_PADRAO.get(nome, "")
    url = base_url if base_url is not None else (amb.get("LLM_BASE_URL") or "")
    tempo = timeout or int(amb.get("LLM_TIMEOUT") or 300)

    if nome == "ollama":
        return OllamaProvider(
            modelo, url, timeout=tempo,
            modelo_embeddings=modelo_embeddings
            or amb.get("LLM_EMBED_MODEL")
            or "nomic-embed-text",
        )

    if nome == "anthropic":
        return AnthropicProvider(modelo, url, chave, timeout=tempo)

    if nome in _ALIASES_OPENAI:
        return OpenAICompatProvider(
            modelo, url or BASES_CONHECIDAS.get(nome, ""), chave, timeout=tempo
        )

    raise LLMError(
        f"LLM_PROVIDER={nome!r} desconhecido. Use: {' | '.join(provedores_disponiveis())}"
    )

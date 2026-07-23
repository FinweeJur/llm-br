"""Resolução de provedor a partir do ambiente e dos argumentos."""
import pytest

from llm_br import LLMError, get_llm
from llm_br.providers.anthropic import AnthropicProvider
from llm_br.providers.ollama import OllamaProvider
from llm_br.providers.openai_compat import OpenAICompatProvider


def test_sem_configuracao_nenhuma_cai_no_local():
    """Local-first é o default — a lib não deve exigir chave para funcionar."""
    llm = get_llm(env={})
    assert isinstance(llm, OllamaProvider)
    assert llm.base_url == "http://localhost:11434"


@pytest.mark.parametrize(
    "provider,classe",
    [
        ("ollama", OllamaProvider),
        ("anthropic", AnthropicProvider),
        ("deepseek", OpenAICompatProvider),
        ("maritaca", OpenAICompatProvider),
        ("openai_compat", OpenAICompatProvider),
    ],
)
def test_provider_por_ambiente(provider, classe):
    assert isinstance(get_llm(env={"LLM_PROVIDER": provider}), classe)


def test_deepseek_e_maritaca_usam_o_mesmo_adapter_com_urls_diferentes():
    ds = get_llm("deepseek", env={})
    mar = get_llm("maritaca", env={})
    assert type(ds) is type(mar)
    assert ds.base_url != mar.base_url
    assert "deepseek" in ds.base_url and "maritaca" in mar.base_url


def test_auto_usa_nuvem_quando_ha_chave():
    llm = get_llm(env={"LLM_PROVIDER": "auto", "LLM_API_KEY": "abc"})
    assert isinstance(llm, OpenAICompatProvider)


def test_auto_cai_no_local_sem_chave():
    llm = get_llm(env={"LLM_PROVIDER": "auto"})
    assert isinstance(llm, OllamaProvider)


def test_argumento_explicito_vence_o_ambiente():
    """Um app com config própria (Django, YAML) passa tudo na chamada."""
    llm = get_llm("ollama", modelo="qwen2.5", env={"LLM_PROVIDER": "anthropic",
                                                   "LLM_MODEL": "claude-sonnet-5"})
    assert isinstance(llm, OllamaProvider)
    assert llm.modelo == "qwen2.5"


def test_provider_invalido_diz_quais_existem():
    with pytest.raises(LLMError) as exc:
        get_llm("inexistente", env={})
    assert "ollama" in str(exc.value)


def test_identificacao_registra_provedor_e_modelo():
    """Comparar pontuação entre modelos diferentes não é válido — o registro
    precisa dizer qual modelo produziu cada análise."""
    assert get_llm("ollama", modelo="llama3.1:8b", env={}).identificacao == "ollama:llama3.1:8b"

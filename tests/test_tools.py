"""Registro de ferramentas e os dois formatos de schema."""
import pytest

from llm_br.tools import Registro


@pytest.fixture
def registro():
    r = Registro()

    @r.tool("saldo", "Saldo em aberto de um cliente",
            {"cliente_id": {"type": "integer", "description": "ID do cliente"}},
            obrigatorios=("cliente_id",))
    def saldo(cliente_id):
        return {"saldo": cliente_id * 10}

    return r


def test_executa_a_ferramenta_registrada(registro):
    assert registro.executar("saldo", cliente_id=3) == {"saldo": 30}


def test_ferramenta_inexistente_lista_as_disponiveis(registro):
    with pytest.raises(KeyError) as exc:
        registro.executar("nao_existe")
    assert "saldo" in str(exc.value)


def test_aceita_ferramenta_com_parametro_chamado_nome(registro):
    """O 1º argumento de executar() é positional-only justamente por isto:
    'nome' é um kwarg comum em ferramenta de domínio."""
    @registro.tool("criar_lead", "Cria lead", {"nome": {"type": "string"}})
    def criar(nome):
        return {"criado": nome}

    assert registro.executar("criar_lead", nome="Fulano") == {"criado": "Fulano"}


def test_schema_anthropic_usa_input_schema(registro):
    spec = registro.specs_anthropic()[0]
    assert spec["name"] == "saldo"
    assert spec["input_schema"]["required"] == ["cliente_id"]
    assert "cliente_id" in spec["input_schema"]["properties"]


def test_schema_ollama_usa_function_parameters(registro):
    spec = registro.specs_ollama()[0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "saldo"
    assert spec["function"]["parameters"]["required"] == ["cliente_id"]


def test_specs_escolhe_o_formato_pelo_provedor(registro):
    from llm_br import get_llm

    anthropic = registro.specs(get_llm("anthropic", env={}))
    ollama = registro.specs(get_llm("ollama", env={}))
    assert "input_schema" in anthropic[0]
    assert ollama[0]["type"] == "function"


def test_registros_sao_isolados(registro):
    """Dois agentes com ferramentas diferentes não podem se contaminar."""
    outro = Registro()
    assert registro.nomes() == ["saldo"]
    assert outro.nomes() == []

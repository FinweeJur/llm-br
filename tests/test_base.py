"""Extração de JSON, retentativa e os defaults do contrato."""
import pytest

from llm_br import LLMError, Mensagem, extrair_json, separar_system
from llm_br.base import LLMProvider, com_retentativa


class TestExtrairJson:
    def test_json_limpo(self):
        assert extrair_json('{"a": 1}') == {"a": 1}

    def test_dentro_de_cerca_de_codigo(self):
        assert extrair_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_cerca_sem_a_palavra_json(self):
        assert extrair_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_com_conversa_antes_e_depois(self):
        assert extrair_json('Claro! Segue:\n{"a": 1}\nEspero ter ajudado.') == {"a": 1}

    def test_nao_mascara_json_valido_que_contem_chaves_no_texto(self):
        """O parse direto vem antes do recorte justamente por isto."""
        assert extrair_json('{"texto": "use {chaves} assim"}') == {"texto": "use {chaves} assim"}

    def test_resposta_sem_json_levanta_com_trecho(self):
        with pytest.raises(LLMError) as exc:
            extrair_json("desculpe, não entendi")
        assert "não entendi" in str(exc.value)

    def test_vazio_levanta(self):
        with pytest.raises(LLMError):
            extrair_json("")


def test_separar_system_junta_multiplas_e_preserva_ordem():
    msgs = [Mensagem("system", "A"), Mensagem("user", "1"),
            Mensagem("system", "B"), Mensagem("assistant", "2")]
    system, resto = separar_system(msgs)
    assert system == "A\n\nB"
    assert [m.content for m in resto] == ["1", "2"]


class TestRetentativa:
    def test_devolve_na_primeira_quando_da_certo(self):
        assert com_retentativa(lambda: "ok", dormir=lambda _: None) == "ok"

    def test_repete_ate_conseguir(self):
        tentativas = []

        def falha_duas_vezes():
            tentativas.append(1)
            if len(tentativas) < 3:
                raise LLMError("instável")
            return "ok"

        assert com_retentativa(falha_duas_vezes, dormir=lambda _: None) == "ok"
        assert len(tentativas) == 3

    def test_desiste_apos_o_limite(self):
        with pytest.raises(LLMError):
            com_retentativa(lambda: (_ for _ in ()).throw(LLMError("sempre falha")),
                            tentativas=2, dormir=lambda _: None)

    def test_erro_nao_reententavel_falha_na_hora(self):
        """Chave inválida não melhora esperando."""
        tentativas = []

        def chave_ruim():
            tentativas.append(1)
            erro = LLMError("chave inválida")
            erro.reententavel = False
            raise erro

        with pytest.raises(LLMError):
            com_retentativa(chave_ruim, dormir=lambda _: None)
        assert len(tentativas) == 1

    def test_espera_cresce_exponencialmente(self):
        esperas = []

        def sempre_falha():
            raise LLMError("x")

        with pytest.raises(LLMError):
            com_retentativa(sempre_falha, tentativas=4, espera_inicial=1,
                            dormir=esperas.append)
        assert esperas == [1, 2, 4]


class ProvedorSoDeTarefa(LLMProvider):
    """Implementa o mínimo: só `gerar_texto`."""

    nome = "fake"

    def gerar_texto(self, prompt, *, system="", temperatura=0.2):
        return f"[system={system}] {prompt}"


class TestDefaultsDoContrato:
    def test_chat_funciona_num_provedor_que_so_sabe_gerar_texto(self):
        """Um provedor de ETL continua utilizável numa tela de chat."""
        resp = ProvedorSoDeTarefa("m").chat(
            [Mensagem("system", "seja breve"), Mensagem("user", "oi")]
        )
        assert "seja breve" in resp.texto and "oi" in resp.texto
        assert resp.provider == "fake"

    def test_stream_default_devolve_tudo_de_uma_vez(self):
        assert list(ProvedorSoDeTarefa("m").stream([Mensagem("user", "oi")])) == [
            "[system=] user: oi"
        ]

    def test_embeddings_default_e_none_e_nao_levanta(self):
        assert ProvedorSoDeTarefa("m").embeddings(["a"]) is None

    def test_disponivel_e_false_quando_o_provedor_falha(self):
        class Quebrado(LLMProvider):
            def gerar_texto(self, prompt, *, system="", temperatura=0.2):
                raise LLMError("fora do ar")

        assert Quebrado("m").disponivel() is False

    def test_suporta_reflete_os_recursos_declarados(self):
        from llm_br.providers.ollama import OllamaProvider

        o = OllamaProvider("m")
        assert o.suporta("embeddings") and o.suporta("tools")
        assert not ProvedorSoDeTarefa("m").suporta("tools")

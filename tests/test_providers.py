"""Comportamento HTTP dos adapters, com a rede simulada."""
from unittest.mock import MagicMock, patch

import pytest

from llm_br import LLMError, Mensagem
from llm_br.providers.ollama import OllamaProvider
from llm_br.providers.openai_compat import OpenAICompatProvider


def resposta(status=200, json_data=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    r.text = text
    return r


class TestOllama:
    def test_gerar_json_pede_decodificacao_restrita(self):
        """format=json no Ollama impede o modelo de emitir token que quebre o
        JSON — é garantia real, não instrução. Precisa ser enviado."""
        with patch("requests.post", return_value=resposta(json_data={"response": '{"a": 1}'})) as post:
            assert OllamaProvider("m").gerar_json("p") == {"a": 1}
        assert post.call_args.kwargs["json"]["format"] == "json"

    def test_gerar_texto_nao_pede_formato(self):
        with patch("requests.post", return_value=resposta(json_data={"response": " oi "})) as post:
            assert OllamaProvider("m").gerar_texto("p") == "oi"
        assert "format" not in post.call_args.kwargs["json"]

    def test_json_sujo_ainda_e_recuperado(self):
        with patch("requests.post", return_value=resposta(json_data={"response": '```json\n{"a":1}\n```'})):
            assert OllamaProvider("m").gerar_json("p") == {"a": 1}

    def test_servico_fora_do_ar_vira_LLMError_com_a_url(self):
        import requests

        with patch("requests.post", side_effect=requests.RequestException("recusou")):
            with pytest.raises(LLMError) as exc:
                OllamaProvider("m", "http://maquina:11434").gerar_texto("p")
        assert "maquina:11434" in str(exc.value)

    def test_embeddings_devolve_none_em_vez_de_explodir(self):
        """RAG é opcional: o chamador precisa poder cair na busca simples."""
        import requests

        with patch("requests.post", side_effect=requests.RequestException("fora")):
            assert OllamaProvider("m").embeddings(["a"]) is None

    def test_embeddings_usa_o_modelo_de_embeddings_e_nao_o_de_chat(self):
        with patch("requests.post", return_value=resposta(json_data={"embeddings": [[0.1]]})) as post:
            assert OllamaProvider("chat-model").embeddings(["a"]) == [[0.1]]
        assert post.call_args.kwargs["json"]["model"] == "nomic-embed-text"

    def test_chat_com_tools_executa_e_devolve_o_resultado_ao_modelo(self):
        chamou = []

        def executar(nome, **kw):
            chamou.append((nome, kw))
            return {"saldo": 42}

        respostas = [
            resposta(json_data={"message": {"content": "", "tool_calls": [
                {"function": {"name": "saldo", "arguments": {"id": 1}}}]}}),
            resposta(json_data={"message": {"content": "O saldo é 42."}}),
        ]
        with patch("requests.post", side_effect=respostas):
            r = OllamaProvider("m").chat_com_tools("sys", [], "qual o saldo?", [], executar)

        assert chamou == [("saldo", {"id": 1})]
        assert r["texto"] == "O saldo é 42."
        assert r["tool_calls"][0]["name"] == "saldo"

    def test_erro_da_ferramenta_volta_ao_modelo_em_vez_de_derrubar(self):
        def executar(nome, **kw):
            raise RuntimeError("banco fora")

        respostas = [
            resposta(json_data={"message": {"content": "", "tool_calls": [
                {"function": {"name": "x", "arguments": {}}}]}}),
            resposta(json_data={"message": {"content": "Não consegui consultar."}}),
        ]
        with patch("requests.post", side_effect=respostas):
            r = OllamaProvider("m").chat_com_tools("", [], "?", [], executar)
        assert "banco fora" in r["tool_calls"][0]["resultado"]

    def test_loop_de_tools_tem_teto(self):
        """Modelo teimoso não pode virar laço infinito."""
        sempre_chama = resposta(json_data={"message": {"content": "", "tool_calls": [
            {"function": {"name": "x", "arguments": {}}}]}})
        with patch("requests.post", return_value=sempre_chama) as post:
            r = OllamaProvider("m").chat_com_tools("", [], "?", [], lambda n, **k: {}, max_iters=3)
        assert post.call_count == 3
        assert "Não consegui concluir" in r["texto"]


class TestOpenAICompat:
    def _ok(self, texto):
        return resposta(json_data={"choices": [{"message": {"content": texto}}]})

    def test_monta_a_rota_padrao_do_mercado(self):
        with patch("requests.post", return_value=self._ok("oi")) as post:
            OpenAICompatProvider("m", "https://api.exemplo.com", "k").gerar_texto("p")
        assert post.call_args.args[0] == "https://api.exemplo.com/v1/chat/completions"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer k"

    def test_json_mode_usa_response_format(self):
        with patch("requests.post", return_value=self._ok('{"a":1}')) as post:
            assert OpenAICompatProvider("m", "https://x", "k").gerar_json("p") == {"a": 1}
        assert post.call_args.kwargs["json"]["response_format"] == {"type": "json_object"}

    def test_chave_invalida_nao_e_retentada(self):
        """401 não melhora esperando — falhar em 1 chamada, não em 3."""
        with patch("requests.post", return_value=resposta(status=401)) as post:
            with pytest.raises(LLMError) as exc:
                OpenAICompatProvider("m", "https://x", "ruim").gerar_texto("p")
        assert post.call_count == 1
        assert "chave" in str(exc.value).lower()

    def test_falha_transitoria_e_retentada(self):
        respostas = [resposta(status=503, text="indisponível"), self._ok("consegui")]
        with patch("requests.post", side_effect=respostas), \
             patch("llm_br.base.time.sleep"):
            assert OpenAICompatProvider("m", "https://x", "k").gerar_texto("p") == "consegui"

    def test_sem_base_url_diz_o_que_falta(self):
        with pytest.raises(LLMError) as exc:
            OpenAICompatProvider("m", "", "k").gerar_texto("p")
        assert "base_url" in str(exc.value)

    def test_chat_preserva_o_system_como_mensagem(self):
        with patch("requests.post", return_value=self._ok("r")) as post:
            OpenAICompatProvider("m", "https://x", "k").chat(
                [Mensagem("system", "seja breve"), Mensagem("user", "oi")]
            )
        enviadas = post.call_args.kwargs["json"]["messages"]
        assert enviadas[0] == {"role": "system", "content": "seja breve"}
        assert enviadas[1]["content"] == "oi"

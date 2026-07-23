"""Provedor local via Ollama.

O `format: "json"` do Ollama é *constrained decoding* de verdade: o modelo
fica impedido de emitir token que quebre o JSON, em vez de ser só instruído
a se comportar. Por isso `gerar_json` o usa mesmo tendo `extrair_json` como
rede de segurança.
"""
from __future__ import annotations

import json
from typing import Iterator

from ..base import LLMError, LLMProvider, Mensagem, RespostaLLM, extrair_json

PADRAO_HOST = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    nome = "ollama"
    RECURSOS = ("stream", "tools", "embeddings", "json_nativo")

    def __init__(self, modelo: str, base_url: str = "", api_key: str = "",
                 timeout: int = 600, modelo_embeddings: str = "nomic-embed-text"):
        super().__init__(modelo, base_url or PADRAO_HOST, api_key, timeout)
        self.modelo_embeddings = modelo_embeddings

    def _post(self, rota: str, payload: dict, timeout: int | None = None) -> dict:
        import requests

        try:
            resp = requests.post(
                f"{self.base_url}{rota}", json=payload, timeout=timeout or self.timeout
            )
        except requests.RequestException as e:
            raise LLMError(f"Ollama inacessível em {self.base_url}: {e}") from e
        if resp.status_code >= 400:
            raise LLMError(f"Ollama respondeu {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    # ── nível tarefa ────────────────────────────────────────────────
    def _gerar(self, prompt: str, system: str, temperatura: float, formato: str | None) -> str:
        payload: dict = {
            "model": self.modelo,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperatura},
        }
        if system:
            payload["system"] = system
        if formato:
            payload["format"] = formato
        return self._post("/api/generate", payload).get("response", "")

    def gerar_texto(self, prompt: str, *, system: str = "", temperatura: float = 0.2) -> str:
        return self._gerar(prompt, system, temperatura, None).strip()

    def gerar_json(self, prompt: str, *, system: str = "", temperatura: float = 0.0) -> dict:
        return extrair_json(self._gerar(prompt, system, temperatura, "json"))

    # ── nível conversa ──────────────────────────────────────────────
    def chat(self, mensagens: list[Mensagem], **kwargs) -> RespostaLLM:
        dados = self._post("/api/chat", {
            "model": self.modelo,
            "messages": [{"role": m.role, "content": m.content} for m in mensagens],
            "stream": False,
        })
        return RespostaLLM(
            texto=dados["message"]["content"],
            modelo=self.modelo,
            provider=self.nome,
            meta={"raw": dados},
        )

    def stream(self, mensagens: list[Mensagem], **kwargs) -> Iterator[str]:
        import requests

        payload = {
            "model": self.modelo,
            "messages": [{"role": m.role, "content": m.content} for m in mensagens],
            "stream": True,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout, stream=True
            )
        except requests.RequestException as e:
            raise LLMError(f"Ollama inacessível em {self.base_url}: {e}") from e
        for linha in resp.iter_lines():
            if not linha:
                continue
            parte = json.loads(linha)
            conteudo = (parte.get("message") or {}).get("content")
            if conteudo:
                yield conteudo

    # ── tool calling ────────────────────────────────────────────────
    def chat_com_tools(self, system: str, historico: list[Mensagem], user_text: str,
                       tools: list[dict], executar, max_iters: int = 5) -> dict:
        """Loop de function-calling no formato do Ollama.

        `executar(nome, **args) -> dict` é injetado pelo app: a biblioteca
        conhece o protocolo de chamada, não as ferramentas do negócio.
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages += [{"role": m.role, "content": m.content} for m in historico]
        messages.append({"role": "user", "content": user_text})

        chamadas = []
        for _ in range(max_iters):
            dados = self._post("/api/chat", {
                "model": self.modelo, "messages": messages,
                "tools": tools or None, "stream": False,
            })
            msg = dados["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return {"texto": msg.get("content", ""), "tool_calls": chamadas}

            messages.append({"role": "assistant", "content": msg.get("content", ""),
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                nome = tc["function"]["name"]
                args = tc["function"].get("arguments") or {}
                try:
                    conteudo = json.dumps(executar(nome, **args), ensure_ascii=False)
                except Exception as exc:
                    conteudo = json.dumps({"erro": str(exc)}, ensure_ascii=False)
                chamadas.append({"name": nome, "args": args, "resultado": conteudo})
                messages.append({"role": "tool", "content": conteudo})

        return {"texto": "Não consegui concluir após várias chamadas de ferramenta.",
                "tool_calls": chamadas}

    # ── embeddings ──────────────────────────────────────────────────
    def embeddings(self, textos: list[str]) -> list[list[float]] | None:
        if not textos:
            return None
        try:
            dados = self._post("/api/embed",
                               {"model": self.modelo_embeddings, "input": textos},
                               timeout=180)
        except LLMError:
            return None  # modelo não baixado / serviço fora — o chamador cai no fallback
        return dados.get("embeddings") or None

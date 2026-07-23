"""Provedor para qualquer API compatível com OpenAI `/v1/chat/completions`.

Cobre **DeepSeek** e **Maritaca (Sabiá)** com o mesmo código — as duas expõem
esse formato, que virou o padrão de facto do mercado. Trocar entre elas é
mudar provider, base_url e chave; nada de código.

É por isso que a lib não depende do LiteLLM: o ganho dele aqui seria quase
todo este arquivo. Se algum dia for preciso um provedor exótico, ele entra
como mais um adapter atrás do mesmo contrato, sem tocar em quem consome.
"""
from __future__ import annotations

from ..base import LLMError, LLMProvider, Mensagem, RespostaLLM, com_retentativa, extrair_json, separar_system

# Endpoints conhecidos, para o app não precisar decorar URL.
BASES_CONHECIDAS = {
    "deepseek": "https://api.deepseek.com",
    "maritaca": "https://chat.maritaca.ai/api",
}


class OpenAICompatProvider(LLMProvider):
    nome = "openai_compat"
    RECURSOS = ("json_nativo",)

    def _chamar(self, mensagens: list[dict], temperatura: float, json_mode: bool) -> str:
        import requests

        if not self.base_url:
            raise LLMError("base_url não configurada para este provedor.")

        payload: dict = {
            "model": self.modelo,
            "messages": mensagens,
            "temperature": temperatura,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        def uma_tentativa() -> str:
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                raise LLMError(f"provedor inacessível em {self.base_url}: {e}") from e

            if resp.status_code == 401:
                # Chave errada não melhora com retentativa — falhar na hora.
                erro = LLMError("chave de API inválida ou ausente.")
                erro.reententavel = False
                raise erro
            if resp.status_code >= 400:
                raise LLMError(f"provedor respondeu {resp.status_code}: {resp.text[:200]}")
            return resp.json()["choices"][0]["message"]["content"]

        return com_retentativa(uma_tentativa)

    def _montar(self, prompt: str, system: str) -> list[dict]:
        return ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]

    def gerar_texto(self, prompt: str, *, system: str = "", temperatura: float = 0.2) -> str:
        return self._chamar(self._montar(prompt, system), temperatura, False).strip()

    def gerar_json(self, prompt: str, *, system: str = "", temperatura: float = 0.0) -> dict:
        return extrair_json(self._chamar(self._montar(prompt, system), temperatura, True))

    def chat(self, mensagens: list[Mensagem], **kwargs) -> RespostaLLM:
        system, resto = separar_system(mensagens)
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m.role, "content": m.content} for m in resto
        ]
        texto = self._chamar(msgs, kwargs.get("temperatura", 0.2), False)
        return RespostaLLM(texto=texto, modelo=self.modelo, provider=self.nome)

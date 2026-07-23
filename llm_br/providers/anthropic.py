"""Provedor via API Claude (Anthropic). Usa o SDK oficial.

O import de `anthropic` é feito dentro dos métodos: a lib não deve exigir o
SDK de quem só roda modelo local.
"""
from __future__ import annotations

import json
from typing import Iterator

from ..base import LLMError, LLMProvider, Mensagem, RespostaLLM, extrair_json, separar_system

PADRAO_MAX_TOKENS = 1024


class AnthropicProvider(LLMProvider):
    nome = "anthropic"
    RECURSOS = ("stream", "tools")

    def _client(self):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("Pacote 'anthropic' não instalado: pip install llm-br[anthropic]") from e
        if not self.api_key:
            raise LLMError("chave de API da Anthropic não configurada.")
        return anthropic.Anthropic(api_key=self.api_key)

    def chat(self, mensagens: list[Mensagem], **kwargs) -> RespostaLLM:
        system, resto = separar_system(mensagens)
        msg = self._client().messages.create(
            model=self.modelo,
            max_tokens=kwargs.get("max_tokens", PADRAO_MAX_TOKENS),
            system=system or None,
            messages=[{"role": m.role, "content": m.content} for m in resto],
        )
        return RespostaLLM(
            texto=msg.content[0].text,
            modelo=self.modelo,
            provider=self.nome,
            meta={"stop_reason": msg.stop_reason},
        )

    def stream(self, mensagens: list[Mensagem], **kwargs) -> Iterator[str]:
        system, resto = separar_system(mensagens)
        with self._client().messages.stream(
            model=self.modelo,
            max_tokens=kwargs.get("max_tokens", PADRAO_MAX_TOKENS),
            system=system or None,
            messages=[{"role": m.role, "content": m.content} for m in resto],
        ) as stream:
            yield from stream.text_stream

    def gerar_texto(self, prompt: str, *, system: str = "", temperatura: float = 0.2) -> str:
        return self.chat(
            [Mensagem("system", system), Mensagem("user", prompt)] if system
            else [Mensagem("user", prompt)]
        ).texto.strip()

    def gerar_json(self, prompt: str, *, system: str = "", temperatura: float = 0.0) -> dict:
        # A API não tem modo JSON nativo; pedimos no system e validamos na saída.
        instrucao = "Responda APENAS com um objeto JSON válido, sem cercas de código."
        return extrair_json(
            self.gerar_texto(prompt, system=f"{system}\n\n{instrucao}".strip(), temperatura=temperatura)
        )

    def chat_com_tools(self, system: str, historico: list[Mensagem], user_text: str,
                       tools: list[dict], executar, max_iters: int = 5) -> dict:
        """Loop de tool-calling no formato `tools`/`tool_use` da API Claude."""
        client = self._client()
        messages = [{"role": m.role, "content": m.content} for m in historico]
        messages.append({"role": "user", "content": user_text})

        chamadas = []
        for _ in range(max_iters):
            resp = client.messages.create(
                model=self.modelo, max_tokens=PADRAO_MAX_TOKENS,
                system=system or None, tools=tools or None, messages=messages,
            )
            textos = [b.text for b in resp.content if b.type == "text"]
            usos = [b for b in resp.content if b.type == "tool_use"]
            if not usos:
                return {"texto": "".join(textos), "tool_calls": chamadas}

            messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
            resultados = []
            for tb in usos:
                try:
                    conteudo = json.dumps(executar(tb.name, **(tb.input or {})), ensure_ascii=False)
                    erro = False
                except Exception as exc:
                    conteudo = json.dumps({"erro": str(exc)}, ensure_ascii=False)
                    erro = True
                chamadas.append({"name": tb.name, "args": tb.input, "resultado": conteudo})
                resultados.append({"type": "tool_result", "tool_use_id": tb.id,
                                   "content": conteudo, "is_error": erro})
            messages.append({"role": "user", "content": resultados})

        return {"texto": "Não consegui concluir após várias chamadas de ferramenta.",
                "tool_calls": chamadas}

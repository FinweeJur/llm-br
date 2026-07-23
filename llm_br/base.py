"""Contratos da camada LLM.

Um provedor implementa dois níveis:

- **Tarefa** (`gerar_texto`, `gerar_json`): um prompt, uma resposta. É o que
  pipelines de ETL e classificadores usam.
- **Conversa** (`chat`, `stream`): histórico de mensagens, com streaming. É o
  que telas de chat usam.

Nem todo provedor precisa de tudo; `chat_com_tools` e `embeddings` são
opcionais e declarados por capacidade (ver `suporta`).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class LLMError(RuntimeError):
    """Falha de comunicação, de configuração ou de formato do provedor."""


@dataclass
class Mensagem:
    """Mensagem de uma conversa. role: 'system' | 'user' | 'assistant'."""

    role: str
    content: str


@dataclass
class RespostaLLM:
    texto: str
    modelo: str
    provider: str
    meta: dict = field(default_factory=dict)


class LLMProvider:
    """Base dos provedores. Subclasses implementam ao menos `gerar_texto`.

    `nome` identifica o adapter (ollama, anthropic, openai_compat); `modelo`
    identifica o modelo concreto. Os dois juntos formam `identificacao`, que
    é o que se grava junto de uma análise — comparar pontuações produzidas
    por modelos diferentes não é válido, então o registro precisa dizer qual
    modelo produziu cada uma.
    """

    nome: str = "base"

    def __init__(self, modelo: str, base_url: str = "", api_key: str = "", timeout: int = 300):
        self.modelo = modelo
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def identificacao(self) -> str:
        return f"{self.nome}:{self.modelo}"

    # ── nível tarefa ────────────────────────────────────────────────
    def gerar_texto(self, prompt: str, *, system: str = "", temperatura: float = 0.2) -> str:
        raise NotImplementedError

    def gerar_json(self, prompt: str, *, system: str = "", temperatura: float = 0.0) -> dict:
        raise NotImplementedError

    # ── nível conversa ──────────────────────────────────────────────
    def chat(self, mensagens: list[Mensagem], **kwargs) -> RespostaLLM:
        """Resposta completa a partir de um histórico.

        Default: achata o histórico em um prompt e delega para `gerar_texto`,
        para que um provedor só de tarefa continue utilizável numa tela de
        chat sem precisar reimplementar nada.
        """
        system, resto = separar_system(mensagens)
        prompt = "\n\n".join(f"{m.role}: {m.content}" for m in resto)
        texto = self.gerar_texto(prompt, system=system, temperatura=kwargs.get("temperatura", 0.2))
        return RespostaLLM(texto=texto, modelo=self.modelo, provider=self.nome)

    def stream(self, mensagens: list[Mensagem], **kwargs) -> Iterator[str]:
        """Default: sem streaming real, devolve a resposta inteira de uma vez."""
        yield self.chat(mensagens, **kwargs).texto

    # ── opcionais ───────────────────────────────────────────────────
    def chat_com_tools(self, system: str, historico: list[Mensagem], user_text: str,
                       tools: list[dict], executar, max_iters: int = 5) -> dict:
        """Loop de tool-calling. Nem todo provedor implementa.

        Falha com mensagem explícita em vez de AttributeError: quem chama
        costuma ser um agente que já escolheu o provedor por variável de
        ambiente, e um erro opaco aqui manda o dedo para o lugar errado.
        Cheque antes com `suporta("tools")`.
        """
        raise LLMError(
            f"provedor {self.nome!r} não suporta tool-calling. "
            f"Use suporta('tools') antes de chamar, ou troque de provedor."
        )

    def embeddings(self, textos: list[str]) -> list[list[float]] | None:
        """Vetores de significado, para RAG. `None` quando indisponível.

        Nunca levanta: embedding é recurso opcional e o chamador precisa
        poder cair num método mais simples (busca por palavra-chave).
        """
        return None

    def suporta(self, recurso: str) -> bool:
        """'stream' | 'tools' | 'embeddings' | 'json_nativo'."""
        return recurso in getattr(self, "RECURSOS", ())

    # ── saúde ───────────────────────────────────────────────────────
    def disponivel(self) -> bool:
        """Checagem barata, para falhar em 1 segundo em vez de na centésima
        chamada de uma fila longa."""
        try:
            self.gerar_texto("responda apenas: ok", temperatura=0.0)
            return True
        except Exception:
            return False


def separar_system(mensagens: list[Mensagem]) -> tuple[str, list[Mensagem]]:
    """Separa as mensagens 'system' do resto (APIs as tratam como campo à parte)."""
    system = "\n\n".join(m.content for m in mensagens if m.role == "system")
    resto = [m for m in mensagens if m.role != "system"]
    return system, resto


def extrair_json(texto: str) -> dict:
    """Extrai o objeto JSON de uma resposta de LLM.

    Mesmo com `format=json`/`response_format`, modelos menores às vezes
    envolvem a resposta em ```json ... ``` ou emendam uma frase antes.
    Tentamos o parse direto primeiro e só depois apelamos para recorte —
    nessa ordem para não mascarar um JSON válido que por acaso contenha
    chaves no texto.
    """
    texto = (texto or "").strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    sem_cerca = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.MULTILINE).strip()
    try:
        return json.loads(sem_cerca)
    except json.JSONDecodeError:
        pass

    inicio, fim = sem_cerca.find("{"), sem_cerca.rfind("}")
    if inicio != -1 and fim > inicio:
        try:
            return json.loads(sem_cerca[inicio : fim + 1])
        except json.JSONDecodeError:
            pass

    raise LLMError(f"resposta não é JSON válido: {texto[:300]}")


def com_retentativa(
    func: Callable[[], Any], *, tentativas: int = 3, espera_inicial: float = 2.0,
    espera_maxima: float = 30.0, dormir: Callable[[float], None] | None = None,
) -> Any:
    """Repete `func` com espera exponencial. Sem dependência externa.

    `dormir` é injetável para que os testes não gastem tempo real. Resolvido
    aqui dentro, e não como valor default do parâmetro: default é avaliado na
    definição do módulo, o que congelaria `time.sleep` e faria um
    `patch("llm_br.base.time.sleep")` não ter efeito nenhum.
    """
    dormir = dormir or time.sleep
    espera = espera_inicial
    for tentativa in range(1, tentativas + 1):
        try:
            return func()
        except LLMError as exc:
            if tentativa == tentativas or not getattr(exc, "reententavel", True):
                raise
            dormir(min(espera, espera_maxima))
            espera *= 2

"""Registro de ferramentas (tools) que um agente LLM pode executar.

A biblioteca conhece o **protocolo** de tool-calling (como declarar o schema
para cada API, como devolver o resultado); as ferramentas em si são do app.

    from llm_br.tools import tool, executar, specs_anthropic

    @tool("saldo_cliente", "Saldo em aberto de um cliente",
          {"cliente_id": {"type": "integer", "description": "ID"}},
          obrigatorios=("cliente_id",))
    def saldo(cliente_id: int) -> dict:
        return {"saldo": ...}

Existe um registro global (as funções de módulo abaixo) porque é assim que
apps costumam usar — declarando com decorator em vários módulos. Quem
precisar de registros isolados — dois agentes com ferramentas diferentes,
ou um teste que não quer herdar o estado de outro — instancia `Registro`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    nome: str
    descricao: str
    parametros: dict = field(default_factory=dict)
    funcao: Callable | None = None
    obrigatorios: tuple = ()


class Registro:
    """Conjunto nomeado de ferramentas."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def tool(self, nome: str, descricao: str, parametros: dict, obrigatorios: tuple = ()):
        """Decorator que registra uma função como ferramenta.

        `parametros` é {nome_do_param: json_schema}, ex.:
            {"pedido_id": {"type": "integer", "description": "ID do pedido"}}
        """
        def decorator(funcao: Callable) -> Callable:
            self._tools[nome] = Tool(nome, descricao, parametros, funcao, obrigatorios)
            return funcao
        return decorator

    def executar(self, nome_ferramenta: str, /, **kwargs) -> dict:
        """Executa a ferramenta registrada.

        `nome_ferramenta` é positional-only de propósito: muita ferramenta de
        domínio tem um parâmetro chamado "nome" (lead, cliente, fornecedor),
        e um parâmetro keyword aqui colidiria com esse kwarg.
        """
        if nome_ferramenta not in self._tools:
            raise KeyError(
                f"Ferramenta não registrada: {nome_ferramenta!r}. "
                f"Disponíveis: {list(self._tools)}"
            )
        return self._tools[nome_ferramenta].funcao(**kwargs)

    def specs_anthropic(self) -> list[dict]:
        """Schema no formato `tools` da API Claude (input_schema)."""
        return [
            {
                "name": t.nome,
                "description": t.descricao,
                "input_schema": {
                    "type": "object",
                    "properties": t.parametros,
                    "required": list(t.obrigatorios),
                },
            }
            for t in self._tools.values()
        ]

    def specs_ollama(self) -> list[dict]:
        """Schema no formato function-calling (estilo OpenAI)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.nome,
                    "description": t.descricao,
                    "parameters": {
                        "type": "object",
                        "properties": t.parametros,
                        "required": list(t.obrigatorios),
                    },
                },
            }
            for t in self._tools.values()
        ]

    def specs(self, provider) -> list[dict]:
        """Schema no formato certo para o provedor dado."""
        return self.specs_anthropic() if provider.nome == "anthropic" else self.specs_ollama()

    def nomes(self) -> list[str]:
        return list(self._tools)

    def limpar(self) -> None:
        self._tools.clear()


# Registro global — a forma usual de uso.
padrao = Registro()

tool = padrao.tool
executar = padrao.executar
specs_anthropic = padrao.specs_anthropic
specs_ollama = padrao.specs_ollama
specs = padrao.specs

__all__ = ["Tool", "Registro", "padrao", "tool", "executar",
           "specs_anthropic", "specs_ollama", "specs"]

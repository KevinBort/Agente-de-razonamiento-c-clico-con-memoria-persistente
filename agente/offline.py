from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

PREGUNTAS_DEMO: list[str] = [
    "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?",
    "¿Y qué productos tenía el último?",
    "Buscá a Martín Gómez y decime cuántos pedidos tuvo y qué contenía el más reciente.",
    "¿Cuántos pedidos tuvo el cliente 1020?",
]


def _llamada(nombre: str, argumentos: dict[str, Any], id_llamada: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": nombre, "args": argumentos, "id": id_llamada, "type": "tool_call"}],
    )


def respuestas_demo() -> list[AIMessage]:
    return [
        _llamada("buscar_pedidos", {"cliente_id": 102}, "call_1"),
        AIMessage(content="El cliente 102 tuvo 3 pedidos por un total de $14.500."),
        _llamada("obtener_detalle_pedido", {"pedido_id": "P-1003"}, "call_2"),
        AIMessage(
            content=(
                "El último pedido del cliente 102 es el P-1003 (2 de octubre de 2026) y contenía "
                "1 Webcam HD ($2.500) y 2 Soportes para notebook ($2.250 c/u), "
                "por un total de $7.000."
            )
        ),
        _llamada("buscar_cliente", {"nombre": "Martín Gómez"}, "call_3"),
        _llamada("buscar_pedidos", {"cliente_id": 205}, "call_4"),
        _llamada("obtener_detalle_pedido", {"pedido_id": "P-2002"}, "call_5"),
        AIMessage(
            content=(
                "Martín Gómez (cliente 205) tuvo 2 pedidos por un total de $18.100. "
                "El más reciente, P-2002 (20 de octubre de 2026), contenía 1 Disco SSD 1TB "
                "($9.800) y 1 Funda protectora ($1.500)."
            )
        ),
        _llamada("buscar_pedidos", {"cliente_id": 1020}, "call_6"),
        AIMessage(
            content=(
                "No encontré ningún cliente con el ID 1020. ¿Te referías al cliente 102 "
                "(Lucía Fernández)? Confirmame el ID o decime el nombre y lo busco."
            )
        ),
    ]


class ModeloGuionado(BaseChatModel):
    respuestas: list[AIMessage]
    indice: int = 0
    llamadas: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "modelo-guionado"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.indice >= len(self.respuestas):
            raise RuntimeError("El guion del modelo simulado se agotó.")
        self.llamadas.append(list(messages))
        respuesta = self.respuestas[self.indice]
        self.indice += 1
        return ChatResult(generations=[ChatGeneration(message=respuesta)])

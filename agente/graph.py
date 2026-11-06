from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agente.state import AgentState

NODO_MODELO = "modelo"
NODO_HERRAMIENTAS = "herramientas"

PROMPT_SISTEMA = """Sos un asistente de consultas sobre clientes y pedidos.
Respondé siempre en español.

Reglas:
- Para obtener datos usá SIEMPRE las herramientas disponibles. Nunca inventes clientes, \
pedidos, fechas ni montos.
- Si una pregunta necesita varios datos, llamá a las herramientas en secuencia: el resultado \
de una te da lo que necesita la siguiente (por ejemplo, buscar_cliente entrega el cliente_id \
que usa buscar_pedidos).
- Si una herramienta devuelve un campo "error", leé el detalle y la sugerencia: reintentá con \
otros argumentos si podés corregirlo vos, o pedile una aclaración al usuario si no.
- Cuando ya tengas todos los datos, respondé de forma breve, con montos en pesos \
(por ejemplo $14.500) y sin llamar más herramientas.
- Usá el historial de la conversación: si el usuario dice "el último" o "ese pedido", \
resolvelo con los datos que ya obtuviste antes."""

GrafoAgente = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def recortar_mensajes(mensajes: Sequence[BaseMessage], maximo: int) -> list[BaseMessage]:
    recortados = trim_messages(
        list(mensajes),
        max_tokens=maximo,
        token_counter=len,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    return list(recortados) or list(mensajes)


def construir_grafo(
    llm: BaseChatModel,
    herramientas: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver[Any] | None,
    max_mensajes_contexto: int = 30,
) -> GrafoAgente:
    llm_con_herramientas = llm.bind_tools(list(herramientas))

    async def nodo_modelo(state: AgentState) -> dict[str, Any]:
        contexto = recortar_mensajes(state["messages"], max_mensajes_contexto)
        respuesta = await llm_con_herramientas.ainvoke([SystemMessage(PROMPT_SISTEMA), *contexto])
        pedidas = len(respuesta.tool_calls) if isinstance(respuesta, AIMessage) else 0
        return {"messages": [respuesta], "llamadas_herramientas": pedidas}

    constructor = StateGraph(AgentState)
    constructor.add_node(NODO_MODELO, nodo_modelo)
    constructor.add_node(NODO_HERRAMIENTAS, ToolNode(list(herramientas)))
    constructor.add_edge(START, NODO_MODELO)
    constructor.add_conditional_edges(
        NODO_MODELO,
        tools_condition,
        {"tools": NODO_HERRAMIENTAS, END: END},
    )
    constructor.add_edge(NODO_HERRAMIENTAS, NODO_MODELO)
    return constructor.compile(checkpointer=checkpointer)

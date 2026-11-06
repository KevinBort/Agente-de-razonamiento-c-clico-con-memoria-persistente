import logging
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from agente.graph import GrafoAgente
from agente.state import AgentState
from agente.trace import TrazaReAct, extraer_texto

logger = logging.getLogger(__name__)


def mensaje_limite_alcanzado(recursion_limit: int) -> str:
    return (
        f"Alcancé el límite de {recursion_limit} pasos de razonamiento sin llegar a una "
        "respuesta. Probá reformular la pregunta con más detalle."
    )


async def ejecutar_turno(
    grafo: GrafoAgente,
    pregunta: str,
    thread_id: str,
    recursion_limit: int,
    turno: int = 1,
    traza: TrazaReAct | None = None,
) -> str:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if traza is not None:
        traza.registrar_pregunta(turno, pregunta)

    entrada = cast(AgentState, {"messages": [HumanMessage(pregunta)]})
    respuesta_final = ""
    try:
        async for actualizacion in grafo.astream(
            entrada,
            config=config,
            stream_mode="updates",
        ):
            for nodo, cambios in actualizacion.items():
                if not cambios:
                    continue
                if traza is not None:
                    traza.registrar_actualizacion(turno, nodo, cambios)
                for mensaje in cambios.get("messages", []):
                    if isinstance(mensaje, AIMessage):
                        if mensaje.tool_calls:
                            nombres = [llamada["name"] for llamada in mensaje.tool_calls]
                            logger.info(
                                "thread=%s turno=%d herramientas=%s", thread_id, turno, nombres
                            )
                        else:
                            respuesta_final = extraer_texto(mensaje.content)
    except GraphRecursionError:
        logger.warning(
            "thread=%s turno=%d límite de recursión (%d)", thread_id, turno, recursion_limit
        )
        respuesta_final = mensaje_limite_alcanzado(recursion_limit)
        if traza is not None:
            traza.registrar_actualizacion(
                turno, "sistema", {"messages": [AIMessage(content=respuesta_final)]}
            )
    return respuesta_final

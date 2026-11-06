from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from agente.graph import construir_grafo, recortar_mensajes
from agente.offline import PREGUNTAS_DEMO, ModeloGuionado, respuestas_demo
from agente.persistence import abrir_checkpointer
from agente.runner import ejecutar_turno, mensaje_limite_alcanzado
from agente.tools import HERRAMIENTAS
from agente.trace import TrazaReAct


def _llamada(nombre: str, args: dict[str, object], id_llamada: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": nombre, "args": args, "id": id_llamada, "type": "tool_call"}],
    )


async def test_razonamiento_multipaso_llama_herramientas_en_secuencia(ruta_db: Path) -> None:
    llm = ModeloGuionado(respuestas=respuestas_demo()[4:8])
    traza = TrazaReAct("multipaso")
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer)
        respuesta = await ejecutar_turno(grafo, PREGUNTAS_DEMO[2], "multipaso", 10, 1, traza)

    assert traza.llamadas_por_turno()[1] == 3
    assert "18.100" in respuesta
    herramientas_usadas = [p.herramienta for p in traza.pasos if p.tipo == "decision_herramienta"]
    assert herramientas_usadas == ["buscar_cliente", "buscar_pedidos", "obtener_detalle_pedido"]


async def test_el_estado_acumula_las_llamadas_con_el_reducer(ruta_db: Path) -> None:
    llm = ModeloGuionado(respuestas=respuestas_demo()[:4])
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer)
        for turno, pregunta in enumerate(PREGUNTAS_DEMO[:2], start=1):
            await ejecutar_turno(grafo, pregunta, "acumula", 10, turno)
        estado = await grafo.aget_state({"configurable": {"thread_id": "acumula"}})
    assert estado.values["llamadas_herramientas"] == 2


async def test_el_agente_recuerda_turnos_previos_con_el_mismo_thread_id(ruta_db: Path) -> None:
    llm = ModeloGuionado(respuestas=respuestas_demo()[:4])
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer)
        await ejecutar_turno(grafo, PREGUNTAS_DEMO[0], "memoria", 10, 1)
        await ejecutar_turno(grafo, PREGUNTAS_DEMO[1], "memoria", 10, 2)

    primera_llamada_del_turno_2 = llm.llamadas[2]
    assert any(m.content == PREGUNTAS_DEMO[0] for m in primera_llamada_del_turno_2)
    assert any(isinstance(m, ToolMessage) and "P-1003" in str(m.content)
               for m in primera_llamada_del_turno_2)


async def test_la_memoria_sobrevive_a_reabrir_la_base_de_datos(ruta_db: Path) -> None:
    primer_llm = ModeloGuionado(respuestas=respuestas_demo()[:2])
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(primer_llm, HERRAMIENTAS, checkpointer)
        await ejecutar_turno(grafo, PREGUNTAS_DEMO[0], "persistente", 10, 1)

    segundo_llm = ModeloGuionado(respuestas=[AIMessage(content="ok")])
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(segundo_llm, HERRAMIENTAS, checkpointer)
        await ejecutar_turno(grafo, "¿y el último?", "persistente", 10, 2)

    assert any(m.content == PREGUNTAS_DEMO[0] for m in segundo_llm.llamadas[0])


async def test_threads_distintos_no_comparten_memoria(ruta_db: Path) -> None:
    llm = ModeloGuionado(respuestas=[*respuestas_demo()[:2], AIMessage(content="ok")])
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer)
        await ejecutar_turno(grafo, PREGUNTAS_DEMO[0], "thread-a", 10, 1)
        await ejecutar_turno(grafo, "hola", "thread-b", 10, 1)

    assert not any(m.content == PREGUNTAS_DEMO[0] for m in llm.llamadas[2])


async def test_error_de_herramienta_vuelve_al_modelo_y_permite_reintento(ruta_db: Path) -> None:
    llm = ModeloGuionado(
        respuestas=[
            _llamada("buscar_pedidos", {"cliente_id": "no-es-un-numero"}, "c1"),
            _llamada("buscar_pedidos", {"cliente_id": 102}, "c2"),
            AIMessage(content="Corregido: 3 pedidos."),
        ]
    )
    traza = TrazaReAct("reintento")
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer)
        respuesta = await ejecutar_turno(grafo, "pedidos del 102", "reintento", 10, 1, traza)

    segunda_llamada = llm.llamadas[1]
    mensaje_error = next(m for m in segunda_llamada if isinstance(m, ToolMessage))
    assert "error" in str(mensaje_error.content).lower()
    assert traza.llamadas_por_turno()[1] == 2
    assert respuesta == "Corregido: 3 pedidos."


async def test_error_de_negocio_se_devuelve_al_modelo_como_observacion(ruta_db: Path) -> None:
    llm = ModeloGuionado(respuestas=respuestas_demo()[8:10])
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer)
        await ejecutar_turno(grafo, PREGUNTAS_DEMO[3], "error-negocio", 10, 1)

    observacion = next(m for m in llm.llamadas[1] if isinstance(m, ToolMessage))
    assert "CLIENTE_NO_ENCONTRADO" in str(observacion.content)


async def test_recursion_limit_corta_los_bucles_infinitos(ruta_db: Path) -> None:
    bucle = [_llamada("buscar_pedidos", {"cliente_id": 102}, f"c{i}") for i in range(30)]
    llm = ModeloGuionado(respuestas=bucle)
    async with abrir_checkpointer(ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer)
        respuesta = await ejecutar_turno(grafo, "bucle", "bucle", 4, 1)

    assert respuesta == mensaje_limite_alcanzado(4)
    assert llm.indice < 10


def test_recortar_mensajes_siempre_empieza_en_un_mensaje_humano() -> None:
    mensajes: list[AnyMessage] = [
        HumanMessage("p1"),
        _llamada("buscar_pedidos", {"cliente_id": 102}, "c1"),
        ToolMessage("{}", tool_call_id="c1", name="buscar_pedidos"),
        AIMessage(content="r1"),
        HumanMessage("p2"),
        AIMessage(content="r2"),
    ]
    recortados = recortar_mensajes(mensajes, 3)
    assert isinstance(recortados[0], HumanMessage)
    assert recortados[0].content == "p2"


def test_recortar_mensajes_no_deja_el_contexto_vacio() -> None:
    mensajes: list[AnyMessage] = [HumanMessage("p1"), AIMessage(content="r1")]
    assert recortar_mensajes(mensajes, 1) == mensajes

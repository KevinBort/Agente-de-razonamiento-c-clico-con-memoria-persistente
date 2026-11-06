import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from agente.config import BASE_DIR, ConfiguracionError, cargar_settings
from agente.graph import construir_grafo
from agente.llm import crear_llm
from agente.offline import PREGUNTAS_DEMO, ModeloGuionado, respuestas_demo
from agente.persistence import abrir_checkpointer
from agente.runner import ejecutar_turno
from agente.tools import HERRAMIENTAS
from agente.trace import TrazaReAct

RUTA_TRAZA_POR_DEFECTO = BASE_DIR / "trazas" / "ejemplo_traza.json"


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo multi-paso con memoria y traza ReAct")
    parser.add_argument("--offline", action="store_true", help="modelo simulado, sin API key")
    parser.add_argument("--thread-id", default=f"demo-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--salida", type=Path, default=RUTA_TRAZA_POR_DEFECTO)
    return parser.parse_args()


async def _demo(args: argparse.Namespace) -> None:
    settings = cargar_settings(requerir_api_key=not args.offline)
    llm: BaseChatModel
    if args.offline:
        llm = ModeloGuionado(respuestas=respuestas_demo())
        origen = "modelo simulado (offline): las decisiones del LLM vienen de un guion"
    else:
        llm = crear_llm(settings)
        origen = f"{settings.proveedor}:{settings.modelo}"

    traza = TrazaReAct(
        args.thread_id,
        {
            "origen_del_modelo": origen,
            "recursion_limit": settings.recursion_limit,
            "generada_el": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )

    async with abrir_checkpointer(settings.ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer, settings.max_mensajes_contexto)
        for turno, pregunta in enumerate(PREGUNTAS_DEMO, start=1):
            respuesta = await ejecutar_turno(
                grafo, pregunta, args.thread_id, settings.recursion_limit, turno, traza
            )
            print(f"\n[Turno {turno}] Usuario: {pregunta}\nAgente: {respuesta}")

        config: RunnableConfig = {"configurable": {"thread_id": args.thread_id}}
        estado = await grafo.aget_state(config)

    traza.guardar_json(args.salida)
    print("\n--- Traza ReAct ---\n" + traza.a_texto())
    print(f"\nThread: {args.thread_id} | mensajes en memoria: {len(estado.values['messages'])}")
    print(f"Llamadas a herramientas por turno: {traza.llamadas_por_turno()}")
    print(f"Total acumulado en el estado: {estado.values['llamadas_herramientas']}")
    print(f"Traza guardada en: {args.salida}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_demo(_parsear_argumentos()))
    except ConfiguracionError as error:
        sys.exit(f"Error de configuración: {error}")


if __name__ == "__main__":
    main()

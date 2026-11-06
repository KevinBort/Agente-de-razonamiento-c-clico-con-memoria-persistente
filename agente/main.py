import argparse
import asyncio
import logging
import sys

from agente.config import ConfiguracionError, cargar_settings
from agente.graph import construir_grafo
from agente.llm import crear_llm
from agente.persistence import abrir_checkpointer
from agente.runner import ejecutar_turno
from agente.tools import HERRAMIENTAS
from agente.trace import TrazaReAct

COMANDOS_SALIDA = {"salir", "exit", "quit"}


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente ReAct con memoria persistente")
    parser.add_argument("--thread-id", default="sesion-1", help="id de la conversación")
    parser.add_argument("--pregunta", help="una sola pregunta; si se omite, abre un chat")
    parser.add_argument("--verbose", action="store_true", help="muestra la traza ReAct")
    return parser.parse_args()


async def _chat(args: argparse.Namespace) -> None:
    settings = cargar_settings()
    llm = crear_llm(settings)

    async with abrir_checkpointer(settings.ruta_db) as checkpointer:
        grafo = construir_grafo(llm, HERRAMIENTAS, checkpointer, settings.max_mensajes_contexto)
        print(f"Sesión: {args.thread_id} | modelo: {settings.proveedor}:{settings.modelo}")

        turno = 0
        while True:
            if args.pregunta and turno == 1:
                break
            pregunta = args.pregunta or await asyncio.to_thread(input, "\nVos: ")
            if pregunta.strip().lower() in COMANDOS_SALIDA:
                break
            if not pregunta.strip():
                continue

            turno += 1
            traza = TrazaReAct(args.thread_id)
            respuesta = await ejecutar_turno(
                grafo, pregunta, args.thread_id, settings.recursion_limit, turno, traza
            )
            if args.verbose:
                print(f"\n{traza.a_texto()}")
            print(f"\nAgente: {respuesta}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_chat(_parsear_argumentos()))
    except ConfiguracionError as error:
        sys.exit(f"Error de configuración: {error}")


if __name__ == "__main__":
    main()

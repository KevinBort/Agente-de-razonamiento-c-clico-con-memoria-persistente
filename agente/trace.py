import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

TIPO_PREGUNTA = "pregunta_usuario"
TIPO_DECISION = "decision_herramienta"
TIPO_OBSERVACION = "observacion"
TIPO_RESPUESTA = "respuesta_final"


def extraer_texto(contenido: str | list[str | dict[str, Any]]) -> str:
    if isinstance(contenido, str):
        return contenido
    partes: list[str] = []
    for bloque in contenido:
        if isinstance(bloque, str):
            partes.append(bloque)
        elif bloque.get("type") == "text":
            partes.append(str(bloque.get("text", "")))
    return "".join(partes)


def _parsear_resultado(texto: str) -> Any:
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return texto


@dataclass(slots=True)
class PasoTraza:
    paso: int
    turno: int
    tipo: str
    nodo: str
    contenido: str | None = None
    herramienta: str | None = None
    argumentos: dict[str, Any] | None = None
    resultado: Any = None


class TrazaReAct:
    def __init__(self, thread_id: str, metadatos: dict[str, Any] | None = None) -> None:
        self.thread_id = thread_id
        self.metadatos: dict[str, Any] = metadatos or {}
        self.pasos: list[PasoTraza] = []

    def _agregar(self, **campos: Any) -> None:
        self.pasos.append(PasoTraza(paso=len(self.pasos) + 1, **campos))

    def registrar_pregunta(self, turno: int, pregunta: str) -> None:
        self._agregar(turno=turno, tipo=TIPO_PREGUNTA, nodo="usuario", contenido=pregunta)

    def registrar_actualizacion(self, turno: int, nodo: str, actualizacion: dict[str, Any]) -> None:
        mensajes: list[AnyMessage] = actualizacion.get("messages", [])
        for mensaje in mensajes:
            if isinstance(mensaje, AIMessage) and mensaje.tool_calls:
                razonamiento = extraer_texto(mensaje.content) or None
                for llamada in mensaje.tool_calls:
                    self._agregar(
                        turno=turno,
                        tipo=TIPO_DECISION,
                        nodo=nodo,
                        contenido=razonamiento,
                        herramienta=llamada["name"],
                        argumentos=dict(llamada["args"]),
                    )
            elif isinstance(mensaje, AIMessage):
                self._agregar(
                    turno=turno,
                    tipo=TIPO_RESPUESTA,
                    nodo=nodo,
                    contenido=extraer_texto(mensaje.content),
                )
            elif isinstance(mensaje, ToolMessage):
                self._agregar(
                    turno=turno,
                    tipo=TIPO_OBSERVACION,
                    nodo=nodo,
                    herramienta=mensaje.name,
                    resultado=_parsear_resultado(extraer_texto(mensaje.content)),
                )

    def llamadas_por_turno(self) -> dict[int, int]:
        conteo: dict[int, int] = {}
        for paso in self.pasos:
            if paso.tipo == TIPO_DECISION:
                conteo[paso.turno] = conteo.get(paso.turno, 0) + 1
        return conteo

    def a_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "metadatos": self.metadatos,
            "llamadas_a_herramientas_por_turno": {
                str(turno): n for turno, n in self.llamadas_por_turno().items()
            },
            "pasos": [asdict(paso) for paso in self.pasos],
        }

    def guardar_json(self, ruta: Path) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps(self.a_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def a_texto(self) -> str:
        lineas: list[str] = []
        for p in self.pasos:
            if p.tipo == TIPO_PREGUNTA:
                lineas.append(f"\n[Turno {p.turno}] Paso {p.paso} | Usuario: {p.contenido}")
            elif p.tipo == TIPO_DECISION:
                argumentos = ", ".join(f"{k}={v!r}" for k, v in (p.argumentos or {}).items())
                lineas.append(f"  Paso {p.paso} | Acción -> {p.herramienta}({argumentos})")
            elif p.tipo == TIPO_OBSERVACION:
                resultado = json.dumps(p.resultado, ensure_ascii=False)
                lineas.append(f"  Paso {p.paso} | Observación <- {p.herramienta}: {resultado}")
            else:
                lineas.append(f"  Paso {p.paso} | Respuesta: {p.contenido}")
        return "\n".join(lineas).lstrip("\n")

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

Proveedor = Literal["openai", "anthropic"]

MODELOS_POR_DEFECTO: dict[Proveedor, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
VARIABLE_API_KEY: dict[Proveedor, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ConfiguracionError(RuntimeError):
    """Se lanza cuando falta o es inválida alguna variable de entorno."""


@dataclass(frozen=True, slots=True)
class Settings:
    proveedor: Proveedor
    modelo: str
    api_key: str
    ruta_db: Path
    recursion_limit: int
    max_mensajes_contexto: int = 30


def _leer_proveedor() -> Proveedor:
    valor = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if valor == "openai":
        return "openai"
    if valor == "anthropic":
        return "anthropic"
    raise ConfiguracionError(f"LLM_PROVIDER inválido: '{valor}'. Usá 'openai' o 'anthropic'.")


def _leer_api_key(proveedor: Proveedor) -> str:
    nombre = VARIABLE_API_KEY[proveedor]
    valor = os.getenv(nombre, "").strip()
    if not valor or "xxxx" in valor.lower() or valor == "api key xx":
        raise ConfiguracionError(
            f"Falta configurar {nombre} en {BASE_DIR / '.env'} (vacía o con valor de ejemplo)."
        )
    return valor


def _leer_entero_positivo(nombre: str, por_defecto: int) -> int:
    texto = os.getenv(nombre, "").strip() or str(por_defecto)
    try:
        valor = int(texto)
    except ValueError as exc:
        raise ConfiguracionError(f"{nombre} debe ser un entero, recibí '{texto}'.") from exc
    if valor < 1:
        raise ConfiguracionError(f"{nombre} debe ser mayor o igual a 1.")
    return valor


def cargar_settings(requerir_api_key: bool = True) -> Settings:
    proveedor = _leer_proveedor()
    ruta = Path(os.getenv("DB_PATH", "").strip() or "data/checkpoints.sqlite")
    if not ruta.is_absolute():
        ruta = BASE_DIR / ruta
    return Settings(
        proveedor=proveedor,
        modelo=os.getenv("LLM_MODEL", "").strip() or MODELOS_POR_DEFECTO[proveedor],
        api_key=_leer_api_key(proveedor) if requerir_api_key else "",
        ruta_db=ruta,
        recursion_limit=_leer_entero_positivo("RECURSION_LIMIT", 10),
    )


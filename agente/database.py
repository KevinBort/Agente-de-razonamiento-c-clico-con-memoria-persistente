import asyncio
import unicodedata
from dataclasses import dataclass
from datetime import date

LATENCIA_SIMULADA_SEGUNDOS = 0.05


@dataclass(frozen=True, slots=True)
class Cliente:
    cliente_id: int
    nombre: str
    email: str


@dataclass(frozen=True, slots=True)
class ItemPedido:
    producto: str
    cantidad: int
    precio_unitario: int

    @property
    def subtotal(self) -> int:
        return self.cantidad * self.precio_unitario


@dataclass(frozen=True, slots=True)
class Pedido:
    pedido_id: str
    cliente_id: int
    fecha: date
    items: tuple[ItemPedido, ...]

    @property
    def total(self) -> int:
        return sum(item.subtotal for item in self.items)


CLIENTES: tuple[Cliente, ...] = (
    Cliente(102, "Lucía Fernández", "lucia.fernandez@example.com"),
    Cliente(205, "Martín Gómez", "martin.gomez@example.com"),
    Cliente(310, "Carolina Ruiz", "carolina.ruiz@example.com"),
)

PEDIDOS: tuple[Pedido, ...] = (
    Pedido("P-1001", 102, date(2026, 8, 14), (
        ItemPedido("Teclado mecánico", 1, 3500),
        ItemPedido("Mouse inalámbrico", 1, 1000),
    )),
    Pedido("P-1002", 102, date(2026, 9, 3), (
        ItemPedido("Monitor 24 pulgadas", 1, 3000),
    )),
    Pedido("P-1003", 102, date(2026, 10, 2), (
        ItemPedido("Webcam HD", 1, 2500),
        ItemPedido("Soporte para notebook", 2, 2250),
    )),
    Pedido("P-2001", 205, date(2026, 9, 5), (
        ItemPedido("Auriculares", 1, 5200),
        ItemPedido("Cable HDMI", 2, 800),
    )),
    Pedido("P-2002", 205, date(2026, 10, 20), (
        ItemPedido("Disco SSD 1TB", 1, 9800),
        ItemPedido("Funda protectora", 1, 1500),
    )),
)


def normalizar(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto.lower().strip())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


async def _simular_latencia() -> None:
    await asyncio.sleep(LATENCIA_SIMULADA_SEGUNDOS)


async def obtener_cliente(cliente_id: int) -> Cliente | None:
    await _simular_latencia()
    return next((c for c in CLIENTES if c.cliente_id == cliente_id), None)


async def buscar_clientes_por_nombre(nombre: str) -> list[Cliente]:
    await _simular_latencia()
    consulta = normalizar(nombre)
    if not consulta:
        return []
    return [c for c in CLIENTES if consulta in normalizar(c.nombre)]


async def listar_ids_clientes() -> list[int]:
    await _simular_latencia()
    return [c.cliente_id for c in CLIENTES]


async def listar_pedidos_de_cliente(cliente_id: int) -> list[Pedido]:
    await _simular_latencia()
    pedidos = [p for p in PEDIDOS if p.cliente_id == cliente_id]
    return sorted(pedidos, key=lambda p: p.fecha)


async def obtener_pedido(pedido_id: str) -> Pedido | None:
    await _simular_latencia()
    buscado = pedido_id.strip().upper()
    return next((p for p in PEDIDOS if p.pedido_id == buscado), None)

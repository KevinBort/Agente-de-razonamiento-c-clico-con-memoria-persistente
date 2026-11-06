from typing import Any

from langchain_core.tools import BaseTool, tool

from agente import database


@tool(parse_docstring=True)
async def buscar_cliente(nombre: str) -> dict[str, Any]:
    """Busca clientes en la base de datos por nombre o apellido y devuelve su cliente_id.

    Usá esta herramienta PRIMERO cuando el usuario mencione a un cliente por su nombre
    (por ejemplo "Martín Gómez") y no conozcas su identificador numérico. El cliente_id
    que devuelve es el dato que necesitan las otras herramientas (buscar_pedidos).
    La búsqueda ignora mayúsculas y tildes y acepta nombres parciales ("gomez").
    No la uses si el usuario ya te dio un cliente_id numérico.
    Si no hay coincidencias devuelve un campo "error"; en ese caso pedile al usuario
    que confirme el nombre. Si hay varias coincidencias, pedile que aclare cuál es.

    Args:
        nombre: Nombre completo o parcial del cliente, por ejemplo "Lucía Fernández" o "Gómez".
    """
    clientes = await database.buscar_clientes_por_nombre(nombre)
    if not clientes:
        return {
            "error": "CLIENTE_NO_ENCONTRADO",
            "detalle": f"No hay clientes cuyo nombre contenga '{nombre}'.",
            "sugerencia": "Pedile al usuario que revise la ortografía o que dé el cliente_id.",
        }
    return {
        "cantidad": len(clientes),
        "coincidencias": [
            {"cliente_id": c.cliente_id, "nombre": c.nombre, "email": c.email} for c in clientes
        ],
    }


@tool(parse_docstring=True)
async def buscar_pedidos(cliente_id: int) -> dict[str, Any]:
    """Lista los pedidos de un cliente y devuelve la cantidad de pedidos y el total gastado.

    Usá esta herramienta cuando el usuario pregunte cuántos pedidos hizo un cliente, cuánto
    gastó en total, cuándo compró o cuál fue su pedido más reciente. Requiere el cliente_id
    numérico: si solo tenés el nombre, llamá antes a buscar_cliente.
    Devuelve "cantidad_pedidos", "total" (suma en pesos de todos los pedidos) y "pedidos",
    una lista ordenada del más antiguo al más reciente (el último elemento es el último
    pedido), con pedido_id, fecha y total de cada uno. NO incluye los productos: para ver
    qué contenía un pedido usá obtener_detalle_pedido con su pedido_id.
    Si el cliente no existe devuelve un campo "error" con "ids_parecidos" (posibles
    correcciones): no inventes datos, confirmá con el usuario cuál es el ID correcto.

    Args:
        cliente_id: Identificador numérico del cliente, por ejemplo 102.
    """
    cliente = await database.obtener_cliente(cliente_id)
    if cliente is None:
        ids = await database.listar_ids_clientes()
        parecidos = [i for i in ids if str(i) in str(cliente_id) or str(cliente_id) in str(i)]
        return {
            "error": "CLIENTE_NO_ENCONTRADO",
            "detalle": f"No existe un cliente con cliente_id={cliente_id}.",
            "ids_parecidos": parecidos,
            "sugerencia": "Confirmá el ID con el usuario o buscá al cliente por nombre.",
        }
    pedidos = await database.listar_pedidos_de_cliente(cliente_id)
    return {
        "cliente_id": cliente.cliente_id,
        "cliente": cliente.nombre,
        "cantidad_pedidos": len(pedidos),
        "total": sum(p.total for p in pedidos),
        "pedidos": [
            {"pedido_id": p.pedido_id, "fecha": p.fecha.isoformat(), "total": p.total}
            for p in pedidos
        ],
    }


@tool(parse_docstring=True)
async def obtener_detalle_pedido(pedido_id: str) -> dict[str, Any]:
    """Devuelve los productos, cantidades y precios de UN pedido específico.

    Usá esta herramienta cuando el usuario quiera saber qué compró un cliente, qué contenía
    un pedido o cuánto costó cada producto. Necesita el pedido_id con formato "P-1003", que
    se obtiene de buscar_pedidos (no lo adivines ni lo inventes). Para consultar "el último
    pedido" tomá el pedido_id del último elemento de la lista que devuelve buscar_pedidos.
    Si el pedido no existe devuelve un campo "error".

    Args:
        pedido_id: Identificador del pedido, por ejemplo "P-1003".
    """
    pedido = await database.obtener_pedido(pedido_id)
    if pedido is None:
        return {
            "error": "PEDIDO_NO_ENCONTRADO",
            "detalle": f"No existe el pedido '{pedido_id}'.",
            "sugerencia": "Obtené un pedido_id válido con buscar_pedidos.",
        }
    return {
        "pedido_id": pedido.pedido_id,
        "cliente_id": pedido.cliente_id,
        "fecha": pedido.fecha.isoformat(),
        "items": [
            {
                "producto": i.producto,
                "cantidad": i.cantidad,
                "precio_unitario": i.precio_unitario,
                "subtotal": i.subtotal,
            }
            for i in pedido.items
        ],
        "total": pedido.total,
    }


HERRAMIENTAS: list[BaseTool] = [buscar_cliente, buscar_pedidos, obtener_detalle_pedido]

from agente.tools import (
    HERRAMIENTAS,
    buscar_cliente,
    buscar_pedidos,
    obtener_detalle_pedido,
)


async def test_buscar_pedidos_replica_el_ejemplo_del_enunciado() -> None:
    resultado = await buscar_pedidos.ainvoke({"cliente_id": 102})
    assert resultado["cantidad_pedidos"] == 3
    assert resultado["total"] == 14500
    assert resultado["pedidos"][-1]["pedido_id"] == "P-1003"


async def test_buscar_pedidos_cliente_inexistente_devuelve_error_con_sugerencias() -> None:
    resultado = await buscar_pedidos.ainvoke({"cliente_id": 1020})
    assert resultado["error"] == "CLIENTE_NO_ENCONTRADO"
    assert resultado["ids_parecidos"] == [102]


async def test_buscar_cliente_ignora_tildes_y_mayusculas() -> None:
    resultado = await buscar_cliente.ainvoke({"nombre": "MARTIN gomez"})
    assert resultado["coincidencias"][0]["cliente_id"] == 205


async def test_buscar_cliente_sin_coincidencias() -> None:
    resultado = await buscar_cliente.ainvoke({"nombre": "Nadie Existente"})
    assert resultado["error"] == "CLIENTE_NO_ENCONTRADO"


async def test_detalle_pedido_calcula_total() -> None:
    resultado = await obtener_detalle_pedido.ainvoke({"pedido_id": "p-1003"})
    assert resultado["total"] == 7000
    assert len(resultado["items"]) == 2


async def test_detalle_pedido_inexistente() -> None:
    resultado = await obtener_detalle_pedido.ainvoke({"pedido_id": "P-9999"})
    assert resultado["error"] == "PEDIDO_NO_ENCONTRADO"


def test_todas_las_herramientas_tienen_descripcion_detallada() -> None:
    for herramienta in HERRAMIENTAS:
        assert len(herramienta.description) > 150, herramienta.name
        for argumento in herramienta.args.values():
            assert argumento.get("description"), herramienta.name

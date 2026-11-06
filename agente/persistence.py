from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def abrir_checkpointer(ruta_db: Path) -> AsyncIterator[AsyncSqliteSaver]:
    ruta_db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(ruta_db)) as checkpointer:
        yield checkpointer

from pathlib import Path

import pytest


@pytest.fixture
def ruta_db(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints.sqlite"

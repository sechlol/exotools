import os
from pathlib import Path

import pytest
from astropy.table import QTable


_CURRENT_DIR = Path(os.path.realpath(__file__)).parent
_TEST_ASSETS_DIR = _CURRENT_DIR / "test_assets"

TEST_TMP_DIR = _CURRENT_DIR / "tmp"
TEST_ASSETS_QTABLES = _TEST_ASSETS_DIR / "qtables"


@pytest.fixture(scope="session")
def all_test_qtables() -> dict[str, QTable]:
    """Load all qtables in TEST_ASSETS_QTABLES as a dict"""
    qtables = {}
    for qtable in TEST_ASSETS_QTABLES.iterdir():
        if qtable.suffix == ".ecsv":
            qtables[qtable.stem] = QTable.read(qtable)
    return qtables

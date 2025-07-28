import os
from pathlib import Path

import pytest
from astropy.table import QTable

from exotools.utils.qtable_utils import QTableHeader, RootQTableHeader

_CURRENT_DIR = Path(os.path.realpath(__file__)).parent
_TEST_ASSETS_DIR = _CURRENT_DIR / "test_assets"

TEST_TMP_DIR = _CURRENT_DIR / "tmp"
TEST_ASSETS_QTABLES = _TEST_ASSETS_DIR / "qtables"


@pytest.fixture(scope="session")
def all_test_qtables() -> dict[str, QTable]:
    return load_all_test_qtables()


@pytest.fixture(scope="session")
def all_test_headers() -> dict[str, QTableHeader]:
    return load_all_test_headers()


@pytest.fixture(scope="session")
def all_test_qtables_and_headers() -> dict[str, tuple[QTable, QTableHeader]]:
    return load_all_test_qtables_and_headers()


def load_all_test_qtables() -> dict[str, QTable]:
    """Load all qtables in TEST_ASSETS_QTABLES as a dict"""
    qtables = {}
    for path in TEST_ASSETS_QTABLES.iterdir():
        if path.suffix == ".ecsv":
            qtables[path.stem] = QTable.read(path)
    return qtables


def load_all_test_headers() -> dict[str, QTableHeader]:
    """Load all headers in TEST_ASSETS_QTABLES as a dict"""
    headers = {}
    for path in TEST_ASSETS_QTABLES.iterdir():
        if path.suffix == ".json":
            with open(path, "r") as f:
                headers[path.stem.removesuffix("_header")] = RootQTableHeader.model_validate_json(f.read()).root
    return headers


def load_all_test_qtables_and_headers() -> dict[str, tuple[QTable, QTableHeader]]:
    headers = load_all_test_headers()
    qtables = load_all_test_qtables()
    assert set(headers.keys()) == set(qtables.keys())
    return {name: (qtable, headers[name]) for name, qtable in qtables.items()}

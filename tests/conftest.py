import pytest
from astropy.table import QTable

from tests.paths import TEST_ASSETS_QTABLES


@pytest.fixture(scope="session")
def all_test_qtables() -> dict[str, QTable]:
    """Load all qtables in TEST_ASSETS_QTABLES as a dict"""
    qtables = {}
    for qtable in TEST_ASSETS_QTABLES.iterdir():
        if qtable.suffix == ".ecsv":
            qtables[qtable.stem] = QTable.read(qtable)
    return qtables

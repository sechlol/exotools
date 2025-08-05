import os
from pathlib import Path

import pytest
from astropy.table import QTable
from lightkurve import LightCurve

from exotools import GaiaDB, KnownExoplanetsDataset, StarSystemDB
from exotools.datasets import GaiaParametersDataset
from exotools.db.lightcurve_db import load_lightcurve
from exotools.io import EcsvStorage
from exotools.utils.qtable_utils import QTableHeader, RootQTableHeader

_CURRENT_DIR = Path(os.path.realpath(__file__)).parent
_TEST_ASSETS_DIR = _CURRENT_DIR / "assets"

TEST_TMP_DIR = _CURRENT_DIR / "tmp"
TEST_ASSETS_QTABLES = _TEST_ASSETS_DIR / "qtables"
TEST_ASSETS_LC = _TEST_ASSETS_DIR / "lightcurves"
_TEST_STORAGE = EcsvStorage(TEST_ASSETS_QTABLES)


@pytest.fixture(scope="module")
def all_test_qtables() -> dict[str, QTable]:
    return load_all_test_qtables()


@pytest.fixture(scope="module")
def all_test_headers() -> dict[str, QTableHeader]:
    return load_all_test_headers()


@pytest.fixture(scope="module")
def all_test_qtables_and_headers() -> dict[str, tuple[QTable, QTableHeader]]:
    return load_all_test_qtables_and_headers()


@pytest.fixture(scope="module")
def all_test_lightcurves() -> dict[str, LightCurve]:
    return load_all_test_lightcurves()


def load_all_test_qtables() -> dict[str, QTable]:
    """Load all qtables in TEST_ASSETS_QTABLES as a dict"""
    qtables = {}
    for path in TEST_ASSETS_QTABLES.iterdir():
        if path.suffix == ".ecsv":
            qtables[path.stem] = _TEST_STORAGE.read_qtable(path.stem)
    return qtables


def load_all_test_lightcurves() -> dict[str, LightCurve]:
    all_data = {}
    for path, dirs, files in os.walk(TEST_ASSETS_LC):
        for file_name in files:
            if ".fits" in file_name:
                file_path = Path(path) / file_name
                all_data[file_name.removesuffix(".fits")] = load_lightcurve(file_path)

    return all_data


def load_all_test_headers() -> dict[str, QTableHeader]:
    """Load all headers in TEST_ASSETS_QTABLES as a dict"""
    headers = {}
    for path in TEST_ASSETS_QTABLES.iterdir():
        if path.suffix == ".json":
            with open(path, "r") as f:
                headers[path.stem.removesuffix("_header")] = RootQTableHeader.model_validate_json(f.read()).root
    return headers


def load_all_test_qtables_and_headers() -> dict[str, tuple[QTable, QTableHeader]]:
    """Load all qtables and headers in TEST_ASSETS_QTABLES as a dict"""
    qtables = load_all_test_qtables()
    headers = load_all_test_headers()

    result = {}
    for name in qtables:
        if name in headers:
            result[name] = (qtables[name], headers[name])
    return result


# Dataset-specific fixtures for testing
@pytest.fixture(scope="module")
def known_exoplanets_test_data(all_test_qtables_and_headers) -> tuple[QTable, QTableHeader]:
    """Test data for known exoplanets dataset"""
    return all_test_qtables_and_headers["known_exoplanets"]


@pytest.fixture(scope="module")
def candidate_exoplanets_test_data(all_test_qtables_and_headers) -> tuple[QTable, QTableHeader]:
    """Test data for candidate exoplanets dataset"""
    return all_test_qtables_and_headers["candidate_exoplanets"]


@pytest.fixture(scope="module")
def gaia_parameters_test_data(all_test_qtables_and_headers) -> tuple[QTable, QTableHeader]:
    """Test data for gaia parameters dataset"""
    return all_test_qtables_and_headers["gaia"]


@pytest.fixture(scope="module")
def tess_observations_test_data(all_test_qtables_and_headers) -> tuple[QTable, QTableHeader]:
    """Test data for TESS observations dataset"""
    return all_test_qtables_and_headers["tess_observations"]


@pytest.fixture(scope="module")
def tess_tic_test_data(all_test_qtables_and_headers) -> tuple[QTable, QTableHeader]:
    """Test data for TESS observations dataset"""
    return all_test_qtables_and_headers["tess_tic"]


@pytest.fixture(scope="module")
def tess_tic_by_id_test_data(all_test_qtables_and_headers) -> tuple[QTable, QTableHeader]:
    """Test data for TESS observations dataset"""
    return all_test_qtables_and_headers["tess_tic_by_id"]


@pytest.fixture(scope="module")
def star_system_test_db() -> StarSystemDB:
    """Test data for StarSystem dataset"""
    return KnownExoplanetsDataset(storage=_TEST_STORAGE).load_star_system_dataset()


@pytest.fixture(scope="module")
def gaia_test_db() -> GaiaDB:
    """Test data for ExoDB dataset"""
    return GaiaParametersDataset(storage=_TEST_STORAGE).load_gaia_parameters_dataset()


@pytest.fixture(scope="module")
def lightcurve_test_paths() -> dict[int, Path]:
    """Test paths for lightcurve FITS files"""
    path_map = {}

    # Iterate through TIC ID directories
    for path, dirs, files in os.walk(TEST_ASSETS_LC):
        for file in files:
            if ".fits" in file:
                obs_id = int(file.removesuffix(".fits"))
                path_map[obs_id] = Path(path) / file

    return path_map

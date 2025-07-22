from typing import Optional

from .exo_db import ExoDB
from .gaia_db import GaiaDB
from .lightcurve_db import LightcurveDB
from .starsystem_db import StarSystemDB
from .tic_db import TicDB
from .toi_db import CandidateDB
from .urls_db import UrlsDB
from utils.io import read_qtable, get_file_paths_in_subfolder


def load_star_system_db() -> StarSystemDB:
    dataset = read_qtable(
        file_path=configs.DATASET_KNOWN_EXOPLANETS, file_name=configs.NAME_REDUCED_TRANSITING_EXOPLANETS
    )
    ExoDB.convert_time_columns(dataset)
    dataset = StarSystemDB.preprocess_dataset(dataset)
    return StarSystemDB(dataset)


def load_sunlike_star_tic_db() -> TicDB:
    dataset = read_qtable(configs.DATASET_SUNLIKE_STARS)
    return TicDB(dataset)


def load_toi_tic_db() -> TicDB:
    dataset = read_qtable(configs.DATASET_SUNLIKE_STARS, file_name="toi")
    return TicDB(dataset)


def load_url_db() -> UrlsDB:
    return UrlsDB(read_qtable(configs.DATASET_URLS))


def load_lightcurve_db() -> LightcurveDB:
    downloaded_lc = get_file_paths_in_subfolder(configs.MAST_FOLDER, file_extension="fits")
    dataset = LightcurveDB.path_map_to_qtable(downloaded_lc)
    return LightcurveDB(dataset)

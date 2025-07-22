from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.table import QTable

from ._downloaders.exoplanets_downloader import KnownExoplanetsDownloader
from ._downloaders.gaia_downloader import GaiaDownloader
from ._loaders.exo_db import ExoDB
from ._loaders.gaia_db import GaiaDB
from .utils.qtable_utils import read_qtable


class KnownExoplanetsDataset:
    _DATASET_NAME_EXO = "known_exoplanets"
    _DATASET_NAME_GAIA = "gaia_astro_parameters"

    def __init__(self, storage_folder_path: Path):
        self._folder_path = storage_folder_path

    def load_known_exoplanets_dataset(self, with_gaia_star_data: bool = True) -> Optional[ExoDB]:
        gaia_db = None
        if with_gaia_star_data:
            try:
                gaia_qtable = read_qtable(file_path=self._folder_path, file_name=self._DATASET_NAME_GAIA)
            except ValueError:
                print(
                    "Gaia dataset not found. You need to download it first by "
                    "calling download_known_exoplanets(with_gaia_star_data=True)."
                )
                return None
            gaia_db = _create_gaia_db(gaia_qtable)
        try:
            exo_qtable = read_qtable(file_path=self._folder_path, file_name=self._DATASET_NAME_EXO)
        except ValueError:
            print("Exoplanets dataset not found. You need to download it first by calling download_known_exoplanets().")
            return None

        return _create_exo_db(exo_dataset=exo_qtable, gaia_db=gaia_db)

    def download_known_exoplanets(self, limit: Optional[int] = None, with_gaia_star_data: bool = False) -> ExoDB:
        self._folder_path.mkdir(parents=True, exist_ok=True)

        print("Preparing to download known exoplanets dataset...")
        exo_qtable = KnownExoplanetsDownloader().download(
            limit=limit,
            out_folder_path=self._folder_path,
            out_file_name=self._DATASET_NAME_EXO,
        )

        if with_gaia_star_data:
            gaia_ids = np.unique(exo_qtable["gaia_id"].value).tolist()
            gaia_db = self._download_gaia_dr3_data(gaia_ids=gaia_ids)
        else:
            gaia_db = None

        return _create_exo_db(exo_qtable, gaia_db)

    def _download_gaia_dr3_data(self, gaia_ids: Sequence[int]):
        print(f"Preparing to download Gaia DR3 data for {len(gaia_ids)} stars...")
        gaia_qtable = GaiaDownloader().download_by_id(
            ids=gaia_ids,
            out_folder_path=self._folder_path,
            out_file_name=self._DATASET_NAME_GAIA,
        )

        return _create_gaia_db(gaia_qtable)


def _create_exo_db(exo_dataset: QTable, gaia_db: Optional[GaiaDB] = None) -> ExoDB:
    ExoDB.preprocess_dataset(exo_dataset)
    ExoDB.compute_bounds(exo_dataset)
    ExoDB.convert_time_columns(exo_dataset)

    if gaia_db:
        ExoDB.impute_stellar_parameters(exo_dataset, gaia_db.view)
    return ExoDB(exo_dataset)


def _create_gaia_db(gaia_dataset: QTable) -> GaiaDB:
    GaiaDB.impute_radius(gaia_dataset)
    GaiaDB.compute_mean_temperature(gaia_dataset)
    GaiaDB.compute_habitable_zone(gaia_dataset)
    return GaiaDB(gaia_dataset)

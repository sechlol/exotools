from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.table import QTable

from .downloaders.exoplanets_downloader import KnownExoplanetsDownloader
from .downloaders.gaia_downloader import GaiaDownloader
from .db.exo_db import ExoDB
from .db.gaia_db import GaiaDB
from .db.starsystem_db import StarSystemDB
from .utils.exoplanet_dataset_reducer import reduce_exoplanet_dataset
from .utils.qtable_utils import read_qtable, save_qtable


class KnownExoplanetsDataset:
    _DATASET_EXO = "known_exoplanets"
    _DATASET_EXO_REDUCED = "known_exoplanets_reduced"
    _DATASET_GAIA = "known_gaia_astro_parameters"

    def __init__(self, storage_folder_path: Path):
        self._folder_path = storage_folder_path

    def load_known_exoplanets_dataset(self, with_gaia_star_data: bool = True) -> Optional[ExoDB]:
        gaia_db = None
        if with_gaia_star_data:
            try:
                gaia_qtable = read_qtable(file_path=self._folder_path, file_name=self._DATASET_GAIA)
            except ValueError:
                print(
                    "Gaia dataset not found. You need to download it first by "
                    "calling download_known_exoplanets(with_gaia_star_data=True)."
                )
                return None
            gaia_db = _create_gaia_db(gaia_qtable)
        try:
            exo_qtable = read_qtable(file_path=self._folder_path, file_name=self._DATASET_EXO)
        except ValueError:
            print(
                "Known Exoplanets dataset not found. "
                "You need to download it first by calling download_known_exoplanets()."
            )
            return None

        return _create_exo_db(exo_dataset=exo_qtable, gaia_db=gaia_db)

    def load_star_system_dataset(self) -> Optional[StarSystemDB]:
        try:
            # Try to load reduced dataset
            reduced_exo_dataset = read_qtable(file_path=self._folder_path, file_name=self._DATASET_EXO_REDUCED)
            return _create_star_system_db(reduced_exo_dataset)
        except ValueError:
            # If it doesn't exist, compute it from the full datasets
            try:
                gaia_qtable = read_qtable(file_path=self._folder_path, file_name=self._DATASET_GAIA)
                exo_qtable = read_qtable(file_path=self._folder_path, file_name=self._DATASET_EXO)
            except ValueError:
                print(
                    "Gaia dataset not found. You need to download it first by "
                    "calling download_known_exoplanets(with_gaia_star_data=True)."
                )
                return None

            return self._create_star_system_db_from_scratch(exo_dataset=exo_qtable, gaia_dataset=gaia_qtable)

    def download_known_exoplanets(self, limit: Optional[int] = None, with_gaia_star_data: bool = False) -> ExoDB:
        self._folder_path.mkdir(parents=True, exist_ok=True)

        print("Preparing to download known exoplanets dataset...")
        exo_qtable = KnownExoplanetsDownloader().download(
            limit=limit,
            out_folder_path=self._folder_path,
            out_file_name=self._DATASET_EXO,
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
            out_file_name=self._DATASET_GAIA,
        )

        return _create_gaia_db(gaia_qtable)

    def _create_star_system_db_from_scratch(self, exo_dataset: QTable, gaia_dataset: QTable) -> StarSystemDB:
        gaia_db = _create_gaia_db(gaia_dataset)

        # Disable parsing Time columns; we need them as Quantities to copy units to the transiting qtable.
        exo_db = _create_exo_db(exo_dataset=exo_dataset, gaia_db=gaia_db, convert_time_columns=False)

        # Reduce exoplanet dataset to a compact representation
        reduced_exo_dataset, header = reduce_exoplanet_dataset(exo_db=exo_db)

        # Store the reduced dataset for future use
        save_qtable(
            table=reduced_exo_dataset,
            header=header,
            file_path=self._folder_path,
            file_name=self._DATASET_EXO_REDUCED,
        )

        return _create_star_system_db(reduced_exo_dataset)


def _create_exo_db(exo_dataset: QTable, gaia_db: Optional[GaiaDB] = None, convert_time_columns: bool = True) -> ExoDB:
    ExoDB.preprocess_dataset(exo_dataset)
    ExoDB.compute_bounds(exo_dataset)

    # It's useful to disable parsing Time columns if we need them as Quantities,
    # for example to copy units to another qtable.
    if convert_time_columns:
        ExoDB.convert_time_columns(exo_dataset)

    if gaia_db:
        ExoDB.impute_stellar_parameters(exo_dataset, gaia_db.view)
    return ExoDB(exo_dataset)


def _create_gaia_db(gaia_dataset: QTable) -> GaiaDB:
    GaiaDB.impute_radius(gaia_dataset)
    GaiaDB.compute_mean_temperature(gaia_dataset)
    GaiaDB.compute_habitable_zone(gaia_dataset)
    return GaiaDB(gaia_dataset)


def _create_star_system_db(reduced_exo_dataset: QTable) -> StarSystemDB:
    # Now it's safe to parse Time columns
    ExoDB.convert_time_columns(reduced_exo_dataset)
    reduced_exo_dataset = StarSystemDB.preprocess_dataset(reduced_exo_dataset)
    return StarSystemDB(reduced_exo_dataset)

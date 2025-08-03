import logging
from typing import Optional, Sequence

import numpy as np
from astropy.table import QTable

from exotools.datasets.gaia_parameters import GaiaParametersDataset
from exotools.db import ExoDB, GaiaDB, StarSystemDB
from exotools.downloaders import KnownExoplanetsDownloader
from exotools.io import BaseStorage

from ._exoplanet_dataset_reducer import reduce_exoplanet_dataset
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class KnownExoplanetsDataset(BaseDataset):
    _DATASET_EXO = "known_exoplanets"

    def __init__(self, dataset_tag: Optional[str] = None, storage: Optional[BaseStorage] = None):
        super().__init__(dataset_name=self._DATASET_EXO, dataset_tag=dataset_tag, storage=storage)
        self._gaia_dataset = GaiaParametersDataset(dataset_tag=self._DATASET_EXO, storage=storage)
        self._reduced_dataset_name = self.name + "_reduced"

    def load_known_exoplanets_dataset(self, with_gaia_star_data: bool = False) -> Optional[ExoDB]:
        gaia_db = None
        if with_gaia_star_data:
            gaia_db = self.load_gaia_dataset_of_known_exoplanets()
            if gaia_db is None:
                return None
        try:
            exo_qtable = self._storage.read_qtable(table_name=self.name)
        except ValueError:
            logger.error(
                "Known Exoplanets dataset not found. "
                "You need to download it first by calling download_known_exoplanets(store=True)."
            )
            return None

        return _create_exo_db(exo_dataset=exo_qtable, gaia_db=gaia_db)

    def load_star_system_dataset(self) -> Optional[StarSystemDB]:
        try:
            # Try to load reduced dataset
            reduced_exo_dataset = self._storage.read_qtable(table_name=self._reduced_dataset_name)
            return _create_star_system_db(reduced_exo_dataset)
        except ValueError:
            # If it doesn't exist, compute it from the full datasets
            gaia_db = self.load_gaia_dataset_of_known_exoplanets()
            if gaia_db is None:
                return None

            try:
                exo_qtable = self._storage.read_qtable(table_name=self.name)
            except ValueError:
                logger.error(
                    "Known Exoplanets dataset not found. "
                    "You need to download it first by calling download_known_exoplanets(store=True)."
                )
                return None

            return self._create_star_system_db_from_scratch(exo_dataset=exo_qtable, gaia_db=gaia_db)

    def load_gaia_dataset_of_known_exoplanets(self) -> Optional[GaiaDB]:
        gaia_db = self._gaia_dataset.load_gaia_parameters_dataset()
        if gaia_db is None:
            logger.error(
                "Gaia dataset not found. You need to download it first by "
                "calling download_known_exoplanets(with_gaia_star_data=True, store=True)."
            )
        return gaia_db

    def download_known_exoplanets(
        self,
        with_gaia_star_data: bool = False,
        store: bool = True,
        limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> ExoDB:
        logger.info("Preparing to download known exoplanets dataset...")
        exo_qtable, exo_header = KnownExoplanetsDownloader().download(limit=limit, columns=columns)

        if store:
            self._storage.write_qtable(table=exo_qtable, header=exo_header, table_name=self.name, override=True)

        if with_gaia_star_data:
            gaia_ids = np.unique(exo_qtable["gaia_id"].value).tolist()
            gaia_db = self._gaia_dataset.download_gaia_parameters(gaia_ids=gaia_ids, store=store)
        else:
            gaia_db = None

        return _create_exo_db(exo_qtable, gaia_db)

    def _create_star_system_db_from_scratch(self, exo_dataset: QTable, gaia_db: GaiaDB) -> StarSystemDB:
        # Disable parsing Time columns; we need them as Quantities to copy units to the transiting qtable.
        exo_db = _create_exo_db(exo_dataset=exo_dataset, gaia_db=gaia_db, convert_time_columns=False)

        # Reduce exoplanet dataset to a compact representation
        reduced_exo_dataset, header = reduce_exoplanet_dataset(exo_db=exo_db)

        # Store the reduced dataset for future use
        self._storage.write_qtable(
            table=reduced_exo_dataset,
            header=header,
            table_name=self._reduced_dataset_name,
            override=True,
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


def _create_star_system_db(reduced_exo_dataset: QTable) -> StarSystemDB:
    # Now it's safe to parse Time columns
    ExoDB.convert_time_columns(reduced_exo_dataset)
    reduced_exo_dataset = StarSystemDB.preprocess_dataset(reduced_exo_dataset)
    return StarSystemDB(reduced_exo_dataset)

import logging
from typing import Any, Optional, Sequence

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
    """
    Dataset class for accessing and managing confirmed exoplanets data.

    This class provides functionality to download, store, and retrieve data about known
    exoplanets from the NASA Exoplanet Archive. It also supports integration with Gaia
    stellar data and can generate star system representations that combine exoplanet
    and stellar information.
    """

    _DATASET_EXO = "known_exoplanets"

    def __init__(self, dataset_tag: Optional[str] = None, storage: Optional[BaseStorage] = None):
        """
        Initialize a KnownExoplanetsDataset instance.

        Args:
            dataset_tag: Tag to identify this specific dataset instance, it will be used as a postfix
                for all the storage keys.
            storage: Storage backend for persisting dataset information. Defaults to in-memory storage.
        """
        super().__init__(dataset_name=self._DATASET_EXO, dataset_tag=dataset_tag, storage=storage)
        self._gaia_dataset = GaiaParametersDataset(dataset_tag=self._DATASET_EXO, storage=storage)
        self._reduced_dataset_name = self.name + "_reduced"

    def load_known_exoplanets_dataset(self, with_gaia_star_data: bool = False) -> Optional[ExoDB]:
        """
        Load previously stored known exoplanets dataset.

        Attempts to load known exoplanets data from the configured storage backend,
        optionally including associated Gaia stellar data.

        Args:
            with_gaia_star_data: Whether to include Gaia stellar data. Default is False.

        Returns:
            Database object containing the loaded exoplanets data,
            or None if no data is found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """
        gaia_db = None
        if with_gaia_star_data:
            gaia_db = self.load_gaia_dataset_of_known_exoplanets()
            if gaia_db is None:
                return None
        try:
            exo_qtable = self._storage.read_qtable(table_name=self.name)
        except ValueError:
            return None

        return _create_exo_db(exo_dataset=exo_qtable, gaia_db=gaia_db)

    def load_star_system_dataset(self) -> Optional[StarSystemDB]:
        """
        Load previously stored star system dataset.

        Attempts to load a reduced representation of star systems from the configured storage backend.
        If the reduced dataset doesn't exist, it will attempt to compute it from the full exoplanet
        and Gaia datasets.

        Returns:
            Database object containing star system data,
            or None if the required data is not found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """
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
                return None

            return self._create_star_system_db_from_scratch(exo_dataset=exo_qtable, gaia_db=gaia_db)

    def load_gaia_dataset_of_known_exoplanets(self) -> Optional[GaiaDB]:
        """
        Load previously stored Gaia data for known exoplanets' host stars.

        Attempts to load Gaia stellar data associated with known exoplanets
        from the configured storage backend.

        Returns:
            Database object containing Gaia stellar data,
            or None if no data is found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """
        return self._gaia_dataset.load_gaia_parameters_dataset()

    def download_known_exoplanets(
        self,
        with_gaia_star_data: bool = False,
        store: bool = True,
        limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        where: Optional[dict[str, Any | list[Any]]] = None,
    ) -> ExoDB:
        """
        Download known exoplanets data from NASA Exoplanet Archive.

        Retrieves data about confirmed exoplanets and optionally their host stars' Gaia data,
        and stores it in the configured storage backend.

        Args:
            with_gaia_star_data: Whether to also download Gaia data for the host stars. Default is False.
            store: Whether to store the downloaded data in the storage backend. Default is True.
            limit: Maximum number of exoplanets to retrieve. Default is None (no limit).
            columns: Specific columns to retrieve. Default is None (all available columns).
            where: Additional filters to apply to the data.

        Returns:
            Database object containing the downloaded exoplanets data.

        Raises:
            Various exceptions may be raised by the underlying downloader if the
            download fails.
        """
        logger.info("Preparing to download known exoplanets dataset...")
        # TODO: some columns are mandatory for data processing, add an exception or add them as default
        exo_qtable, exo_header = KnownExoplanetsDownloader().download(limit=limit, columns=columns, where=where)

        if store:
            self._storage.write_qtable(table=exo_qtable, header=exo_header, table_name=self.name, override=True)

        if with_gaia_star_data:
            gaia_ids = np.unique(exo_qtable["gaia_id"].value).tolist()
            gaia_db = self._gaia_dataset.download_gaia_parameters(gaia_ids=gaia_ids, store=store)
        else:
            gaia_db = None

        return _create_exo_db(exo_qtable, gaia_db)

    def _create_star_system_db_from_scratch(self, exo_dataset: QTable, gaia_db: GaiaDB) -> StarSystemDB:
        """
        Create a star system database from exoplanet and Gaia datasets.

        Processes the full exoplanet and Gaia datasets to create a reduced representation
        of star systems, and stores this reduced dataset for future use.

        Args:
            exo_dataset: The dataset containing exoplanet data.
            gaia_db: Database containing Gaia stellar data.

        Returns:
            Database object containing the processed star system data.
        """
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
    """
    Create an ExoDB instance from an exoplanet dataset.

    Preprocesses the dataset and optionally integrates Gaia stellar data.

    Args:
        exo_dataset: The dataset containing exoplanet data.
        gaia_db: Optional database containing Gaia stellar data to integrate.
        convert_time_columns: Whether to convert time columns to astropy Time objects.

    Returns:
        Database object for accessing and querying exoplanet data.
    """
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
    """
    Create a StarSystemDB instance from a reduced exoplanet dataset.

    Processes the reduced dataset to prepare it for use as a star system database.

    Args:
        reduced_exo_dataset: The reduced dataset containing star system data.

    Returns:
        Database object for accessing and querying star system data.
    """
    # Now it's safe to parse Time columns
    ExoDB.convert_time_columns(reduced_exo_dataset)
    reduced_exo_dataset = StarSystemDB.preprocess_dataset(reduced_exo_dataset)
    return StarSystemDB(reduced_exo_dataset)

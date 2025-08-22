import logging
from typing import Any, Optional, Sequence

from astropy.table import QTable

from exotools.db import ExoDB, StarSystemDB
from exotools.downloaders import PlanetarySystemsCompositeDownloader
from exotools.io import BaseStorage

from ._exoplanet_dataset_reducer import reduce_exoplanet_dataset
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class PlanetarySystemsCompositeDataset(BaseDataset):
    """
    Dataset class for accessing and managing confirmed exoplanets data.

    This class provides functionality to download, store, and retrieve data about known
    exoplanets from the NASA Exoplanet Archive. It also supports integration with Gaia
    stellar data and can generate star system representations that combine exoplanet
    and stellar information.
    """

    _DATASET_COMP = "ps_composite"

    def __init__(self, dataset_tag: Optional[str] = None, storage: Optional[BaseStorage] = None):
        """
        Initialize a KnownExoplanetsDataset instance.

        Args:
            dataset_tag: Tag to identify this specific dataset instance, it will be used as a postfix
                for all the storage keys.
            storage: Storage backend for persisting dataset information. Defaults to in-memory storage.
        """
        super().__init__(dataset_name=self._DATASET_COMP, dataset_tag=dataset_tag, storage=storage)
        self._reduced_dataset_name = self.name + "_reduced"

    def load_composite_dataset(self, with_name: Optional[str] = None) -> Optional[ExoDB]:
        """
        Load previously stored known exoplanets dataset.

        Attempts to load known exoplanets data from the configured storage backend,
        optionally including associated Gaia stellar data.

        Args:
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Database object containing the loaded exoplanets data,
            or None if no data is found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """

        try:
            table_name = self.name + (f"_{with_name}" if with_name else "")
            exo_qtable = self._storage.read_qtable(table_name=table_name)
        except ValueError:
            return None

        return _create_exo_db(exo_dataset=exo_qtable)

    def load_star_system_dataset(self, with_name: Optional[str] = None) -> Optional[StarSystemDB]:
        """
        Load previously stored star system dataset.

        Attempts to load a reduced representation of star systems from the configured storage backend.
        If the reduced dataset doesn't exist, it will attempt to compute it from the full exoplanet
        and Gaia datasets.

        Args:
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Database object containing star system data,
            or None if the required data is not found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """
        reduced_name = self._reduced_dataset_name + (f"_{with_name}" if with_name else "")
        try:
            # Try to load reduced dataset
            reduced_exo_dataset = self._storage.read_qtable(table_name=reduced_name)
            return _create_star_system_db(reduced_exo_dataset)
        except ValueError:
            # If it doesn't exist, compute it from the full datasets
            try:
                table_name = self.name + (f"_{with_name}" if with_name else "")
                exo_qtable = self._storage.read_qtable(table_name=table_name)
            except ValueError:
                return None

            return self._create_star_system_db_from_scratch(exo_dataset=exo_qtable, with_name=with_name)

    def download_composite_dataset(
        self,
        store: bool = True,
        limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        where: Optional[dict[str, Any | list[Any]]] = None,
        with_name: Optional[str] = None,
    ) -> ExoDB:
        """
        Download known exoplanets data from NASA Exoplanet Archive.

        Retrieves data about confirmed exoplanets and optionally their host stars' Gaia data,
        and stores it in the configured storage backend.

        Args:
            store: Whether to store the downloaded data in the storage backend. Default is True.
            limit: Maximum number of exoplanets to retrieve. Default is None (no limit).
            columns: Specific columns to retrieve. Default is None (all available columns).
            where: Additional filters to apply to the data.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Database object containing the downloaded exoplanets data.

        Raises:
            Various exceptions may be raised by the underlying downloader if the
            download fails.
        """
        logger.info("Preparing to download known exoplanets dataset...")
        # TODO: some columns are mandatory for data processing, add an exception or add them as default
        exo_qtable, exo_header = PlanetarySystemsCompositeDownloader().download(
            limit=limit,
            columns=columns,
            where=where,
        )

        if store:
            table_name = self.name + (f"_{with_name}" if with_name else "")
            self._storage.write_qtable(table=exo_qtable, header=exo_header, table_name=table_name, override=True)

        return _create_exo_db(exo_qtable)

    def _create_star_system_db_from_scratch(self, exo_dataset: QTable, with_name: Optional[str] = None) -> StarSystemDB:
        """
        Create a star system database from exoplanet and Gaia datasets.

        Processes the full exoplanet and Gaia datasets to create a reduced representation
        of star systems, and stores this reduced dataset for future use.

        Args:
            exo_dataset: The dataset containing exoplanet data.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Database object containing the processed star system data.
        """
        # Disable parsing Time columns; we need them as Quantities to copy units to the transiting qtable.
        exo_db = _create_exo_db(exo_dataset=exo_dataset)

        # Reduce exoplanet dataset to a compact representation
        reduced_exo_dataset, header = reduce_exoplanet_dataset(exo_db=exo_db)

        # Store the reduced dataset for future use
        reduced_name = self._reduced_dataset_name + (f"_{with_name}" if with_name else "")
        self._storage.write_qtable(
            table=reduced_exo_dataset,
            header=header,
            table_name=reduced_name,
            override=True,
        )

        return _create_star_system_db(reduced_exo_dataset)


def _create_exo_db(exo_dataset: QTable) -> ExoDB:
    """
    Create an ExoDB instance from an exoplanet dataset.

    Preprocesses the dataset and optionally integrates Gaia stellar data.

    Args:
        exo_dataset: The dataset containing exoplanet data.

    Returns:
        Database object for accessing and querying exoplanet data.
    """
    ExoDB.preprocess_dataset(exo_dataset)
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
    reduced_exo_dataset = StarSystemDB.preprocess_dataset(reduced_exo_dataset)
    return StarSystemDB(reduced_exo_dataset)

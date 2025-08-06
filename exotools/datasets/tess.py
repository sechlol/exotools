import logging
from typing import Optional, Sequence

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import TessMetaDB, TicDB
from exotools.downloaders import TessCatalogDownloader, TessObservationsDownloader
from exotools.io import BaseStorage

logger = logging.getLogger(__name__)


class TessDataset(BaseDataset):
    """
    Dataset class for accessing and managing TESS (Transiting Exoplanet Survey Satellite) data.

    This class provides functionality to download, store, and retrieve TESS observation metadata
    and TIC (TESS Input Catalog) target information. It interfaces with the MAST archive
    to retrieve TESS-related data.
    """

    _DATASET_TESS = "tess"

    def __init__(
        self,
        dataset_tag: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize a TessDataset instance.

        Args:
            dataset_tag: Tag to identify this specific dataset instance, it will be used as a postfix for all the
                storage keys.
            storage: Storage backend for persisting dataset information. Defaults to in-memory storage.
            username: MAST username for authentication. Optional for fetching observation metadata, but required for
                querying the TIC.
            password: MAST password for authentication. Optional for fetching observation metadata, but required for
                querying the TIC.
        """
        super().__init__(dataset_name=self._DATASET_TESS, dataset_tag=dataset_tag, storage=storage)
        self._catalog_downloader = TessCatalogDownloader(username, password) if username and password else None
        self._observations_name = self.name + "_observations"
        self._tic_name = self.name + "_tic"

    def download_observation_metadata(
        self, targets_tic_id: Sequence[int], store: bool = True, with_name: Optional[str] = None
    ) -> TessMetaDB:
        """
        Download TESS observation metadata for specified TIC IDs.

        Retrieves observation metadata for the given TIC IDs from the MAST archive
        and optionally stores it in the configured storage backend.

        Args:
            targets_tic_id: List of TIC IDs to retrieve observation metadata for.
            store: Whether to store the downloaded data in the storage backend. Default is True.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            TessMetaDB: Database object containing the downloaded observation metadata.

        Raises:
            Various exceptions may be raised by the underlying downloader if the
            download fails.
        """
        logger.info(f"Preparing to download TESS observation list for {len(targets_tic_id)} objects...")
        meta_qtable, meta_header = TessObservationsDownloader().download_by_id(targets_tic_id)

        if store:
            table_name = self._observations_name + (f"_{with_name}" if with_name else "")
            self._storage.write_qtable(meta_qtable, meta_header, table_name, override=True)

        return TessMetaDB(meta_dataset=meta_qtable)

    def download_tic_targets(
        self,
        limit: Optional[int] = None,
        star_mass_range: Optional[tuple[float, float]] = None,
        priority_threshold: Optional[float] = None,
        store: bool = False,
        with_name: Optional[str] = None,
    ) -> TicDB:
        """
        Searches the TESS Input Catalog for targets matching the given criteria
        and optionally stores the results in the configured storage backend.

        Args:
            limit: Maximum number of targets to retrieve. Default is None (no limit).
            star_mass_range: Range of stellar masses to filter by,
                specified as (min_mass, max_mass) in solar masses. Default is None (no filtering).
            priority_threshold: Minimum priority value for targets.
                Default is None (no filtering).
            store (bool): Whether to store the search results in the storage backend.
                Default is False.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            TicDB: Database object containing the search results.

        Raises:
            ValueError: If username and password were not provided during initialization.
            Various exceptions may be raised by the underlying downloader if the
            download fails.
        """
        if self._catalog_downloader is None:
            raise ValueError("You need to provide a username and password to download the TIC dataset.")
        if star_mass_range is not None:
            self._catalog_downloader.star_mass_range = star_mass_range
        if priority_threshold is not None:
            self._catalog_downloader.priority_threshold = priority_threshold

        catalog_qtable, catalog_header = self._catalog_downloader.download(limit=limit)

        if store:
            table_name = self._tic_name + (f"_{with_name}" if with_name else "")
            self._storage.write_qtable(catalog_qtable, catalog_header, table_name, override=True)

        return TicDB(dataset=catalog_qtable)

    def download_tic_targets_by_ids(
        self, tic_ids: Sequence[int], store: bool = True, with_name: Optional[str] = None
    ) -> TicDB:
        """
        Download TIC target information for specific TIC IDs.

        Retrieves detailed information for the specified TIC IDs from the TESS Input Catalog
        and optionally stores it in the configured storage backend.

        Args:
            tic_ids: List of TIC IDs to retrieve information for.
            store: Whether to store the downloaded data in the storage backend. Default is True.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            TicDB: Database object containing the downloaded TIC target information.

        Raises:
            ValueError: If username and password were not provided during initialization.
            Various exceptions may be raised by the underlying downloader if the
            download fails.
        """
        if self._catalog_downloader is None:
            raise ValueError("You need to provide a username and password to download the TIC dataset.")
        catalog_qtable, catalog_header = self._catalog_downloader.download_by_id(tic_ids)

        if store:
            table_name = self._tic_name + (f"_{with_name}" if with_name else "")
            self._storage.write_qtable(
                table=catalog_qtable, header=catalog_header, table_name=table_name, override=True
            )

        return TicDB(dataset=catalog_qtable)

    def load_observation_metadata(self, with_name: Optional[str] = None) -> Optional[TessMetaDB]:
        """
        Load previously stored TESS observation metadata.

        Attempts to load TESS observation metadata from the configured storage backend.

        Args:
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Optional[TessMetaDB]: Database object containing the loaded observation metadata,
                or None if no data is found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """
        table_name = self._observations_name + (f"_{with_name}" if with_name else "")
        try:
            meta_qtable = self._storage.read_qtable(table_name=table_name)
        except ValueError:
            return None

        return TessMetaDB(meta_dataset=meta_qtable)

    def load_tic_target_dataset(self, with_name: Optional[str] = None) -> Optional[TicDB]:
        """
        Load previously stored TIC target dataset.

        Attempts to load TIC target data from the configured storage backend.
        This loads data that was previously stored via the search_tic_targets method.

        Args:
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Optional[TicDB]: Database object containing the loaded TIC target data,
                or None if no data is found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """
        table_name = self._tic_name + (f"_{with_name}" if with_name else "")
        try:
            tic_qtable = self._storage.read_qtable(table_name=table_name)
        except ValueError:
            return None

        return TicDB(dataset=tic_qtable)

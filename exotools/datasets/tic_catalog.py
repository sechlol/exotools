import logging
from typing import Optional, Sequence

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import TicDB
from exotools.downloaders import TessCatalogDownloader
from exotools.io import BaseStorage

logger = logging.getLogger(__name__)


class TicCatalogDataset(BaseDataset):
    """
    Dataset class for accessing and managing TESS (Transiting Exoplanet Survey Satellite) catalog data.

    This class provides functionality to download, store, and retrieve TESS observation metadata
    and TIC (TESS Input Catalog) target information. It interfaces with the MAST archive
    to retrieve TESS-related data.
    """

    _DATASET_NAME = "tic_catalog"
    _catalog_downloader: Optional[TessCatalogDownloader] = None

    def __init__(
        self,
        dataset_tag: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
    ):
        """
        Args:
            dataset_tag: Tag to identify this specific dataset instance, it will be used as a postfix for all the
                storage keys.
            storage: Storage backend for persisting dataset information. Defaults to in-memory storage.
        """
        super().__init__(dataset_name=self._DATASET_NAME, dataset_tag=dataset_tag, storage=storage)

    @classmethod
    def authenticate_casjobs(cls, username: str, password: str):
        """
        Authenticate with the MAST CasJobs service using the provided username and password.
        Authentication is required for querying the TIC

        Create an account at
        https://mastweb.stsci.edu/mcasjobs/CreateAccount.aspx

        Args:
            username: MAST username for authentication. Optional for fetching observation metadata, but required for
                querying the TIC.
            password: MAST password for authentication. Optional for fetching observation metadata, but required for
                querying the TIC.
        """
        cls._catalog_downloader = TessCatalogDownloader(username=username, password=password)

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
            raise ValueError("You need to call TicCatalogDataset.authenticate() to query the TIC dataset.")
        if star_mass_range is not None:
            self._catalog_downloader.star_mass_range = star_mass_range
        if priority_threshold is not None:
            self._catalog_downloader.priority_threshold = priority_threshold

        catalog_qtable, catalog_header = self._catalog_downloader.download(limit=limit)

        if store:
            table_name = self.name + (f"_{with_name}" if with_name else "")
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
            raise ValueError("You need to call TicCatalogDataset.authenticate() to query the TIC dataset.")
        catalog_qtable, catalog_header = self._catalog_downloader.download_by_id(tic_ids)

        if store:
            table_name = self.name + (f"_{with_name}" if with_name else "")
            self._storage.write_qtable(
                table=catalog_qtable, header=catalog_header, table_name=table_name, override=True
            )

        return TicDB(dataset=catalog_qtable)

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
        table_name = self.name + (f"_{with_name}" if with_name else "")
        try:
            tic_qtable = self._storage.read_qtable(table_name=table_name)
        except ValueError:
            return None

        return TicDB(dataset=tic_qtable)

import logging
from typing import Optional

from astropy.table import QTable

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import CandidateDB
from exotools.downloaders import CandidateExoplanetsDownloader
from exotools.io import BaseStorage

logger = logging.getLogger(__name__)


class CandidateExoplanetsDataset(BaseDataset):
    """
    Dataset class for accessing and managing candidate exoplanets data.

    This class provides functionality to download, store, and retrieve candidate exoplanets
    data from the NASA Exoplanet Archive. It handles the retrieval and storage of exoplanet candidates
    that have not yet been confirmed as actual exoplanets.
    """

    _DATASET_NAME_CANDIDATES = "candidate_exoplanets"

    def __init__(self, dataset_tag: Optional[str] = None, storage: Optional[BaseStorage] = None):
        """
        Initialize a CandidateExoplanetsDataset instance.

        Args:
            dataset_tag: Tag to identify this specific dataset instance, it will be used as a postfix
                for all the storage keys.
            storage: Storage backend for persisting dataset information. Defaults to in-memory storage.
        """
        super().__init__(dataset_name=self._DATASET_NAME_CANDIDATES, dataset_tag=dataset_tag, storage=storage)

    def load_candidate_exoplanets_dataset(self, with_name: Optional[str] = None) -> Optional[CandidateDB]:
        """
        Load previously stored candidate exoplanets dataset, with an optional distinctive name.
        Attempts to load candidate exoplanets data from the configured storage backend.

        Returns:
            Database object containing the loaded candidate exoplanets data,
            or None if no data is found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """
        table_name = self.name + (f"_{with_name}" if with_name else "")
        try:
            candidate_qtable = self._storage.read_qtable(table_name=table_name)
        except ValueError:
            return None

        return _create_candidate_db(candidate_dataset=candidate_qtable)

    def download_candidate_exoplanets(
        self,
        limit: Optional[int] = None,
        store: bool = True,
        with_name: Optional[str] = None,
    ) -> CandidateDB:
        """
        Retrieves candidate exoplanets data from NASA Exoplanet Archive and optionally
        stores it in the configured storage backend.

        Args:
            limit: Maximum number of candidates to retrieve. Default is None (no limit).
            store: Whether to store the downloaded data in the storage backend. Default is True.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name

        Returns:
            Database object containing the downloaded candidate exoplanets data.

        Raises:
            Various exceptions may be raised by the underlying downloader if the
            download fails.
        """
        logger.info("Preparing to download candidate exoplanets dataset...")
        candidate_qtable, candidate_header = CandidateExoplanetsDownloader().download(limit=limit)

        if store:
            table_name = self.name + (f"_{with_name}" if with_name else "")
            self._storage.write_qtable(
                table=candidate_qtable,
                header=candidate_header,
                table_name=table_name,
            )

        return _create_candidate_db(candidate_qtable)


def _create_candidate_db(candidate_dataset: QTable) -> CandidateDB:
    return CandidateDB(candidate_dataset)

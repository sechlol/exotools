import logging
from typing import Optional, Sequence

from astroquery import mast

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import TicObsDB
from exotools.downloaders import TessObservationsDownloader
from exotools.io import BaseStorage

logger = logging.getLogger(__name__)


class TicObservationsDataset(BaseDataset):
    _DATASET_NAME = "tic_observations"

    def __init__(
        self,
        dataset_tag: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
    ):
        super().__init__(dataset_name=self._DATASET_NAME, dataset_tag=dataset_tag, storage=storage)

    @classmethod
    def authenticate_mast(cls, mast_token: str):
        """
        Authenticate with the MAST archive using the provided MAST token.
        Get a MAST token from https://auth.mast.stsci.edu/tokens

        Args:
            mast_token: MAST token for authentication.
        """
        mast.Observations.login(token=mast_token)

    def download_observation_metadata(
        self,
        targets_tic_id: Sequence[int],
        store: bool = True,
        with_name: Optional[str] = None,
    ) -> TicObsDB:
        """
        Download TESS observation metadata for specified TIC IDs.

        Retrieves observation metadata for the given TIC IDs from the MAST archive
        and optionally stores it in the configured storage backend.

        Args:
            targets_tic_id: List of TIC IDs to retrieve observation metadata for.
            store: Whether to store the downloaded data in the storage backend. Default is True.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            TicObsDB: Database object containing the downloaded observation metadata.

        Raises:
            Various exceptions may be raised by the underlying downloader if the
            download fails.
        """
        logger.info(f"Preparing to download TESS observation list for {len(targets_tic_id)} objects...")
        meta_qtable, meta_header = TessObservationsDownloader().download_by_id(targets_tic_id)

        if store:
            table_name = self.name + (f"_{with_name}" if with_name else "")
            self._storage.write_qtable(meta_qtable, meta_header, table_name, override=True)

        return TicObsDB(meta_dataset=meta_qtable)

    def load_observation_metadata(self, with_name: Optional[str] = None) -> Optional[TicObsDB]:
        """
        Load previously stored TESS observation metadata.

        Attempts to load TESS observation metadata from the configured storage backend.

        Args:
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Optional[TicObsDB]: Database object containing the loaded observation metadata,
                or None if no data is found in storage.

        Raises:
            Various exceptions may be raised by the underlying storage backend if the
            load operation fails for reasons other than missing data.
        """
        table_name = self.name + (f"_{with_name}" if with_name else "")
        try:
            meta_qtable = self._storage.read_qtable(table_name=table_name)
        except ValueError:
            return None

        return TicObsDB(meta_dataset=meta_qtable)

import logging
from typing import Optional, Sequence

from astropy.table import QTable

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import GaiaDB
from exotools.downloaders import GaiaDownloader
from exotools.io import BaseStorage

logger = logging.getLogger(__name__)


class GaiaParametersDataset(BaseDataset):
    _DATASET_GAIA = "gaia"

    def __init__(self, dataset_tag: Optional[str] = None, storage: Optional[BaseStorage] = None):
        super().__init__(dataset_name=self._DATASET_GAIA, dataset_tag=dataset_tag, storage=storage)

    @staticmethod
    def authenticate(username: str, password: str):
        # This import causes network operations. It's included locally to avoid overheads at import time
        from astroquery.gaia import Gaia

        Gaia.login(user=username, password=password)

    def load_gaia_parameters_dataset(self, with_name: Optional[str] = None) -> Optional[GaiaDB]:
        """
        Load Gaia parameters dataset from storage.

        Args:
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            GaiaDB: Database containing Gaia parameters, or None if not found.
        """
        table_name = self.name + (f"_{with_name}" if with_name else "")
        try:
            gaia_qtable = self._storage.read_qtable(table_name=table_name)
            return self._create_gaia_db(gaia_qtable)
        except ValueError:
            return None

    def download_gaia_parameters(
        self, gaia_ids: Sequence[int], store: bool = True, with_name: Optional[str] = None
    ) -> GaiaDB:
        """
        Download Gaia DR3 data for the given Gaia IDs.

        Args:
            gaia_ids: Sequence of Gaia IDs to download data for.
            store: Whether to store the downloaded data.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            GaiaDB: Database containing the downloaded Gaia parameters.
        """
        logger.info(f"Preparing to download Gaia DR3 data for {len(gaia_ids)} stars...")
        gaia_qtable, gaia_header = GaiaDownloader().download_by_id(ids=gaia_ids)
        logger.info(f"Downloaded {len(gaia_qtable)} stars")

        if store:
            table_name = self.name + (f"_{with_name}" if with_name else "")
            self._storage.write_qtable(
                table=gaia_qtable,
                header=gaia_header,
                table_name=table_name,
                override=True,
            )

        return self._create_gaia_db(gaia_qtable)

    @staticmethod
    def _create_gaia_db(gaia_dataset: QTable) -> GaiaDB:
        GaiaDB.impute_radius(gaia_dataset)
        GaiaDB.compute_mean_temperature(gaia_dataset)
        GaiaDB.compute_habitable_zone(gaia_dataset)
        return GaiaDB(gaia_dataset)

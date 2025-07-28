from typing import Optional, Sequence

from astropy.table import QTable

from exotools.db.gaia_db import GaiaDB
from exotools.downloaders.gaia_downloader import GaiaDownloader
from exotools.io.base_storage_wrapper import StorageWrapper


class GaiaParametersDataset:
    _DATASET_GAIA = "known_gaia_astro_parameters"

    def __init__(self, storage: StorageWrapper):
        self._storage = storage

    def load_gaia_parameters_dataset(self) -> Optional[GaiaDB]:
        """
        Load Gaia parameters dataset from storage.

        Returns:
            GaiaDB: Database containing Gaia parameters, or None if not found.
        """
        try:
            gaia_qtable = self._storage.read_qtable(table_name=self._DATASET_GAIA)
            return self._create_gaia_db(gaia_qtable)
        except ValueError:
            print("Gaia dataset not found. You need to download it first by " "calling download_gaia_parameters().")
            return None

    def download_gaia_parameters(self, gaia_ids: Sequence[int], store: bool = True) -> GaiaDB:
        """
        Download Gaia DR3 data for the given Gaia IDs.

        Args:
            gaia_ids: Sequence of Gaia IDs to download data for.
            store: Whether to store the downloaded data.

        Returns:
            GaiaDB: Database containing the downloaded Gaia parameters.
        """
        print(f"Preparing to download Gaia DR3 data for {len(gaia_ids)} stars...")

        gaia_qtable, gaia_header = GaiaDownloader().download_by_id(ids=gaia_ids)
        if store:
            self._storage.write_qtable(
                table=gaia_qtable,
                header=gaia_header,
                table_name=self._DATASET_GAIA,
                override=True,
            )

        return self._create_gaia_db(gaia_qtable)

    @staticmethod
    def _create_gaia_db(gaia_dataset: QTable) -> GaiaDB:
        """
        Create a GaiaDB instance from a QTable dataset.

        Args:
            gaia_dataset: QTable containing Gaia data.

        Returns:
            GaiaDB: Database containing processed Gaia parameters.
        """
        GaiaDB.impute_radius(gaia_dataset)
        GaiaDB.compute_mean_temperature(gaia_dataset)
        GaiaDB.compute_habitable_zone(gaia_dataset)
        return GaiaDB(gaia_dataset)

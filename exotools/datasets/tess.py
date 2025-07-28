from typing import Sequence, Optional

from exotools.db import TicDB, TessMetaDB
from exotools.downloaders import TessCatalogDownloader, TessObservationsDownloader
from exotools.io import BaseStorage


class TessDataset:
    _OBSERVATIONS_NAME = "tess_observations"
    _TIC_NAME = "tess_tic"
    _TIC_BY_ID_NAME = "tess_tic_by_id"

    def __init__(self, storage: BaseStorage, username: Optional[str] = None, password: Optional[str] = None):
        self._storage = storage
        self._catalog_downloader = TessCatalogDownloader(username, password) if username and password else None

    def download_observation_metadata(self, targets_tic_id: Sequence[int], store: bool = True) -> TessMetaDB:
        print(f"Preparing to download TESS observation list for {len(targets_tic_id)} objects...")
        meta_qtable, meta_header = TessObservationsDownloader().download_by_id(targets_tic_id)

        if store:
            self._storage.write_qtable(meta_qtable, meta_header, self._OBSERVATIONS_NAME, override=True)

        return TessMetaDB(meta_dataset=meta_qtable)

    def search_tic_targets(
        self,
        limit: Optional[int] = None,
        star_mass_range: Optional[tuple[float, float]] = None,
        priority_threshold: Optional[float] = None,
        store: bool = False,
    ) -> TicDB:
        if self._catalog_downloader is None:
            raise ValueError("You need to provide a username and password to download the TIC dataset.")
        if star_mass_range is not None:
            self._catalog_downloader.star_mass_range = star_mass_range
        if priority_threshold is not None:
            self._catalog_downloader.priority_threshold = priority_threshold

        catalog_qtable, catalog_header = self._catalog_downloader.download(limit=limit)

        if store:
            self._storage.write_qtable(catalog_qtable, catalog_header, self._TIC_NAME, override=True)

        return TicDB(dataset=catalog_qtable)

    def download_tic_targets_by_ids(self, tic_ids: Sequence[int], store: bool = False) -> TicDB:
        if self._catalog_downloader is None:
            raise ValueError("You need to provide a username and password to download the TIC dataset.")
        catalog_qtable, catalog_header = self._catalog_downloader.download_by_id(tic_ids)

        if store:
            self._storage.write_qtable(catalog_qtable, catalog_header, self._TIC_BY_ID_NAME, override=True)

        return TicDB(dataset=catalog_qtable)

    def load_observation_metadata(self) -> Optional[TessMetaDB]:
        try:
            meta_qtable = self._storage.read_qtable(table_name=self._OBSERVATIONS_NAME)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TessMetaDB(meta_dataset=meta_qtable)

    def load_tic_target_dataset(self) -> Optional[TicDB]:
        try:
            tic_qtable = self._storage.read_qtable(table_name=self._TIC_NAME)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TicDB(dataset=tic_qtable)

    def load_tic_target_dataset_by_id(self) -> Optional[TicDB]:
        try:
            tic_qtable = self._storage.read_qtable(table_name=self._TIC_BY_ID_NAME)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TicDB(dataset=tic_qtable)

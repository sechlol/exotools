from typing import Sequence, Optional

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import TicDB, TessMetaDB
from exotools.downloaders import TessCatalogDownloader, TessObservationsDownloader
from exotools.io import BaseStorage


class TessDataset(BaseDataset):
    _DATASET_TESS = "tess"

    def __init__(
        self,
        dataset_tag: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        super().__init__(dataset_name=self._DATASET_TESS, dataset_tag=dataset_tag, storage=storage)
        self._catalog_downloader = TessCatalogDownloader(username, password) if username and password else None
        self._observations_name = self.name + "_observations"
        self._tic_name = self.name + "_tic"
        self._tic_by_id_name = self.name + "_tic_by_id"

    def download_observation_metadata(self, targets_tic_id: Sequence[int], store: bool = True) -> TessMetaDB:
        print(f"Preparing to download TESS observation list for {len(targets_tic_id)} objects...")
        meta_qtable, meta_header = TessObservationsDownloader().download_by_id(targets_tic_id)

        if store:
            self._storage.write_qtable(meta_qtable, meta_header, self._observations_name, override=True)

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
            self._storage.write_qtable(catalog_qtable, catalog_header, self._tic_name, override=True)

        return TicDB(dataset=catalog_qtable)

    def download_tic_targets_by_ids(self, tic_ids: Sequence[int], store: bool = False) -> TicDB:
        if self._catalog_downloader is None:
            raise ValueError("You need to provide a username and password to download the TIC dataset.")
        catalog_qtable, catalog_header = self._catalog_downloader.download_by_id(tic_ids)

        if store:
            self._storage.write_qtable(catalog_qtable, catalog_header, self._tic_by_id_name, override=True)

        return TicDB(dataset=catalog_qtable)

    def load_observation_metadata(self) -> Optional[TessMetaDB]:
        try:
            meta_qtable = self._storage.read_qtable(table_name=self._observations_name)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TessMetaDB(meta_dataset=meta_qtable)

    def load_tic_target_dataset(self) -> Optional[TicDB]:
        try:
            tic_qtable = self._storage.read_qtable(table_name=self._tic_name)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TicDB(dataset=tic_qtable)

    def load_tic_target_dataset_by_id(self) -> Optional[TicDB]:
        try:
            tic_qtable = self._storage.read_qtable(table_name=self._tic_by_id_name)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TicDB(dataset=tic_qtable)

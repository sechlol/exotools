from pathlib import Path
from typing import Sequence, Optional

from .downloaders.tess_catalog_downloader import TessCatalogDownloader
from .downloaders.tess_observations_downloader import TessObservationsDownloader
from .db.tic_db import TicDB
from .db.urls_db import TessMetaDB
from .utils.qtable_utils import read_qtable


class TessDataset:
    _OBSERVATIONS_NAME = "tess_observations"
    _TIC_NAME = "tess_tic"
    _TIC_BY_ID_NAME = "tess_tic_by_id"

    def __init__(self, base_folder_path: Path, username: Optional[str] = None, password: Optional[str] = None):
        self._folder_path = base_folder_path
        self._catalog_downloader = TessCatalogDownloader(username, password) if username and password else None

    def download_observation_metadata(self, targets_tic_id: Sequence[int], store: bool = False) -> TessMetaDB:
        self._folder_path.mkdir(parents=True, exist_ok=True)

        print(f"Preparing to download TESS observation list for {len(targets_tic_id)} objects...")
        meta_qtable = TessObservationsDownloader().download_by_id(
            targets_tic_id,
            out_folder_path=self._folder_path if store else None,
            out_file_name=self._OBSERVATIONS_NAME if store else None,
        )

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

        catalog_qtable = self._catalog_downloader.download(
            limit=limit,
            out_folder_path=self._folder_path if store else None,
            out_file_name=self._TIC_NAME if store else None,
        )

        return TicDB(dataset=catalog_qtable)

    def download_tic_targets_by_ids(self, tic_ids: Sequence[int], store: bool = False) -> TicDB:
        if self._catalog_downloader is None:
            raise ValueError("You need to provide a username and password to download the TIC dataset.")
        catalog_qtable = self._catalog_downloader.download_by_id(
            tic_ids,
            out_folder_path=self._folder_path if store else None,
            out_file_name=self._TIC_BY_ID_NAME if store else None,
        )

        return TicDB(dataset=catalog_qtable)

    def load_observation_metadata(self) -> Optional[TessMetaDB]:
        try:
            meta_qtable = read_qtable(file_path=self._folder_path, file_name=self._OBSERVATIONS_NAME)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TessMetaDB(meta_dataset=meta_qtable)

    def load_tic_target_dataset(self) -> Optional[TicDB]:
        try:
            tic_qtable = read_qtable(file_path=self._folder_path, file_name=self._TIC_NAME)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TicDB(dataset=tic_qtable)

    def load_tic_target_dataset_by_id(self) -> Optional[TicDB]:
        try:
            tic_qtable = read_qtable(file_path=self._folder_path, file_name=self._TIC_BY_ID_NAME)
        except ValueError:
            print(
                "Stored TIC dataset not found. You need to download it first by "
                "calling download_tic_targets(store=True)."
            )
            return None

        return TicDB(dataset=tic_qtable)

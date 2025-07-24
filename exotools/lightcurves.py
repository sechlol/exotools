from pathlib import Path
from typing import Optional

from .downloaders.lightcurve_downloader import LightcurveDownloader
from .db.lightcurve_db import LightcurveDB
from .db.urls_db import TessMetaDB
from .utils.download import DownloadParams
from .utils.io import get_file_paths_in_subfolder


class LightcurveDataset:
    _DATASET_LIGHTCURVES = "lightcurves"

    def __init__(self, storage_folder_path: Path, override_existing: bool = False, verbose: bool = False):
        self._folder_path = storage_folder_path / self._DATASET_LIGHTCURVES
        self._downloader = LightcurveDownloader(override_existing=override_existing, verbose=verbose)

    def download_lightcurves_from_tess_db(self, tess_db: TessMetaDB) -> Optional[LightcurveDB]:
        download_params = [
            DownloadParams(
                url=row["dataURL"],
                download_path=str(self._folder_path / str(row["tic_id"]) / f"{row['obs_id']}.fits"),
            )
            for row in tess_db.view
        ]

        print(f"Downloading {len(download_params)} lightcurves")
        downloaded_paths = self._downloader.download_fits_parallel(download_params)
        print(f"Downloaded {len(downloaded_paths)} lightcurves")

        return self.load_lightcurve_dataset()

    def load_lightcurve_dataset(self) -> Optional[LightcurveDB]:
        downloaded_paths = get_file_paths_in_subfolder(self._folder_path, file_extension="fits")
        if len(downloaded_paths) == 0:
            return None

        dataset = LightcurveDB.path_map_to_qtable(downloaded_paths)
        return LightcurveDB(dataset)

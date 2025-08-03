import logging
from pathlib import Path
from typing import Optional

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import LightcurveDB, TessMetaDB
from exotools.downloaders import LightcurveDownloader
from exotools.utils.download import DownloadParams

logger = logging.getLogger(__name__)


class LightcurveDataset(BaseDataset):
    _DATASET_LIGHTCURVES = "lightcurves"

    def __init__(
        self,
        lc_storage_path: Path,
        dataset_tag: Optional[str] = None,
        override_existing: bool = False,
        verbose: bool = False,
    ):
        super().__init__(dataset_name=self._DATASET_LIGHTCURVES, dataset_tag=dataset_tag, storage=None)
        self._folder_path = lc_storage_path / self.name
        self._downloader = LightcurveDownloader(override_existing=override_existing, verbose=verbose)

    def download_lightcurves_from_tess_db(self, tess_db: TessMetaDB) -> Optional[LightcurveDB]:
        download_params = [
            DownloadParams(
                url=row["dataURL"],
                download_path=str(self._folder_path / str(row["tic_id"]) / f"{row['obs_id']}.fits"),
            )
            for row in tess_db.view
        ]

        logger.info(f"Downloading {len(download_params)} lightcurves")
        downloaded_paths = self._downloader.download_fits_parallel(download_params)
        logger.info(f"Downloaded {len(downloaded_paths)} lightcurves")

        return self.load_lightcurve_dataset()

    def load_lightcurve_dataset(self) -> Optional[LightcurveDB]:
        downloaded_paths = _get_file_paths_in_subfolder(self._folder_path, file_extension="fits")
        if len(downloaded_paths) == 0:
            return None

        dataset = LightcurveDB.path_map_to_qtable(downloaded_paths)
        return LightcurveDB(dataset)


def _get_file_paths_in_subfolder(
    parent_path: Path,
    file_extension: Optional[str] = None,
    match_name: Optional[str] = None,
) -> dict[int, list[Path]]:
    if not file_extension and not match_name:
        raise ValueError("At least one between file_extension and match_name should be given")

    if not parent_path.exists():
        return {}

    subfolder_dict = {}
    pattern = match_name if match_name else f"*.{file_extension}"

    # Iterate over each subfolder
    for subfolder in parent_path.iterdir():
        if subfolder.is_dir():
            fits_files = list(subfolder.glob(pattern))
            if fits_files:
                subfolder_dict[int(subfolder.name)] = [Path(file) for file in fits_files]

    return subfolder_dict

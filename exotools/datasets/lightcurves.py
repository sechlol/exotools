import logging
from pathlib import Path
from typing import Optional

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import LightcurveDB, TicObsDB
from exotools.downloaders import LightcurveDownloader
from exotools.utils.download import DownloadParams

logger = logging.getLogger(__name__)


class LightcurveDataset(BaseDataset):
    """
    Dataset class for accessing and managing astronomical lightcurve data.

    This class provides functionality to download, store, and retrieve lightcurve data
    from various astronomical sources, primarily TESS (Transiting Exoplanet Survey Satellite).
    Unlike other datasets that use the BaseStorage interface, lightcurves are stored directly
    as FITS files in the filesystem.
    """

    _DATASET_LIGHTCURVES = "lightcurves"

    def __init__(
        self,
        lc_storage_path: Path,
        dataset_tag: Optional[str] = None,
        override_existing: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize a LightcurveDataset instance.

        Args:
            lc_storage_path: Base directory where lightcurve files will be stored.
            dataset_tag: Tag to identify this specific dataset instance, it will be used as a postfix
                for the storage directory.
            override_existing: Whether to overwrite existing lightcurve files when downloading.
                Default is False.
            verbose: Whether to output detailed progress information during downloads.
                Default is False.
        """
        super().__init__(dataset_name=self._DATASET_LIGHTCURVES, dataset_tag=dataset_tag, storage=None)
        self._folder_path = lc_storage_path / self.name
        self._downloader = LightcurveDownloader(override_existing=override_existing, verbose=verbose)

    def download_lightcurves_from_tess_db(
        self, tess_db: TicObsDB, with_name: Optional[str] = None
    ) -> Optional[LightcurveDB]:
        """
        Download lightcurves for targets in a TESS metadata database.

        For each observation in the provided TESS metadata database, downloads the
        corresponding lightcurve FITS file and stores it in the configured directory.
        Files are organized in subdirectories by TIC ID.

        Args:
            tess_db: Database containing TESS observation metadata with URLs to lightcurve files.
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Database object containing information about the downloaded lightcurves,
            or None if no lightcurves were downloaded.

        Raises:
            Various exceptions may be raised by the underlying downloader if the
            download fails.
        """
        folder_path = self._folder_path
        if with_name:
            folder_path = folder_path / with_name

        download_params = [
            DownloadParams(
                url=row["dataURL"],
                download_path=str(folder_path / str(row["tic_id"]) / f"{row['obs_id']}.fits"),
            )
            for row in tess_db.view
        ]

        logger.info(f"Downloading {len(download_params)} lightcurves")
        downloaded_paths = self._downloader.download_fits_parallel(download_params)
        logger.info(f"Downloaded {len(downloaded_paths)} lightcurves")

        return self.load_lightcurve_dataset(with_name=with_name)

    def load_lightcurve_dataset(self, with_name: Optional[str] = None) -> Optional[LightcurveDB]:
        """
        Load previously downloaded lightcurve files.

        Scans the configured directory for lightcurve FITS files and creates a database
        object to access them.

        Args:
            with_name: A distinctive name to give the dataset, it will be used as a postfix for the artifact name.

        Returns:
            Database object containing information about the available lightcurves,
            or None if no lightcurve files were found.
        """
        folder_path = self._folder_path
        if with_name:
            folder_path = folder_path / with_name

        downloaded_paths = _get_file_paths_in_subfolder(folder_path, file_extension="fits")
        if len(downloaded_paths) == 0:
            return None

        dataset = LightcurveDB.path_map_to_qtable(downloaded_paths)
        return LightcurveDB(dataset)


def _get_file_paths_in_subfolder(
    parent_path: Path,
    file_extension: Optional[str] = None,
    match_name: Optional[str] = None,
) -> dict[int, list[Path]]:
    """
    Retrieve file paths organized by subfolder name (interpreted as integer ID).

    Scans a directory structure where each subfolder is named with an integer ID,
    and collects files matching the specified criteria within each subfolder.

    Args:
        parent_path: Directory containing the subfolders to scan.
        file_extension: File extension to filter by (without the dot). Default is None.
        match_name: Filename pattern to match. Default is None.

    Returns:
        Dictionary mapping subfolder names (as integers) to lists of file paths.

    Raises:
        ValueError: If neither file_extension nor match_name is provided.
    """
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

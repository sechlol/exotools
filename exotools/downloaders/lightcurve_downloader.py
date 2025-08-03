import contextlib
import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import lightkurve as lk
import numpy as np
from joblib import Parallel, delayed
from requests import HTTPError
from tqdm.auto import tqdm

from exotools.utils.download import DownloadParams
from exotools.utils.observations_fix import Observations

logger = logging.Logger(__name__)


class LightcurveDownloader:
    def __init__(self, override_existing: bool = False, verbose: bool = False):
        self._override_existing = override_existing
        self._verbose = verbose

    def download_one_lc(self, download_args: DownloadParams) -> Optional[Path]:
        # download_path = self._base_folder_path / str(params.tic_id) / f"{params.obs_id}.fits"
        download_path = Path(download_args.download_path)
        download_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._override_existing and download_path.exists():
            return download_path

        # Silences print() for the download_file() calls
        output_stream = sys.stdout if self._verbose else io.StringIO()
        with contextlib.redirect_stdout(output_stream):
            status, message, _ = Observations.download_file(download_args.url, local_path=download_path)

        if status == "COMPLETE":
            return download_path

        logger.error(f"Error downloading {download_args.url}: {message}")
        return None

    def download_fits_multiple(self, download_args: Sequence[DownloadParams]) -> list[Path]:
        return [result for param in tqdm(download_args) if (result := self.download_one_lc(param)) is not None]

    def download_fits_parallel(self, download_args: Sequence[DownloadParams]) -> list[Path]:
        """tid, obs_id, url"""
        parallel_iterator = Parallel(n_jobs=os.cpu_count() - 1, return_as="generator_unordered")
        tqdm_iterator = tqdm(
            parallel_iterator(delayed(self.download_one_lc)(x) for x in download_args),
            total=len(download_args),
            desc="Downloading FITS files",
        )
        return [r for r in tqdm_iterator if r is not None]


def search_available_lightcurve_data(star_name: str, exp_time_s: int = 120) -> Optional[lk.LightCurveCollection]:
    search_results = _search_mast_target(star_name, verbose=False)
    if len(search_results) == 0:
        return None

    return _download_lightcurve_data(search_result=search_results, exp_time=exp_time_s)


def _download_lightcurve_data(
    search_result: lk.SearchResult, authors: Optional[Sequence[str]] = None, exp_time: Optional[int] = None
) -> Optional[lk.LightCurveCollection]:
    if authors is None:
        authors = ["SPOC", "Kepler", "K2"]
    filters = np.isin(search_result.table["provenance_name"], authors)

    if exp_time:
        filters = filters & (search_result.table["exptime"] == exp_time)

    selected_ids = search_result.table[filters]["#"].tolist()
    to_download = search_result[selected_ids]

    try:
        return to_download.download_all()
    except HTTPError as http_e:
        logger.error("Server timeout while downloading lightcurve")
        logger.error(repr(http_e))
        return None
    except lk.search.SearchError as se:
        logger.error("Search error while downloading lightcurve")
        logger.error(repr(se))
        return None


def _search_mast_target(target_name: str, verbose=False) -> lk.SearchResult:
    """
    Searches the target name into the database. Optionally prints the search results
    to the standard output
    """
    results = lk.search_lightcurve(target_name)

    if verbose:
        logger.info(f"### Search Result for target: '{target_name}' ###")
        for i, r in enumerate(results):
            logger.info(f"{i}: {r}, {r.exptime[0]}")

    return results

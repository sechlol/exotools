import contextlib
import io
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import lightkurve as lk
import numpy as np
from astropy.io import fits
from astropy.time import Time
from joblib import Parallel, delayed
from lightkurve import LightCurve, LightCurveCollection
from requests import HTTPError
from tqdm.auto import tqdm

from src.exotools.utils.observations_fix import Observations


_logger = logging.Logger("LightcurveDownloader")


@dataclass
class DownloadParams:
    tic_id: str | int
    obs_id: str | int
    url: str
    verbose: bool = False


class LightcurveLoader:
    def __init__(self, base_folder_path: Path):
        self._base_folder_path = base_folder_path

    def download_fits(self, params: DownloadParams) -> Optional[Path]:
        download_path = self._base_folder_path / str(params.tic_id) / f"{params.obs_id}.fits"
        download_path.parent.mkdir(parents=True, exist_ok=True)

        if download_path.exists():
            return download_path

        # Silences print() for the download_file() calls
        output_stream = sys.stdout if params.verbose else io.StringIO()
        with contextlib.redirect_stdout(output_stream):
            status, message, _ = Observations.download_file(params.url, local_path=download_path)

        if status == "COMPLETE":
            return download_path

        print(f"Error downloading {params.tic_id} {params.url}: {message}")
        return None

    def download_fits_multiple(self, ids_and_urls: list[tuple[str, str, str]]) -> list[Path]:
        return [
            x
            for tid, obs_id, url in ids_and_urls
            if (x := self.download_fits(DownloadParams(tid, obs_id, url))) is not None
        ]

    def download_fits_parallel(self, ids_and_urls: list[tuple[str, str, str]]) -> list[Path]:
        """tid, obs_id, url"""
        parallel_iterator = Parallel(n_jobs=os.cpu_count() - 1, return_as="generator_unordered")
        return [
            result
            for result in tqdm(
                parallel_iterator(delayed(self.download_fits)(DownloadParams(*params)) for params in ids_and_urls),
                total=len(ids_and_urls),
                desc="Downloading FITS files",
            )
            if result is not None
        ]


def load_lightcurve(fits_file_path: Path | str) -> LightCurve:
    # This line stores all the additional information from the fits file. But takes more time to execute
    # return lightkurve.utils.read(downloaded)

    with fits.open(str(fits_file_path)) as hdul:
        lightcurve_data = hdul["LIGHTCURVE"].data
        time_array: np.ndarray = lightcurve_data["TIME"]
        flux: np.ndarray = lightcurve_data["PDCSAP_FLUX"]
        error: np.ndarray = lightcurve_data["PDCSAP_FLUX_ERR"]
        valid_range = ~np.isnan(time_array) & ~np.isnan(flux)

        # Convert to JD time
        time = Time(time_array[valid_range] + 2457000, format="jd", scale="tdb")
        flux = flux[valid_range]

        return LightCurve(time=time, flux=flux, flux_err=error[valid_range], meta=dict(hdul[0].header))


def load_lightcurve_collection(paths: list[Path]) -> LightCurveCollection:
    lightcurves = [load_lightcurve(p).remove_outliers() for p in paths]
    lightcurves = [(lc if np.median(lc.flux.value) > 0 else None) for lc in lightcurves]
    return LightCurveCollection([lc for lc in lightcurves if lc is not None])


def search_available_lightcurve_data(star_name: str) -> lk.SearchResult:
    return search_mast_target(star_name, verbose=False)


def download_lightcurve_data(
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
        _logger.error(f"Server timeout while downloading lightcurve")
        _logger.error(repr(http_e))
        return None
    except lk.search.SearchError as se:
        _logger.error(f"Search error while downloading lightcurve")
        _logger.error(repr(se))
        return None


def copy_lightcurve(lightcurve: LightCurve, with_flux: Optional[np.ndarray] = None) -> LightCurve:
    if with_flux is None:
        return lightcurve.copy(copy_data=True)

    lc = LightCurve(time=lightcurve.time.copy(), flux=with_flux.copy())
    lc.meta = lightcurve.meta
    return lc


def search_mast_target(target_name: str, verbose=False) -> lk.SearchResult:
    """
    Searches the target name into the database. Optionally prints the search results
    to the standard output
    """
    results = lk.search_lightcurve(target_name)

    if verbose:
        print(f"### Search Result for target: '{target_name}' ###")
        for i, r in enumerate(results):
            print(i, r, r.exptime[0])

    return results

from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from lightkurve import LightCurve, LightCurveCollection
from typing_extensions import Self

from .base_db import BaseDB
from .lightcurve_plus import LightCurvePlus


class LightcurveDB(BaseDB):
    """
    Dtypes:
    ------------------
    obs_id     int64
    tic_id     int64
    path      object
    ------------------
    """

    def __init__(self, dataset: QTable):
        super().__init__(dataset=dataset, id_field="obs_id")

    @property
    def tic_ids(self) -> np.ndarray:
        return self.view["tic_id"].value

    @property
    def obs_id(self) -> np.ndarray:
        return self.view["obs_id"].value

    @property
    def unique_tic_ids(self) -> np.ndarray:
        return np.unique(self.tic_ids)

    @property
    def unique_obs_ids(self) -> np.ndarray:
        return np.unique(self.obs_id)

    def _factory(self, dataset: QTable) -> Self:
        return LightcurveDB(dataset)

    def select_by_tic_ids(self, tic_ids: np.ndarray) -> Self:
        return self.where(tic_id=tic_ids)

    def load_by_tic(self, tic_id: int, start_time_at_zero: bool = False) -> Optional[list[LightCurvePlus]]:
        paths = self.view[["path", "obs_id"]][self.view["tic_id"] == tic_id]
        if len(paths) == 0:
            return None

        # Sort lightcurves chronologically
        lcs = [LightCurvePlus(self.load_lightcurve(row["path"]), obs_id=row["obs_id"]) for row in paths]
        lcs = sorted(lcs, key=lambda x: x.time[0])
        if start_time_at_zero:
            for lc in lcs:
                lc.start_at_zero()
        return lcs

    def load_stitched_by_tic(self, tic_id: int, start_time_at_zero: bool = False) -> Optional[LightCurvePlus]:
        lcs = self.load_by_tic(tic_id, start_time_at_zero=False)
        if not lcs:
            return None
        stitched = LightCurveCollection([lc.lc for lc in lcs]).stitch()
        lc_plus = LightCurvePlus(stitched)
        return lc_plus.start_at_zero() if start_time_at_zero else lc_plus

    def load_by_obs_id(self, obs_id: int, start_time_at_zero: bool = False) -> Optional[LightCurvePlus]:
        path = self.view["path"][self.view["obs_id"] == obs_id]
        if len(path) == 0:
            return None
        lc = LightCurvePlus(self.load_lightcurve(path[0]))
        if start_time_at_zero:
            lc = lc.start_at_zero()
        return lc

    def load_collections_by_tics(self, tic_ids: list[int]) -> list[Optional[LightCurveCollection]]:
        return [self.load_by_tic(tic) for tic in tic_ids]

    def load_stitched_by_tics(self, tic_ids: list[int]) -> list[Optional[LightCurvePlus]]:
        return [self.load_stitched_by_tic(tic) for tic in tic_ids]

    def load_by_obs_ids(self, obs_ids: list[int]) -> list[Optional[LightCurvePlus]]:
        return [self.load_by_obs_id(obs) for obs in obs_ids]

    @staticmethod
    def path_map_to_qtable(path_map: dict[int, list[Path]]) -> QTable:
        tabular_data = [
            {"tic_id": tic, "obs_id": int(path.stem), "path": str(path)}
            for tic, paths in path_map.items()
            for path in paths
        ]
        return QTable(tabular_data)

    @staticmethod
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

    @staticmethod
    def load_lightcurve_collection(paths: list[Path]) -> LightCurveCollection:
        lightcurves = [LightcurveDB.load_lightcurve(p).remove_outliers() for p in paths]
        lightcurves = [(lc if np.median(lc.flux.value) > 0 else None) for lc in lightcurves]
        return LightCurveCollection([lc for lc in lightcurves if lc is not None])

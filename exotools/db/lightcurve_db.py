import warnings
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

SECONDS_PER_DAY = 1.0 / 86400.0


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

            # Read metadata from headers
            meta = dict(hdul[0].header)
            meta.update(dict(hdul[1].header))

            # Build the JD(TDB) timestamps from header keywords
            ref_int = meta.get("BJDREFI", 0)
            ref_fractional = meta.get("BJDREFF", 0)
            time_unit = (meta.get("TIMEUNIT", "d") or "d").lower()
            time_sys = (meta.get("TIMESYS", "TDB") or "TDB").lower()
            time_ref = (meta.get("TIMEREF", "LOCAL") or "LOCAL").upper()

            # -------------------------------------------------------------------
            # TIMEUNIT tells us the units of the `TIME` column:
            #   - Usually 'd' (days), meaning no conversion needed.
            #   - Sometimes seconds ('s'), which we must convert to days.
            #   - The `startswith("s")` check is used to cover cases like 'sec',
            #     'seconds', or 's' without having to hardcode each spelling.
            #
            # This ensures the calculation works even if the keyword value changes
            # slightly across data releases or instruments.
            # -------------------------------------------------------------------
            if time_unit in {"d", "day", "days"}:
                factor = 1.0
            elif time_unit.startswith("s"):
                factor = SECONDS_PER_DAY
            else:
                # Default: assume days, but log a warning for unusual units
                warnings.warn(f"Unexpected TIMEUNIT='{time_unit}', assuming days.")
                factor = 1.0

            # Warn if times are not barycentric
            if time_ref != "SOLARSYSTEM":
                warnings.warn(
                    f"TIMEREF='{time_ref}' indicates times are not barycentric; "
                    "consider applying barycentric correction if needed."
                )

            # Compute barycentric Julian dates (BJD_TDB) and construct astropy Time object
            jd = (ref_int + ref_fractional) + time_array[valid_range] * factor
            time = Time(jd, format="jd", scale=time_sys)

            return LightCurve(time=time, flux=flux[valid_range], flux_err=error[valid_range], meta=meta)

    @staticmethod
    def load_lightcurve_collection(paths: list[Path | str]) -> LightCurveCollection:
        lightcurves = [LightcurveDB.load_lightcurve(p).remove_outliers() for p in paths]
        lightcurves = [(lc if np.median(lc.flux.value) > 0 else None) for lc in lightcurves]
        return LightCurveCollection([lc for lc in lightcurves if lc is not None])

    @staticmethod
    def load_lightcurve_plus(fits_file_path: Path | str) -> LightCurvePlus:
        return LightCurvePlus(LightcurveDB.load_lightcurve(fits_file_path))

    @staticmethod
    def load_lightcurve_plus_from_collection(paths: list[Path | str]) -> LightCurvePlus:
        collection = LightcurveDB.load_lightcurve_collection(paths)
        return LightCurvePlus(collection.stitch())

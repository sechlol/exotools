from pathlib import Path
from typing import Optional

import numpy as np
from astropy.table import QTable
from lightkurve import LightCurveCollection

from .base_db import BaseDB
from .lightcurve_plus import LightCurvePlus
from .lightcurve_loader import load_lightcurve


class LightcurveDB(BaseDB):
    def __init__(self, dataset: QTable):
        super().__init__(dataset=dataset, id_field="obs_id")

    @property
    def unique_tic_ids(self) -> np.ndarray:
        return np.unique(self.view["tic_id"].value)

    @property
    def unique_obs_ids(self) -> np.ndarray:
        return np.unique(self.view["obs_id"].value)

    def _factory(self, dataset: QTable) -> "LightcurveDB":
        return LightcurveDB(dataset)

    def select_by_tic_ids(self, tic_ids: np.ndarray) -> "LightcurveDB":
        return self.select_by_mask(np.isin(self.view["tic_id"], tic_ids))

    def load_by_tic(self, tic_id: int, start_time_at_zero: bool = False) -> Optional[list[LightCurvePlus]]:
        paths = self.view[["path", "obs_id"]][self.view["tic_id"] == tic_id]
        if len(paths) == 0:
            return None

        # Sort lightcurves chronologically
        lcs = [LightCurvePlus(load_lightcurve(row["path"]), obs_id=row["obs_id"]) for row in paths]
        lcs = sorted(lcs, key=lambda x: x.time[0])
        if start_time_at_zero:
            t0 = lcs[0].time_x[0]
            for lc in lcs:
                lc.shift_time(-t0)
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
        lc = LightCurvePlus(load_lightcurve(path[0]))
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
            {"tic_id": tic, "obs_id": int(path.stem), "path": path} for tic, paths in path_map.items() for path in paths
        ]
        return QTable(tabular_data)

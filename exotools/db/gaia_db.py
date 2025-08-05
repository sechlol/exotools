import numpy as np
from astropy.table import QTable
from typing_extensions import Self

from .base_db import BaseDB

_ID_FIELD = "gaia_id"


class GaiaDB(BaseDB):
    def _factory(self, dataset: QTable) -> Self:
        return GaiaDB(dataset)

    def __init__(self, gaia_dataset: QTable):
        super().__init__(gaia_dataset, id_field=_ID_FIELD)

    @property
    def tic_ids(self) -> np.ndarray:
        return self.view["tic_id"].value

    @property
    def gaia_ids(self) -> np.ndarray:
        return self.view["gaia_ids"].value

    @property
    def unique_tic_ids(self) -> np.ndarray:
        return np.unique(self.tic_ids)

    @property
    def unique_gaia_ids(self) -> np.ndarray:
        return np.unique(self.gaia_ids)

    @staticmethod
    def impute_radius(dataset: QTable) -> QTable:
        """
        Creates a new column "radius" as the average of the available estimations. Fixes nan values where possible
        """
        r1 = dataset["radius_flame"]
        r2 = dataset["radius_gspphot"]

        # Take average of the two observations where both are present
        r3 = (r1 + r2) / 2
        r1_fill = r1.mask ^ r3.mask
        r2_fill = r2.mask ^ r3.mask
        r3[r1_fill] = r1[r1_fill]
        r3[r2_fill] = r2[r2_fill]
        r3.mask = r1.mask & r2.mask
        dataset["radius"] = r3
        return dataset

    @staticmethod
    def compute_mean_temperature(dataset: QTable) -> QTable:
        """
        Creates a new column "teff_mean" as the average of the available estimations.
        """
        fields = ["teff_gspphot", "teff_gspspec", "teff_esphs", "teff_espucd", "teff_msc1", "teff_msc2"]
        subset = dataset[fields].to_pandas()
        dataset["teff_mean"] = subset.mean(axis=1)
        return dataset

    @staticmethod
    def compute_habitable_zone(dataset: QTable) -> QTable:
        """
        https://www.planetarybiology.com/calculating_habitable_zone.html
        Whitmire, Daniel; Reynolds, Ray, (1996). Circumstellar habitable zones: astronomical considerations.
        In: Doyle, Laurence (ed.). Circumstellar Habitable Zones, 117-142. Travis House Publications, Menlo Park.
        """
        valid_luminosity = np.where(dataset["lum_flame"] > 0, dataset["lum_flame"], np.nan)
        dataset["inner_hz"] = GaiaDB.inner_hz(valid_luminosity)
        dataset["outer_hz"] = GaiaDB.outer_hz(valid_luminosity)
        return dataset

    @staticmethod
    def inner_hz(luminosity: np.ndarray):
        return np.sqrt(luminosity / 1.1)

    @staticmethod
    def outer_hz(luminosity: np.ndarray):
        return np.sqrt(luminosity / 0.53)

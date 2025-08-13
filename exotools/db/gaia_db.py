import numpy as np
from astropy.table import QTable
from typing_extensions import Self

from exotools.utils.masked_operations import safe_average

from .base_db import BaseDB

_ID_FIELD = "gaia_id"


class GaiaDB(BaseDB):
    def _factory(self, dataset: QTable) -> Self:
        return GaiaDB(dataset)

    def __init__(self, gaia_dataset: QTable):
        super().__init__(gaia_dataset, id_field=_ID_FIELD)

    @property
    def gaia_ids(self) -> np.ndarray:
        return self.view["gaia_id"].value

    @property
    def unique_gaia_ids(self) -> np.ndarray:
        return np.unique(self.gaia_ids)

    @staticmethod
    def impute_radius(dataset: QTable) -> QTable:
        """
        Creates a new column "radius" as the average of the available estimations. Fixes nan values where possible
        """
        # Take average of the two observations where both are present
        dataset["radius"] = safe_average(dataset, ["radius_flame", "radius_gspphot"])
        return dataset

    @staticmethod
    def compute_mean_temperature(dataset: QTable) -> QTable:
        """
        Creates a new column "teff_mean" as the average of the available estimations.
        """
        fields = ["teff_gspphot", "teff_gspspec", "teff_esphs", "teff_espucd", "teff_msc1", "teff_msc2"]
        dataset["teff_mean"] = safe_average(dataset, fields)
        return dataset

    @staticmethod
    def compute_habitable_zone(dataset: QTable) -> QTable:
        """
        https://www.planetarybiology.com/calculating_habitable_zone.html
        Whitmire, Daniel; Reynolds, Ray, (1996). Circumstellar habitable zones: astronomical considerations.
        In: Doyle, Laurence (ed.). Circumstellar Habitable Zones, 117-142. Travis House Publications, Menlo Park.
        """
        valid_luminosity = np.where(dataset["lum_flame"] > 0, dataset["lum_flame"], np.nan)
        dataset["inner_hz"] = np.sqrt(valid_luminosity / 1.1)
        dataset["outer_hz"] = np.sqrt(valid_luminosity / 0.53)
        return dataset

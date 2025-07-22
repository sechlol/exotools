import astropy.units as u
import numpy as np
from astropy.table import QTable, join
from astropy.time import Time

from .base_db import BaseDB

_ID_FIELD = "tic_id"
_PARAMETER_JD = ["pl_tranmid"]


class ExoDB(BaseDB):
    def __init__(self, exoplanets_dataset: QTable):
        super().__init__(exoplanets_dataset, id_field=_ID_FIELD)

    @property
    def gaia_id(self) -> np.ndarray:
        return self.view["gaia_id"].value

    def _factory(self, dataset: QTable) -> "ExoDB":
        return ExoDB(dataset)

    def get_star_names(self) -> list[str]:
        return np.unique(self.view["hostname"]).tolist()

    def get_default_records(self) -> "ExoDB":
        return self._factory(self.view[self.view["default_flag"] == 1])

    def get_tess_planets(self) -> "ExoDB":
        condition = np.char.find(self.view["disc_telescope"], "TESS") != -1
        return self._factory(self.view[condition])

    def get_kepler_planets(self) -> "ExoDB":
        condition = np.char.find(self.view["disc_telescope"], "Kepler") != -1
        return self._factory(self.view[condition])

    def get_transiting_planets(self, kepler_or_tess_only: bool = False) -> "ExoDB":
        condition = self.view["tran_flag"] == 1
        if kepler_or_tess_only:
            telescope_condition = np.char.find(self.view["disc_telescope"], "TESS") != -1
            telescope_condition |= np.char.find(self.view["disc_telescope"], "Kepler") != -1
            condition &= telescope_condition
        return self._factory(self.view[condition])

    @staticmethod
    def preprocess_dataset(dataset: QTable):
        # Set lowercase hostname for faster retrieval
        dataset["hostname_lowercase"] = np.char.lower(dataset["hostname"].tolist())

    @staticmethod
    def impute_stellar_parameters(dataset: QTable, gaia_data: QTable):
        subset_exo = dataset["gaia_id", "st_rad", "pl_ratror", "pl_ratdor", "pl_rade", "pl_orbsmax"]
        subset_gaia = gaia_data[["gaia_id", "radius"]]

        # Join exoplanets dataset with Gaia dataset
        joined = join(subset_exo, subset_gaia, keys="gaia_id", join_type="left")
        valid_radius = ~joined["radius"].mask

        # Find the properties that can be repaired
        recoverable_radius = joined["st_rad"].mask & valid_radius
        recoverable_ratror = joined["pl_ratror"].mask & ~joined["pl_rade"].mask & valid_radius
        recoverable_ratdor = joined["pl_ratdor"].mask & ~joined["pl_orbsmax"].mask & valid_radius

        # Needs to sort the dataset, as the joined table gets sorted by key
        dataset.sort(keys="gaia_id")

        # Repair
        dataset["st_rad"][recoverable_radius] = joined["radius"][recoverable_radius]
        dataset["pl_ratror"][recoverable_ratror] = (
            joined["pl_rade"][recoverable_ratror].to(u.solRad) / joined["radius"][recoverable_ratror]
        )
        dataset["pl_ratdor"][recoverable_ratdor] = (
            joined["pl_orbsmax"][recoverable_ratdor].to(u.solRad) / joined["radius"][recoverable_ratdor]
        )
        dataset["st_rad_gaia"] = joined["radius"]

    @staticmethod
    def compute_bounds(dataset: QTable):
        # Set all the missing error fields to be 0
        for err_param in _get_error_bounds_from_table(dataset):
            dataset[err_param].fill_value = 0

        parameters_with_err = [c.removesuffix("err1") for c in dataset.colnames if "err1" in c]
        for c in parameters_with_err:
            dataset[f"{c}_upper"] = dataset[c] + dataset[f"{c}err1"]
            dataset[f"{c}_lower"] = dataset[c] + dataset[f"{c}err2"]

    @staticmethod
    def convert_time_columns(dataset: QTable):
        columns_to_convert = _get_limits_from_table(_PARAMETER_JD, include_input=True)
        for c in columns_to_convert:
            dataset[c] = Time(dataset[c], format="jd", scale="tdb")


# TODO: find a better name for these methods
def _get_limits_from_table(columns: list[str], include_input: bool = False) -> list[str]:
    all_columns = []
    for c in columns:
        all_columns.extend([f"{c}_upper", f"{c}_lower"])
    if include_input:
        all_columns.extend(columns)
    return all_columns


def _get_columns_with_error_bounds(dataset: QTable) -> list[str]:
    return [c.removesuffix("err1") for c in dataset.colnames if "err1" in c]


def _get_error_bounds_from_table(dataset: QTable) -> list[str]:
    return [c for c in dataset.colnames if c.endswith("err1") or c.endswith("err2")]

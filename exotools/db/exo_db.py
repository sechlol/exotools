import logging

import astropy.units as u
from astropy.table import QTable, join
from typing_extensions import Self

from .ps_db import PsDB

logger = logging.getLogger(__name__)


class ExoDB(PsDB):
    def _factory(self, dataset: QTable) -> Self:
        return ExoDB(dataset)

    def get_default_records(self) -> Self:
        return self._factory(self.view[self.view["default_flag"] == 1])

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

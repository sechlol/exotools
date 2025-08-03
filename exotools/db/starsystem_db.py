from typing import Optional, Self

from astropy.table import QTable

from .exo_db import ExoDB
from .star_system import StarSystem

_ID_FIELD = "tic_id"


class StarSystemDB(ExoDB):
    def __init__(self, transit_dataset: QTable):
        super().__init__(transit_dataset)

    def _factory(self, dataset: QTable) -> Self:
        return StarSystemDB(dataset)

    def get_valid_planets(self) -> Self:
        return StarSystemDB(self.view[self.view["pl_valid_flag"]])

    def get_star_system_from_star_name(self, star_name: str) -> Optional[StarSystem]:
        system_data = self.view[self.view["hostname_lowercase"] == star_name.lower()]
        if len(system_data) == 0:
            return None
        return StarSystem(star_name=star_name, data=system_data)

    def get_star_system_from_tic_id(self, tic_id: int) -> Optional[StarSystem]:
        system_data = self.view[self.view["tic_id"] == tic_id]
        if len(system_data) == 0:
            return None
        star_name = system_data["hostname"][0]
        return StarSystem(star_name=star_name, data=system_data)

    @staticmethod
    def preprocess_dataset(dataset: QTable) -> QTable:
        dataset.sort("rowupdate", reverse=True)
        dataset.add_index("pl_name")
        return dataset.filled(0)

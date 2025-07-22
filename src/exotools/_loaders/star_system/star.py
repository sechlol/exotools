from astropy.table import Row
from astropy.units import Quantity

from .uncertain_data import UncertainDataSource, UncertainValue


class Star(UncertainDataSource):
    def __init__(self, star_name: str, data: Row):
        super().__init__(data)
        self._name: str = star_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def tic_id(self):
        return self._row["tic_id"]

    @property
    def radius(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("st_rad")

    @property
    def mass(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("st_mass")

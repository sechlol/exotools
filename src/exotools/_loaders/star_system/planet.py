from astropy.table import Row
from astropy.time import Time
from astropy.units import Quantity

from .uncertain_data import UncertainDataSource, UncertainValue


class Planet(UncertainDataSource):
    def __init__(self, name: str, data: Row):
        super().__init__(data)
        self._name = name

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self) -> str:
        return (
            f"{self.name}, "
            f"r: {self.radius.central}, "
            f"p: {self.orbital_period.central}, "
            f"dur: {self.transit_duration.upper}, "
            f"mid: {self.transit_midpoint.central}, "
            f"depth: {self.transit_depth.central}, "
            # f"axis: {self.semimajor_axis.central}, "
            f"{'' if self.has_mandatory_parameters else '(MISSING PARAMETERS)'}"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def has_mandatory_parameters(self) -> bool:
        return self._row["pl_valid_flag"]

    @property
    def radius(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("pl_rade")

    @property
    def mass(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("pl_masse")

    @property
    def density(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("pl_dens")

    @property
    def eccentricity(self) -> UncertainValue[float]:
        return self._uncertain_value_from_cache("pl_orbeccen")

    @property
    def orbital_period(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("pl_orbper")

    @property
    def parameter_of_periastron(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("pl_orblper")

    @property
    def orbital_inclination(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("pl_orbincl")

    @property
    def semimajor_axis(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("pl_orbsmax")

    @property
    def transit_midpoint(self) -> UncertainValue[Time]:
        return self._uncertain_value_from_cache("pl_tranmid")

    @property
    def transit_duration(self) -> UncertainValue[Quantity]:
        return self._uncertain_value_from_cache("pl_trandur")

    @property
    def transit_depth(self) -> UncertainValue[float]:
        return self._uncertain_value_from_cache("pl_trandep")

    @property
    def impact_parameter(self) -> UncertainValue[float]:
        return self._uncertain_value_from_cache("pl_imppar")

    @property
    def radius_to_stellar_ratio(self) -> UncertainValue[float]:
        return self._uncertain_value_from_cache("pl_ratror")

    @property
    def semimajor_axis_to_stellar_ratio(self) -> UncertainValue[float]:
        return self._uncertain_value_from_cache("pl_ratdor")

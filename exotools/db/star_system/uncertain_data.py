from dataclasses import dataclass
from typing import Generic, TypeVar

from astropy.table import Row

T = TypeVar("T")


@dataclass
class UncertainValue(Generic[T]):
    central: T
    lower: T
    upper: T


class UncertainDataSource:
    def __init__(self, data: Row):
        self._value_cache = {}
        self._row = data

    def _uncertain_value_from_cache(self, parameter_name: str) -> UncertainValue:
        if parameter_name not in self._value_cache:
            c = self._row[parameter_name]
            self._value_cache[parameter_name] = UncertainValue(
                central=c,
                lower=min(c, self._row[f"{parameter_name}_lower"]),
                upper=max(c, self._row[f"{parameter_name}_upper"]),
            )
        return self._value_cache[parameter_name]

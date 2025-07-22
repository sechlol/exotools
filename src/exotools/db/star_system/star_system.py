import logging
from typing import Optional, List, Dict

import numpy as np
from astropy.table import QTable

from .planet import Planet
from .star import Star

_logger = logging.Logger("StarSystem")


class StarSystem:
    def __init__(self, star_name: str, data: QTable):
        self._star_name: str = star_name
        self._ds = data
        self._tic_id = self._ds["tic_id"][0]

        planets_name: List[str] = np.unique(self._ds["pl_name"]).tolist()
        self._planets: Dict[str, Planet] = {name: _get_planet_from_table(self._ds, name) for name in planets_name}
        self._star = _get_star_from_table(self._ds, star_name)

        # Assumes there is only one row per planet
        assert len(planets_name) == len(self._ds), (
            f"Detected multiple entries for planets of star {star_name}. "
            f"Make sure there is only one entry per planets when creating "
            f"a StarSystem."
        )

    def __str__(self) -> str:
        return self.to_string()

    @property
    def tic_id(self) -> Optional[int]:
        return self._tic_id

    @property
    def star_name(self) -> str:
        return self._star_name

    @property
    def planets_count(self) -> int:
        return len(self._planets)

    @property
    def planets_name(self) -> List[str]:
        return list(self._planets.keys())

    @property
    def planets(self) -> List[Planet]:
        return list(self._planets.values())

    @property
    def star(self) -> Star:
        return self._star

    @property
    def has_valid_planets(self) -> bool:
        return np.all([p.has_mandatory_parameters for p in self.planets])

    def to_string(self) -> str:
        rows = [
            f"StarSystem {self.star_name} (TIC_ID: {self.tic_id}) with {len(self._planets)} planets.",
            f"\t* Star {self.star.radius.central}, {self.star.mass.central}",
        ]
        for planet in self.planets:
            rows.append(f"\t- {planet}")
        return "\n".join(rows)

    def get_planet_from_name(self, planet_name: str) -> Optional[Planet]:
        return self._planets.get(planet_name)

    def get_planet_from_letter(self, planet_letter: str) -> Optional[Planet]:
        return self.get_planet_from_name(planet_name=f"{self.star_name} {planet_letter}")


def _get_planet_from_table(table: QTable, planet_name: str) -> Optional[Planet]:
    planet_data = table[table["pl_name"] == planet_name]
    if len(planet_data) == 0:
        return None

    return Planet(planet_name, planet_data[0])


def _get_star_from_table(table: QTable, star_name: str) -> Star:
    star_parameters = [col for col in table.columns if col.startswith("st_")]
    return Star(star_name=star_name, data=table[star_parameters][0])

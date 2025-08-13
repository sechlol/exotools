import warnings
from math import ceil
from typing import Any, Optional

import numpy as np
from astropy.time import Time, TimeDelta
from astropy.units import Quantity
from lightkurve import FoldedLightCurve, LightCurve
from typing_extensions import Self

from .star_system import Planet


class LightCurvePlus:
    def __init__(self, lightcurve: LightCurve, obs_id: Optional[int] = None):
        self.lc: LightCurve = _convert_time_to_jd(lightcurve)
        self._time_shift = TimeDelta(0, format=self.lc.time.format, scale=self.lc.time.scale)
        self._obs_id = obs_id
        self._warn_if_not_barycentric()

    @property
    def time_x(self) -> np.ndarray:
        return self.lc.time.value

    @property
    def time(self) -> Time:
        return self.lc.time

    @property
    def flux_y(self) -> np.ndarray:
        return self.lc.flux.value

    @property
    def flux(self) -> np.ndarray:
        return self.lc.flux

    @property
    def tic_id(self) -> int:
        return self.meta["TICID"]

    @property
    def obs_id(self) -> Optional[int]:
        return self._obs_id

    @property
    def meta(self) -> dict[str, Any]:
        return self.lc.meta

    @property
    def time_bjd(self) -> np.ndarray:
        """Absolute BJD in TDB (days) as a NumPy array."""
        # Convert to TDB explicitly to be unambiguous
        return np.asarray(self.time.tdb.jd, dtype=float)

    @property
    def time_elapsed(self) -> np.ndarray:
        """
        Days since first cadence (relative timeline), independent of BJDREF*.
        """
        bjd = self.time_bjd
        return bjd - bjd[0]

    @property
    def time_btjd(self) -> np.ndarray:
        """
        TESS BTJD in days, i.e., BJD_TDB − (BJDREFI + BJDREFF).
        """

        refi = self.meta.get("BJDREFI")
        reff = self.meta.get("BJDREFF")
        if refi is None and reff is None:
            # TESS convention; safe fallback for BTJD if headers were stripped
            warnings.warn("BJDREFI/BJDREFF not found in meta; assuming 2457000.0 (TESS default) for BTJD.")
            refi, reff = 2457000, 0.0
        else:
            refi = 0 if refi is None else refi
            reff = 0.0 if reff is None else reff
        bjd_ref = float(refi) + float(reff)

        return self.time_bjd - bjd_ref

    def _warn_if_not_barycentric(self) -> None:
        """Warn if TIMEREF suggests times are not barycentric."""
        timeref = (self.meta.get("TIMEREF") or "").upper()
        if timeref and timeref != "SOLARSYSTEM":
            warnings.warn(
                f"TIMEREF='{timeref}' indicates times may not be barycentric; "
                "BJD/BTJD semantics assume barycentric timing."
            )

    def to_numpy(self) -> np.ndarray:
        return np.array([self.time_x, self.flux_y]).T

    def remove_outliers(self) -> Self:
        return LightCurvePlus(self.lc.remove_outliers())

    def normalize(self) -> Self:
        return LightCurvePlus(self.lc.normalize())

    def get_first_transit_value(self, planet: Planet) -> Time:
        i = self.get_transit_first_index(planet)
        return self.lc.time[i]

    def get_transit_first_index(self, planet: Planet) -> int:
        """
        Get the index of the first transit in the light curve time series.
        """
        return _find_fist_transit_index(
            time=self.time_x, period=planet.orbital_period.central.value, midpoint=self._get_aligned_midpoint(planet)
        )

    def shift_time(self, shift: float | Quantity) -> Self:
        delta = TimeDelta(shift, format=self.lc.time.format, scale=self.lc.time.scale)
        self._time_shift += delta
        self.lc.time += delta
        return self

    def start_at_zero(self) -> Self:
        return self.shift_time(shift=-self.lc.time[0].value)

    def get_transit_phase(self, planet: Planet) -> np.ndarray:
        return _get_phase(
            time=self.time_x, period=planet.orbital_period.central.value, midpoint=self._get_aligned_midpoint(planet)
        )

    def get_transit_mask(self, planet: Planet, duration_increase_percent: float = 0) -> np.ndarray:
        """
        Args:
            planet: planet with transit information
            duration_increase_percent: increases the transit duration by a given percentage (0 to 1).
            This changes the size of the masked regions

        Returns: a boolean array were 1 corresponds to planet transits
        """
        return self.lc.create_transit_mask(
            period=planet.orbital_period.central,
            transit_time=self._get_aligned_midpoint(planet),
            duration=planet.transit_duration.central + duration_increase_percent * planet.transit_duration.central,
        )

    def get_transit_count(self, planet: Planet) -> int:
        # mask          = 000011100000011100
        # mask[:-1]     = 00001110000001110
        # mask[1:]      = 00011100000011100
        # xor_mask      = 00010010000010010
        mask = self.get_transit_mask(planet=planet)
        xor_mask = mask[:-1] ^ mask[1:]
        return ceil(xor_mask.sum() / 2)

    def get_combined_transit_mask(self, planets: list[Planet]) -> np.ndarray:
        return self.lc.create_transit_mask(
            period=[p.orbital_period.central for p in planets],
            transit_time=[self._get_aligned_midpoint(p) for p in planets],
            duration=[p.transit_duration.central for p in planets],
        )

    def fold_with_planet(self, planet: Planet, normalize_time: bool = False) -> FoldedLightCurve:
        return self.lc.fold(
            epoch_time=self._get_aligned_midpoint(planet),
            period=planet.orbital_period.central,
            normalize_phase=normalize_time,
        )

    def copy_with_flux(self, flux: np.ndarray) -> Self:
        lc = copy_lightcurve(self.lc, with_flux=flux)
        return LightCurvePlus(lc)

    def _get_aligned_midpoint(self, planet: Planet) -> float:
        return (planet.transit_midpoint.central + self._time_shift).value

    def fold(self, period=None, epoch_time=None, epoch_phase=0, wrap_phase=None, normalize_phase=False):
        return self.lc.fold(
            period=period,
            epoch_time=epoch_time,
            epoch_phase=epoch_phase,
            wrap_phase=wrap_phase,
            normalize_phase=normalize_phase,
        )

    def __len__(self) -> int:
        return len(self.time_x)

    def __sub__(self, other):
        if isinstance(other, LightCurvePlus):
            return LightCurvePlus(lightcurve=self.lc - other.lc)
        return LightCurvePlus(lightcurve=self.lc - other)

    def __add__(self, other):
        if isinstance(other, LightCurvePlus):
            return LightCurvePlus(lightcurve=self.lc + other.lc)
        return LightCurvePlus(lightcurve=self.lc + other)

    def __getitem__(self, index) -> Self:
        return LightCurvePlus(self.lc[index])


def copy_lightcurve(lightcurve: LightCurve, with_flux: Optional[np.ndarray] = None) -> LightCurve:
    if with_flux is None:
        return lightcurve.copy(copy_data=True)

    lc = LightCurve(time=lightcurve.time.copy(), flux=with_flux.copy())
    lc.meta = lightcurve.meta
    return lc


def _btjd_to_jd_time(time: Time) -> Time:
    return Time(val=time.value + 2457000, format="jd", scale="tdb")


def _convert_time_to_jd(lc: LightCurve) -> LightCurve:
    if lc.time.scale != "tdb":
        raise ValueError(f"Time scale {lc.time.scale} unknown/unsupported.")

    if lc.time.format == "jd":
        return lc
    elif lc.time.format == "btjd":
        new_t = _btjd_to_jd_time(lc.time)
        return LightCurve(time=new_t, flux=lc.flux, flux_err=lc.flux_err, meta=lc.meta)
    raise ValueError(f"Time format {lc.time.format} unknown/unsupported.")


def _get_phase(time: np.ndarray, period: float, midpoint: float) -> np.ndarray:
    k = np.round((time - midpoint) / period)
    closest_event_time = midpoint + k * period
    return np.abs(closest_event_time - time)


def _find_fist_transit_index(time: np.ndarray, period: float, midpoint: float, step: int = 100) -> int:
    phase = _get_phase(time=time, midpoint=midpoint, period=period)

    i = 1
    length = len(time) - 1
    while i < length and phase[i - 1] < phase[i]:
        i = min(i + step, length)
    while i < length and phase[i - 1] > phase[i]:
        i = min(i + step, length)
    while i > 0 and phase[i - 1] < phase[i]:
        i = max(i - step, -1)
    return i

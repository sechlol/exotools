import warnings
from math import ceil
from typing import Any, Optional

import numpy as np
from astropy.time import Time, TimeDelta
from astropy.units import Quantity
from lightkurve import FoldedLightCurve, LightCurve
from typing_extensions import Self

from exotools.utils.array_utils import (
    get_contiguous_interval_indices,
    get_contiguous_intervals,
    get_gaps_interval_indices,
    get_gaps_intervals,
)

from .star_system import Planet


class LightCurvePlus:
    def __init__(self, lightcurve: LightCurve, obs_id: Optional[int] = None):
        # Store original format information
        self._original_time_format = lightcurve.meta.get("_ORIGINAL_TIME_FORMAT", "btjd")

        # Use the lightcurve as-is, preserving its original time format
        self.lc: LightCurve = lightcurve

        # TimeDelta doesn't support all Time formats, so use 'sec' format for compatibility
        self._time_shift = TimeDelta(0, format="sec", scale=self.lc.time.scale)
        self._obs_id = obs_id
        self._warn_if_not_barycentric()

    @property
    def time_system(self) -> str:
        """Return the current time system (format/scale combination)."""
        return f"{self.lc.time.format.upper()}/{self.lc.time.scale.upper()}"

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
    def jd_time(self) -> np.ndarray:
        """Julian Date as a NumPy array."""
        if self.lc.time.format == "jd":
            # Already in JD format, return directly
            return np.asarray(self.lc.time.value, dtype=float)
        else:
            # Convert to JD
            return np.asarray(self.lc.time.jd, dtype=float)

    @property
    def bjd_time(self) -> np.ndarray:
        """Absolute BJD in TDB (days) as a NumPy array."""
        if self.lc.time.format == "jd" and self.lc.time.scale == "tdb":
            # Already in BJD_TDB format, return directly
            return np.asarray(self.lc.time.value, dtype=float)
        else:
            # Convert to TDB explicitly to be unambiguous
            return np.asarray(self.lc.time.tdb.jd, dtype=float)

    @property
    def elapsed_time(self) -> np.ndarray:
        """
        Days since first cadence (relative timeline), independent of BJDREF*.
        """
        bjd = self.bjd_time
        return bjd - bjd[0]

    @property
    def btjd_time(self) -> np.ndarray:
        """
        TESS BTJD in days, i.e., BJD_TDB − (BJDREFI + BJDREFF).
        """
        if self.lc.time.format == "btjd":
            # Already in BTJD format, return directly
            return np.asarray(self.lc.time.value, dtype=float)

        # Need to convert from other format to BTJD
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

        return self.bjd_time - bjd_ref

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

    def remove_nans(self) -> Self:
        return LightCurvePlus(self.lc.remove_nans(), obs_id=self._obs_id)

    def remove_outliers(self) -> Self:
        return LightCurvePlus(self.lc.remove_outliers(), obs_id=self._obs_id)

    def normalize(self) -> Self:
        return LightCurvePlus(self.lc.normalize(), obs_id=self._obs_id)

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
        # Use 'sec' format for TimeDelta compatibility, but convert to days if needed
        if isinstance(shift, (int, float)):
            # Assume shift is in the same units as the time (days for astronomical data)
            delta = TimeDelta(shift * 86400, format="sec", scale=self.lc.time.scale)  # Convert days to seconds
        else:
            delta = TimeDelta(shift, format="sec", scale=self.lc.time.scale)
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
        return LightCurvePlus(lc, obs_id=self._obs_id)

    def find_time_gaps_i(self, greater_than_median: float = 10.0) -> list[tuple[int, int]]:
        """
        Find time gaps in the lightcurve based on time step analysis.

        Identifies locations where the time difference between consecutive points
        exceeds the median time step by a specified factor, indicating data gaps
        or interruptions in observations.

        Args:
            greater_than_median: Threshold multiplier for gap detection. Gaps are
                identified where time_diff > median_time_step * greater_than_median.


        Returns:
            List of index tuples (i, i+1) where each tuple represents the indices
            immediately before and after a detected gap. The gap occurs between
            time[i] and time[i+1].
        """
        return get_gaps_interval_indices(x=self.time_x, greater_than_median=greater_than_median)

    def find_time_gaps_x(self, greater_than_median: float = 10.0) -> list[tuple[float, float]]:
        """
        Find time gaps in the lightcurve and return actual time values.

        Identifies locations where the time difference between consecutive points
        exceeds the median time step by a specified factor, returning the actual
        time values at gap boundaries rather than indices.

        Args:
            greater_than_median: Threshold multiplier for gap detection. Gaps are
                identified where time_diff > median_time_step * greater_than_median.

        Returns:
            List of time value tuples (t1, t2) where each tuple represents the
            actual time values immediately before and after a detected gap.
            The gap occurs between time t1 and time t2.

        See Also:
            find_time_gaps_i: Returns the same gaps as index pairs instead of time values.
        """
        return get_gaps_intervals(x=self.time_x, greater_than_median=greater_than_median)

    def find_contiguous_time_i(self, greater_than_median: float = 10.0) -> list[tuple[int, int]]:
        """
        Find contiguous time intervals in the lightcurve based on time step analysis.

        Identifies regions where time differences between consecutive points remain
        below the threshold, indicating continuous observation periods without
        significant gaps.

        Args:
            greater_than_median: Threshold multiplier for gap detection. Contiguous
                intervals are where time_diff <= median_time_step * greater_than_median.

        Returns:
            List of index tuples (start, end) where each tuple represents the start
            and end indices (inclusive) of a contiguous time interval.
        """
        return get_contiguous_interval_indices(x=self.time_x, greater_than_median=greater_than_median)

    def find_contiguous_time_x(self, greater_than_median: float = 10.0) -> list[tuple[float, float]]:
        """
        Find contiguous time intervals in the lightcurve and return actual time values.

        Identifies regions where time differences between consecutive points remain
        below the threshold, returning the actual time values at the boundaries
        of contiguous intervals.

        Args:
            greater_than_median: Threshold multiplier for gap detection. Contiguous
                intervals are where time_diff <= median_time_step * greater_than_median.

        Returns:
            List of time value tuples (t_start, t_end) where each tuple represents
            the actual time values at the start and end of a contiguous interval.
        """
        return get_contiguous_intervals(x=self.time_x, greater_than_median=greater_than_median)

    def to_jd_time(self) -> Self:
        """Convert the light curve time to plain Julian Date (JD) *representation* in place.

        JD is the continuous count of days since 4713 BCE (noon), independent of location;
        the *scale* (UTC, TT, TDB, …) is tracked separately. This method puts the times
        in `format="jd"` while preserving the existing time *scale* and reference frame.

        When your times are already barycentric (e.g., TESS BJD_TDB), converting to JD
        does not change the numeric values—it only standardizes the representation.

        Examples
        --------
        Suppose your first cadence is BJD_TDB = 2458354.123456:
        >>> lc.time.format, lc.time.scale
        ('jd', 'tdb')
        >>> lc.time[0].value
        2458354.123456
        >>> lc.to_jd_time().lc.time[0].value  # still JD in TDB scale
        2458354.123456

        Returns
        -------
        Self
            Returns self for method chaining.
        """
        if self.lc.time.format != "jd":
            self.lc = _convert_time_to_bjd(self.lc)
        return self

    def to_btjd_time(self) -> Self:
        """Convert the light curve time to BTJD (Barycentric TESS Julian Date) in place.

        BTJD is a TESS-specific convenience: BTJD ≡ BJD_TDB − (BJDREFI + BJDREFF).
        For standard SPOC products, (BJDREFI, BJDREFF) = (2457000, 0), so BTJD = BJD_TDB − 2457000.
        This keeps the *barycentric* reference and the TDB time scale, but shifts
        the zero-point so numbers are ~10^3 instead of ~2.4×10^6.

        Examples
        --------
        >>> # Starting from BJD_TDB = 2458354.123456 (TESS Year 1)
        >>> lc.to_btjd_time().lc.time[0].value
        1354.123456      # 2458354.123456 - 2457000.0

        >>> # Converting back to BJD_TDB (see to_bjd_time) restores the 2.458e6 magnitude.
        >>> lc.to_bjd_time().lc.time[0].value
        2458354.123456

        Returns
        -------
        Self
            Returns self for method chaining.
        """
        if self.lc.time.format != "btjd":
            self.lc = _convert_time_to_btjd(self.lc)
        return self

    def to_bjd_time(self) -> Self:
        """Convert the light curve time to Barycentric Julian Date (BJD_TDB) in place.

        **BJD** is simply JD evaluated at the Solar System Barycenter (SSB).
        For TESS, timestamps are already referenced to the SSB with `TIMESYS='TDB'`
        and `TIMEREF='SOLARSYSTEM'`, so BJD_TDB is the physically correct absolute time.
        Numerically, BJD_TDB equals JD in the TDB scale when the reference location is barycentric.

        This method ensures the output is **BJD_TDB** (absolute, not offset), which is
        what you want for comparing absolute epochs (e.g., transit mid-times) across sectors
        or with literature ephemerides.

        Examples
        --------
        >>> # From BTJD back to absolute BJD_TDB:
        >>> lc.to_btjd_time().lc.time[0].value
        1354.123456
        >>> lc.to_bjd_time().lc.time[0].value
        2458354.123456  # adds back (BJDREFI + BJDREFF) = 2457000.0

        >>> # If already BJD_TDB, calling again is a no-op:
        >>> lc.time.format, lc.time.scale
        ('jd', 'tdb')
        >>> lc.to_bjd_time().lc.time.format
        'jd'

        Returns
        -------
        Self
            Returns self for method chaining.
        """
        # BJD_TDB is represented as JD in the TDB scale with a barycentric reference.
        # If your internal representation uses a custom 'btjd' format, this will add back
        # the header offset (BJDREFI + BJDREFF). Otherwise, it's effectively a no-op.
        if self.lc.time.format != "jd":
            self.lc = _convert_time_to_bjd(self.lc)
        return self

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
        return LightCurvePlus(self.lc[index], obs_id=self._obs_id)


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


def _convert_time_to_btjd(lc: LightCurve) -> LightCurve:
    """Convert lightcurve time to BTJD format."""
    if lc.time.scale != "tdb":
        raise ValueError(f"Time scale {lc.time.scale} unknown/unsupported.")

    if lc.time.format == "btjd":
        return lc
    elif lc.time.format == "jd":
        # Convert JD to BTJD by subtracting reference
        refi = lc.meta.get("BJDREFI", 2457000)
        reff = lc.meta.get("BJDREFF", 0.0)
        bjd_ref = float(refi) + float(reff)
        new_t = Time(lc.time.value - bjd_ref, format="btjd", scale="tdb")
        return LightCurve(time=new_t, flux=lc.flux, flux_err=lc.flux_err, meta=lc.meta)
    raise ValueError(f"Time format {lc.time.format} unknown/unsupported.")


def _convert_time_to_bjd(lc: LightCurve) -> LightCurve:
    """Convert lightcurve time to BJD format (same as JD for TDB scale)."""
    # For TDB scale, BJD is the same as JD
    return _convert_time_to_jd(lc)


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

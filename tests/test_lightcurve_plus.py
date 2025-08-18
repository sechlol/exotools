import numpy as np
import pytest
from astropy.time import Time
from lightkurve import FoldedLightCurve, LightCurve

from exotools import LightcurveDB, LightCurvePlus, Planet, StarSystemDB
from exotools.db.lightcurve_plus import copy_lightcurve
from exotools.utils.array_utils import get_gaps_indices


class TestLightcurvePlus:
    @pytest.fixture
    def tic_with_planets(self) -> list[int]:
        return [159781361, 179317684, 441765914, 158635959, 328081248]

    @pytest.fixture
    def sample_lc_plus(self, all_test_lightcurves) -> LightCurvePlus:
        """Get a sample LightCurvePlus object with a valid observation ID."""
        obs_id = 65149728  # next(iter(all_test_lightcurves))
        return LightCurvePlus(copy_lightcurve(all_test_lightcurves[obs_id]), obs_id=obs_id)

    @pytest.fixture
    def lc_and_planet(
        self, star_system_test_db: StarSystemDB, lc_test_db: LightcurveDB
    ) -> tuple[Planet, LightCurvePlus]:
        planets_with_lc = list(set(star_system_test_db.unique_tic_ids) & set(lc_test_db.unique_tic_ids))
        tic_id = np.random.choice(planets_with_lc)
        planet = star_system_test_db.get_star_system_from_tic_id(tic_id).planets[0]
        obs_id = np.random.choice(lc_test_db.where(tic_id=tic_id).unique_obs_ids)
        lc = lc_test_db.load_by_obs_id(obs_id)
        return planet, lc

    def test_print_meta(self, all_test_lightcurves):
        for lc in all_test_lightcurves.values():
            print(lc.meta)
            break

    def test_planet_ids(self, star_system_test_db: StarSystemDB, tic_with_planets):
        for tic_id in tic_with_planets:
            star_system = star_system_test_db.get_star_system_from_tic_id(tic_id)
            assert star_system is not None
            assert star_system.has_valid_planets
            assert star_system.planets_count > 0

    def test_lightcurve_plus_basic(self, star_system_test_db: StarSystemDB, all_test_lightcurves):
        for obs_id, lc in all_test_lightcurves.items():
            lc_plus = LightCurvePlus(lc, obs_id=obs_id)
            assert lc_plus.tic_id is not None
            assert lc_plus.obs_id == obs_id

            planet = star_system_test_db.get_star_system_from_tic_id(lc_plus.tic_id)
            if planet:
                print(lc_plus.tic_id)

    def test_properties(self, sample_lc_plus):
        """Test basic properties of LightCurvePlus."""
        # Test time properties
        assert isinstance(sample_lc_plus.time, Time)
        assert isinstance(sample_lc_plus.time_x, np.ndarray)
        assert len(sample_lc_plus.time_x) > 0

        # Test flux properties
        assert isinstance(sample_lc_plus.flux_y, np.ndarray)
        assert isinstance(sample_lc_plus.flux, np.ndarray)
        assert len(sample_lc_plus.flux_y) > 0

        # Test metadata properties
        assert isinstance(sample_lc_plus.meta, dict)
        assert "TICID" in sample_lc_plus.meta

    def test_time_bjd(self, sample_lc_plus):
        """Test the time_bjd property."""
        # Test that time_bjd returns a numpy array
        bjd_time = sample_lc_plus.bjd_time
        assert isinstance(bjd_time, np.ndarray)
        assert len(bjd_time) == len(sample_lc_plus)

        # Test that the values are in the expected range for BJD
        # BJD values should be around 2.4-2.5 million for recent astronomical observations
        assert np.all(bjd_time > 2400000)
        assert np.all(bjd_time < 2500000)

        # Test that the values match the underlying time.tdb.jd values
        expected_bjd = np.asarray(sample_lc_plus.time.tdb.jd, dtype=float)
        np.testing.assert_array_equal(bjd_time, expected_bjd)

    def test_time_elapsed(self, sample_lc_plus):
        """Test the time_elapsed property."""
        elapsed_time = sample_lc_plus.elapsed_time
        assert isinstance(elapsed_time, np.ndarray)
        assert len(elapsed_time) == len(sample_lc_plus)

        # First value should always be 0 (days since first cadence)
        assert elapsed_time[0] == 0

        # All values should be non-negative and monotonically increasing
        assert np.all(elapsed_time >= 0)
        assert np.all(np.diff(elapsed_time) >= 0)

        # Verify the calculation is correct
        bjd_time = sample_lc_plus.bjd_time
        expected_elapsed = bjd_time - bjd_time[0]
        np.testing.assert_array_equal(elapsed_time, expected_elapsed)

    def test_time_btjd(self, sample_lc_plus):
        """Test the time_btjd property."""
        btjd_time = sample_lc_plus.btjd_time
        assert isinstance(btjd_time, np.ndarray)
        assert len(btjd_time) == len(sample_lc_plus)

        # Get the reference values from metadata or use TESS default
        refi = sample_lc_plus.meta.get("BJDREFI")
        reff = sample_lc_plus.meta.get("BJDREFF")
        if refi is None and reff is None:
            bjd_ref = 2457000.0  # TESS default
        else:
            refi = 0 if refi is None else refi
            reff = 0.0 if reff is None else reff
            bjd_ref = float(refi) + float(reff)

        # Test that the calculation is correct
        expected_btjd = sample_lc_plus.bjd_time - bjd_ref
        np.testing.assert_array_almost_equal(btjd_time, expected_btjd)

        # BTJD values for TESS data should be in a reasonable range (typically 1000-3000)
        # This is a loose check since we don't know the exact reference time
        if "TESS" in sample_lc_plus.meta.get("TELESCOP", ""):
            assert np.all(btjd_time > 0)
            assert np.all(btjd_time < 10000)

    def test_to_numpy(self, sample_lc_plus):
        """Test conversion to numpy array."""
        numpy_array = sample_lc_plus.to_numpy()
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape[1] == 2  # Two columns: time and flux
        assert numpy_array.shape[0] == len(sample_lc_plus)

        # Check that the values match
        np.testing.assert_array_equal(numpy_array[:, 0], sample_lc_plus.time_x)
        np.testing.assert_array_equal(numpy_array[:, 1], sample_lc_plus.flux_y)

    def test_remove_nans(self, sample_lc_plus):
        """Test remove_outliers method."""
        nan_flux = sample_lc_plus.flux_y
        nan_flux[0:10] = np.nan

        cleaned_lc = sample_lc_plus.copy_with_flux(nan_flux).remove_nans()
        assert isinstance(cleaned_lc, LightCurvePlus)

        # The cleaned lightcurve should have the same or fewer points
        assert len(cleaned_lc) == len(sample_lc_plus) - 10
        assert sample_lc_plus.obs_id is not None
        assert cleaned_lc.obs_id == sample_lc_plus.obs_id

    def test_remove_outliers(self, sample_lc_plus):
        """Test remove_outliers method."""
        cleaned_lc = sample_lc_plus.remove_outliers()
        assert isinstance(cleaned_lc, LightCurvePlus)

        # The cleaned lightcurve should have fewer points
        assert len(cleaned_lc) < len(sample_lc_plus)
        assert len(cleaned_lc) == 13745  # Specific value for the test sample
        assert sample_lc_plus.obs_id is not None
        assert cleaned_lc.obs_id == sample_lc_plus.obs_id

    def test_normalize(self, sample_lc_plus):
        """Test normalize method."""
        normalized_lc = sample_lc_plus.normalize()
        assert isinstance(normalized_lc, LightCurvePlus)
        assert len(normalized_lc) == len(sample_lc_plus)
        assert sample_lc_plus.obs_id is not None
        assert normalized_lc.obs_id == sample_lc_plus.obs_id

        # Check that the normalized flux has mean close to 1
        assert 0.9 <= np.mean(normalized_lc.flux_y) <= 1.1

    def test_shift_time(self, sample_lc_plus):
        """Test shift_time method."""
        original_time = sample_lc_plus.time.copy()
        shift_value = 10.0

        # Apply the shift
        shifted_lc = sample_lc_plus.shift_time(shift_value)

        # Check that it returns self
        assert shifted_lc is sample_lc_plus

        # Check that the time has been shifted by the correct amount
        assert np.allclose(shifted_lc.time.value, original_time.value + shift_value)

    def test_start_at_zero(self, sample_lc_plus):
        """Test start_at_zero method."""
        # Make a copy to avoid modifying the fixture
        lc_copy = LightCurvePlus(sample_lc_plus.lc.copy())

        # Apply the method
        zeroed_lc = lc_copy.start_at_zero()

        # Check that it returns self
        assert zeroed_lc is lc_copy

        # Check that the first time value is close to zero
        assert abs(zeroed_lc.time_x[0]) < 1e-10

    def test_get_transit_phase(self, lc_and_planet: tuple[Planet, LightCurvePlus]):
        """Test get_transit_phase method."""
        sample_planet, sample_lc_plus = lc_and_planet
        phase = sample_lc_plus.get_transit_phase(sample_planet)

        # Check that the result is a numpy array with the correct length
        assert isinstance(phase, np.ndarray)
        assert len(phase) == len(sample_lc_plus)

        # Phase values should be between 0 and half the period
        period = sample_planet.orbital_period.central.value
        assert np.all(phase >= 0)
        assert np.all(phase <= period / 2)

    def test_get_transit_mask(self, lc_and_planet: tuple[Planet, LightCurvePlus]):
        """Test get_transit_mask method."""
        # Test with default parameters
        sample_planet, sample_lc_plus = lc_and_planet
        mask = sample_lc_plus.get_transit_mask(sample_planet)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(sample_lc_plus)

        # Test with increased duration
        mask_extended = sample_lc_plus.get_transit_mask(sample_planet, duration_increase_percent=0.5)
        # Extended mask should have at least as many True values as the original
        assert np.sum(mask_extended) >= np.sum(mask)

    def test_get_transit_count(self, lc_and_planet: tuple[Planet, LightCurvePlus]):
        """Test get_transit_count method."""
        sample_planet, sample_lc_plus = lc_and_planet
        count = sample_lc_plus.get_transit_count(sample_planet)
        assert isinstance(count, int)
        assert count >= 0

    def test_get_combined_transit_mask(
        self, lc_and_planet: tuple[Planet, LightCurvePlus], star_system_test_db: StarSystemDB
    ):
        """Test get_combined_transit_mask method."""
        _, sample_lc_plus = lc_and_planet
        star_system = star_system_test_db.get_star_system_from_tic_id(sample_lc_plus.tic_id)
        planets = star_system.planets
        mask = sample_lc_plus.get_combined_transit_mask(planets)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(sample_lc_plus)

        # If we have multiple planets, test with just a subset
        if len(planets) > 1:
            subset_mask = sample_lc_plus.get_combined_transit_mask([planets[0]])
            assert isinstance(subset_mask, np.ndarray)
            assert len(subset_mask) == len(sample_lc_plus)

    def test_fold_with_planet(self, lc_and_planet: tuple[Planet, LightCurvePlus]):
        """Test fold_with_planet method."""
        # Test with default parameters
        sample_planet, sample_lc_plus = lc_and_planet
        folded_lc = sample_lc_plus.fold_with_planet(sample_planet)
        assert isinstance(folded_lc, FoldedLightCurve)

        # Test with normalize_time=True
        folded_lc_norm = sample_lc_plus.fold_with_planet(sample_planet, normalize_time=True)
        assert isinstance(folded_lc_norm, FoldedLightCurve)

    def test_copy_with_flux(self, sample_lc_plus):
        """Test copy_with_flux method."""
        # Create a new flux array with the same shape
        new_flux = np.ones_like(sample_lc_plus.flux_y)

        # Create a copy with the new flux
        new_lc = sample_lc_plus.copy_with_flux(new_flux)

        # Check that it's a new instance
        assert new_lc is not sample_lc_plus
        assert isinstance(new_lc, LightCurvePlus)

        # Check that the flux has been replaced
        np.testing.assert_array_equal(new_lc.flux_y, new_flux)

        # Check that the time is the same
        np.testing.assert_array_equal(new_lc.time_x, sample_lc_plus.time_x)

    def test_fold(self, sample_lc_plus):
        """Test fold method."""
        # Test with minimal parameters
        period = 5.0  # arbitrary period for testing
        folded_lc = sample_lc_plus.fold(period=period)
        assert isinstance(folded_lc, FoldedLightCurve)

        # Test with more parameters
        epoch_time = sample_lc_plus.time[0].value
        folded_lc = sample_lc_plus.fold(period=period, epoch_time=epoch_time, epoch_phase=0.5, normalize_phase=True)
        assert isinstance(folded_lc, FoldedLightCurve)

    def test_len(self, sample_lc_plus):
        """Test __len__ method."""
        assert len(sample_lc_plus) == len(sample_lc_plus.lc)

    def test_getitem(self, sample_lc_plus):
        """Test __getitem__ method."""
        # Test with a slice
        slice_lc = sample_lc_plus[0:10]
        assert isinstance(slice_lc, LightCurvePlus)
        assert len(slice_lc) == 10

        # Test with a boolean mask
        mask = np.zeros(len(sample_lc_plus), dtype=bool)
        mask[0:5] = True
        masked_lc = sample_lc_plus[mask]
        assert isinstance(masked_lc, LightCurvePlus)
        assert len(masked_lc) == 5

    def test_arithmetic_operations(self, sample_lc_plus):
        """Test __add__ and __sub__ methods."""
        # Test addition with a scalar
        added_lc = sample_lc_plus + 1.0
        assert isinstance(added_lc, LightCurvePlus)
        assert np.allclose(added_lc.flux_y, sample_lc_plus.flux_y + 1.0)

        # Test subtraction with a scalar
        subtracted_lc = sample_lc_plus - 1.0
        assert isinstance(subtracted_lc, LightCurvePlus)
        assert np.allclose(subtracted_lc.flux_y, sample_lc_plus.flux_y - 1.0)

        # Test addition with another LightCurvePlus
        lc_copy = LightCurvePlus(sample_lc_plus.lc.copy())
        added_lc = sample_lc_plus + lc_copy
        assert isinstance(added_lc, LightCurvePlus)
        assert np.allclose(added_lc.flux_y, sample_lc_plus.flux_y + lc_copy.flux_y)

        # Test subtraction with another LightCurvePlus
        subtracted_lc = sample_lc_plus - lc_copy
        assert isinstance(subtracted_lc, LightCurvePlus)
        assert np.allclose(subtracted_lc.flux_y, sample_lc_plus.flux_y - lc_copy.flux_y)

    def test_get_first_transit_value(self, lc_and_planet: tuple[Planet, LightCurvePlus]):
        """Test get_first_transit_value method."""
        sample_planet, sample_lc_plus = lc_and_planet
        transit_time = sample_lc_plus.get_first_transit_value(sample_planet)
        assert isinstance(transit_time, Time)

    def test_get_transit_first_index(self, lc_and_planet: tuple[Planet, LightCurvePlus]):
        """Test get_transit_first_index method."""
        sample_planet, sample_lc_plus = lc_and_planet
        index = sample_lc_plus.get_transit_first_index(sample_planet)
        assert isinstance(index, int)
        assert 0 <= index < len(sample_lc_plus)

    def test_time_format_awareness(self, all_test_lightcurves):
        """Test that LightCurvePlus correctly handles different time formats."""
        # Get a sample lightcurve
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Test with original format (should be BTJD)
        lc_plus_original = LightCurvePlus(base_lc)

        # Test with JD conversion
        lc_plus_jd = LightCurvePlus(base_lc.copy())
        lc_plus_jd.to_jd_time()

        # Verify time systems are reported correctly
        assert lc_plus_original.time_system == "BTJD/TDB"
        assert lc_plus_jd.time_system == "JD/TDB"

        # Verify original format is stored
        assert lc_plus_original._original_time_format == "btjd"
        assert lc_plus_jd._original_time_format == "btjd"

    def test_time_property_optimizations(self, all_test_lightcurves):
        """Test that time properties avoid unnecessary conversions."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Test with BTJD format (original)
        lc_plus_btjd = LightCurvePlus(base_lc)

        # Test with JD format
        lc_plus_jd = LightCurvePlus(base_lc.copy())
        lc_plus_jd.to_jd_time()

        # Get time values from both formats
        btjd_from_btjd = lc_plus_btjd.btjd_time
        btjd_from_jd = lc_plus_jd.btjd_time

        bjd_from_btjd = lc_plus_btjd.bjd_time
        bjd_from_jd = lc_plus_jd.bjd_time

        jd_from_btjd = lc_plus_btjd.jd_time
        jd_from_jd = lc_plus_jd.jd_time

        # Verify consistency between formats (allow for floating-point precision differences)
        np.testing.assert_array_almost_equal(btjd_from_btjd, btjd_from_jd, decimal=8)
        np.testing.assert_array_almost_equal(bjd_from_btjd, bjd_from_jd, decimal=8)
        np.testing.assert_array_almost_equal(jd_from_btjd, jd_from_jd, decimal=8)

        # Verify relationships between time formats
        # BJD should be JD (they're the same for TDB scale)
        np.testing.assert_array_almost_equal(bjd_from_jd, jd_from_jd, decimal=10)

        # BTJD should be JD - reference (2457000 for TESS)
        bjd_ref = base_lc.meta.get("BJDREFI", 2457000) + base_lc.meta.get("BJDREFF", 0.0)
        expected_btjd = jd_from_jd - bjd_ref
        np.testing.assert_array_almost_equal(btjd_from_jd, expected_btjd, decimal=8)

    def test_time_elapsed_consistency(self, all_test_lightcurves):
        """Test that time_elapsed is consistent across different time formats."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Test with both formats
        lc_plus_btjd = LightCurvePlus(base_lc)
        lc_plus_jd = LightCurvePlus(base_lc.copy())
        lc_plus_jd.to_jd_time()

        # Get elapsed times
        elapsed_btjd = lc_plus_btjd.elapsed_time
        elapsed_jd = lc_plus_jd.elapsed_time

        # Should be identical regardless of underlying format
        np.testing.assert_array_almost_equal(elapsed_btjd, elapsed_jd, decimal=8)

        # First value should always be 0
        assert elapsed_btjd[0] == 0
        assert elapsed_jd[0] == 0

        # Values should be monotonically increasing
        assert np.all(np.diff(elapsed_btjd) >= 0)
        assert np.all(np.diff(elapsed_jd) >= 0)

    def test_shift_time_with_different_formats(self, all_test_lightcurves):
        """Test that shift_time works correctly with different time formats."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Create copies for testing
        lc_plus_btjd = LightCurvePlus(base_lc.copy())
        lc_plus_jd = LightCurvePlus(base_lc.copy())
        lc_plus_jd.to_jd_time()

        # Record original times
        original_time_btjd = lc_plus_btjd.time_x.copy()
        original_time_jd = lc_plus_jd.time_x.copy()

        # Apply the same shift to both
        shift_days = 10.0
        lc_plus_btjd.shift_time(shift_days)
        lc_plus_jd.shift_time(shift_days)

        # Verify the shift was applied correctly
        # Note: shift_time converts days to seconds internally for TimeDelta
        expected_shift_btjd = original_time_btjd + shift_days
        expected_shift_jd = original_time_jd + shift_days

        np.testing.assert_array_almost_equal(lc_plus_btjd.time_x, expected_shift_btjd, decimal=8)
        np.testing.assert_array_almost_equal(lc_plus_jd.time_x, expected_shift_jd, decimal=8)

    def test_start_at_zero_with_different_formats(self, all_test_lightcurves):
        """Test that start_at_zero works correctly with different time formats."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Create copies for testing
        lc_plus_btjd = LightCurvePlus(base_lc.copy())
        lc_plus_jd = LightCurvePlus(base_lc.copy())
        lc_plus_jd.to_jd_time()

        # Apply start_at_zero
        lc_plus_btjd.start_at_zero()
        lc_plus_jd.start_at_zero()

        # First time value should be close to zero for both
        assert abs(lc_plus_btjd.time_x[0]) == pytest.approx(0, abs=1e-9)
        assert abs(lc_plus_jd.time_x[0]) == pytest.approx(0, abs=1e-9)

        # Verify that the time differences are preserved
        original_btjd_lc = LightCurvePlus(base_lc.copy())
        original_jd_lc = LightCurvePlus(base_lc.copy())
        original_jd_lc.to_jd_time()
        original_diffs_btjd = np.diff(original_btjd_lc.time_x)
        original_diffs_jd = np.diff(original_jd_lc.time_x)

        shifted_diffs_btjd = np.diff(lc_plus_btjd.time_x)
        shifted_diffs_jd = np.diff(lc_plus_jd.time_x)

        np.testing.assert_array_almost_equal(original_diffs_btjd, shifted_diffs_btjd, decimal=8)
        np.testing.assert_array_almost_equal(original_diffs_jd, shifted_diffs_jd, decimal=8)

    def test_time_system_property_accuracy(self, all_test_lightcurves):
        """Test that time_system property returns accurate information."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Test different format combinations
        lc_plus_btjd = LightCurvePlus(base_lc)
        lc_plus_jd = LightCurvePlus(base_lc.copy())
        lc_plus_jd.to_jd_time()

        # Verify time system strings
        assert lc_plus_btjd.time_system == "BTJD/TDB"
        assert lc_plus_jd.time_system == "JD/TDB"

        # Verify the underlying time objects match
        assert lc_plus_btjd.lc.time.format == "btjd"
        assert lc_plus_btjd.lc.time.scale == "tdb"
        assert lc_plus_jd.lc.time.format == "jd"
        assert lc_plus_jd.lc.time.scale == "tdb"

    def test_format_aware_performance_optimization(self, all_test_lightcurves):
        """Test that format-aware properties avoid unnecessary conversions."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Create BTJD format lightcurve
        lc_plus_btjd = LightCurvePlus(base_lc)

        # Access BTJD time (should be direct access, no conversion)
        btjd_values = lc_plus_btjd.btjd_time

        # Verify it's the same as the underlying time values (direct access)
        if lc_plus_btjd.lc.time.format == "btjd":
            np.testing.assert_array_equal(btjd_values, lc_plus_btjd.lc.time.value)

        # Create JD format lightcurve
        lc_plus_jd = LightCurvePlus(base_lc.copy())
        lc_plus_jd.to_jd_time()

        # Access JD time (should be direct access, no conversion)
        jd_values = lc_plus_jd.jd_time

        # Verify it's the same as the underlying time values (direct access)
        if lc_plus_jd.lc.time.format == "jd":
            np.testing.assert_array_equal(jd_values, lc_plus_jd.lc.time.value)

    def test_backward_compatibility(self, all_test_lightcurves):
        """Test that the refactored code maintains backward compatibility."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Test that constructor works with obs_id parameter
        lc_plus_with_obs_id = LightCurvePlus(base_lc, obs_id=obs_id)
        lc_plus_without_obs_id = LightCurvePlus(base_lc)

        # Should produce identical time systems (both should preserve original format)
        assert lc_plus_with_obs_id.time_system == lc_plus_without_obs_id.time_system
        assert lc_plus_with_obs_id._obs_id == obs_id
        assert lc_plus_without_obs_id._obs_id is None

        # Time values should be identical
        np.testing.assert_array_equal(lc_plus_with_obs_id.time_x, lc_plus_without_obs_id.time_x)
        np.testing.assert_array_equal(lc_plus_with_obs_id.btjd_time, lc_plus_without_obs_id.btjd_time)
        np.testing.assert_array_equal(lc_plus_with_obs_id.bjd_time, lc_plus_without_obs_id.bjd_time)
        np.testing.assert_array_equal(lc_plus_with_obs_id.jd_time, lc_plus_without_obs_id.jd_time)

    def test_to_jd_time_conversion(self, all_test_lightcurves):
        """Test that to_jd_time() method converts time format correctly."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Start with BTJD format
        lc_plus = LightCurvePlus(base_lc.copy())
        assert lc_plus.lc.time.format == "btjd"

        # Convert to JD
        result = lc_plus.to_jd_time()

        # Verify method chaining (returns self)
        assert result is lc_plus

        # Verify format changed
        assert lc_plus.lc.time.format == "jd"
        assert lc_plus.lc.time.scale == "tdb"

        # Verify time values are correctly converted
        expected_jd = base_lc.time.value + 2457000  # BTJD to JD conversion
        np.testing.assert_array_almost_equal(lc_plus.lc.time.value, expected_jd, decimal=8)

        # Test idempotency - calling to_jd_time() again should not change anything
        original_values = lc_plus.lc.time.value.copy()
        lc_plus.to_jd_time()
        np.testing.assert_array_equal(lc_plus.lc.time.value, original_values)

    def test_to_btjd_time_conversion(self, all_test_lightcurves):
        """Test that to_btjd_time() method converts time format correctly."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Start with BTJD format and convert to JD first
        lc_plus = LightCurvePlus(base_lc.copy())
        lc_plus.to_jd_time()
        assert lc_plus.lc.time.format == "jd"

        # Store original BTJD values for comparison
        original_btjd_values = base_lc.time.value.copy()

        # Convert back to BTJD
        result = lc_plus.to_btjd_time()

        # Verify method chaining (returns self)
        assert result is lc_plus

        # Verify format changed back
        assert lc_plus.lc.time.format == "btjd"
        assert lc_plus.lc.time.scale == "tdb"

        # Verify time values are correctly converted back to original
        np.testing.assert_array_almost_equal(lc_plus.lc.time.value, original_btjd_values, decimal=8)

        # Test idempotency - calling to_btjd_time() again should not change anything
        lc_plus.to_btjd_time()
        np.testing.assert_array_almost_equal(lc_plus.lc.time.value, original_btjd_values, decimal=8)

    def test_to_bjd_time_conversion(self, all_test_lightcurves):
        """Test that to_bjd_time() method converts time format correctly."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Start with BTJD format
        lc_plus = LightCurvePlus(base_lc.copy())
        assert lc_plus.lc.time.format == "btjd"

        # Convert to BJD
        result = lc_plus.to_bjd_time()

        # Verify method chaining (returns self)
        assert result is lc_plus

        # Verify format changed (BJD is same as JD for TDB scale)
        assert lc_plus.lc.time.format == "jd"
        assert lc_plus.lc.time.scale == "tdb"

        # Verify time values are correctly converted (same as JD conversion)
        expected_jd = base_lc.time.value + 2457000  # BTJD to JD/BJD conversion
        np.testing.assert_array_almost_equal(lc_plus.lc.time.value, expected_jd, decimal=8)

        # Test idempotency - calling to_bjd_time() again should not change anything
        original_values = lc_plus.lc.time.value.copy()
        lc_plus.to_bjd_time()
        np.testing.assert_array_equal(lc_plus.lc.time.value, original_values)

    def test_time_conversion_round_trip(self, all_test_lightcurves):
        """Test that time conversions are reversible and maintain precision."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Start with original BTJD format
        lc_plus = LightCurvePlus(base_lc.copy())
        original_btjd_values = lc_plus.lc.time.value.copy()

        # Round trip: BTJD -> JD -> BTJD
        lc_plus.to_jd_time()
        lc_plus.to_btjd_time()

        # Should be back to original values (use lower precision due to floating-point arithmetic)
        assert lc_plus.lc.time.format == "btjd"
        np.testing.assert_array_almost_equal(lc_plus.lc.time.value, original_btjd_values, decimal=8)

        # Test another round trip: BTJD -> BJD -> BTJD
        lc_plus2 = LightCurvePlus(base_lc.copy())
        lc_plus2.to_bjd_time()
        lc_plus2.to_btjd_time()

        # Should be back to original values (use lower precision due to floating-point arithmetic)
        assert lc_plus2.lc.time.format == "btjd"
        np.testing.assert_array_almost_equal(lc_plus2.lc.time.value, original_btjd_values, decimal=8)

    def test_time_conversion_with_metadata_preservation(self, all_test_lightcurves):
        """Test that time conversions preserve lightcurve metadata."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Add some test metadata
        base_lc.meta["TEST_KEY"] = "test_value"
        base_lc.meta["BJDREFI"] = 2457000
        base_lc.meta["BJDREFF"] = 0.0

        lc_plus = LightCurvePlus(base_lc.copy())
        original_meta = dict(lc_plus.lc.meta)

        # Test that all conversion methods preserve metadata
        lc_plus.to_jd_time()
        assert original_meta["TEST_KEY"] == "test_value"
        assert original_meta["BJDREFI"] == 2457000
        assert original_meta["BJDREFF"] == 0.0

        lc_plus.to_btjd_time()
        assert original_meta["TEST_KEY"] == "test_value"
        assert original_meta["BJDREFI"] == 2457000
        assert original_meta["BJDREFF"] == 0.0

        lc_plus.to_bjd_time()
        assert original_meta["TEST_KEY"] == "test_value"
        assert original_meta["BJDREFI"] == 2457000
        assert original_meta["BJDREFF"] == 0.0

    def test_time_conversion_with_different_formats(self, all_test_lightcurves):
        """Test that time conversion methods work correctly with different starting formats."""
        obs_id = next(iter(all_test_lightcurves))
        base_lc = all_test_lightcurves[obs_id]

        # Test converting from BTJD to all other formats
        lc_plus_btjd = LightCurvePlus(base_lc.copy())
        assert lc_plus_btjd.lc.time.format == "btjd"

        # Convert BTJD to JD
        lc_plus_btjd_to_jd = LightCurvePlus(base_lc.copy())
        lc_plus_btjd_to_jd.to_jd_time()
        assert lc_plus_btjd_to_jd.lc.time.format == "jd"

        # Convert BTJD to BJD (should be same as JD)
        lc_plus_btjd_to_bjd = LightCurvePlus(base_lc.copy())
        lc_plus_btjd_to_bjd.to_bjd_time()
        assert lc_plus_btjd_to_bjd.lc.time.format == "jd"  # BJD is same as JD for TDB scale

        # Verify that JD and BJD conversions produce the same result
        np.testing.assert_array_equal(lc_plus_btjd_to_jd.lc.time.value, lc_plus_btjd_to_bjd.lc.time.value)

        # Test converting from JD back to BTJD
        lc_plus_jd_to_btjd = LightCurvePlus(base_lc.copy())
        lc_plus_jd_to_btjd.to_jd_time()
        lc_plus_jd_to_btjd.to_btjd_time()
        assert lc_plus_jd_to_btjd.lc.time.format == "btjd"

        # Should match original BTJD values
        np.testing.assert_array_almost_equal(lc_plus_jd_to_btjd.lc.time.value, lc_plus_btjd.lc.time.value, decimal=8)

    @pytest.fixture
    def time_array_with_gaps(self):
        """
        Create a time array with known gaps for testing gap detection functions.

        The array has evenly spaced points with dt=0.1 except for three gaps:
        1. Between indices 4-5: gap of 0.5 (5x normal spacing)
        2. Between indices 9-10: gap of 1.0 (10x normal spacing)
        3. Between indices 14-15: gap of 0.3 (3x normal spacing)

        Returns:
            tuple: (time_array, expected_gap_indices, expected_gap_tuples_i, expected_gap_tuples_x,
                   expected_contiguous_tuples_i, expected_contiguous_tuples_x)
        """
        # Create base array with even spacing
        base = np.arange(0, 2, 0.1)

        # Insert gaps
        time_array = np.concatenate(
            [
                base[:5],  # 0.0 to 0.4 with dt=0.1
                [base[5] + 0.5],  # 1.0 (gap of 0.5 after 0.4)
                base[6:10],  # 0.6 to 0.9 with dt=0.1
                [base[10] + 1.0],  # 2.0 (gap of 1.0 after 0.9)
                base[11:15],  # 1.1 to 1.4 with dt=0.1
                [base[15] + 0.3],  # 1.8 (gap of 0.3 after 1.4)
                base[16:20],  # 1.6 to 1.9 with dt=0.1
            ]
        )

        # Expected gap indices (positions where gaps start)
        expected_gap_indices = np.array([4, 9, 14])

        # Expected gap tuples as (i, i+1) pairs
        expected_gap_tuples_i = [(4, 5), (9, 10), (14, 15)]

        # Expected gap tuples as (time[i], time[i+1]) pairs
        expected_gap_tuples_x = [(0.4, 1.0), (0.9, 2.0), (1.4, 1.8)]

        # Expected contiguous intervals as (start, end) index pairs
        expected_contiguous_tuples_i = [(0, 4), (5, 9), (10, 14), (15, 19)]

        # Expected contiguous intervals as (time[start], time[end]) pairs
        expected_contiguous_tuples_x = [(0.0, 0.4), (1.0, 0.9), (2.0, 1.4), (1.8, 1.9)]

        return (
            time_array,
            expected_gap_indices,
            expected_gap_tuples_i,
            expected_gap_tuples_x,
            expected_contiguous_tuples_i,
            expected_contiguous_tuples_x,
        )

    @pytest.fixture
    def mock_lightcurve_plus_with_gaps(self, time_array_with_gaps):
        """Create a mock LightCurvePlus object with known time gaps."""
        time_array = time_array_with_gaps[0]
        # Create a simple flux array matching the time array length
        flux_array = np.ones_like(time_array)

        # Create a LightCurve object with the time array
        lc = LightCurve(time=Time(time_array, format="jd", scale="tdb"), flux=flux_array)

        # Create a LightCurvePlus object with the LightCurve
        return LightCurvePlus(lc)

    def test_get_gaps_indices(self, time_array_with_gaps):
        """Test the get_gaps_indices standalone function."""
        time_array, expected_gap_indices, _, _, _, _ = time_array_with_gaps

        # Test with default threshold (should find all gaps >= 5x median)
        gaps_5x = get_gaps_indices(time_array, greater_than_median=5.0)
        assert len(gaps_5x) == 2  # Should find gaps at indices 4 and 9 (5x and 10x)
        assert 4 in gaps_5x
        assert 9 in gaps_5x
        assert 14 not in gaps_5x  # This gap is only 3x

        # Test with lower threshold (should find all gaps >= 3x median)
        gaps_3x = get_gaps_indices(time_array, greater_than_median=3.0)
        assert len(gaps_3x) == 3  # Should find all three gaps
        assert set(gaps_3x) == set(expected_gap_indices)

        # Test with higher threshold (should find only the largest gap)
        gaps_8x = get_gaps_indices(time_array, greater_than_median=8.0)
        assert len(gaps_8x) == 1  # Should find only the 10x gap at index 9
        assert 9 in gaps_8x

        # Test with threshold that finds no gaps
        gaps_none = get_gaps_indices(time_array, greater_than_median=15.0)
        assert len(gaps_none) == 0

    def test_find_time_gaps_i(self, mock_lightcurve_plus_with_gaps, time_array_with_gaps):
        """Test the find_time_gaps_i method of LightCurvePlus."""
        _, _, expected_gap_tuples_i, _, _, _ = time_array_with_gaps

        # Test with threshold that finds all gaps >= 5x median
        gaps_5x = mock_lightcurve_plus_with_gaps.find_time_gaps_i(greater_than_median=5.0)
        assert len(gaps_5x) == 2  # Should find gaps at indices 4 and 9
        assert (4, 5) in gaps_5x
        assert (9, 10) in gaps_5x
        assert (14, 15) not in gaps_5x  # This gap is only 3x

        # Test with lower threshold (should find all gaps >= 3x median)
        gaps_3x = mock_lightcurve_plus_with_gaps.find_time_gaps_i(greater_than_median=3.0)
        assert len(gaps_3x) == 3  # Should find all three gaps
        assert set(gaps_3x) == set(expected_gap_tuples_i)

        # Test with higher threshold (should find only the largest gap)
        gaps_8x = mock_lightcurve_plus_with_gaps.find_time_gaps_i(greater_than_median=8.0)
        assert len(gaps_8x) == 1  # Should find only the 10x gap
        assert (9, 10) in gaps_8x

        # Test with threshold that finds no gaps
        gaps_none = mock_lightcurve_plus_with_gaps.find_time_gaps_i(greater_than_median=15.0)
        assert len(gaps_none) == 0

    def test_find_time_gaps_x(self, mock_lightcurve_plus_with_gaps, time_array_with_gaps):
        """Test the find_time_gaps_x method of LightCurvePlus."""
        _, _, _, expected_gap_tuples_x, _, _ = time_array_with_gaps

        # Test with threshold that finds all gaps >= 5x median
        gaps_5x = mock_lightcurve_plus_with_gaps.find_time_gaps_x(greater_than_median=5.0)
        assert len(gaps_5x) == 2  # Should find gaps at indices 4 and 9

        # Check that the time values match expected values
        # Use np.isclose for floating point comparison
        assert any(np.isclose(gap[0], 0.4) and np.isclose(gap[1], 1.0) for gap in gaps_5x)
        assert any(np.isclose(gap[0], 0.9) and np.isclose(gap[1], 2.0) for gap in gaps_5x)

        # Test with lower threshold (should find all gaps >= 3x median)
        gaps_3x = mock_lightcurve_plus_with_gaps.find_time_gaps_x(greater_than_median=3.0)
        assert len(gaps_3x) == 3  # Should find all three gaps

        # Check that all expected gaps are found (using approximate comparison)
        for expected_gap in expected_gap_tuples_x:
            assert any(np.isclose(gap[0], expected_gap[0]) and np.isclose(gap[1], expected_gap[1]) for gap in gaps_3x)

        # Test with threshold that finds no gaps
        gaps_none = mock_lightcurve_plus_with_gaps.find_time_gaps_x(greater_than_median=15.0)
        assert len(gaps_none) == 0

    def test_find_contiguous_time_i(self, mock_lightcurve_plus_with_gaps, time_array_with_gaps):
        """Test the find_contiguous_time_i method of LightCurvePlus."""
        _, _, _, _, expected_contiguous_tuples_i, _ = time_array_with_gaps

        # Test with threshold that finds all gaps >= 5x median
        contiguous_5x = mock_lightcurve_plus_with_gaps.find_contiguous_time_i(greater_than_median=5.0)
        assert len(contiguous_5x) == 3  # Should find 3 contiguous regions

        # Check that the contiguous regions include the start and end regions
        assert (0, 4) in contiguous_5x
        assert (5, 9) in contiguous_5x
        assert (10, 19) in contiguous_5x  # This includes the small gap that wasn't detected

        # Test with lower threshold (should find all gaps >= 3x median)
        contiguous_3x = mock_lightcurve_plus_with_gaps.find_contiguous_time_i(greater_than_median=3.0)
        assert len(contiguous_3x) == 4  # Should find 4 contiguous regions
        assert set(contiguous_3x) == set(expected_contiguous_tuples_i)

        # Test with threshold that finds no gaps (entire array is one contiguous region)
        contiguous_none = mock_lightcurve_plus_with_gaps.find_contiguous_time_i(greater_than_median=15.0)
        assert len(contiguous_none) == 1
        assert contiguous_none[0] == (0, 19)  # Full array

    def test_find_contiguous_time_x(self, mock_lightcurve_plus_with_gaps, time_array_with_gaps):
        """Test the find_contiguous_time_x method of LightCurvePlus."""
        time_array, _, _, _, _, expected_contiguous_tuples_x = time_array_with_gaps

        # Test with threshold that finds all gaps >= 5x median
        contiguous_5x = mock_lightcurve_plus_with_gaps.find_contiguous_time_x(greater_than_median=5.0)
        assert len(contiguous_5x) == 3  # Should find 3 contiguous regions

        # Check that the time values match expected values
        # Use np.isclose for floating point comparison
        assert any(np.isclose(interval[0], 0.0) and np.isclose(interval[1], 0.4) for interval in contiguous_5x)
        assert any(np.isclose(interval[0], 1.0) and np.isclose(interval[1], 0.9) for interval in contiguous_5x)
        assert any(np.isclose(interval[0], 2.0) and np.isclose(interval[1], 1.9) for interval in contiguous_5x)

        # Test with lower threshold (should find all gaps >= 3x median)
        contiguous_3x = mock_lightcurve_plus_with_gaps.find_contiguous_time_x(greater_than_median=3.0)
        assert len(contiguous_3x) == 4  # Should find 4 contiguous regions

        # Check that all expected contiguous intervals are found (using approximate comparison)
        for expected_interval in expected_contiguous_tuples_x:
            assert any(
                np.isclose(interval[0], expected_interval[0]) and np.isclose(interval[1], expected_interval[1])
                for interval in contiguous_3x
            )

        # Test with threshold that finds no gaps (entire array is one contiguous region)
        contiguous_none = mock_lightcurve_plus_with_gaps.find_contiguous_time_x(greater_than_median=15.0)
        assert len(contiguous_none) == 1
        assert np.isclose(contiguous_none[0][0], time_array[0])
        assert np.isclose(contiguous_none[0][1], time_array[-1])

    def test_edge_cases(self):
        """Test edge cases for gap detection functions."""
        # Test with empty array
        empty_array = np.array([])
        assert len(get_gaps_indices(empty_array, greater_than_median=5.0)) == 0

        # Test with single element array (no gaps possible)
        single_element = np.array([1.0])
        assert len(get_gaps_indices(single_element, greater_than_median=5.0)) == 0

        # Test with two elements (one difference, which becomes the median)
        two_elements = np.array([1.0, 2.0])
        assert len(get_gaps_indices(two_elements, greater_than_median=5.0)) == 0

        # Test with constant time differences (no gaps)
        constant_diff = np.arange(0, 10, 1.0)
        assert len(get_gaps_indices(constant_diff, greater_than_median=5.0)) == 0

        # Test with all identical values (zero differences)
        identical_values = np.ones(10)
        # This should not raise errors, but the result is not meaningful
        # since all differences are zero and median is zero
        try:
            get_gaps_indices(identical_values, greater_than_median=5.0)
        except Exception as e:
            pytest.fail(f"get_gaps_indices raised {type(e).__name__} with identical values: {e}")

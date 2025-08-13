import numpy as np
import pytest
from astropy.time import Time
from lightkurve import FoldedLightCurve

from exotools import LightcurveDB, LightCurvePlus, Planet, StarSystemDB


class TestLightcurvePlus:
    @pytest.fixture
    def tic_with_planets(self) -> list[int]:
        return [159781361, 179317684, 441765914, 158635959, 328081248]

    @pytest.fixture
    def sample_lc_plus(self, all_test_lightcurves):
        """Get a sample LightCurvePlus object with a valid observation ID."""
        obs_id = next(iter(all_test_lightcurves))
        return LightCurvePlus(all_test_lightcurves[obs_id], obs_id=obs_id)

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
        bjd_time = sample_lc_plus.time_bjd
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
        elapsed_time = sample_lc_plus.time_elapsed
        assert isinstance(elapsed_time, np.ndarray)
        assert len(elapsed_time) == len(sample_lc_plus)

        # First value should always be 0 (days since first cadence)
        assert elapsed_time[0] == 0

        # All values should be non-negative and monotonically increasing
        assert np.all(elapsed_time >= 0)
        assert np.all(np.diff(elapsed_time) >= 0)

        # Verify the calculation is correct
        bjd_time = sample_lc_plus.time_bjd
        expected_elapsed = bjd_time - bjd_time[0]
        np.testing.assert_array_equal(elapsed_time, expected_elapsed)

    def test_time_btjd(self, sample_lc_plus):
        """Test the time_btjd property."""
        btjd_time = sample_lc_plus.time_btjd
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
        expected_btjd = sample_lc_plus.time_bjd - bjd_ref
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

    def test_remove_outliers(self, sample_lc_plus):
        """Test remove_outliers method."""
        cleaned_lc = sample_lc_plus.remove_outliers()
        assert isinstance(cleaned_lc, LightCurvePlus)
        # The cleaned lightcurve should have the same or fewer points
        assert len(cleaned_lc) <= len(sample_lc_plus)

    def test_normalize(self, sample_lc_plus):
        """Test normalize method."""
        normalized_lc = sample_lc_plus.normalize()
        assert isinstance(normalized_lc, LightCurvePlus)
        assert len(normalized_lc) == len(sample_lc_plus)

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

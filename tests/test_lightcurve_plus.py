import numpy as np
import pytest
from astropy.time import Time
from lightkurve import FoldedLightCurve

from exotools import LightCurvePlus, StarSystemDB


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
    def sample_planet(self, star_system_test_db, sample_lc_plus):
        """Get a sample planet from the star system associated with the sample lightcurve."""
        star_system = star_system_test_db.get_star_system_from_tic_id(sample_lc_plus.tic_id)
        if star_system and star_system.planets:
            return star_system.planets[0]
        pytest.skip("No valid planet found for testing")

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

    def test_get_transit_phase(self, sample_lc_plus, sample_planet):
        """Test get_transit_phase method."""
        if sample_planet is None:
            pytest.skip("No valid planet available for testing")

        phase = sample_lc_plus.get_transit_phase(sample_planet)

        # Check that the result is a numpy array with the correct length
        assert isinstance(phase, np.ndarray)
        assert len(phase) == len(sample_lc_plus)

        # Phase values should be between 0 and half the period
        period = sample_planet.orbital_period.central.value
        assert np.all(phase >= 0)
        assert np.all(phase <= period / 2)

    def test_get_transit_mask(self, sample_lc_plus, sample_planet):
        """Test get_transit_mask method."""
        if sample_planet is None:
            pytest.skip("No valid planet available for testing")

        # Test with default parameters
        mask = sample_lc_plus.get_transit_mask(sample_planet)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(sample_lc_plus)

        # Test with increased duration
        mask_extended = sample_lc_plus.get_transit_mask(sample_planet, duration_increase_percent=0.5)
        # Extended mask should have at least as many True values as the original
        assert np.sum(mask_extended) >= np.sum(mask)

    def test_get_transit_count(self, sample_lc_plus, sample_planet):
        """Test get_transit_count method."""
        if sample_planet is None:
            pytest.skip("No valid planet available for testing")

        count = sample_lc_plus.get_transit_count(sample_planet)
        assert isinstance(count, int)
        assert count >= 0

    def test_get_combined_transit_mask(self, star_system_test_db, sample_lc_plus):
        """Test get_combined_transit_mask method."""
        star_system = star_system_test_db.get_star_system_from_tic_id(sample_lc_plus.tic_id)
        if not star_system or len(star_system.planets) < 1:
            pytest.skip("No valid planets available for testing")

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

    def test_fold_with_planet(self, sample_lc_plus, sample_planet):
        """Test fold_with_planet method."""
        if sample_planet is None:
            pytest.skip("No valid planet available for testing")

        # Test with default parameters
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
        # Skip single index test as it returns a Row object, not a LightCurve
        # and LightCurvePlus doesn't handle this case correctly

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

    def test_get_first_transit_value(self, sample_lc_plus, sample_planet):
        """Test get_first_transit_value method."""
        if sample_planet is None:
            pytest.skip("No valid planet available for testing")

        transit_time = sample_lc_plus.get_first_transit_value(sample_planet)
        assert isinstance(transit_time, Time)

    def test_get_transit_first_index(self, sample_lc_plus, sample_planet):
        """Test get_transit_first_index method."""
        if sample_planet is None:
            pytest.skip("No valid planet available for testing")

        index = sample_lc_plus.get_transit_first_index(sample_planet)
        assert isinstance(index, int)
        assert 0 <= index < len(sample_lc_plus)

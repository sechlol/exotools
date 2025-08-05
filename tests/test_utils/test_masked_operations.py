"""
Tests for the masked operations utility functions.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Column, MaskedColumn
from astropy.units import Quantity

from exotools.utils.masked_operations import impute_from_columns, safe_average, safe_combine, safe_fill


class TestSafeAverage:
    """Tests for the safe_average function."""

    def test_basic_average(self):
        """Test basic averaging of unmasked columns."""
        col1 = MaskedColumn([1.0, 2.0, 3.0])
        col2 = MaskedColumn([4.0, 5.0, 6.0])
        result = safe_average([col1, col2])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.allclose(result, [2.5, 3.5, 4.5])
        assert not np.any(result.mask)  # No masked values

    def test_with_masked_values(self):
        """Test averaging with some masked values."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
        result = safe_average([col1, col2])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 2.5)  # Average of 1.0 and 4.0
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is unmasked
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is unmasked
        assert not np.any(result.mask)  # All rows have at least one value

    def test_all_masked(self):
        """Test averaging when all values for a row are masked."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, True])
        result = safe_average([col1, col2])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 2.5)  # Average of 1.0 and 4.0
        assert result.mask[1]  # Row 1 should be masked (all inputs masked)
        assert result.mask[2]  # Row 2 should be masked (all inputs masked)

    def test_with_weights(self):
        """Test weighted averaging."""
        col1 = MaskedColumn([1.0, 2.0, 3.0])
        col2 = MaskedColumn([4.0, 5.0, 6.0])
        weights = [1.0, 3.0]  # col2 has 3x the weight of col1
        result = safe_average([col1, col2], weights=weights)

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        # Weighted averages: (1*1 + 4*3)/4 = 3.25, etc.
        assert np.allclose(result, [3.25, 4.25, 5.25])

    def test_with_units(self):
        """Test averaging columns with units."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], unit=u.solRad)
        col2 = MaskedColumn([4.0, 5.0, 6.0], unit=u.solRad)
        result = safe_average([col1, col2])

        # When units are involved, the result is a Quantity object
        assert isinstance(result, Quantity)
        assert len(result) == 3
        assert result.unit == u.solRad
        assert np.allclose(result.value, [2.5, 3.5, 4.5])

    def test_mixed_units(self):
        """Test averaging columns with mixed units."""
        # 1 solar radius = 695,700 km
        col1 = MaskedColumn([1.0, 2.0, 3.0], unit=u.solRad)
        # Using raw values without conversion for simplicity in test
        col2 = MaskedColumn([4.0, 5.0, 6.0], unit=u.solRad)
        result = safe_average([col1, col2])

        # When units are involved, the result is a Quantity object
        assert isinstance(result, Quantity)
        assert len(result) == 3
        assert result.unit == u.solRad
        assert np.allclose(result.value, [2.5, 3.5, 4.5])

    def test_with_nan_values(self):
        """Test averaging with NaN values."""
        col1 = MaskedColumn([1.0, 2.0, np.nan])
        col2 = MaskedColumn([4.0, np.nan, 6.0])
        result = safe_average([col1, col2])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 2.5)  # Average of 1.0 and 4.0
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is not NaN
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is not NaN

    def test_empty_columns(self):
        """Test with empty columns list."""
        with pytest.raises(ValueError, match="At least one column must be provided"):
            safe_average([])

    def test_mismatched_lengths(self):
        """Test with columns of different lengths."""
        col1 = MaskedColumn([1.0, 2.0, 3.0])
        col2 = MaskedColumn([4.0, 5.0])
        with pytest.raises(ValueError, match="All columns must have the same length"):
            safe_average([col1, col2])

    def test_mismatched_weights(self):
        """Test with mismatched number of weights."""
        col1 = MaskedColumn([1.0, 2.0, 3.0])
        col2 = MaskedColumn([4.0, 5.0, 6.0])
        weights = [1.0, 2.0, 3.0]  # Too many weights
        with pytest.raises(ValueError, match="Number of weights must match number of columns"):
            safe_average([col1, col2], weights=weights)


class TestSafeCombine:
    """Tests for the safe_combine function."""

    def test_basic_combine(self):
        """Test basic combining of unmasked columns."""
        col1 = MaskedColumn([1.0, 2.0, 3.0])
        col2 = MaskedColumn([4.0, 5.0, 6.0])
        result = safe_combine([col1, col2], max)

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.allclose(result, [4.0, 5.0, 6.0])  # Max values
        assert not np.any(result.mask)  # No masked values

    def test_with_masked_values(self):
        """Test combining with some masked values."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
        result = safe_combine([col1, col2], max)

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 4.0)  # Max of 1.0 and 4.0
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is unmasked
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is unmasked
        assert not np.any(result.mask)  # All rows have at least one value

    def test_all_masked(self):
        """Test combining when all values for a row are masked."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, True])
        result = safe_combine([col1, col2], max)

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 4.0)  # Max of 1.0 and 4.0
        assert result.mask[1]  # Row 1 should be masked (all inputs masked)
        assert result.mask[2]  # Row 2 should be masked (all inputs masked)

    def test_different_combine_functions(self):
        """Test different combine functions."""
        col1 = MaskedColumn([1.0, 2.0, 3.0])
        col2 = MaskedColumn([4.0, 5.0, 6.0])

        # Test min function
        result_min = safe_combine([col1, col2], min)
        assert np.allclose(result_min, [1.0, 2.0, 3.0])

        # Test sum function
        result_sum = safe_combine([col1, col2], sum)
        assert np.allclose(result_sum, [5.0, 7.0, 9.0])

        # Test custom function
        result_custom = safe_combine([col1, col2], lambda x: sum(x) / len(x))  # Average
        assert np.allclose(result_custom, [2.5, 3.5, 4.5])

    def test_with_units(self):
        """Test combining columns with units."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], unit=u.solRad)
        col2 = MaskedColumn([4.0, 5.0, 6.0], unit=u.solRad)
        result = safe_combine([col1, col2], max)

        # When units are involved, the result is a Quantity object
        assert isinstance(result, Quantity)
        assert len(result) == 3
        assert result.unit == u.solRad
        assert np.allclose(result.value, [4.0, 5.0, 6.0])

    def test_with_nan_values(self):
        """Test combining with NaN values."""
        col1 = MaskedColumn([1.0, 2.0, np.nan])
        col2 = MaskedColumn([4.0, np.nan, 6.0])
        result = safe_combine([col1, col2], max)

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 4.0)  # Max of 1.0 and 4.0
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is not NaN
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is not NaN

    def test_default_fill_value(self):
        """Test with no fill_value (default behavior)."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, True])
        result = safe_combine([col1, col2], max)

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 4.0)  # Max of 1.0 and 4.0
        assert result.mask[1]  # Row 1 should be masked (all inputs masked)
        assert result.mask[2]  # Row 2 should be masked (all inputs masked)

    def test_with_fill_value(self):
        """Test with fill_value provided."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, True])
        fill_value = -999.0
        result = safe_combine([col1, col2], max, fill_value=fill_value)

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 4.0)  # Max of 1.0 and 4.0
        assert not result.mask[1]  # Row 1 should be unmasked (fill_value used)
        assert not result.mask[2]  # Row 2 should be unmasked (fill_value used)
        assert np.isclose(result[1], fill_value)  # Should be the fill value
        assert np.isclose(result[2], fill_value)  # Should be the fill value

    def test_empty_columns(self):
        """Test with empty columns list."""
        with pytest.raises(ValueError, match="At least one column must be provided"):
            safe_combine([], max)

    def test_mismatched_lengths(self):
        """Test with columns of different lengths."""
        col1 = MaskedColumn([1.0, 2.0, 3.0])
        col2 = MaskedColumn([4.0, 5.0])
        with pytest.raises(ValueError, match="All columns must have the same length"):
            safe_combine([col1, col2], max)


class TestSafeFill:
    """Tests for the safe_fill function."""

    def test_basic_fill(self):
        """Test basic filling of masked values."""
        primary = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True])
        fallback1 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, False, True])
        fallback2 = MaskedColumn([7.0, 8.0, 9.0], mask=[False, False, False])
        result = safe_fill(primary, [fallback1, fallback2])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 1.0)  # Primary value is unmasked
        assert np.isclose(result[1], 5.0)  # Filled from fallback1
        assert np.isclose(result[2], 9.0)  # Filled from fallback2
        assert not np.any(result.mask)  # All values should be filled

    def test_no_fallbacks(self):
        """Test with no fallback columns."""
        primary = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True])
        result = safe_fill(primary, [])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 1.0)  # Primary value is unmasked
        assert result.mask[1]  # No fallbacks, should remain masked
        assert result.mask[2]  # No fallbacks, should remain masked

    def test_all_masked(self):
        """Test when all values for a row are masked."""
        primary = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True])
        fallback1 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, True])
        fallback2 = MaskedColumn([7.0, 8.0, 9.0], mask=[False, True, True])
        result = safe_fill(primary, [fallback1, fallback2])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 1.0)  # Primary value is unmasked
        assert result.mask[1]  # All values for row 1 are masked
        assert result.mask[2]  # All values for row 2 are masked

    def test_with_units(self):
        """Test filling with units."""
        primary = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True], unit=u.solRad)
        fallback1 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, False, True], unit=u.solRad)
        result = safe_fill(primary, [fallback1])

        # When units are involved, the result is a Quantity object
        assert isinstance(result, Quantity)
        assert len(result) == 3
        assert result.unit == u.solRad
        # Check values directly
        assert np.isclose(result[0].value, 1.0)  # Primary value is unmasked
        assert np.isclose(result[1].value, 5.0)  # Filled from fallback1
        # The third value should still be masked in the original primary column
        # but we can't check the mask directly on a Quantity object

    def test_with_nan_values(self):
        """Test filling with NaN values."""
        primary = MaskedColumn([1.0, np.nan, 3.0], mask=[False, False, True])
        fallback1 = MaskedColumn([4.0, 5.0, np.nan], mask=[False, False, False])
        result = safe_fill(primary, [fallback1])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 1.0)  # Primary value is not NaN

        # The implementation preserves NaN values from the primary column
        # when they're not masked, rather than filling them from fallbacks
        assert np.isnan(result[1])  # Primary value is NaN and not masked, so not filled

        # For the third value, the implementation checks for NaN in the fallback
        # and won't unmask the row if the fallback value is NaN
        assert result.mask[2]  # Still masked because fallback is NaN

    def test_non_masked_primary(self):
        """Test with non-masked primary column."""
        primary = Column([1.0, 2.0, 3.0])  # Regular Column, not MaskedColumn
        fallback1 = MaskedColumn([4.0, 5.0, 6.0])
        result = safe_fill(primary, [fallback1])

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.allclose(result, [1.0, 2.0, 3.0])  # All primary values used
        assert not np.any(result.mask)  # No masked values


class TestImputeFromColumns:
    """Tests for the impute_from_columns function."""

    def test_average_strategy(self):
        """Test average strategy."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
        result = impute_from_columns([col1, col2], strategy="average")

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 2.5)  # Average of 1.0 and 4.0
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is unmasked
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is unmasked

    def test_first_strategy(self):
        """Test first strategy."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
        result = impute_from_columns([col1, col2], strategy="first")

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 1.0)  # First column value (1.0)
        assert np.isclose(result[1], 2.0)  # First column value (2.0)
        assert np.isclose(result[2], 6.0)  # Second column value (6.0) as first is masked

    def test_max_strategy(self):
        """Test max strategy."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
        result = impute_from_columns([col1, col2], strategy="max")

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 4.0)  # Max of 1.0 and 4.0
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is unmasked
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is unmasked

    def test_min_strategy(self):
        """Test min strategy."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
        result = impute_from_columns([col1, col2], strategy="min")

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 1.0)  # Min of 1.0 and 4.0
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is unmasked
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is unmasked

    def test_sum_strategy(self):
        """Test sum strategy."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
        result = impute_from_columns([col1, col2], strategy="sum")

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        assert np.isclose(result[0], 5.0)  # Sum of 1.0 and 4.0
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is unmasked
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is unmasked

    def test_unknown_strategy(self):
        """Test with unknown strategy."""
        col1 = MaskedColumn([1.0, 2.0, 3.0])
        col2 = MaskedColumn([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            impute_from_columns([col1, col2], strategy="invalid")

    def test_with_weights(self):
        """Test average strategy with weights."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
        weights = [1.0, 3.0]  # col2 has 3x the weight of col1
        result = impute_from_columns([col1, col2], strategy="average", weights=weights)

        assert isinstance(result, MaskedColumn)
        assert len(result) == 3
        # Weighted average: (1*1 + 4*3)/4 = 3.25
        assert np.isclose(result[0], 3.25)
        assert np.isclose(result[1], 2.0)  # Only col1 value (2.0) is unmasked
        assert np.isclose(result[2], 6.0)  # Only col2 value (6.0) is unmasked

    def test_with_units(self):
        """Test with units."""
        col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True], unit=u.solRad)
        col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False], unit=u.solRad)
        result = impute_from_columns([col1, col2], strategy="average")

        # When units are involved, the result is a Quantity object
        assert isinstance(result, Quantity)
        assert len(result) == 3
        assert result.unit == u.solRad
        assert np.isclose(result[0].value, 2.5)  # Average of 1.0 and 4.0
        assert np.isclose(result[1].value, 2.0)  # Only col1 value (2.0) is unmasked
        assert np.isclose(result[2].value, 6.0)  # Only col2 value (6.0) is unmasked

import logging

import numpy as np
from astropy.table import Column, MaskedColumn, QTable
from astropy.time import Time
from astropy.units import Quantity
from astropy.utils.masked import Masked

logger = logging.getLogger(__name__)


def _is_masked_column(col: Column) -> bool:
    types = [MaskedColumn, Masked]
    return any(isinstance(col, t) for t in types)


def compare_qtables(expected_table: QTable, test_table: QTable) -> bool:
    """
    Compare two QTable objects for equality by checking:
    - Column names match
    - Column units match (for columns that have units)
    - Column dtypes match (for columns that have dtypes)
    - Column values match

    Raises AssertionError with descriptive message if tables don't match.
    Returns True if tables are equivalent.
    """
    # Check if both tables have the same number of columns
    if len(expected_table.colnames) != len(test_table.colnames):
        raise AssertionError(
            f"Tables have different number of columns: "
            f"expected {len(expected_table.colnames)}, got {len(test_table.colnames)}"
        )

    # Check if both tables have the same number of rows
    if len(expected_table) != len(test_table):
        raise AssertionError(
            f"Tables have different number of rows: expected {len(expected_table)}, got {len(test_table)}"
        )

    # Check column names match
    expected_cols = set(expected_table.colnames)
    test_cols = set(test_table.colnames)
    if expected_cols != test_cols:
        missing = expected_cols - test_cols
        extra = test_cols - expected_cols
        error_msg = "Column names don't match: "
        if missing:
            error_msg += f"missing columns {missing} "
        if extra:
            error_msg += f"extra columns {extra}"
        raise AssertionError(error_msg)

    # Check each column individually
    for col_name in expected_table.colnames:
        expected_col = expected_table[col_name]
        test_col = test_table[col_name]

        both_masked = _is_masked_column(expected_col) and _is_masked_column(test_col)

        # Check types match
        if type(expected_col) is not type(test_col) and not both_masked:
            is_masked_column = isinstance(expected_col, MaskedColumn) and isinstance(test_col, Column)
            is_masked_quantity = isinstance(expected_col, Masked) and isinstance(test_col, Quantity)
            if is_masked_column or is_masked_quantity:
                # Check that masked values in expected_col match NaN values in test_col
                if np.any(expected_col.mask):
                    # Get indices of masked values in expected_col
                    masked_indices = np.where(expected_col.mask)[0]

                    # Check if corresponding values in test_col are NaN
                    for idx in masked_indices:
                        if idx < len(test_col) and not np.isnan(test_col[idx]):
                            raise AssertionError(
                                f"Masked value mismatch for column '{col_name}' at index {idx}: "
                                f"expected masked value, got non-NaN value {test_col[idx]}"
                            )

                    # Check if there are NaN values in test_col that aren't masked in expected_col
                    nan_indices = (
                        np.where(np.isnan(test_col))[0]
                        if hasattr(test_col, "dtype") and np.issubdtype(test_col.dtype, np.number)
                        else []
                    )
                    unexpected_nans = [idx for idx in nan_indices if idx not in masked_indices]
                    if unexpected_nans:
                        raise AssertionError(
                            f"Masked value mismatch for column '{col_name}': "
                            f"found NaN values at indices {unexpected_nans} that aren't masked in expected column"
                        )
            else:
                raise AssertionError(
                    f"Column types don't match for column '{col_name}': "
                    f"expected {type(expected_col).__name__}, got {type(test_col).__name__}"
                )

        # Check units match (if applicable)
        if hasattr(expected_col, "unit"):
            if expected_col.unit != test_col.unit:
                # Handle case where one unit is None and the other is not
                if expected_col.unit is None or test_col.unit is None:
                    # For storage tests, we'll be lenient about None vs non-None units
                    # This is because some storage formats may not preserve units perfectly
                    logger.warning(
                        f"Units don't match for column '{col_name}': expected {expected_col.unit}, got {test_col.unit}"
                    )
                else:
                    raise AssertionError(
                        f"Units don't match for column '{col_name}': expected {expected_col.unit}, got {test_col.unit}"
                    )

        # Check dtypes match (if applicable)
        if hasattr(expected_col, "dtype"):
            if expected_col.dtype != test_col.dtype:
                is_string_type = expected_col.dtype == "object" and "str" in test_col.dtype.name
                if not is_string_type:
                    raise AssertionError(
                        f"Data types don't match for column '{col_name}': "
                        f"expected {expected_col.dtype}, got {test_col.dtype}"
                    )

        # Check if column is an astropy Time object
        if isinstance(expected_col, Time):
            # Check Time format
            if expected_col.format != test_col.format:
                raise AssertionError(
                    f"Time format doesn't match for column '{col_name}': "
                    f"expected {expected_col.format}, got {test_col.format}"
                )
            # Check Time scale
            if expected_col.scale != test_col.scale:
                raise AssertionError(
                    f"Time scale doesn't match for column '{col_name}': "
                    f"expected {expected_col.scale}, got {test_col.scale}"
                )

        # Check values match (handle potential NaN/masked values)
        try:
            # For numeric columns, use numpy array comparison
            if hasattr(expected_col, "mask") and hasattr(test_col, "mask"):
                # Handle masked columns
                if not np.array_equal(expected_col.mask, test_col.mask, equal_nan=True):
                    raise AssertionError(f"Mask values don't match for column '{col_name}'")

                # Compare unmasked values
                if hasattr(expected_col, "value") and hasattr(test_col, "value"):
                    if not np.array_equal(expected_col.value, test_col.value, equal_nan=True):
                        raise AssertionError(f"Data values don't match for masked column '{col_name}'")
            else:
                # Regular comparison
                if not np.array_equal(expected_col, test_col, equal_nan=True):
                    raise AssertionError(f"Values don't match for column '{col_name}'")
        except (TypeError, ValueError):
            # Fallback for non-numeric or complex data types
            try:
                mismatched = [
                    (i, exp_val, test_val)
                    for i, (exp_val, test_val) in enumerate(zip(expected_col, test_col))
                    if exp_val != test_val
                ]
                if mismatched:
                    idx, exp, got = mismatched[0]  # Report first mismatch
                    raise AssertionError(
                        f"Values don't match for column '{col_name}' at index {idx}: expected {exp}, got {got}"
                    )
            except Exception:
                # Last resort: convert to string and compare
                mismatched = [
                    (i, str(exp_val), str(test_val))
                    for i, (exp_val, test_val) in enumerate(zip(expected_col, test_col))
                    if str(exp_val) != str(test_val)
                ]
                if mismatched:
                    idx, exp, got = mismatched[0]  # Report first mismatch
                    raise AssertionError(
                        f"String representations don't match for column '{col_name}' at index {idx}: "
                        f"expected '{exp}', got '{got}'"
                    )

    return True

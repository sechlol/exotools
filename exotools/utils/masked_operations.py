"""
Utility functions for safely handling operations on masked columns.

These functions provide a consistent way to perform operations on masked columns
while properly handling the masking.
"""

from typing import Any, Callable, Optional, Union

import numpy as np
from astropy.table import Column, MaskedColumn


def safe_average(
    columns: list[Union[MaskedColumn, Column, np.ndarray]],
    weights: Optional[list[float]] = None,
) -> MaskedColumn:
    """
    Safely average multiple columns, respecting masks.

    For each row, the average is computed using only unmasked values.
    If all values for a row are masked, the result will be masked.

    Parameters
    ----------
    columns : list[Union[MaskedColumn, Column, np.ndarray]]
        List of columns to average
    weights : Optional[list[float]], optional
        Weights to apply to each column, by default None (equal weights)

    Returns
    -------
    MaskedColumn
        A masked column containing the average of unmasked values

    Examples
    --------
    >>> from astropy.table import MaskedColumn
    >>> import numpy as np
    >>> col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
    >>> col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
    >>> safe_average([col1, col2])
    MaskedColumn([2.5, 2.0, 6.0], mask=[False, False, False])
    """
    if not columns:
        raise ValueError("At least one column must be provided")

    # Ensure all columns have the same length
    length = len(columns[0])
    for col in columns:
        if len(col) != length:
            raise ValueError("All columns must have the same length")

    # Set default weights if not provided
    if weights is None:
        weights = [1.0] * len(columns)
    elif len(weights) != len(columns):
        raise ValueError("Number of weights must match number of columns")

    # Determine if we need to handle units
    has_units = any(hasattr(col, "unit") for col in columns)
    unit = None
    if has_units:
        # Use the unit from the first column that has one
        for col in columns:
            if hasattr(col, "unit") and col.unit is not None:
                unit = col.unit
                break

    # Initialize result arrays
    result_values = np.zeros(length)
    result_mask = np.ones(length, dtype=bool)  # Start with all masked

    # For each row
    for i in range(length):
        valid_values = []
        valid_weights = []

        # Collect unmasked values and their weights
        for col_idx, col in enumerate(columns):
            # Check if the value is masked
            is_masked = hasattr(col, "mask") and col.mask[i]

            if not is_masked and not np.isnan(col[i]):
                # Extract the raw value without units for calculation
                if hasattr(col, "unit") and col.unit is not None:
                    # If the column has units, get the raw value
                    if hasattr(col[i], "value"):
                        valid_values.append(col[i].value)
                    else:
                        valid_values.append(col[i])
                else:
                    valid_values.append(col[i])
                valid_weights.append(weights[col_idx])

        # If we have valid values, compute weighted average
        if valid_values:
            total_weight = sum(valid_weights)
            if total_weight > 0:
                result_values[i] = sum(v * w for v, w in zip(valid_values, valid_weights)) / total_weight
                result_mask[i] = False  # Unmask this row

    # Create the result column
    result = MaskedColumn(result_values, mask=result_mask)

    # Add the unit if needed
    if unit is not None:
        result = result * unit

    return result


def safe_combine(
    columns: list[Union[MaskedColumn, Column, np.ndarray]],
    combine_func: Callable[[list[Any]], Any],
    default_mask: bool = True,
) -> MaskedColumn:
    """
    Safely combine multiple columns using a custom function, respecting masks.

    For each row, the function is applied to unmasked values only.
    If all values for a row are masked, the result will be masked.

    Parameters
    ----------
    columns : list[Union[MaskedColumn, Column, np.ndarray]]
        List of columns to combine
    combine_func : Callable[[list[Any]], Any]
        Function to apply to unmasked values for each row
    default_mask : bool, optional
        Default mask value when no valid values are found, by default True (masked)

    Returns
    -------
    MaskedColumn
        A masked column containing the combined values

    Examples
    --------
    >>> from astropy.table import MaskedColumn
    >>> import numpy as np
    >>> col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
    >>> col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
    >>> safe_combine([col1, col2], max)
    MaskedColumn([4.0, 2.0, 6.0], mask=[False, False, False])
    """
    if not columns:
        raise ValueError("At least one column must be provided")

    # Ensure all columns have the same length
    length = len(columns[0])
    for col in columns:
        if len(col) != length:
            raise ValueError("All columns must have the same length")

    # Determine if we need to handle units
    has_units = any(hasattr(col, "unit") for col in columns)
    unit = None
    if has_units:
        # Use the unit from the first column that has one
        for col in columns:
            if hasattr(col, "unit") and col.unit is not None:
                unit = col.unit
                break

    # Initialize result arrays
    result_values = np.zeros(length)
    result_mask = np.ones(length, dtype=bool) if default_mask else np.zeros(length, dtype=bool)

    # For each row
    for i in range(length):
        valid_values = []

        # Collect unmasked values
        for col in columns:
            # Check if the value is masked
            is_masked = hasattr(col, "mask") and col.mask[i]

            if not is_masked and not np.isnan(col[i]):
                # Extract the raw value without units for calculation
                if hasattr(col, "unit") and col.unit is not None:
                    # If the column has units, get the raw value
                    if hasattr(col[i], "value"):
                        valid_values.append(col[i].value)
                    else:
                        valid_values.append(col[i])
                else:
                    valid_values.append(col[i])

        # If we have valid values, apply the combine function
        if valid_values:
            result_values[i] = combine_func(valid_values)
            result_mask[i] = False  # Unmask this row

    # Create the result column
    result = MaskedColumn(result_values, mask=result_mask)

    # Add the unit if needed
    if unit is not None:
        result = result * unit

    return result


def safe_fill(
    primary_column: Union[MaskedColumn, Column, np.ndarray],
    fallback_columns: list[Union[MaskedColumn, Column, np.ndarray]],
) -> MaskedColumn:
    """
    Fill masked values in the primary column with values from fallback columns.

    For each row, if the primary column is masked, the first unmasked value
    from the fallback columns is used. If all values are masked, the result
    will be masked.

    Parameters
    ----------
    primary_column : Union[MaskedColumn, Column, np.ndarray]
        The primary column to fill
    fallback_columns : list[Union[MaskedColumn, Column, np.ndarray]]
        List of columns to use as fallbacks, in order of preference

    Returns
    -------
    MaskedColumn
        A masked column with values filled from fallback columns where possible

    Examples
    --------
    >>> from astropy.table import MaskedColumn
    >>> import numpy as np
    >>> primary = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, True])
    >>> fallback1 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, False, True])
    >>> fallback2 = MaskedColumn([7.0, 8.0, 9.0], mask=[False, False, False])
    >>> safe_fill(primary, [fallback1, fallback2])
    MaskedColumn([1.0, 5.0, 9.0], mask=[False, False, False])
    """
    # Get the unit from the primary column if it has one
    unit = None
    if hasattr(primary_column, "unit") and primary_column.unit is not None:
        unit = primary_column.unit

    # Initialize result with primary column
    if hasattr(primary_column, "mask"):
        # For MaskedColumn, get the data and mask
        if hasattr(primary_column, "value"):
            # For Quantity objects, get the raw value
            result_values = primary_column.value.copy()
        else:
            result_values = primary_column.data.copy()
        result_mask = primary_column.mask.copy()
    else:
        # For regular arrays, no mask
        result_values = np.array(primary_column)
        result_mask = np.zeros(len(primary_column), dtype=bool)

    # For each row that's masked in the primary column
    for i in range(len(result_values)):
        if result_mask[i]:
            # Try each fallback column
            for col in fallback_columns:
                is_masked = hasattr(col, "mask") and col.mask[i]

                if not is_masked and not np.isnan(col[i]):
                    # Extract the raw value without units
                    if hasattr(col, "unit") and col.unit is not None:
                        if hasattr(col[i], "value"):
                            result_values[i] = col[i].value
                        else:
                            result_values[i] = col[i]
                    else:
                        result_values[i] = col[i]
                    result_mask[i] = False
                    break

    # Create the result column
    result = MaskedColumn(result_values, mask=result_mask)

    # Add the unit if needed
    if unit is not None:
        result = result * unit

    return result


def impute_from_columns(
    columns: list[Union[MaskedColumn, Column, np.ndarray]],
    strategy: str = "average",
    weights: Optional[list[float]] = None,
) -> MaskedColumn:
    """
    Impute values from multiple columns using the specified strategy.

    Parameters
    ----------
    columns : list[Union[MaskedColumn, Column, np.ndarray]]
        List of columns to impute from
    strategy : str, optional
        Strategy to use for imputation, by default "average"
        Options: "average", "first", "max", "min", "sum"
    weights : Optional[list[float]], optional
        Weights to use for average strategy, by default None

    Returns
    -------
    MaskedColumn
        A masked column with imputed values

    Examples
    --------
    >>> from astropy.table import MaskedColumn
    >>> import numpy as np
    >>> col1 = MaskedColumn([1.0, 2.0, 3.0], mask=[False, False, True])
    >>> col2 = MaskedColumn([4.0, 5.0, 6.0], mask=[False, True, False])
    >>> impute_from_columns([col1, col2], strategy="average")
    MaskedColumn([2.5, 2.0, 6.0], mask=[False, False, False])
    >>> impute_from_columns([col1, col2], strategy="first")
    MaskedColumn([1.0, 2.0, 6.0], mask=[False, False, False])
    """
    if strategy == "average":
        return safe_average(columns, weights)
    elif strategy == "first":
        # Use the first column as primary and others as fallbacks
        primary, *fallbacks = columns
        return safe_fill(primary, fallbacks)
    elif strategy == "max":
        return safe_combine(columns, max)
    elif strategy == "min":
        return safe_combine(columns, min)
    elif strategy == "sum":
        return safe_combine(columns, sum)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

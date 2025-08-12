"""
Utility functions for safely handling operations on masked columns.

These functions provide a consistent way to perform operations on masked columns
while properly handling the masking.
"""

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from astropy.table import Column, MaskedColumn, QTable
from astropy.units import Quantity
from astropy.utils.masked import Masked


def safe_average_columns(columns: list[Union[MaskedColumn, Column, np.ndarray]]) -> Masked | Column | Quantity:
    if len(columns) == 0:
        raise ValueError("At least one column must be provided")

    # Check if all columns have the same unit
    units = [c.unit for c in columns if hasattr(c, "unit") and c.unit is not None]

    # Assert units are all the same:
    if len(units) > 1 and not all(u == units[0] for u in units):
        raise ValueError("All columns must have the same unit")

    nan_values = np.hstack([np.isnan(c)[:, np.newaxis] for c in columns])
    nan_values = nan_values.filled(True) if hasattr(nan_values, "filled") else nan_values

    masked = np.hstack(
        [c.mask[:, np.newaxis] if hasattr(c, "mask") else np.full(shape=(len(c), 1), fill_value=False) for c in columns]
    )
    masked = masked | nan_values
    final_mask = masked.all(axis=1)
    non_masked_count = (masked == 0).sum(axis=1)

    to_sum = np.zeros(len(columns[0]))
    for i, c in enumerate(columns):
        filled = c.filled(0) if hasattr(c, "filled") else c
        filled[nan_values[:, i]] = 0
        to_sum = to_sum + filled

    avg = to_sum / non_masked_count
    avg = Quantity(avg, units[0]) if units else Column(avg)

    return Masked(avg, mask=final_mask) if final_mask.any() else avg


def safe_average(dataset: QTable, columns: Sequence[str]) -> Masked | Column | Quantity:
    return safe_average_columns([dataset[c] for c in columns])


def safe_combine(
    columns: list[Union[MaskedColumn, Column, np.ndarray]],
    combine_func: Callable[[list[Any]], Any],
    fill_value: Optional[Any] = None,
) -> MaskedColumn:
    """
    Safely combine multiple columns using a custom function, respecting masks.

    For each row, the function is applied to unmasked values only.
    If all values for a row are masked, the result will be masked unless
    a fill_value is provided.

    Parameters
    ----------
    columns : list[Union[MaskedColumn, Column, np.ndarray]]
        List of columns to combine
    combine_func : Callable[[list[Any]], Any]
        Function to apply to unmasked values for each row
    fill_value : Optional[Any], optional
        Value to use for rows where all input values are masked, by default None.
        If None, rows with all masked values will be masked in the output.
        If provided, these rows will be unmasked and filled with this value.

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
    result_mask = np.ones(length, dtype=bool)  # Start with all masked

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
        elif fill_value is not None:
            # Use fill value for rows with no valid values
            result_values[i] = fill_value
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
        return safe_average_columns(columns)
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

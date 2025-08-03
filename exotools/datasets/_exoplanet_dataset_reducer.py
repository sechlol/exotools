import numpy as np
import pandas as pd
from astropy.table import QTable

from exotools.db import ExoDB
from exotools.utils.qtable_utils import get_empty_table_header, TableColumnInfo, QTableHeader


def _get_subset_df(table: QTable) -> pd.DataFrame:
    """Select a subset of data from the main exoplanet dataset, including upper and lower bounds"""
    err_cols = []
    dataset_columns = ["tic_id", "gaia_id", "disc_telescope", "rowupdate"]
    star_columns = ["hostname", "hostname_lowercase", "st_rad", "st_rad_gaia", "st_mass"]
    planet_columns = [
        "pl_name",
        "pl_rade",
        "pl_masse",
        "pl_dens",
        "pl_orbeccen",
        "pl_orbper",
        "pl_orblper",
        "pl_orbincl",
        "pl_orbsmax",
        "pl_tranmid",
        "pl_trandur",
        "pl_trandep",
        "pl_imppar",
        "pl_ratror",
        "pl_ratdor",
    ]
    fields = dataset_columns + star_columns + planet_columns
    for p in fields:
        if f"{p}_lower" in table.colnames:
            err_cols.extend([f"{p}_lower", f"{p}_upper"])
    return table[fields + err_cols].to_pandas()


def _reduce_group(group: pd.DataFrame) -> pd.Series:
    """
    For each column, select the first element that is not null.
    Parameters:
        - group: a dataframe grouped by planet name, and sorted by "rowupdate" in descending order
    """
    first_non_null = group.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else np.nan)
    return first_non_null


def _reduce_df(table: QTable) -> pd.DataFrame:
    """
    Reduce all the planets having multiple rows to only one single row, taking the most updated not-null value
    available in the dataset.
    """
    df = _get_subset_df(table)
    # Sort by update date
    sorted_df = df.sort_values("rowupdate", ascending=False)

    # Group by 'pl_name' and apply reduce_group
    grouped = sorted_df.groupby("pl_name", as_index=True)
    reduced_groups = grouped.apply(_reduce_group, include_groups=False).reset_index(drop=False)
    return reduced_groups


def _flag_invalid_planets(dataset: pd.DataFrame):
    """
    Adds a pl_valid_flag to the planets which have all the required parameters
    """
    # Without these we can't fit the transits
    mandatory_fields = ["pl_rade", "pl_trandur", "pl_tranmid", "pl_orbsmax"]

    # Add the validation flag to the dataset
    dataset["pl_valid_flag"] = ~dataset[mandatory_fields].isna().any(axis=1)


def reduce_exoplanet_dataset(exo_db: ExoDB) -> tuple[QTable, QTableHeader]:
    """
    Post-processes the Known exoplanets dataset to select only the transiting planets, and reducing multiple
    entries for each planet to only one. Additionally, impute some missing values using GAIA data.
    """
    # Load datasets, limiting to only Tess and Kepler transiting planets
    exo_db = exo_db.get_transiting_planets(kepler_or_tess_only=False)

    # Reduce exoplanet dataset
    reduced_df = _reduce_df(exo_db.dataset_copy)
    _flag_invalid_planets(reduced_df)

    # Assign units and convert to QTable
    units_map = {c: exo_db.view[c].unit for c in reduced_df.columns if c in exo_db.view.colnames}
    reduced_table = QTable.from_pandas(reduced_df, units=units_map)

    # Assign units to reduced dataset and convert to QTable
    reduced_header = get_empty_table_header(reduced_table)
    reduced_header["pl_valid_flag"] = TableColumnInfo(
        unit=None, description="True if the planet has all the parameters to determine transit events"
    )

    return reduced_table, reduced_header

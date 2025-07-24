from typing import Optional, Sequence
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import QTable, MaskedColumn
from astropy.units import Unit

from exotools.utils.qtable_utils import TableColumnInfo, QTableHeader
from exotools.utils.unit_mapper import UNIT_MAPPER
from .dataset_downloader import fix_unrecognized_units, override_units, DatasetDownloader
from .tap_service import ExoService, TapService


def _get_error_parameters(parameters: list[str], include_original: Optional[bool] = False) -> list[str]:
    errs = [f"{p}err{i}" for p in parameters for i in [1, 2]]
    if include_original:
        return parameters + errs
    return errs


class KnownExoplanetsDownloader(DatasetDownloader):
    """
    Data source: Nasa Exoplanet Archive (table: Planetary Systems)
    https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
    https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
    """

    _table_name = "ps"

    # NOTE: the units of pl_tranmid and its boundaries are mistakenly labelled as "day". It should be "hours"
    #   see https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
    _unit_overrides = {p: u.hour for p in _get_error_parameters(["pl_trandur"], True)}

    def __init__(self):
        self._exo_service = ExoService()

    def _download(self, limit: Optional[int] = None) -> QTable:
        # There are some unnecessary fields that cause troubles when saving the DF, we should remove them
        fields = ",".join([f for f in self._exo_service.get_field_names(self._table_name) if not f.endswith("str")])

        limit_clause = f"top {limit}" if limit else ""
        query_str = f"select {limit_clause} {fields} from {self._table_name}"

        print(f"Querying {self._exo_service.url} (synchronous)...")
        dataset = self._exo_service.query(query_str, sync=True)
        n_planets = len(pd.unique(dataset["pl_name"]))
        n_records = len(dataset)

        print(f"DONE! Collected {n_planets} unique planets, for a total of {n_records} records.")
        return dataset

    def _clean_and_fix(self, table: QTable) -> QTable:
        _parse_ids(table)
        fix_unrecognized_units(table=table, units_map=UNIT_MAPPER)
        override_units(table=table, unit_overrides=self._unit_overrides)

        return table

    def _get_table_header(self, table: QTable) -> QTableHeader:
        return _get_fixed_table_header(table=table, table_name=self._table_name, tap_service=self._exo_service)

    def _download_by_id(self, ids: Sequence[int]) -> QTable:
        raise NotImplementedError("KnownExoplanetsDownloader does not support this download method")


def _get_fixed_table_header(table: QTable, table_name: str, tap_service: TapService) -> QTableHeader:
    descriptions = tap_service.get_field_descriptions(table_name)
    return {
        c: TableColumnInfo(
            unit=table[c].unit.name if isinstance(table[c].unit, Unit) else None, description=descriptions[c]
        )
        for c in table.columns
        if c in descriptions
    }


def _parse_ids(table: QTable):
    """
    Parses some relevant IDs to their numerical representation, from string to masked integer
    """
    fill_value = -1
    tic_ids = np.char.split(table["tic_id"].astype(str), " ")
    gaia_ids = np.char.split(table["gaia_id"].astype(str), " ")

    # Extract the desired elements and convert to pandas nullable integer type
    tic_id_extracted = [int(item[1]) if len(item) > 1 else fill_value for item in tic_ids]
    gaia_id_extracted = [int(item[2]) if len(item) > 2 else fill_value for item in gaia_ids]

    table["tic_id"] = MaskedColumn(tic_id_extracted, fill_value=fill_value, dtype="int64")
    table["gaia_id"] = MaskedColumn(gaia_id_extracted, fill_value=fill_value, dtype="int64")

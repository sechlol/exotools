from typing import Optional

import pandas as pd
import astropy.units as u
from astropy.table import QTable

from exotools.utils.qtable_utils import QTableHeader, fix_unrecognized_units, override_units
from exotools.utils.unit_mapper import UNIT_MAPPER

from .dataset_downloader import DatasetDownloader
from .exoplanets_downloader import _get_fixed_table_header, _get_error_parameters
from .tap_service import ExoService


class CandidateExoplanetsDownloader(DatasetDownloader):
    """
    Data source: Nasa Exoplanet Archive (table: TESS Objects of Interest Table Data Columns)
    https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
    https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html
    """

    def _download_by_id(self, ids: list[int]) -> QTable:
        raise NotImplementedError("CandidateExoplanetsDownloader does not support this download method")

    _table_name = "toi"

    # originally marked as "ppm", but it's not a recognized unit
    _unit_overrides = {p: (u.Unit("%") * 1e-6) for p in _get_error_parameters(["pl_trandep"], True)}

    def __init__(self):
        self._exo_service = ExoService()

    def _download(self, limit: Optional[int] = None) -> QTable:
        limit_clause = f"top {limit}" if limit else ""
        query_str = f"select {limit_clause} * from {self._table_name}"

        print(f"Downloading Candidate exoplanets...")
        dataset = self._exo_service.query(query_str, sync=True)
        n_unique = len(pd.unique(dataset["toi"]))

        print(f"DONE! Collected {n_unique} unique candidates, for a total of {len(dataset)} records.")
        return dataset

    def _clean_and_fix(self, table: QTable) -> QTable:
        fix_unrecognized_units(table=table, units_map=UNIT_MAPPER)
        override_units(table=table, unit_overrides=self._unit_overrides)
        table.rename_column("tid", "tic_id")
        return table

    def _get_table_header(self, table: QTable) -> QTableHeader:
        return _get_fixed_table_header(table=table, table_name=self._table_name, tap_service=self._exo_service)

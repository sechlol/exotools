from typing import Optional

import pandas as pd
from astropy import units as u
from astropy.table import QTable

from .dataset_downloader import DatasetDownloader, fix_unrecognized_units
from .exoplanets_downloader import _get_fixed_table_header
from src.exotools.utils.qtable_utils import QTableHeader
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
    _unit_map = {
        "days": u.day,
        "BJD": u.day,
        "hours": u.hour,
        "R_Earth": u.R_earth,
        "R_Sun": u.solRad,
        "cm/s**2": u.cm / u.s**2,
    }

    def __init__(self):
        self._exo_service = ExoService()

    def _download(self, limit: Optional[int] = None) -> QTable:
        limit_clause = f"top {limit}" if limit else ""
        query_str = f"select {limit_clause} * from {self._table_name}"

        print(f"Downloading Candidate exoplanets: \n{query_str}")
        dataset = self._exo_service.query(query_str, sync=True)
        n_unique = len(pd.unique(dataset["toi"]))

        print(f"DONE! Collected {n_unique} unique candidates, for a total of {len(dataset)} records.")
        return dataset

    def _clean_and_fix(self, table: QTable) -> QTable:
        fix_unrecognized_units(table=table, units_map=self._unit_map)
        table.rename_column("tid", "tic_id")
        return table

    def _get_table_header(self, table: QTable) -> QTableHeader:
        return _get_fixed_table_header(table=table, table_name=self._table_name, tap_service=self._exo_service)

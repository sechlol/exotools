import logging
from typing import Optional, Sequence

import astropy.units as u
import pandas as pd
from astropy.table import QTable

from exotools.utils.qtable_utils import QTableHeader
from exotools.utils.unit_mapper import UNIT_MAPPER

from ._utils import fix_unrecognized_units, override_units
from .dataset_downloader import DatasetDownloader
from .exoplanets_downloader import fill_error_bounds, get_error_parameters, get_fixed_table_header
from .tap_service import ExoService, TapService

logger = logging.getLogger(__name__)


class CandidateExoplanetsDownloader(DatasetDownloader):
    """
    Data source: Nasa Exoplanet Archive (table: TESS Objects of Interest Table Data Columns)
    https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
    https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html
    """

    _table_name = "toi"
    _index_col = "tid"

    # "pl_trandep" unit is "ppm", but it's not a recognized unit
    _unit_overrides = {p: (u.Unit("%") * 1e-6) for p in get_error_parameters(["pl_trandep"], True)}

    _exo_service: Optional[TapService] = None

    def _initialize_services(self):
        if self._exo_service is None:
            self._exo_service = ExoService()

    def _download_by_id(self, ids: list[int], **kwargs) -> QTable:
        raise NotImplementedError("CandidateExoplanetsDownloader does not support this download method")

    def _download(self, limit: Optional[int] = None, columns: Optional[Sequence[str]] = None, **kwargs) -> QTable:
        if columns:
            columns = set(columns)
            columns.add(self._index_col)
            fields = ",".join(columns)
        else:
            fields = "*"

        limit_clause = f"top {limit}" if limit else ""
        query_str = f"select {limit_clause} {fields} from {self._table_name}"

        logger.info("Downloading Candidate exoplanets...")
        dataset = self._exo_service.query(query_str)
        n_unique = len(pd.unique(dataset["toi"]))

        logger.info(f"DONE! Collected {n_unique} unique candidates, for a total of {len(dataset)} records.")
        return dataset

    def _clean_and_fix(self, table: QTable) -> QTable:
        fill_error_bounds(table)
        fix_unrecognized_units(table=table, units_map=UNIT_MAPPER)
        override_units(table=table, unit_overrides=self._unit_overrides)
        table.rename_column(self._index_col, "tic_id")
        return table

    def _get_table_header(self, table: QTable) -> QTableHeader:
        return get_fixed_table_header(table=table, table_name=self._table_name, tap_service=self._exo_service)

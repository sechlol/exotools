import logging
from typing import Any, Optional, Sequence

import pandas as pd
from astropy.table import QTable

from exotools.utils.qtable_utils import QTableHeader
from exotools.utils.unit_mapper import UNIT_MAPPER
from exotools.utils.warning_utils import silence_warnings

from ._utils import fix_unrecognized_units
from .dataset_downloader import DatasetDownloader
from .exoplanets_downloader import _get_where_clause, fill_error_bounds, get_fixed_table_header, parse_ids
from .tap_service import ExoService, TapService

logger = logging.getLogger(__name__)


def _get_error_parameters(parameters: list[str], include_original: Optional[bool] = False) -> list[str]:
    errs = [f"{p}err{i}" for p in parameters for i in [1, 2]]
    if include_original:
        return parameters + errs
    return errs


class PlanetarySystemsCompositeDownloader(DatasetDownloader):
    """
    Data source: Nasa Exoplanet Archive (table: Planetary Systems Composite Parameters [PSCompPars])
    https://exoplanetarchive.ipac.caltech.edu/docs/API_PSCompPars_columns.html

    Note about using the table:
    https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html

    TAP service info:
    https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
    """

    _table_name = "pscomppars"
    _mandatory_fields = ["tic_id", "gaia_id", "hostname", "pl_name"]

    _exo_service: Optional[TapService] = None

    def _initialize_services(self):
        if self._exo_service is None:
            self._exo_service = ExoService()

    def _download_by_id(self, ids: list[int], **kwargs) -> QTable:
        raise NotImplementedError("PlanetarySystemsCompositeDownloader does not support this download method")

    def _download(
        self,
        limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        where: Optional[dict[str, Any | list[Any]]] = None,
        **kwargs,
    ) -> QTable:
        limit_clause = f"top {limit}" if limit else ""
        where_clause = _get_where_clause(where=where)
        fields = self._get_fields_to_query(columns=columns, use_cached_fields=True)

        query_str = f"select {limit_clause} {fields} from {self._table_name} {where_clause}"

        logger.info("Downloading Candidate exoplanets...")
        with silence_warnings():
            dataset = self._exo_service.query(query_str)
        n_unique = len(pd.unique(dataset["objectid"]))

        logger.info(f"DONE! Collected {n_unique} unique planets.")
        return dataset

    def _clean_and_fix(self, table: QTable) -> QTable:
        parse_ids(table)
        fill_error_bounds(table)

        # drop _str columns
        columns_to_keep = [c for c in table.colnames if "_str" not in c and not c.endswith("str")]
        table = table[columns_to_keep]

        # Fix units
        fix_unrecognized_units(table=table, units_map=UNIT_MAPPER)
        return table

    def _get_fields_to_query(self, use_cached_fields: bool = False, columns: Optional[Sequence[str]] = None) -> str:
        if columns:
            col_set = set(columns)
            col_set.update(self._mandatory_fields)
            fields = list(col_set)
        elif use_cached_fields:
            fields = "*"
        else:
            # There are some unnecessary fields that cause troubles when saving the DF, we should remove them
            fields = [f for f in self._exo_service.get_field_names(self._table_name) if not f.endswith("str")]

        return ",".join(fields)

    def _get_table_header(self, table: QTable) -> QTableHeader:
        return get_fixed_table_header(table=table, table_name=self._table_name, tap_service=self._exo_service)

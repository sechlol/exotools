from typing import Optional

import pyvo as vo
from astropy.table import QTable
from pyvo.dal.vosi import VOSITables

from src.exotools.utils.qtable_utils import QTableHeader, TableColumnInfo


class TapService:
    """
    Unfortunately, this method is not suitable for large queries, as it doesn't support authentication.
    Use casjobs instead for large queries (see download_dataset.query_ctl_casjob())
    """

    def __init__(self, url: str):
        self._url = url
        self._service = vo.dal.TAPService(self._url)
        self._tables: Optional[VOSITables] = None

    @property
    def url(self) -> str:
        return self._url

    def get_table_names(self) -> list[str]:
        return [t.name for t in self._service.tables]

    def get_table_schemas(self) -> dict[str, QTableHeader]:
        return {name: self.get_field_info(name) for name in self.get_table_names()}

    def get_field_info(self, table_name: str) -> QTableHeader:
        tables: VOSITables = self._service.tables
        if table_name not in tables:
            raise KeyError(f"{table_name} not in TapService {self._url}")
        return self._get_fields_info(tables[table_name])

    def get_field_units(self, table_name: str) -> dict[str, str]:
        tables: VOSITables = self._service.tables
        if table_name not in tables:
            raise KeyError(f"{table_name} not in TapService {self._url}")
        return {c.name: c.unit for c in tables[table_name].columns}

    def get_field_descriptions(self, table_name: str) -> dict[str, str]:
        tables: VOSITables = self._service.tables
        if table_name not in tables:
            raise KeyError(f"{table_name} not in TapService {self._url}")
        return {c.name: c.description for c in tables[table_name].columns}

    def get_field_names(self, table_name: str) -> list[str]:
        tables: VOSITables = self._service.tables
        if table_name not in tables:
            raise KeyError(f"{table_name} not in TapService {self._url}")
        return [c.name for c in tables[table_name].columns]

    def query(self, query_string: str, sync: bool = False) -> QTable:
        if sync:
            result = self._service.run_sync(query_string)
        else:
            result = self._service.run_async(query_string)
        return result.to_qtable()

    @staticmethod
    def _get_fields_info(table) -> QTableHeader:
        return {
            # NOTE: column.name was previously column.feature_name, after this method started to fail for GAIA data downloader
            column.name: TableColumnInfo(unit=column.unit, description=column.description)
            for column in table.columns
        }


def ExoService() -> TapService:
    return TapService("https://exoplanetarchive.ipac.caltech.edu/TAP/")


def TicService() -> TapService:
    return TapService("https://mast.stsci.edu/vo-tap/api/v0.1/tic/")


def GaiaService() -> TapService:
    return TapService("https://gea.esac.esa.int/tap-server/tap/")

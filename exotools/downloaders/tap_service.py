from typing import Iterable, Iterator, Optional

import pyvo as vo
from astropy.table import QTable, vstack
from pyvo.dal.vosi import VOSITables
from tqdm import tqdm

from exotools.utils.qtable_utils import QTableHeader, TableColumnInfo


class TapService:
    """
    TODO: try out astroquery.TapPlus, seems better than pyvo?
    Unfortunately, this service is not suitable for large queries, as it doesn't support authentication.
    Use 'casjobs' package instead for large queries
    """

    MAX_CHUNK_SIZE = 2000

    def __init__(self, url: str):
        self._url = url
        self._service = vo.dal.TAPService(self._url)

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

    def query(self, query_string: str) -> QTable:
        result = self._service.run_sync(query_string)
        return result.to_qtable()

    def query_chunks_iterative(
        self,
        select_fields: str | Iterable[str],
        from_clause: str,
        where_clause: str,
        order_by: str,
        limit: Optional[int] = None,
    ) -> Iterator[QTable]:
        if not limit:
            # get total number of records
            count_query = f"select count(*) {from_clause} {where_clause}"
            total_records = self.query(count_query).values()
        else:
            total_records = limit

        fields = select_fields if isinstance(select_fields, str) else ",".join(select_fields)
        query = f"select top {self.MAX_CHUNK_SIZE} {fields} {from_clause} {where_clause} order by {order_by} asc"
        query += " offset {offset}"

        total_chunks = (total_records + self.MAX_CHUNK_SIZE - 1) // self.MAX_CHUNK_SIZE
        for offset in tqdm(range(0, total_records, self.MAX_CHUNK_SIZE), total=total_chunks, desc="Downloading chunks"):
            yield self.query(query.format(offset=offset))

    def query_chunks(
        self,
        select_fields: str | Iterable[str],
        from_clause: str,
        where_clause: str,
        order_by: str,
        limit: Optional[int] = None,
    ) -> QTable:
        iterator = self.query_chunks_iterative(select_fields, from_clause, where_clause, order_by, limit)
        return vstack(list(iterator))

    @staticmethod
    def _get_fields_info(table) -> QTableHeader:
        return {
            # NOTE: column.name was previously column.feature_name, after this method started to fail for GAIA data downloader
            column.name: TableColumnInfo(
                unit=column.unit,
                description=column.description,
                dtype=column.datatype.content,
            )
            for column in table.columns
        }


def ExoService() -> TapService:
    return TapService("https://exoplanetarchive.ipac.caltech.edu/TAP/")


def TicService() -> TapService:
    return TapService("https://mast.stsci.edu/vo-tap/api/v0.1/tic/")


def GaiaService() -> TapService:
    return TapService("https://gea.esac.esa.int/tap-server/tap/")

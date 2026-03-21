import logging
import time
from functools import cache
from typing import Iterable, Iterator, Optional

import pyvo as vo
from astropy.table import QTable, vstack
from pyvo.dal.vosi import VOSITables
from tqdm import tqdm

from exotools.utils.qtable_utils import QTableHeader, TableColumnInfo

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF = 5


class TapService:
    """
    TODO: try out astroquery.TapPlus, seems better than pyvo?
    Unfortunately, this service is not suitable for large queries, as it doesn't support authentication.
    Use 'casjobs' package instead for large queries
    """

    MAX_CHUNK_SIZE = 2000

    def __init__(self, url: str, max_retries: int = _DEFAULT_MAX_RETRIES, retry_backoff: float = _DEFAULT_RETRY_BACKOFF):
        self._url = url
        self._service = vo.dal.TAPService(self._url)
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    @property
    def url(self) -> str:
        return self._url

    @cache
    def _get_tables(self) -> VOSITables:
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._service.tables
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    wait = self._retry_backoff * attempt
                    logger.warning(
                        "Failed to fetch table metadata from %s (attempt %d/%d): %s. Retrying in %.0fs...",
                        self._url, attempt, self._max_retries, e, wait,
                    )
                    time.sleep(wait)
        raise ConnectionError(
            f"Failed to fetch table metadata from {self._url} after {self._max_retries} attempts"
        ) from last_error

    def get_table_names(self) -> list[str]:
        return [t.name for t in self._get_tables()]

    def get_table_schemas(self) -> dict[str, QTableHeader]:
        return {name: self.get_field_info(name) for name in self.get_table_names()}

    def get_field_info(self, table_name: str) -> QTableHeader:
        tables = self._get_tables()
        if table_name not in tables:
            raise KeyError(f"{table_name} not in TapService {self._url}")
        return self._get_fields_info(tables[table_name])

    def get_field_units(self, table_name: str) -> dict[str, str]:
        tables = self._get_tables()
        if table_name not in tables:
            raise KeyError(f"{table_name} not in TapService {self._url}")
        return {c.name: c.unit for c in tables[table_name].columns}

    def get_field_descriptions(self, table_name: str) -> dict[str, str]:
        tables = self._get_tables()
        if table_name not in tables:
            raise KeyError(f"{table_name} not in TapService {self._url}")
        return {c.name: c.description for c in tables[table_name].columns}

    def get_field_names(self, table_name: str) -> list[str]:
        tables = self._get_tables()
        if table_name not in tables:
            raise KeyError(f"{table_name} not in TapService {self._url}")
        return [c.name for c in tables[table_name].columns]

    def query(self, query_string: str, timeout: float = 300.0) -> QTable:
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._service.run_async(query_string, timeout=timeout)
                return result.to_qtable()
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    wait = self._retry_backoff * attempt
                    logger.warning(
                        "Query to %s failed (attempt %d/%d): %s. Retrying in %.0fs...",
                        self._url, attempt, self._max_retries, e, wait,
                    )
                    time.sleep(wait)
        raise ConnectionError(
            f"Query to {self._url} failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

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

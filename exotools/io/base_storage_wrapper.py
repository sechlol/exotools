import abc
from pathlib import Path
from typing import Optional

from astropy.table import QTable

from exotools.utils.qtable_utils import QTableHeader


class BaseStorage(abc.ABC):
    @abc.abstractmethod
    def root_path(self) -> Path:
        pass

    @abc.abstractmethod
    def write_json(self, data: dict, name: str, override: bool = False):
        pass

    @abc.abstractmethod
    def read_json(self, name: str) -> dict:
        pass

    @abc.abstractmethod
    def read_qtable(self, table_name: str) -> Optional[QTable]:
        pass

    @abc.abstractmethod
    def write_qtable(self, table: QTable, header: QTableHeader, table_name: str, override: bool = False):
        pass

    @abc.abstractmethod
    def read_qtable_header(self, table_name: str) -> Optional[QTableHeader]:
        pass

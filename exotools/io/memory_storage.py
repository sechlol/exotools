import copy
from pathlib import Path
from typing import Optional

from astropy.table import QTable

from exotools.io import BaseStorage
from exotools.utils.qtable_utils import QTableHeader


class MemoryStorage(BaseStorage):
    _memory = {}

    def __init__(self, name: str = "default"):
        self._mem_key = Path(f"/memory/{name}")

    def root_path(self) -> Path:
        return self._mem_key

    def _get_prefixed_key(self, name: str, suffix: str) -> Path:
        """Create a key with instance name prefix and specified suffix"""
        return (self._mem_key / name).with_suffix(suffix)

    def write_json(self, data: dict, name: str, override: bool = False):
        """
        Store JSON data directly in memory without serialization.

        Args:
            data: Dictionary data to store
            name: Path-like name to use as key
            override: Whether to override existing data
        """
        key = self._get_prefixed_key(name, ".json")

        if key in self._memory and not override:
            raise ValueError(f"Data with key '{key}' already exists. Use override=True to overwrite.")

        # Store a deep copy of the data to prevent external modifications
        self._memory[key] = copy.deepcopy(data)

    def read_json(self, name: str) -> dict:
        """
        Read JSON data from memory.

        Args:
            name: Path-like name used as key

        Returns:
            Dictionary data
        """
        key = self._get_prefixed_key(name, ".json")

        if key not in self._memory:
            raise ValueError(f"Data with key '{key}' does not exist.")

        # Return a deep copy to prevent external modifications
        return copy.deepcopy(self._memory[key])

    def read_qtable(self, table_name: str) -> Optional[QTable]:
        """
        Read QTable from memory.

        Args:
            table_name: Path-like name used as key

        Returns:
            QTable object or None if not found
        """
        data_key = self._get_prefixed_key(table_name, ".qtable")

        if data_key not in self._memory:
            raise ValueError(f"QTable with key '{data_key}' does not exist.")

        # Return a copy of the stored QTable
        return self._memory[data_key].copy()

    def write_qtable(self, table: QTable, header: QTableHeader, table_name: str, override: bool = False):
        """
        Store QTable and its header in memory.

        Args:
            table: QTable to store
            header: QTable header metadata
            table_name: Path-like name to use as key
            override: Whether to override existing data
        """
        data_key = self._get_prefixed_key(table_name, ".qtable")
        header_key = self._get_prefixed_key(f"{table_name}_header", ".json")

        if data_key in self._memory and not override:
            raise ValueError(f"QTable with key '{data_key}' already exists. Use override=True to overwrite.")

        # Store copies of both table and header
        self._memory[data_key] = table.copy()
        self._memory[header_key] = copy.deepcopy(header)

    def read_qtable_header(self, table_name: str) -> Optional[QTableHeader]:
        """
        Read QTable header from memory.

        Args:
            table_name: Path-like name used as key

        Returns:
            QTableHeader object or None if not found
        """
        header_key = self._get_prefixed_key(f"{table_name}_header", ".json")

        if header_key not in self._memory:
            return None

        # Return a deep copy of the stored header
        return copy.deepcopy(self._memory[header_key])

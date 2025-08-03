import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5
from astropy.table import QTable

from exotools.io import BaseStorage
from exotools.utils.qtable_utils import RootQTableHeader, QTableHeader


class Hdf5Storage(BaseStorage):
    _JSON_GROUP = "json"
    _TABLE_GROUP = "table"
    _HEADER_GROUP = "header"

    def __init__(self, hdf5_file_path: Path, root_group: str = ""):
        self._hdf5_path = hdf5_file_path
        self._root_group_path = root_group

    def root_path(self) -> Path:
        return self._hdf5_path

    def _table_group(self, name: str) -> str:
        return f"{self._root_group_path}/{name}/{self._TABLE_GROUP}"

    def _header_group(self, name: str) -> str:
        return f"{self._root_group_path}/{name}/{self._HEADER_GROUP}"

    def _json_group(self, with_name: Optional[str] = None) -> str:
        return f"{self._root_group_path}/{self._JSON_GROUP}" + (f"/{with_name}" if with_name is not None else "")

    def write_json(self, data: dict, name: str, override: bool = False):
        """Writes a dictionary to the HDF5 file at /group_path/name"""
        group_path = self._json_group()

        with h5py.File(self._hdf5_path, "a") as f:
            if group_path in f:
                if override:
                    print("WARNING: overwriting a dataset will not free memory on disk")
                    del f[group_path]
                else:
                    raise ValueError(
                        f"Json file {name} already exists in {self._hdf5_path}. Use override=True to overwrite."
                    )

            group = f.require_group(group_path)
            group.create_dataset(name, data=json.dumps(data))

    def read_json(self, name: str) -> dict:
        """Reads a dictionary from the HDF5 file at /group_path/name"""
        with h5py.File(self._hdf5_path, "r") as f:
            json_str = f[self._json_group(with_name=name)][()].decode("utf-8")
            return json.loads(json_str)

    def read_qtable_header(self, table_name: str) -> Optional[QTableHeader]:
        header_path = self._header_group(table_name)
        with h5py.File(self._hdf5_path, "r") as f:
            json_str = f[header_path][()].decode("utf-8")
            return RootQTableHeader.model_validate_json(json_str).root

    def write_qtable(self, table: QTable, header: QTableHeader, table_name: str, override: bool = False):
        table_path = self._table_group(table_name)
        header_path = self._header_group(table_name)
        parent_path = f"{self._root_group_path}/{table_name}"

        # Preprocess the table to handle object dtypes
        processed_table = _preprocess_table_for_hdf5(table)

        with h5py.File(self._hdf5_path, "a") as f:
            # Check if table exists
            if parent_path in f:
                if override:
                    print("WARNING: overwriting a dataset will not free memory on disk")
                    # Delete the entire group containing the table and header
                    del f[parent_path]
                else:
                    raise ValueError(
                        f"Table {table_name} already exists in {self._hdf5_path.name}. Use override=True to overwrite."
                    )

            # Create parent group if needed
            f.require_group(parent_path)

            # Create header dataset
            f.create_dataset(header_path, data=RootQTableHeader(root=header).model_dump_json())

            # Write table data
            write_table_hdf5(
                table=processed_table,
                output=f,
                path=table_path,
                compression=7,
                append=True,
                overwrite=True,  # Always use overwrite=True since we've already handled the override logic
                serialize_meta=True,
            )

    def read_qtable(self, table_name: str) -> Optional[QTable]:
        table_path = self._table_group(table_name)
        header_path = self._header_group(table_name)

        with h5py.File(self._hdf5_path, "r") as f:
            if table_path not in f:
                raise ValueError(f"read_qtable(): given path does not exist: {table_path}")

            # Read the table from HDF5
            table = read_table_hdf5(input=f, path=table_path)

            # Convert byte strings to Unicode strings
            for col_name in table.colnames:
                col = table[col_name]
                if col.dtype.kind == "S":  # If it's a byte string
                    # Convert to Unicode string
                    table[col_name] = [s.decode("utf-8") if isinstance(s, bytes) else s for s in col]

            # Read header to get additional metadata
            if header_path in f:
                header_json = f[header_path][()].decode("utf-8")
                header = RootQTableHeader.model_validate_json(header_json).root

                # Apply column descriptions from header
                for col_name, col_info in header.items():
                    if col_name in table.colnames and col_info.description:
                        table[col_name].description = col_info.description

            return table


def _preprocess_table_for_hdf5(table: QTable) -> QTable:
    """
    Preprocess a QTable to make it compatible with HDF5 storage.
    Handles object dtypes by converting them to serializable formats.

    Args:
        table: The QTable to preprocess

    Returns:
        A new QTable with HDF5-compatible dtypes
    """
    # Create a copy to avoid modifying the original
    processed_table = table.copy()

    # Process each column
    for col_name in processed_table.colnames:
        col = processed_table[col_name]

        # Handle object dtypes
        if col.dtype == np.dtype("O"):
            # Convert to string if it contains string-like objects
            if all(isinstance(x, (str, bytes, type(None))) for x in col):
                processed_table[col_name] = [str(x) if x is not None else "" for x in col]
            else:
                # For other object types, convert to JSON strings
                try:
                    processed_table[col_name] = [json.dumps(x) if x is not None else "" for x in col]
                except (TypeError, ValueError):
                    # If JSON serialization fails, convert to string representation
                    processed_table[col_name] = [str(x) if x is not None else "" for x in col]

    return processed_table

import json
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5
from astropy.table import QTable

from exotools.utils.qtable_utils import RootQTableHeader, QTableHeader
from .base_storage_wrapper import StorageWrapper


class Hdf5Wrapper(StorageWrapper):
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
        return self._root_group_path + "/" + self._TABLE_GROUP + (f"/{with_name}" or "")

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

        with h5py.File(self._hdf5_path, "a") as f:
            if table_path in f:
                if override:
                    print("WARNING: overwriting a dataset will not free memory on disk")
                    del f[table_path]
                    del f[header_path]
                else:
                    raise ValueError(
                        f"Table {table_name} already exists in {self._hdf5_path.name}. Use override=True to overwrite."
                    )

            f.create_dataset(header_path, data=RootQTableHeader(root=header).model_dump_json())

            write_table_hdf5(
                table=table,
                output=f,
                path=table_path,
                compression=7,
                append=True,
                overwrite=override,
                serialize_meta=True,
            )

    def read_qtable(self, table_name: str) -> Optional[QTable]:
        table_path = self._table_group(table_name)

        with h5py.File(self._hdf5_path, "r") as f:
            if table_path not in f:
                raise ValueError(f"read_qtable(): given path does not exist: {table_path}")

            return read_table_hdf5(input=f, path=table_path)


def _convert_nullable_ints_to_float(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Convert nullable integer columns to float, due to a bug in pandas not being able to write them to hdf5
    https://github.com/pandas-dev/pandas/issues/26144
    """
    converted_columns = {}

    for col_name in df.columns:
        dtype = df[col_name].dtype

        # Check if column has nullable integer dtype
        if pd.api.types.is_extension_array_dtype(dtype) and "Int" in dtype.name:
            if dtype.name == "Int64":
                df[col_name] = df[col_name].astype("float64")
            else:
                df[col_name] = df[col_name].astype("float32")

            converted_columns[col_name] = dtype.name
    return df, converted_columns


def _convert_floats_to_nullable_int(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    for col_name, original_dtype in columns.items():
        df[col_name] = df[col_name].astype(original_dtype)
    return df

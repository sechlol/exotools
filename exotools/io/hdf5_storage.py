import json
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
from astropy.table import QTable

from exotools.utils.qtable_utils import RootQTableHeader, QTableHeader


class Hdf5Wrapper:
    _JSON_GROUP = "json"
    _TABLE_GROUP = "table"
    _HEADER_GROUP = "header"

    def __init__(self, hdf5_file_path: Path, root_group: str = "/"):
        self._hdf5_path = hdf5_file_path
        self._root_group_path = root_group

    def write_json(self, data: dict, name: str):
        """Writes a dictionary to the HDF5 file at /group_path/name"""

        with h5py.File(self._hdf5_path, "a") as f:
            group = f.require_group(self._root_group_path)
            if name in group:
                raise ValueError(
                    f"Dataset {group}/{name} already exists in {self._hdf5_path.name}. "
                    f"Use override=True to overwrite."
                )

            group.create_dataset(name, data=json.dumps(data))

    def read_json(self, name: str) -> dict:
        """Reads a dictionary from the HDF5 file at /group_path/name"""
        with h5py.File(self._hdf5_path, "r") as f:
            json_str = f[self._root_group_path + "/" + name][()].decode("utf-8")
            return json.loads(json_str)

    def get_structure(self, depth: int = 2) -> list[str]:
        """Reads all the high level groups and keys, at most 2 levels deep"""
        if not self._hdf5_path.exists():
            return []

        structure = []
        try:
            with h5py.File(self._hdf5_path, "r") as f:

                def visit_func(name, obj):
                    # Scan up to the desired depth
                    if name.count("/") <= depth:
                        if isinstance(obj, h5py.Group):
                            structure.append(f"GROUP: {name}")
                        elif isinstance(obj, h5py.Dataset):
                            structure.append(f"DATASET: {name}")

                f.visititems(visit_func)
        except (OSError, KeyError):
            # File doesn't exist or is corrupted
            pass
        return structure

    def read_qtable(self, table_name: str) -> Optional[QTable]:
        header = self.read_json(table_name + "_header")
        header = RootQTableHeader.model_validate(header).root
        units = {key: info.unit for key, info in header.items()} if header else None
        df = pd.read_hdf(self._hdf5_path, table_name)
        return QTable.from_pandas(df, units=units)

    def write_qtable(self, table: QTable, header: QTableHeader, table_name: str):
        data_path = f"{self._root_group_path}/{table_name}"
        table_path = f"{data_path}/{self._TABLE_GROUP}"
        header_path = f"{data_path}/{self._HEADER_GROUP}"

        with h5py.File(self._hdf5_path, "a") as f:
            if data_path in f:
                raise ValueError(f"Table {table_name} already exists in {self._hdf5_path.name}.")

            # Store table header
            root_model = RootQTableHeader(root=header)
            f.create_dataset(header_path, data=root_model.model_dump_json())

            # Convert to Pandas
            df = table.to_pandas().reset_index()
            if "index" in df:
                df = df.drop(columns="index")

            df, nullable_columns = _convert_nullable_ints_to_float(df)
            #
            # for col_name, dtype_name in nullable_columns.items():
            #     f[f"{header_path}/nullables/{col_name}"] = dtype_name

        df.to_hdf(self._hdf5_path, key=table_path, complevel=9, format="table")


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

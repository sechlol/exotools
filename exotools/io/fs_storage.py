import abc
import json
from abc import ABC
from pathlib import Path
from typing import Optional

import pandas as pd
from astropy import units as u
from astropy.table import MaskedColumn, QTable

from exotools.utils.qtable_utils import QTableHeader, RootQTableHeader
from exotools.utils.unit_mapper import UNIT_MAPPER

from .base_storage import BaseStorage


class FsStorage(BaseStorage, ABC):
    def __init__(self, root_path: Path | str):
        self._root_path = Path(root_path)

    @property
    def root_path(self) -> Path:
        return self._root_path

    @property
    @abc.abstractmethod
    def _suffix(self) -> str:
        pass

    @abc.abstractmethod
    def _save_qtable(self, table: QTable, table_path: Path, override: bool):
        pass

    @abc.abstractmethod
    def _read_qtable(self, table_path: Path, header_path: Path) -> QTable:
        pass

    def write_json(self, data: dict, name: str, override: bool = False):
        data_path = (self._root_path / name).with_suffix(".json")
        if data_path.exists() and not override:
            raise ValueError(f"File {name} already exists in {self._root_path}. Use override=True to overwrite.")

        with open(data_path, "w") as f:
            f.write(json.dumps(data))

    def read_json(self, name: str) -> dict:
        data_path = (self._root_path / name).with_suffix(".json")

        if not data_path.exists():
            raise ValueError(f"File {name} does not exist in {self._root_path}.")
        with open(data_path, "r") as f:
            return json.load(f)

    def write_qtable(self, table: QTable, header: QTableHeader, table_name: str, override: bool = False):
        data_path, header_path = _get_qtable_paths(file_path=self._root_path, suffix=self._suffix, file_name=table_name)

        if data_path.exists() and not override:
            raise ValueError(f"File {table_name} already exists in {self._root_path}. Use override=True to overwrite.")

        data_path.parent.mkdir(parents=True, exist_ok=True)

        _write_qtable_header(header_path, header)
        self._save_qtable(table, data_path, override)

    def read_qtable(self, table_name: str) -> Optional[QTable]:
        table_path, header_path = _get_qtable_paths(
            file_path=self._root_path,
            suffix=self._suffix,
            file_name=table_name,
        )

        # Read data and assign units
        if not table_path.exists():
            raise ValueError(f"read_qtable(): given path does not exist: {table_path}")

        return self._read_qtable(table_path, header_path)

    def read_qtable_header(self, table_name: str) -> Optional[QTableHeader]:
        _, header_path = _get_qtable_paths(
            file_path=self._root_path,
            suffix=self._suffix,
            file_name=table_name,
        )
        return _read_qtable_header(header_path)


class FeatherStorage(FsStorage):
    @property
    def _suffix(self) -> str:
        return ".feather"

    def _read_qtable(self, table_path: Path, header_path: Path) -> QTable:
        # Load raw DataFrame without unit conversion
        df = pd.read_feather(table_path)
        table = QTable.from_pandas(df)
        col_names = table.colnames

        # Load header metadata
        header = _read_qtable_header(header_path)
        if not header:
            return table

        # Assign units after construction
        for col, info in header.items():
            if col not in col_names or info.unit is None:
                continue

            unit = UNIT_MAPPER.get(info.unit, u.Unit(info.unit))
            if "dex" in info.unit:
                table[col] = table[col].value * unit
            else:
                table[col].unit = unit

        return table

    def _save_qtable(self, table: QTable, table_path: Path, override: bool):
        # Store table data in feather format
        df = table.to_pandas().reset_index()
        if "index" in df:
            df = df.drop(columns="index")
        df.to_feather(table_path)


class EcsvStorage(FsStorage):
    @property
    def _suffix(self) -> str:
        return ".ecsv"

    def _save_qtable(self, table: QTable, table_path: Path, override: bool):
        table.write(table_path, overwrite=override, serialize_method={MaskedColumn: "data_mask"})

    def _read_qtable(self, table_path: Path, header_path: Path) -> QTable:
        return QTable.read(table_path)


def _get_qtable_paths(file_path: Path, suffix: str, file_name: Optional[str]) -> tuple[Path, Path]:
    file_name = file_name or (file_path.stem if file_path.suffix != "" else "data")
    folder_path = file_path if file_name or file_path.suffix != "" else file_path.parent
    header_path = (folder_path / f"{file_name}_header").with_suffix(".json")
    table_path = (folder_path / f"{file_name}").with_suffix(suffix)
    return table_path, header_path


def _write_qtable_header(header_path: Path, header: QTableHeader):
    if header is not None:
        root_model = RootQTableHeader(root=header)

        # Store table unit info in json format
        with open(header_path, "w") as f:
            f.write(root_model.model_dump_json(indent=4))


def _read_qtable_header(header_path: Path) -> Optional[QTableHeader]:
    if not header_path.exists():
        return None

    with open(header_path, "r") as file:
        return RootQTableHeader.model_validate_json(file.read()).root

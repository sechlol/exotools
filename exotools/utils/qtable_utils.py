from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pydantic
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity, UnrecognizedUnit


class TimeInfo(pydantic.BaseModel):
    format: str
    scale: str


class TableColumnInfo(pydantic.BaseModel):
    description: Optional[str] = None
    unit: Optional[str] = None
    dtype: Optional[str] = None
    time_info: Optional[TimeInfo] = None


QTableHeader = dict[str, TableColumnInfo]


class RootQTableHeader(pydantic.RootModel):
    root: QTableHeader


def get_empty_table_header(table: QTable) -> QTableHeader:
    header = {}
    for column_name in table.columns:
        header[column_name] = TableColumnInfo(
            unit=table[column_name].unit.name if isinstance(table[column_name].unit, u.Unit) else None,
            description="---",
        )
    return header


def save_qtable(
    table: QTable,
    header: Optional[QTableHeader],
    file_path: Path,
    file_name: Optional[str] = None,
) -> Path:
    data_path, header_path = _get_qtable_paths(file_path, file_name)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if header is not None:
        root_model = RootQTableHeader(root=header)

        # Store table unit info in json format
        with open(header_path, "w") as f:
            f.write(root_model.model_dump_json(indent=4))

    # Store table data in feather format
    df = table.to_pandas().reset_index()
    if "index" in df:
        df = df.drop(columns="index")
    df.to_feather(data_path)

    return data_path


def get_header_from_table(table: QTable) -> QTableHeader:
    header = {}
    for col, col_name in zip(table.itercols(), table.colnames):
        try:
            header[col_name] = TableColumnInfo(
                description=col.description,
                unit=str(col.unit) if col.unit else None,
                dtype=col.dtype.name,
            )
        except AttributeError as e:
            if isinstance(col, Quantity):
                header[col_name] = TableColumnInfo(unit=str(col.unit), dtype=col.dtype.name)
            elif isinstance(col, Time):
                header[col_name] = TableColumnInfo(time_info=TimeInfo(format=col.format, scale=col.scale))
            else:
                raise e
    return header


def read_qtable(file_path: Path, file_name: Optional[str] = None) -> QTable:
    data_path, header_path = _get_qtable_paths(file_path=file_path, file_name=file_name)

    # Read header information with column units
    header = _read_qtable_header(header_path)
    units = {key: info.unit for key, info in header.items()} if header else None

    # Read data and assign units
    if not data_path.exists():
        raise ValueError(f"read_qtable(): given path does not exist: {data_path}")

    df = pd.read_feather(data_path)
    return QTable.from_pandas(df, units=units)


def read_qtable_header(file_path: Path, file_name: Optional[str] = None) -> Optional[QTableHeader]:
    _, header_path = _get_qtable_paths(file_path=file_path, file_name=file_name)
    return _read_qtable_header(header_path)


def _get_qtable_paths(file_path: Path, file_name: Optional[str]) -> tuple[Path, Path]:
    file_name = file_name or (file_path.stem if file_path.suffix != "" else "data")
    folder_path = file_path if file_name or file_path.suffix != "" else file_path.parent
    header_path = folder_path / f"{file_name}_header.json"
    data_path = folder_path / f"{file_name}.feather"
    return data_path, header_path


def _write_header(file_path: Path, comment: Optional[str] = None):
    header = f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    if comment:
        header += "# " + comment.replace("\n", "\n# ") + "\n"

    with open(file_path, "w") as f:
        f.write(header)


def _read_qtable_header(header_path: Path) -> Optional[QTableHeader]:
    if not header_path.exists():
        return None

    with open(header_path, "r") as file:
        return RootQTableHeader.model_validate_json(file.read()).root


def fix_unrecognized_units(table: QTable, units_map: dict[str, u.Unit]):
    """
    Fix incorrect units that cannot be parsed from the queried table.
    """
    # Assign unrecognized units
    for c in table.colnames:
        unit = table[c].unit
        if isinstance(unit, UnrecognizedUnit) and unit.name in units_map:
            table[c] = table[c].value * units_map[unit.name]


def override_units(table: QTable, unit_overrides: dict[str, u.Unit]):
    """
    Override units that are mistakenly labelled in the source table
    """
    for c, unit in unit_overrides.items():
        table[c] = table[c].value * unit

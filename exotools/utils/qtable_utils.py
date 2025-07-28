from typing import Optional

import pydantic
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity


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

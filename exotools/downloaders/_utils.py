import logging

from astropy import units as u
from astropy.table import QTable
from astropy.units import UnrecognizedUnit

logger = logging.getLogger("downloaders")


def fix_unrecognized_units(table: QTable, units_map: dict[str, u.Unit]):
    """
    Fix incorrect units that cannot be parsed from the queried table.
    """
    # Assign unrecognized units
    for c in table.colnames:
        unit = table[c].unit
        if isinstance(unit, UnrecognizedUnit) and unit.name in units_map:
            try:
                table[c] = table[c].value * units_map[unit.name]
            except Exception as e:
                logger.warning(f"Failed to set unit for column {c}: {e}")
                continue


def override_units(table: QTable, unit_overrides: dict[str, u.Unit]):
    """
    Override units that are mistakenly labelled in the source table
    """
    table_columns = set(table.colnames)
    for c, unit in unit_overrides.items():
        if c in table_columns:
            table[c] = table[c].value * unit

from astropy import units as u
from astropy.table import QTable
from astropy.units import UnrecognizedUnit


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

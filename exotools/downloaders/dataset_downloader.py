from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, Iterator, Sequence

from astropy import units as u
from astropy.table import QTable
from astropy.units import UnrecognizedUnit

from exotools.utils.qtable_utils import QTableHeader, save_qtable

_DEFAULT_FILENAME = "dataset"


class DatasetDownloader(ABC):
    @abstractmethod
    def _download(self, limit: Optional[int] = None) -> QTable:
        """
        Download data from the dataset.

        Args:
            limit (Optional[int]): Maximum number of rows to download.

        Returns:
            QTable: Downloaded data as a QTable.
        """
        pass

    @abstractmethod
    def _download_by_id(self, ids: Sequence[int]) -> QTable:
        """
        Download data from the dataset, filtering by certain IDs.

        Returns:
            QTable: Downloaded data as a QTable.
        """
        pass

    @abstractmethod
    def _clean_and_fix(self, table: QTable) -> QTable:
        """
        Performs necessary cleanup and fixes to the downloaded data
        """
        pass

    @abstractmethod
    def _get_table_header(self, table: QTable) -> QTableHeader:
        """
        creates a header with units and description of each field in the table, to be persisted in a file
        """
        pass

    def download(
        self,
        limit: Optional[int] = None,
        out_folder_path: Optional[Path | str] = None,
        out_file_name: Optional[str] = None,
    ) -> QTable:
        """
        Download data from the dataset and store it.

        Args:
            limit: Maximum number of rows to download.
            out_folder_path: if given, persists the dataset to the specified folder, overriding existing ones
            out_file_name: if given together with out_folder_path, store the file with the given name

        Returns:
            QTable: Downloaded data as a QTable.
        """
        # Check preconditions and create parent folders if provided
        if out_folder_path:
            if not out_folder_path.is_dir():
                raise ValueError(f"The provided path must be a directory. Given: {out_folder_path}")
            out_folder_path.mkdir(parents=True, exist_ok=True)
            out_file_name = out_file_name or _DEFAULT_FILENAME

        if out_file_name and not out_folder_path:
            raise ValueError("If out_file_name is provided, out_folder_path must also be provided.")

        # Download
        raw_data = self._download(limit=limit)

        # Fix table units
        cleaned_table = self._clean_and_fix(raw_data)

        # Fetch metadata
        table_header = self._get_table_header(cleaned_table)

        # Store
        if out_folder_path:
            out_folder_path = Path(out_folder_path)
            out_file_name = out_file_name or _DEFAULT_FILENAME
            save_qtable(table=cleaned_table, header=table_header, file_path=out_folder_path, file_name=out_file_name)

        return cleaned_table

    def download_by_id(
        self,
        ids: Sequence[int],
        out_folder_path: Optional[Path | str] = None,
        out_file_name: Optional[str] = None,
    ) -> QTable:
        """
        Download data from the dataset by selecting fields by ID.

        Args:
            ids: list of IDs to download
            out_folder_path: if given, persists the dataset to the specified folder, overriding existing ones
            out_file_name: if given together with out_folder_path, store the file with the given name

        Returns:
            QTable: Downloaded data as a QTable.
        """
        # Download
        raw_data = self._download_by_id(ids=ids)

        # Fix table units
        cleaned_table = self._clean_and_fix(raw_data)

        # Store
        table_header = self._get_table_header(cleaned_table)
        if out_folder_path:
            out_folder_path = Path(out_folder_path)
            out_file_name = out_file_name or _DEFAULT_FILENAME
            save_qtable(table=cleaned_table, header=table_header, file_path=out_folder_path, file_name=out_file_name)
        return cleaned_table


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


def iterate_chunks(ids: Sequence, chunk_size: int) -> Iterator[Sequence]:
    """Yield successive chunks of ids from the list."""
    for i in range(0, len(ids), chunk_size):
        yield ids[i : i + chunk_size]

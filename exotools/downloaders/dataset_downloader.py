from abc import abstractmethod, ABC
from typing import Optional, Iterator, Sequence

from astropy.table import QTable

from exotools.utils.qtable_utils import QTableHeader

_DEFAULT_FILENAME = "dataset"


class DatasetDownloader(ABC):
    @abstractmethod
    def _download(self, limit: Optional[int] = None, **kwargs) -> QTable:
        """
        Download data from the dataset.

        Args:
            limit (Optional[int]): Maximum number of rows to download.

        Returns:
            QTable: Downloaded data as a QTable.
        """
        pass

    @abstractmethod
    def _download_by_id(self, ids: Sequence[int], **kwargs) -> QTable:
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

    def download(self, limit: Optional[int] = None, **kwargs) -> tuple[QTable, QTableHeader]:
        """
        Download data from the dataset and store it.

        Args:
            limit: Maximum number of rows to download.

        Returns:
            QTable: Downloaded data as a QTable.
            QTableHeader: Table header with info on data types, units and field descriptions
        """

        # Download
        raw_data = self._download(limit=limit, kwargs=kwargs)

        # Fix table units
        cleaned_table = self._clean_and_fix(raw_data)

        # Fetch metadata
        table_header = self._get_table_header(cleaned_table)

        return cleaned_table, table_header

    def download_by_id(self, ids: Sequence[int], **kwargs) -> tuple[QTable, QTableHeader]:
        """
        Download data from the dataset by selecting fields by ID.

        Args:
            ids: list of IDs to download


        Returns:
            QTable: Downloaded data as a QTable.
        """
        # Download
        raw_data = self._download_by_id(ids=ids, kwargs=kwargs)

        # Fix table units
        cleaned_table = self._clean_and_fix(raw_data)

        # Fetch metadata
        table_header = self._get_table_header(cleaned_table)

        return cleaned_table, table_header


def iterate_chunks(ids: Sequence, chunk_size: int) -> Iterator[Sequence]:
    """Yield successive chunks of ids from the list."""
    for i in range(0, len(ids), chunk_size):
        yield ids[i : i + chunk_size]

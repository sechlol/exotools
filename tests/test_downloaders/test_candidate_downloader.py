from unittest.mock import MagicMock, patch

import pytest
from astropy.table import QTable

from exotools.downloaders import CandidateExoplanetsDownloader
from exotools.downloaders.exoplanets_downloader import _get_error_parameters
from exotools.utils.qtable_utils import QTableHeader


class TestCandidateDownloader:
    @pytest.fixture
    def columns_with_units_to_fix(self) -> list[str]:
        return _get_error_parameters(["pl_trandep"], True)

    @pytest.fixture
    def tap_qtable_data(
        self,
        candidate_exoplanets_test_data: tuple[QTable, QTableHeader],
        columns_with_units_to_fix: list[str],
    ):
        in_table = candidate_exoplanets_test_data[0].copy()
        # Modify the table "tic_id" column name to "tid" to match the raw data from the service
        in_table.rename_column("tic_id", "tid")

        # Remove units from columns containing "pl_trandep" to simulate raw data
        for col_name in in_table.colnames:
            if col_name in columns_with_units_to_fix:
                if hasattr(in_table[col_name], "unit") and in_table[col_name].unit is not None:
                    # Store the original data but remove the unit
                    data = in_table[col_name].data
                    in_table[col_name] = data

        return in_table

    @pytest.fixture
    def descriptions(self, candidate_exoplanets_test_data: tuple[QTable, QTableHeader]):
        header = candidate_exoplanets_test_data[1]
        return {name: info.description for name, info in header.items()}

    def test_download_smoketest(
        self,
        tap_qtable_data: QTable,
        descriptions: dict[str, str],
        columns_with_units_to_fix: list[str],
    ):
        # Mock ExoService .query() to return our modified table
        with patch("exotools.downloaders.candidate_exoplanets_downloader.ExoService") as mock_exo_service_factory:
            mock_exo_service = MagicMock()
            mock_exo_service.query.return_value = tap_qtable_data
            # Mock the get_field_descriptions method to return our descriptions
            mock_exo_service.get_field_descriptions.return_value = descriptions
            mock_exo_service_factory.return_value = mock_exo_service

            # Create downloader and call download
            downloader = CandidateExoplanetsDownloader()
            result, _ = downloader.download()  # Unpack the tuple

            # Assert that query was called with "from toi" table
            mock_exo_service.query.assert_called_once()
            query_str = mock_exo_service.query.call_args[0][0]
            assert "from toi" in query_str

            # Assert that get_field_descriptions was called with the correct table name
            mock_exo_service.get_field_descriptions.assert_called_once_with("toi")

            # Verify the result
            assert result is not None
            assert len(result) > 0

            # Assert that column "tid" was renamed to "tic_id"
            assert "tic_id" in result.colnames
            assert "tid" not in result.colnames

            # Assert that pl_trandep units are restored
            for col_name in result.colnames:
                if col_name in columns_with_units_to_fix:
                    assert hasattr(result[col_name], "unit")
                    assert result[col_name].unit is not None

    def test_with_limit(self, tap_qtable_data: QTable, descriptions: dict[str, str]):
        with patch("exotools.downloaders.candidate_exoplanets_downloader.ExoService") as mock_exo_service_factory:
            # Create a mock QTable with required columns
            limit = 10
            reduced_qtable = tap_qtable_data[:limit]
            mock_exo_service = MagicMock()
            mock_exo_service.query.return_value = reduced_qtable
            mock_exo_service.get_field_descriptions.return_value = descriptions
            mock_exo_service_factory.return_value = mock_exo_service

            downloader = CandidateExoplanetsDownloader()
            result, _ = downloader.download(limit=10)  # Unpack the tuple

            # Verify limit was included in the query
            mock_exo_service.query.assert_called_once()
            query_str = mock_exo_service.query.call_args[0][0]
            assert f"top {limit}" in query_str
            assert len(result) == limit

    def test_with_columns(self, tap_qtable_data: QTable, descriptions: dict[str, str]):
        selected_cols = tap_qtable_data.colnames[:10]
        reduced_qtable = tap_qtable_data[selected_cols]
        selected_descriptions = {name: descriptions.get(name, "test") for name in selected_cols}

        assert "tid" in selected_cols

        with patch("exotools.downloaders.candidate_exoplanets_downloader.ExoService") as mock_exo_service_factory:
            mock_exo_service = MagicMock()
            mock_exo_service.query.return_value = reduced_qtable.copy()
            mock_exo_service.get_field_descriptions.return_value = selected_descriptions
            mock_exo_service_factory.return_value = mock_exo_service

            downloader = CandidateExoplanetsDownloader()
            result, _ = downloader.download(columns=selected_cols)  # Unpack the tuple

            mock_exo_service.query.assert_called_once()
            query_str = mock_exo_service.query.call_args[0][0]

            # Check that all columns are in the query
            for col in selected_cols:
                assert col in query_str

            # Also check that tid (index column) was automatically added
            assert "tic_id" in result.colnames
            assert "tid" not in result.colnames

            assert len(result) == len(reduced_qtable)
            assert set(result.colnames).symmetric_difference(set(reduced_qtable.colnames)) == {"tic_id", "tid"}

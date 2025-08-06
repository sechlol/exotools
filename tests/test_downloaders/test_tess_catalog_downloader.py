from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import QTable

from exotools.downloaders import TessCatalogDownloader
from exotools.utils.qtable_utils import QTableHeader


class TestTessCatalogDownloader:
    @pytest.fixture
    def tess_raw_data(self, tic_catalog_test_data) -> QTable:
        """Create a modified copy of the test data to simulate raw data from TIC service"""
        # Get test data and make a copy to modify
        return tic_catalog_test_data[0].copy()

    @pytest.fixture
    def tess_ids(self) -> list[int]:
        """Return a list of TIC IDs for testing"""
        return [123456789, 234567890, 345678901]

    def test_casjob_download(self):
        """
        TODO: test_download() mocks the call to TessCatalogDownloader._query_ctl_casjob()
        but it should be tested properly at some point
        """
        pytest.skip()

    def test_download(self, tess_raw_data: QTable):
        """Test download functionality"""
        with patch.object(TessCatalogDownloader, "_query_ctl_casjob") as mock_query, patch.object(
            TessCatalogDownloader, "_get_table_header"
        ) as mock_get_header:
            # Setup the mock to return our test data
            mock_query.return_value = tess_raw_data

            # Mock the header creation to avoid KeyError
            mock_header = QTableHeader()
            mock_get_header.return_value = mock_header

            # Create downloader and call download
            downloader = TessCatalogDownloader(username="test", password="test")
            result, header = downloader.download(limit=100)

            # Assert that _query_ctl_casjob was called
            mock_query.assert_called_once()

            # Check the query string (passed as keyword argument)
            query = mock_query.call_args.kwargs.get("query", "")
            if not query:  # If not found in kwargs, check positional args
                args = mock_query.call_args.args
                if len(args) >= 2:
                    query = args[1]

            # Check that the query contains the expected elements
            assert "top 100" in query
            assert "id as tic_id" in query
            assert "gaia as gaia_id" in query
            assert "priority > " in query
            assert "mass between " in query

            # Verify the result
            assert result is not None
            assert len(result) > 0

            # Assert that column names were converted correctly
            assert "tic_id" in result.colnames
            assert "gaia_id" in result.colnames

            # Assert that gaia_id was converted to integer
            assert result["gaia_id"].dtype == np.int64

            # Verify that _get_table_header was called
            mock_get_header.assert_called_once()

    def test_download_with_custom_params(self, tess_raw_data: QTable):
        """Test download with custom star mass range and priority threshold"""
        with patch.object(TessCatalogDownloader, "_query_ctl_casjob") as mock_query, patch.object(
            TessCatalogDownloader, "_get_table_header"
        ) as mock_get_header:
            # Setup the mock to return our test data
            mock_query.return_value = tess_raw_data

            # Mock the header creation to avoid KeyError
            mock_header = QTableHeader()
            mock_get_header.return_value = mock_header

            # Create downloader with custom parameters
            downloader = TessCatalogDownloader(
                username="test", password="test", star_mass_range=(0.5, 2.0), priority_threshold=0.005
            )

            # Call download
            result, header = downloader.download()

            # Check that _query_ctl_casjob was called
            mock_query.assert_called_once()

            # Get the query from the call arguments
            query = mock_query.call_args.kwargs.get("query", "")
            if not query:  # If not found in kwargs, check positional args
                args = mock_query.call_args.args
                if len(args) >= 2:
                    query = args[1]

            # Verify custom parameters in query
            assert "priority > 0.005" in query
            assert "mass between 0.5 and 2.0" in query

            # Verify that _get_table_header was called
            mock_get_header.assert_called_once()

    def test_download_by_id(self, tess_raw_data: QTable, tess_ids: list[int]):
        """Test download_by_id functionality"""
        with patch("exotools.downloaders.tess_catalog_downloader.TicService") as mock_tic_service_factory:
            # Mock TicService
            mock_tic_service = MagicMock()
            mock_tic_service.query.return_value = tess_raw_data
            mock_tic_service_factory.return_value = mock_tic_service

            # Create downloader and call download_by_id
            downloader = TessCatalogDownloader(username="test", password="test")
            result, header = downloader.download_by_id(tess_ids)

            # Assert that TicService.query was called
            mock_tic_service.query.assert_called_once()
            query = mock_tic_service.query.call_args[1]["query_string"]

            # Check that the query contains the expected elements
            for tic_id in tess_ids:
                assert f"'{tic_id}'" in query
            assert "id as tic_id" in query
            assert "gaia as gaia_id" in query

            # Verify the result
            assert result is not None
            assert len(result) > 0

            # Assert that column names were converted correctly
            assert "tic_id" in result.colnames
            assert "gaia_id" in result.colnames

    def test_download_by_id_with_columns(self, tess_raw_data: QTable, tess_ids: list[int]):
        """Test download_by_id with extra columns parameter"""
        extra_columns = ["ra", "dec", "tmag"]

        with patch("exotools.downloaders.tess_catalog_downloader.TicService") as mock_tic_service_factory:
            # Mock TicService
            mock_tic_service = MagicMock()
            mock_tic_service.query.return_value = tess_raw_data
            mock_tic_service_factory.return_value = mock_tic_service

            # Create downloader and call download_by_id with extra columns
            downloader = TessCatalogDownloader(username="test", password="test")
            result, _ = downloader.download_by_id(tess_ids, columns=extra_columns)

            # Check that the query contains the extra columns
            query = mock_tic_service.query.call_args[1]["query_string"]
            for col in extra_columns:
                assert col in query

    def test_download_by_id_chunks(self, tess_raw_data: QTable):
        """Test download_by_id with many IDs that require chunking"""
        # Create a list of 1000 IDs (should create 3 chunks with chunk_size=400)
        many_ids = list(range(1000))

        with patch("exotools.downloaders.tess_catalog_downloader.TicService") as mock_tic_service_factory, patch(
            "exotools.downloaders.tess_catalog_downloader.vstack", return_value=tess_raw_data
        ) as mock_vstack:
            # Mock TicService
            mock_tic_service = MagicMock()
            mock_tic_service.query.return_value = tess_raw_data
            mock_tic_service_factory.return_value = mock_tic_service

            # Create downloader and call download_by_id
            downloader = TessCatalogDownloader(username="test", password="test")
            result, _ = downloader.download_by_id(many_ids)

            # Assert that TicService.query was called multiple times (once per chunk)
            assert mock_tic_service.query.call_count == 3  # 1000 IDs / 400 per chunk = 3 chunks (rounded up)

            # Assert that vstack was called to combine the results
            mock_vstack.assert_called_once()

    def test_property_setters(self):
        """Test property setters for star_mass_range and priority_threshold"""
        downloader = TessCatalogDownloader(username="test", password="test")

        # Test setting star_mass_range
        downloader.star_mass_range = (0.6, 1.5)
        assert downloader.star_mass_range == (0.6, 1.5)

        # Test setting priority_threshold
        downloader.priority_threshold = 0.01
        assert downloader.priority_threshold == 0.01

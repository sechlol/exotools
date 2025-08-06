from unittest.mock import patch

import pandas as pd
import pytest
from astropy.table import QTable

from exotools.downloaders import TessObservationsDownloader


class TestTessObservationsDownloader:
    @pytest.fixture
    def tess_observations_raw_data(self, tic_observations_test_data) -> QTable:
        """Create a modified copy of the test data to simulate raw data from MAST service"""
        # Get test data and make a copy to modify
        in_table = tic_observations_test_data[0].copy()

        # Rename columns to match raw data format if needed
        if "tic_id" in in_table.colnames:
            in_table.rename_column("tic_id", "target_name")
        if "obs_id" in in_table.colnames:
            in_table.rename_column("obs_id", "obsid")

        return in_table

    @pytest.fixture
    def tess_observations_pandas_data(self, tess_observations_raw_data: QTable) -> pd.DataFrame:
        """Convert QTable to pandas DataFrame as returned by Observations.query_criteria_columns_async"""
        # Convert to pandas DataFrame
        df = tess_observations_raw_data.to_pandas()

        # Ensure all URLs end with s_lc.fits to match the filter in _download_by_id
        if "dataURL" in df.columns:
            df["dataURL"] = df["dataURL"].astype(str) + "s_lc.fits"

        return df

    @pytest.fixture
    def tess_ids(self) -> list[int]:
        """Return a list of TIC IDs for testing"""
        return [123456789, 234567890, 345678901]

    def test_download_not_implemented(self):
        """Test that download method raises NotImplementedError"""
        downloader = TessObservationsDownloader()
        with pytest.raises(NotImplementedError):
            downloader.download()

    def test_download_by_id(self, tess_observations_pandas_data: pd.DataFrame, tess_ids: list[int]):
        """Test download_by_id functionality"""
        with patch("exotools.downloaders.tess_observations_downloader.Observations") as mock_observations:
            # Mock Observations.query_criteria_columns_async to return our test data
            mock_observations.query_criteria_columns_async.return_value = tess_observations_pandas_data

            # Create downloader and call download_by_id
            downloader = TessObservationsDownloader()
            result, header = downloader.download_by_id(tess_ids)

            # Assert that Observations.query_criteria_columns_async was called with the correct parameters
            mock_observations.query_criteria_columns_async.assert_called_once()
            call_kwargs = mock_observations.query_criteria_columns_async.call_args[1]

            # Check that the call contains the expected elements
            assert call_kwargs["provenance_name"] == "SPOC"
            assert call_kwargs["filters"] == "TESS"
            assert call_kwargs["project"] == "TESS"
            assert call_kwargs["dataproduct_type"] == "timeseries"
            assert call_kwargs["target_name"] == tess_ids
            assert call_kwargs["t_exptime"] == [119, 121]

            # Verify the result
            assert result is not None
            assert len(result) > 0

            # Assert that column names were converted correctly
            assert "tic_id" in result.colnames
            assert "obs_id" in result.colnames
            assert "target_name" not in result.colnames
            assert "obsid" not in result.colnames

    def test_download_by_id_with_columns(self, tess_observations_pandas_data: pd.DataFrame, tess_ids: list[int]):
        """Test download_by_id with extra columns parameter"""
        extra_columns = ["s_ra", "s_dec", "t_resolution"]

        with patch("exotools.downloaders.tess_observations_downloader.Observations") as mock_observations:
            # Mock Observations.query_criteria_columns_async to return our test data
            mock_observations.query_criteria_columns_async.return_value = tess_observations_pandas_data

            # Create downloader and call download_by_id with extra columns
            downloader = TessObservationsDownloader()
            result, _ = downloader.download_by_id(tess_ids, columns=extra_columns)

            # Check that the query contains the extra columns
            call_kwargs = mock_observations.query_criteria_columns_async.call_args[1]
            columns_arg = call_kwargs["columns"]

            # Verify all extra columns were included in the request
            for col in extra_columns:
                assert col in columns_arg

    def test_download_by_id_chunks(self, tess_observations_pandas_data: pd.DataFrame):
        """Test download_by_id with many IDs that require chunking"""
        # Create a list of 5000 IDs (should create 3 chunks with chunk_size=2000)
        many_ids = list(range(5000))

        with patch("exotools.downloaders.tess_observations_downloader.Observations") as mock_observations, patch(
            "exotools.downloaders.tess_observations_downloader.pd.concat", return_value=tess_observations_pandas_data
        ) as mock_concat:
            # Mock Observations.query_criteria_columns_async to return our test data
            mock_observations.query_criteria_columns_async.return_value = tess_observations_pandas_data

            # Create downloader and call download_by_id
            downloader = TessObservationsDownloader()
            result, _ = downloader.download_by_id(many_ids)

            # Assert that Observations.query_criteria_columns_async was called multiple times (once per chunk)
            assert (
                mock_observations.query_criteria_columns_async.call_count == 3
            )  # 5000 IDs / 2000 per chunk = 3 chunks (rounded up)

            # Assert that pd.concat was called to combine the results
            mock_concat.assert_called_once()

    def test_download_by_id_exception_handling(self, tess_ids: list[int]):
        """Test exception handling in download_by_id"""
        with patch("exotools.downloaders.tess_observations_downloader.Observations") as mock_observations, patch(
            "exotools.downloaders.tess_observations_downloader.logger"
        ) as mock_logger:
            # Mock Observations.query_criteria_columns_async to raise an exception
            mock_observations.query_criteria_columns_async.side_effect = Exception("Test exception")

            # Create downloader and call download_by_id, expecting the exception to be re-raised
            downloader = TessObservationsDownloader()
            with pytest.raises(Exception):
                downloader.download_by_id(tess_ids)

            # Verify that the exception was logged
            mock_logger.error.assert_called_once()
            assert "Exception generated while downloading" in mock_logger.error.call_args[0][0]

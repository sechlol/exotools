from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exotools.downloaders import LightcurveDownloader
from exotools.utils.download import DownloadParams


class TestLightcurveDownloader:
    @pytest.fixture
    def download_params(self) -> list[DownloadParams]:
        """Create test download parameters"""
        return [
            DownloadParams(
                url="https://example.com/test1.fits",
                download_path="/tmp/test/123456789/1000.fits",
            ),
            DownloadParams(
                url="https://example.com/test2.fits",
                download_path="/tmp/test/987654321/2000.fits",
            ),
        ]

    def test_download_one_lc(self, download_params):
        """Test downloading a single lightcurve"""
        with patch("exotools.downloaders.lightcurve_downloader.Observations") as mock_observations, patch(
            "exotools.downloaders.lightcurve_downloader.Path.exists", return_value=False
        ), patch("exotools.downloaders.lightcurve_downloader.Path.mkdir"):
            # Mock the download_file method to return success
            mock_observations.download_file.return_value = ("COMPLETE", "Success", None)

            # Create downloader and call download_one_lc
            downloader = LightcurveDownloader(override_existing=False)
            result = downloader.download_one_lc(download_params[0])

            # Verify the result
            assert result is not None
            assert str(result) == download_params[0].download_path

            # Verify that Observations.download_file was called with the correct parameters
            mock_observations.download_file.assert_called_once_with(
                download_params[0].url, local_path=Path(download_params[0].download_path)
            )

    def test_download_one_lc_existing_file(self, download_params):
        """Test downloading a single lightcurve when the file already exists"""
        with patch("exotools.downloaders.lightcurve_downloader.Observations") as mock_observations, patch(
            "exotools.downloaders.lightcurve_downloader.Path.exists", return_value=True
        ), patch("exotools.downloaders.lightcurve_downloader.Path.mkdir"):
            # Create downloader and call download_one_lc
            downloader = LightcurveDownloader(override_existing=False)
            result = downloader.download_one_lc(download_params[0])

            # Verify the result
            assert result is not None
            assert str(result) == download_params[0].download_path

            # Verify that Observations.download_file was NOT called
            mock_observations.download_file.assert_not_called()

    def test_download_one_lc_override_existing(self, download_params):
        """Test downloading a single lightcurve with override_existing=True"""
        with patch("exotools.downloaders.lightcurve_downloader.Observations") as mock_observations, patch(
            "exotools.downloaders.lightcurve_downloader.Path.exists", return_value=True
        ), patch("exotools.downloaders.lightcurve_downloader.Path.mkdir"):
            # Mock the download_file method to return success
            mock_observations.download_file.return_value = ("COMPLETE", "Success", None)

            # Create downloader with override_existing=True and call download_one_lc
            downloader = LightcurveDownloader(override_existing=True)
            result = downloader.download_one_lc(download_params[0])

            # Verify the result
            assert result is not None
            assert str(result) == download_params[0].download_path

            # Verify that Observations.download_file was called despite file existing
            mock_observations.download_file.assert_called_once()

    def test_download_one_lc_failure(self, download_params):
        """Test downloading a single lightcurve that fails"""
        with patch("exotools.downloaders.lightcurve_downloader.Observations") as mock_observations, patch(
            "exotools.downloaders.lightcurve_downloader.Path.exists", return_value=False
        ), patch("exotools.downloaders.lightcurve_downloader.Path.mkdir"), patch(
            "exotools.downloaders.lightcurve_downloader.logger"
        ) as mock_logger:
            # Mock the download_file method to return failure
            mock_observations.download_file.return_value = ("ERROR", "Failed to download", None)

            # Create downloader and call download_one_lc
            downloader = LightcurveDownloader()
            result = downloader.download_one_lc(download_params[0])

            # Verify the result is None on failure
            assert result is None

            # Verify that error was logged
            mock_logger.error.assert_called_once()

    def test_download_fits_multiple(self, download_params):
        """Test downloading multiple lightcurves sequentially"""
        with patch.object(LightcurveDownloader, "download_one_lc") as mock_download_one_lc:
            # Mock download_one_lc to return Path objects
            mock_download_one_lc.side_effect = [Path(p.download_path) for p in download_params]

            # Create downloader and call download_fits_multiple
            downloader = LightcurveDownloader()
            results = downloader.download_fits_multiple(download_params)

            # Verify the results
            assert len(results) == len(download_params)
            for i, result in enumerate(results):
                assert str(result) == download_params[i].download_path

            # Verify download_one_lc was called for each parameter
            assert mock_download_one_lc.call_count == len(download_params)
            for i, param in enumerate(download_params):
                mock_download_one_lc.assert_any_call(param)

    def test_download_fits_parallel(self, download_params):
        """Test downloading multiple lightcurves in parallel"""
        with patch("exotools.downloaders.lightcurve_downloader.Parallel") as mock_parallel, patch(
            "exotools.downloaders.lightcurve_downloader.delayed"
        ), patch("exotools.downloaders.lightcurve_downloader.os.cpu_count", return_value=4):
            # Mock the parallel execution
            mock_generator = MagicMock()
            mock_generator.__iter__.return_value = [Path(p.download_path) for p in download_params]
            mock_parallel.return_value.return_value = mock_generator

            # Create downloader and call download_fits_parallel
            downloader = LightcurveDownloader()
            results = downloader.download_fits_parallel(download_params)

            # Verify the results
            assert len(results) == len(download_params)
            for result, param in zip(results, download_params):
                assert str(result) == param.download_path

            # Verify Parallel was called with the correct parameters
            mock_parallel.assert_called_once_with(n_jobs=3, return_as="generator_unordered")

    def test_search_available_lightcurve_data(self):
        """Test search_available_lightcurve_data function"""
        with patch("exotools.downloaders.lightcurve_downloader._search_mast_target") as mock_search, patch(
            "exotools.downloaders.lightcurve_downloader._download_lightcurve_data"
        ) as mock_download:
            # Mock the search and download functions
            mock_search_result = MagicMock()
            mock_search_result.__len__.return_value = 5
            mock_search.return_value = mock_search_result

            mock_lc_collection = MagicMock()
            mock_download.return_value = mock_lc_collection

            # Call the function
            from exotools.downloaders.lightcurve_downloader import search_available_lightcurve_data

            result = search_available_lightcurve_data("TIC 123456789", exp_time_s=120)

            # Verify the result
            assert result == mock_lc_collection

            # Verify the search and download functions were called with the correct parameters
            mock_search.assert_called_once_with("TIC 123456789", verbose=False)
            mock_download.assert_called_once_with(search_result=mock_search_result, exp_time=120)

    def test_search_available_lightcurve_data_no_results(self):
        """Test search_available_lightcurve_data function with no results"""
        with patch("exotools.downloaders.lightcurve_downloader._search_mast_target") as mock_search, patch(
            "exotools.downloaders.lightcurve_downloader._download_lightcurve_data"
        ) as mock_download:
            # Mock the search function to return empty results
            mock_search_result = MagicMock()
            mock_search_result.__len__.return_value = 0
            mock_search.return_value = mock_search_result

            # Call the function
            from exotools.downloaders.lightcurve_downloader import search_available_lightcurve_data

            result = search_available_lightcurve_data("TIC 123456789")

            # Verify the result is None
            assert result is None

            # Verify the search function was called but not the download function
            mock_search.assert_called_once()
            mock_download.assert_not_called()

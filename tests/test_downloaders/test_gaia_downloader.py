from unittest.mock import MagicMock, patch

import pytest
from astropy.table import QTable

from exotools.downloaders import GaiaDownloader
from exotools.utils.qtable_utils import QTableHeader, TableColumnInfo


class TestGaiaDownloader:
    @pytest.fixture
    def gaia_raw_data(self, gaia_parameters_test_data: tuple[QTable, QTableHeader]) -> QTable:
        """Create a modified copy of the test data to simulate raw data from Gaia service"""
        # Get test data and make a copy to modify
        in_table = gaia_parameters_test_data[0].copy()

        # Rename gaia_id to source_id to match the raw data from the service
        if "gaia_id" in in_table.colnames:
            in_table.rename_column("gaia_id", "source_id")

        # Convert column names to uppercase to simulate raw Gaia data
        for col_name in list(in_table.colnames):
            in_table.rename_column(col_name, col_name.upper())

        return in_table

    @pytest.fixture
    def gaia_field_info(self, gaia_parameters_test_data: tuple[QTable, QTableHeader]) -> dict:
        """Create field info dictionary for the Gaia service mock"""
        header = gaia_parameters_test_data[1]

        # Create a dictionary with field info for both Gaia tables
        field_info = {}
        for name, info in header.items():
            # Create field info for both tables with uppercase column names
            field_info_obj = TableColumnInfo(unit=info.unit, dtype=info.dtype, description=info.description)
            # Some units in Gaia are stored with quotes
            if info.unit == "dex":
                field_info_obj.unit = "'dex'"

            field_info[name.upper()] = field_info_obj

        return field_info

    @pytest.fixture
    def gaia_ids(self) -> list[str]:
        """Return a list of Gaia IDs for testing"""
        return ["1234567890123456789", "2345678901234567890", "3456789012345678901"]

    def test_download_by_id(self, gaia_raw_data: QTable, gaia_field_info: dict, gaia_ids: list[str]):
        """Test download_by_id functionality"""
        # Create a mock Gaia module
        mock_gaia = MagicMock()
        mock_job = MagicMock()
        mock_job.get_results.return_value = gaia_raw_data
        mock_gaia.launch_job.return_value = mock_job

        with (
            # Patch the import statement instead of the module directly
            patch("astroquery.gaia.Gaia", mock_gaia),
            patch("exotools.downloaders.gaia_downloader.GaiaService") as mock_gaia_service_factory,
        ):
            # Mock GaiaService
            mock_gaia_service = MagicMock()
            mock_gaia_service.get_field_info.side_effect = lambda table_name: gaia_field_info
            mock_gaia_service_factory.return_value = mock_gaia_service

            # Create downloader and call download_by_id
            downloader = GaiaDownloader()
            result, header = downloader.download_by_id(gaia_ids)

            # Assert that Gaia.launch_job was called with the correct query
            mock_gaia.launch_job.assert_called_once()
            query = mock_gaia.launch_job.call_args[0][0]

            # Check that the query contains the expected elements
            for gaia_id in gaia_ids:
                assert f"'{gaia_id}'" in query
            assert "gaiadr3.gaia_source_lite" in query
            assert "gaiadr3.astrophysical_parameters" in query

            # Verify the result
            assert result is not None
            assert len(result) > 0

            # Assert that column names were converted to lowercase
            for col_name in result.colnames:
                assert col_name.islower()

            # Assert that source_id was renamed to gaia_id
            assert "gaia_id" in result.colnames
            assert "source_id" not in result.colnames

            # Assert that GaiaService.get_field_info was called for both tables
            assert mock_gaia_service.get_field_info.call_count == 2
            for table_name in ["gaiadr3.astrophysical_parameters", "gaiadr3.gaia_source_lite"]:
                mock_gaia_service.get_field_info.assert_any_call(table_name)

            # Check header has correct units, especially for dex
            for name, info in header.items():
                if info.unit == "dex":
                    assert info.unit != "'dex'"  # Quotes should be removed

    def test_download_by_id_with_extra_columns(self, gaia_raw_data: QTable, gaia_field_info: dict, gaia_ids: list[str]):
        """Test download_by_id with extra columns parameter"""
        extra_columns = ["ra", "dec", "parallax"]

        # Create a mock Gaia module
        mock_gaia = MagicMock()
        mock_job = MagicMock()
        mock_job.get_results.return_value = gaia_raw_data
        mock_gaia.launch_job.return_value = mock_job

        with (
            # Patch the import statement instead of the module directly
            patch("astroquery.gaia.Gaia", mock_gaia),
            patch("exotools.downloaders.gaia_downloader.GaiaService") as mock_gaia_service_factory,
        ):
            # Mock GaiaService
            mock_gaia_service = MagicMock()
            mock_gaia_service.get_field_info.side_effect = lambda table_name: gaia_field_info
            mock_gaia_service_factory.return_value = mock_gaia_service

            # Create downloader and call download_by_id with extra columns
            downloader = GaiaDownloader()
            result, _ = downloader.download_by_id(gaia_ids, columns=extra_columns)

            # Check that the query contains the extra columns
            query = mock_gaia.launch_job.call_args[0][0]
            for col in extra_columns:
                assert col in query

    def test_download_by_id_chunks(self, gaia_raw_data: QTable, gaia_field_info: dict):
        """Test download_by_id with many IDs that require chunking"""
        # Create a list of 2500 IDs (should create 3 chunks)
        many_ids = [str(i).zfill(19) for i in range(2500)]

        # Create a mock Gaia module
        mock_gaia = MagicMock()
        mock_job = MagicMock()
        mock_job.get_results.return_value = gaia_raw_data
        mock_gaia.launch_job.return_value = mock_job

        with (
            # Patch the import statement instead of the module directly
            patch("astroquery.gaia.Gaia", mock_gaia),
            patch("exotools.downloaders.gaia_downloader.GaiaService") as mock_gaia_service_factory,
            patch("exotools.downloaders.gaia_downloader.vstack", return_value=gaia_raw_data) as mock_vstack,
        ):
            # Mock GaiaService
            mock_gaia_service = MagicMock()
            mock_gaia_service.get_field_info.side_effect = lambda table_name: gaia_field_info
            mock_gaia_service_factory.return_value = mock_gaia_service

            # Create downloader and call download_by_id
            downloader = GaiaDownloader()
            result, _ = downloader.download_by_id(many_ids)

            # Assert that Gaia.launch_job was called multiple times (once per chunk)
            assert mock_gaia.launch_job.call_count == 3  # 2500 IDs / 1000 per chunk = 3 chunks

            # Assert that vstack was called to combine the results
            mock_vstack.assert_called_once()

    def test_download_not_implemented(self):
        """Test that download method raises NotImplementedError"""
        downloader = GaiaDownloader()
        with pytest.raises(NotImplementedError):
            downloader.download()

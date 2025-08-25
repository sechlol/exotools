from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import Column, MaskedColumn, QTable

from exotools.downloaders import PlanetarySystemsCompositeDownloader


class TestPlanetarySystemsCompositeDownloader:
    @pytest.fixture
    def tap_qtable_data(self, planetary_systems_composite_test_data) -> QTable:
        """Create a modified copy of the test data to simulate raw data from TAP service"""
        # Get test data and make a copy to modify
        in_table = planetary_systems_composite_test_data[0].copy()

        # Remove units from columns containing "pl_trandur" to simulate raw data
        for col_name in in_table.colnames:
            if "pl_trandur" in col_name:
                if hasattr(in_table[col_name], "unit") and in_table[col_name].unit is not None:
                    # Store the original data but remove the unit
                    data = in_table[col_name].data
                    in_table[col_name] = data

        # Modify the TIC and Gaia IDs to match the raw format from the service
        tic_ids = in_table["tic_id"]
        gaia_ids = in_table["gaia_id"]

        # Convert to string format as they appear in raw data: "TIC 123456789" and "Gaia DR3 123456789"
        in_table["tic_id"] = Column(["TIC " + str(tid) if tid != -1 else "" for tid in tic_ids])
        in_table["gaia_id"] = Column(["Gaia DR3 " + str(gid) if gid != -1 else "" for gid in gaia_ids])

        return in_table

    @pytest.fixture
    def descriptions(self, planetary_systems_composite_test_data) -> dict[str, str]:
        """Extract descriptions from the test data header"""
        header = planetary_systems_composite_test_data[1]
        return {name: info.description for name, info in header.items()}

    @pytest.fixture
    def columns_with_units_to_fix(self) -> list[str]:
        """Return a list of columns that need unit fixing"""
        return ["pl_trandur", "pl_trandurerr1", "pl_trandurerr2"]

    def test_download_smoketest(self, tap_qtable_data: QTable, descriptions: dict[str, str]):
        """Test basic download functionality"""
        # Mock ExoService .query() to return our modified table
        with patch("exotools.downloaders.ps_comppar_downloader.ExoService") as mock_exo_service_factory:
            mock_exo_service = MagicMock()
            mock_exo_service.query.return_value = tap_qtable_data
            # Mock the get_field_descriptions method to return our descriptions
            mock_exo_service.get_field_descriptions.return_value = descriptions
            mock_exo_service_factory.return_value = mock_exo_service

            # Create downloader and call download
            downloader = PlanetarySystemsCompositeDownloader()
            result, _ = downloader.download()  # Unpack the tuple

            # Assert that query was called with "from ps" table
            mock_exo_service.query.assert_called_once()
            query_str = mock_exo_service.query.call_args[0][0]
            assert "from pscomppars" in query_str

            # Assert that get_field_descriptions was called with the correct table name
            mock_exo_service.get_field_descriptions.assert_called_once_with("pscomppars")

            # Verify the result
            assert result is not None
            assert len(result) > 0

            # Assert that TIC and Gaia IDs were parsed correctly
            assert isinstance(result["tic_id"], MaskedColumn)
            assert isinstance(result["gaia_id"], MaskedColumn)
            assert result["tic_id"].dtype == np.dtype("int64")
            assert result["gaia_id"].dtype == np.dtype("int64")

    def test_with_limit(self, tap_qtable_data: QTable, descriptions: dict[str, str]):
        """Test download with limit parameter"""
        with patch("exotools.downloaders.ps_comppar_downloader.ExoService") as mock_exo_service_factory:
            # Create a mock QTable with required columns
            limit = 10
            reduced_qtable = tap_qtable_data[:limit]
            mock_exo_service = MagicMock()
            mock_exo_service.query.return_value = reduced_qtable
            mock_exo_service.get_field_descriptions.return_value = descriptions
            mock_exo_service_factory.return_value = mock_exo_service

            downloader = PlanetarySystemsCompositeDownloader()
            result, _ = downloader.download(limit=limit)  # Unpack the tuple

            # Verify limit was included in the query
            mock_exo_service.query.assert_called_once()
            query_str = mock_exo_service.query.call_args[0][0]
            assert f"top {limit}" in query_str
            assert len(result) == limit

    def test_with_columns(self, tap_qtable_data: QTable, descriptions: dict[str, str]):
        """Test download with columns parameter"""
        # Select a subset of columns, ensuring mandatory fields are included
        mandatory_fields = ["tic_id", "gaia_id", "hostname", "pl_name"]
        additional_cols = ["pl_orbper", "pl_orbsmax", "pl_bmasse"]
        selected_cols = mandatory_fields + additional_cols

        # Create a reduced table with only the selected columns
        reduced_qtable = tap_qtable_data[selected_cols]
        selected_descriptions = {name: descriptions.get(name, "test") for name in selected_cols}

        with patch("exotools.downloaders.ps_comppar_downloader.ExoService") as mock_exo_service_factory:
            mock_exo_service = MagicMock()
            mock_exo_service.query.return_value = reduced_qtable.copy()
            mock_exo_service.get_field_descriptions.return_value = selected_descriptions
            mock_exo_service_factory.return_value = mock_exo_service

            downloader = PlanetarySystemsCompositeDownloader()
            result, _ = downloader.download(columns=additional_cols)  # Unpack the tuple

            mock_exo_service.query.assert_called_once()
            query_str = mock_exo_service.query.call_args[0][0]

            # Check that all columns are in the query
            for col in selected_cols:
                assert col in query_str

            # Check that the result contains all expected columns
            assert set(result.colnames) == set(selected_cols)
            assert len(result) == len(reduced_qtable)

    def test_with_where(self, tap_qtable_data: QTable, descriptions: dict[str, str]):
        """Test download with where parameter"""
        with patch("exotools.downloaders.ps_comppar_downloader.ExoService") as mock_exo_service_factory:
            mock_exo_service = MagicMock()
            mock_exo_service.query.return_value = tap_qtable_data[:5]  # Return a subset as if filtered
            mock_exo_service.get_field_descriptions.return_value = descriptions
            mock_exo_service_factory.return_value = mock_exo_service

            downloader = PlanetarySystemsCompositeDownloader()

            # Test with string condition
            where_condition = {"hostname": "test_host"}
            result, _ = downloader.download(where=where_condition)

            query_str = mock_exo_service.query.call_args[0][0]
            assert "where hostname = 'test_host'" in query_str

            # Reset mock for next test
            mock_exo_service.reset_mock()

            # Test with numeric condition
            where_condition = {"pl_orbper": 5.2}
            mock_exo_service.query.return_value = tap_qtable_data[:3]  # Different subset
            result, _ = downloader.download(where=where_condition)

            query_str = mock_exo_service.query.call_args[0][0]
            assert "where pl_orbper = 5.2" in query_str

            # Reset mock for next test
            mock_exo_service.reset_mock()

            # Test with list condition
            where_condition = {"hostname": ["host1", "host2"]}
            mock_exo_service.query.return_value = tap_qtable_data[:2]  # Different subset
            result, _ = downloader.download(where=where_condition)

            query_str = mock_exo_service.query.call_args[0][0]
            assert "where hostname in ('host1','host2')" in query_str

            # Reset mock for next test
            mock_exo_service.reset_mock()

            # Test with multiple conditions
            where_condition = {"hostname": "test_host", "pl_orbper": 5.2}
            mock_exo_service.query.return_value = tap_qtable_data[:1]  # Single row
            result, _ = downloader.download(where=where_condition)

            query_str = mock_exo_service.query.call_args[0][0]
            assert "where" in query_str
            assert "hostname = 'test_host'" in query_str
            assert "pl_orbper = 5.2" in query_str
            assert "and" in query_str

    def test_download_by_id_not_implemented(self):
        """Test that download_by_id raises NotImplementedError"""
        downloader = PlanetarySystemsCompositeDownloader()
        with pytest.raises(NotImplementedError):
            downloader.download_by_id([1, 2, 3])

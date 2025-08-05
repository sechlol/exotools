from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import QTable
from lightkurve import LightCurve

from exotools import LightcurveDB


class TestLightcurveDb:
    @pytest.fixture
    def lightcurve_db(self) -> LightcurveDB:
        # Create a test dataset with the necessary columns
        data = QTable(
            {
                "obs_id": [1, 2, 3, 4],
                "tic_id": [100, 100, 200, 300],
                "path": ["path/to/lc1.fits", "path/to/lc2.fits", "path/to/lc3.fits", "path/to/lc4.fits"],
            }
        )
        return LightcurveDB(dataset=data)

    def test_init(self, lightcurve_db):
        """Test initialization of LightcurveDB."""
        assert len(lightcurve_db) == 4
        assert lightcurve_db._id_column == "obs_id"

    def test_tic_ids_property(self, lightcurve_db):
        """Test the tic_ids property."""
        tic_ids = lightcurve_db.tic_ids
        assert isinstance(tic_ids, np.ndarray)
        assert len(tic_ids) == 4
        assert list(tic_ids) == [100, 100, 200, 300]

    def test_obs_id_property(self, lightcurve_db):
        """Test the obs_id property."""
        obs_ids = lightcurve_db.obs_id
        assert isinstance(obs_ids, np.ndarray)
        assert len(obs_ids) == 4
        assert list(obs_ids) == [1, 2, 3, 4]

    def test_unique_tic_ids_property(self, lightcurve_db):
        """Test the unique_tic_ids property."""
        unique_tic_ids = lightcurve_db.unique_tic_ids
        assert isinstance(unique_tic_ids, np.ndarray)
        assert len(unique_tic_ids) == 3
        assert sorted(list(unique_tic_ids)) == [100, 200, 300]

    def test_unique_obs_ids_property(self, lightcurve_db):
        """Test the unique_obs_ids property."""
        unique_obs_ids = lightcurve_db.unique_obs_ids
        assert isinstance(unique_obs_ids, np.ndarray)
        assert len(unique_obs_ids) == 4
        assert sorted(list(unique_obs_ids)) == [1, 2, 3, 4]

    def test_select_by_tic_ids(self, lightcurve_db):
        """Test the select_by_tic_ids method."""
        # Test with a single TIC ID
        result = lightcurve_db.select_by_tic_ids(np.array([100]))
        assert isinstance(result, LightcurveDB)
        assert len(result) == 2
        assert all(tic_id == 100 for tic_id in result.tic_ids)

        # Test with multiple TIC IDs
        result = lightcurve_db.select_by_tic_ids(np.array([100, 300]))
        assert isinstance(result, LightcurveDB)
        assert len(result) == 3
        assert all(tic_id in [100, 300] for tic_id in result.tic_ids)

        # Test with non-existent TIC ID
        result = lightcurve_db.select_by_tic_ids(np.array([999]))
        assert isinstance(result, LightcurveDB)
        assert len(result) == 0

    @patch("exotools.db.lightcurve_db.load_lightcurve")
    def test_load_by_tic(self, mock_load_lightcurve, lightcurve_db):
        """Test the load_by_tic method."""
        # Mock the load_lightcurve function
        mock_lc1 = MagicMock(spec=LightCurve)
        mock_lc1.time = MagicMock()
        mock_lc1.time.scale = "tdb"
        mock_lc1.time.format = "jd"
        mock_lc1.time.value = [10, 11, 12]

        mock_lc2 = MagicMock(spec=LightCurve)
        mock_lc2.time = MagicMock()
        mock_lc2.time.scale = "tdb"
        mock_lc2.time.format = "jd"
        mock_lc2.time.value = [20, 21, 22]

        mock_load_lightcurve.side_effect = [mock_lc1, mock_lc2]

        # Mock LightCurvePlus to avoid sorting issues
        with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp:
            # Create mock LightCurvePlus instances with time property
            mock_lcp1 = MagicMock()
            mock_lcp1.time = np.array([10, 11, 12])

            mock_lcp2 = MagicMock()
            mock_lcp2.time = np.array([20, 21, 22])

            mock_lcp.side_effect = [mock_lcp1, mock_lcp2]

            # Test loading lightcurves for TIC ID 100
            result = lightcurve_db.load_by_tic(100)

            # Verify the result
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0] == mock_lcp1
            assert result[1] == mock_lcp2

        # Test with start_time_at_zero=True
        mock_load_lightcurve.reset_mock()
        mock_load_lightcurve.side_effect = [mock_lc1, mock_lc2]

        with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp:
            # Create mock LightCurvePlus instances with time property
            mock_lcp1 = MagicMock()
            mock_lcp1.time = np.array([10, 11, 12])
            # Add start_at_zero method that will be called
            mock_lcp1.start_at_zero = MagicMock(return_value=mock_lcp1)

            mock_lcp2 = MagicMock()
            mock_lcp2.time = np.array([20, 21, 22])
            # Add start_at_zero method that will be called
            mock_lcp2.start_at_zero = MagicMock(return_value=mock_lcp2)

            mock_lcp.side_effect = [mock_lcp1, mock_lcp2]

            # Test with start_time_at_zero=True
            result = lightcurve_db.load_by_tic(100, start_time_at_zero=True)

            # Verify the result
            assert isinstance(result, list)
            assert len(result) == 2
            # Check that start_at_zero was called on each LightCurvePlus
            assert mock_lcp1.start_at_zero.call_count == 1
            assert mock_lcp2.start_at_zero.call_count == 1

    @patch("exotools.db.lightcurve_db.load_lightcurve")
    @patch("lightkurve.LightCurveCollection.stitch")
    def test_load_stitched_by_tic(self, mock_stitch, mock_load_lightcurve, lightcurve_db):
        """Test the load_stitched_by_tic method."""
        # Mock the load_lightcurve function and stitch method
        mock_lc1 = MagicMock(spec=LightCurve)
        mock_lc1.time = MagicMock()
        mock_lc1.time.scale = "tdb"
        mock_lc1.time.format = "jd"
        mock_lc1.time.value = [10, 11, 12]

        mock_lc2 = MagicMock(spec=LightCurve)
        mock_lc2.time = MagicMock()
        mock_lc2.time.scale = "tdb"
        mock_lc2.time.format = "jd"
        mock_lc2.time.value = [20, 21, 22]

        mock_load_lightcurve.side_effect = [mock_lc1, mock_lc2]

        mock_stitched = MagicMock(spec=LightCurve)
        mock_stitched.time = MagicMock()
        mock_stitched.time.scale = "tdb"
        mock_stitched.time.format = "jd"
        mock_stitched.time.value = [10, 11, 12, 20, 21, 22]

        mock_stitch.return_value = mock_stitched

        # Mock LightCurvePlus to avoid sorting issues
        with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp:
            # Create mock LightCurvePlus instances with time property
            mock_lcp1 = MagicMock()
            mock_lcp1.time = np.array([10, 11, 12])

            mock_lcp2 = MagicMock()
            mock_lcp2.time = np.array([20, 21, 22])

            # Create a third mock for the stitched result
            mock_lcp_stitched = MagicMock()

            # Set up the side_effect to return all three mocks
            mock_lcp.side_effect = [mock_lcp1, mock_lcp2, mock_lcp_stitched]

            # Mock LightCurveCollection
            with patch("lightkurve.LightCurveCollection") as mock_lcc:
                mock_collection = MagicMock()
                mock_collection.stitch.return_value = mock_stitched
                mock_lcc.return_value = mock_collection

                # Test loading stitched lightcurve for TIC ID 100
                result = lightcurve_db.load_stitched_by_tic(100)

                # Verify the result
                assert result == mock_lcp_stitched

    @patch("exotools.db.lightcurve_db.load_lightcurve")
    def test_load_by_obs_id(self, mock_load_lightcurve, lightcurve_db):
        """Test the load_by_obs_id method."""
        # Mock the load_lightcurve function
        mock_lc = MagicMock(spec=LightCurve)
        mock_lc.time = MagicMock()
        mock_lc.time.scale = "tdb"
        mock_lc.time.format = "jd"
        mock_lc.time.value = [10, 11, 12]

        mock_load_lightcurve.return_value = mock_lc

        # Mock LightCurvePlus to avoid sorting issues
        with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp:
            # Create mock LightCurvePlus instance
            mock_lcp_instance = MagicMock()
            mock_lcp_instance.time = np.array([10, 11, 12])

            mock_lcp.return_value = mock_lcp_instance

            # Test loading lightcurve for obs_id 1
            result = lightcurve_db.load_by_obs_id(1)

            # Verify the result
            assert result == mock_lcp_instance

        # Test with start_time_at_zero=True
        mock_load_lightcurve.reset_mock()
        mock_load_lightcurve.return_value = mock_lc

        with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp:
            # Create mock LightCurvePlus instance with set_time_zero method
            mock_lcp_instance = MagicMock()
            mock_lcp_instance.time = np.array([10, 11, 12])
            mock_lcp_instance.start_at_zero = MagicMock(return_value=mock_lcp_instance)

            mock_lcp.return_value = mock_lcp_instance

            # Test with start_time_at_zero=True
            result = lightcurve_db.load_by_obs_id(1, start_time_at_zero=True)

            # Verify the result
            assert result == mock_lcp_instance
            assert mock_lcp_instance.start_at_zero.call_count == 1

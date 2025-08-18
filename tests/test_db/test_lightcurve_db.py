from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import QTable
from astropy.time import Time
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

    @pytest.fixture
    def mock_lightcurve(self):
        """Create a mock LightCurve object with configurable time values."""

        def _create_mock_lightcurve(time_values):
            mock_lc = MagicMock(spec=LightCurve)
            mock_lc.time = MagicMock()
            mock_lc.time.scale = "tdb"
            mock_lc.time.format = "jd"
            mock_lc.time.value = time_values
            return mock_lc

        return _create_mock_lightcurve

    @pytest.fixture
    def mock_lightcurve_plus(self):
        """Create a mock LightCurvePlus object with configurable time values and zero-time option."""

        def _create_mock_lightcurve_plus(time_values, with_start_at_zero=False):
            mock_lcp = MagicMock()
            mock_lcp.time = np.array(time_values)
            if with_start_at_zero:
                mock_lcp.start_at_zero = MagicMock(return_value=mock_lcp)
            return mock_lcp

        return _create_mock_lightcurve_plus

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

    @patch.object(LightcurveDB, "load_lightcurve")
    def test_load_by_tic(self, mock_load_lightcurve, lightcurve_db, mock_lightcurve, mock_lightcurve_plus):
        """Test the load_by_tic method."""
        # Create mock lightcurves using fixtures
        mock_lc1 = mock_lightcurve([10, 11, 12])
        mock_lc2 = mock_lightcurve([20, 21, 22])
        mock_load_lightcurve.side_effect = [mock_lc1, mock_lc2]

        # Mock LightCurvePlus to avoid sorting issues
        with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp:
            # Create mock LightCurvePlus instances with time property
            mock_lcp1 = mock_lightcurve_plus([10, 11, 12])
            mock_lcp2 = mock_lightcurve_plus([20, 21, 22])
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
            # Create mock LightCurvePlus instances with start_at_zero method
            mock_lcp1 = mock_lightcurve_plus([10, 11, 12], with_start_at_zero=True)
            mock_lcp2 = mock_lightcurve_plus([20, 21, 22], with_start_at_zero=True)
            mock_lcp.side_effect = [mock_lcp1, mock_lcp2]

            # Test with start_time_at_zero=True
            result = lightcurve_db.load_by_tic(100, start_time_at_zero=True)

            # Verify the result
            assert isinstance(result, list)
            assert len(result) == 2
            # Check that start_at_zero was called on each LightCurvePlus
            assert mock_lcp1.start_at_zero.call_count == 1
            assert mock_lcp2.start_at_zero.call_count == 1

    @patch.object(LightcurveDB, "load_by_tic")
    @patch("lightkurve.LightCurveCollection.stitch")
    def test_load_stitched_by_tic(
        self, mock_stitch, mock_load_by_tic, lightcurve_db, mock_lightcurve, mock_lightcurve_plus
    ):
        """Test the load_stitched_by_tic method."""
        # Create mock LightCurvePlus instances
        mock_lcp1 = mock_lightcurve_plus([10, 11, 12])
        mock_lcp2 = mock_lightcurve_plus([20, 21, 22])
        mock_load_by_tic.return_value = [mock_lcp1, mock_lcp2]

        # Create mock for the stitched lightcurve
        mock_stitched = mock_lightcurve([10, 11, 12, 20, 21, 22])
        mock_stitch.return_value = mock_stitched

        # Mock LightCurvePlus to avoid sorting issues
        with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp:
            # Create mock for the stitched LightCurvePlus
            mock_stitched_lcp = mock_lightcurve_plus([10, 11, 12, 20, 21, 22])
            mock_lcp.return_value = mock_stitched_lcp

            # Test loading stitched lightcurve for TIC ID 100
            result = lightcurve_db.load_stitched_by_tic(100)

            # Verify the result
            assert result == mock_stitched_lcp

            # Verify that load_by_tic was called with the correct parameters
            mock_load_by_tic.assert_called_once_with(100, start_time_at_zero=False, load_in_jd_time=False)

            # Verify that stitch was called
            mock_stitch.assert_called_once()

            # Verify that LightCurvePlus was created with the stitched result
            mock_lcp.assert_called_once_with(mock_stitched)

    @patch.object(LightcurveDB, "load_lightcurve")
    def test_load_by_obs_id(self, mock_load_lightcurve, lightcurve_db, mock_lightcurve, mock_lightcurve_plus):
        """Test the load_by_obs_id method."""
        # Create mock lightcurve using fixture
        mock_lc = mock_lightcurve([10, 11, 12])
        mock_load_lightcurve.return_value = mock_lc

        # Mock LightCurvePlus to avoid sorting issues
        with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp:
            # Create mock LightCurvePlus instance
            mock_lcp_instance = mock_lightcurve_plus([10, 11, 12])
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
            mock_lcp_instance = mock_lightcurve_plus([10, 11, 12], with_start_at_zero=True)
            mock_lcp.return_value = mock_lcp_instance

            # Test with start_time_at_zero=True
            result = lightcurve_db.load_by_obs_id(1, start_time_at_zero=True)

            # Verify the result
            assert result == mock_lcp_instance
            assert mock_lcp_instance.start_at_zero.call_count == 1

    def test_load_lightcurve_time_format_detection(self):
        """Test that load_lightcurve correctly detects time format from FITS metadata."""
        # Test BTJD format detection
        with patch("astropy.io.fits.open") as mock_fits_open:
            # Mock FITS file structure for BTJD format
            mock_hdul = MagicMock()
            mock_hdul.__enter__.return_value = mock_hdul
            mock_hdul.__exit__.return_value = None

            # Create proper structured array for FITS data
            time_data = np.array([1599.8, 1599.9, 1600.0])
            flux_data = np.array([1000.0, 1001.0, 1002.0])
            flux_err_data = np.array([10.0, 10.1, 10.2])

            # Mock the data access properly
            mock_hdul["LIGHTCURVE"].data = {
                "TIME": time_data,
                "PDCSAP_FLUX": flux_data,
                "PDCSAP_FLUX_ERR": flux_err_data,
            }

            # Mock headers with BTJD format
            mock_header0 = {"TELESCOP": "TESS"}
            mock_header1 = {
                "TUNIT1": "BJD - 2457000, days",
                "TIMESYS": "TDB",
                "TIMEUNIT": "d",
                "TIMEREF": "SOLARSYSTEM",
                "BJDREFI": 2457000,
                "BJDREFF": 0.0,
            }
            mock_hdul[0].header = mock_header0
            mock_hdul[1].header = mock_header1

            mock_fits_open.return_value = mock_hdul

            # Test with load_in_jd_time=False (should preserve BTJD)
            lc = LightcurveDB.load_lightcurve("test.fits", load_in_jd_time=False)
            assert lc.time.format == "btjd"
            assert lc.time.scale == "tdb"
            assert lc.meta["_ORIGINAL_TIME_FORMAT"] == "btjd"
            np.testing.assert_array_almost_equal(lc.time.value, [1599.8, 1599.9, 1600.0])

            # Test with load_in_jd_time=True (should convert to JD)
            lc_jd = LightcurveDB.load_lightcurve("test.fits", load_in_jd_time=True)
            assert lc_jd.time.format == "jd"
            assert lc_jd.time.scale == "tdb"
            assert lc_jd.meta["_ORIGINAL_TIME_FORMAT"] == "btjd"
            # JD values should be BTJD + 2457000
            expected_jd = np.array([1599.8, 1599.9, 1600.0]) + 2457000
            np.testing.assert_array_almost_equal(lc_jd.time.value, expected_jd)

    def test_load_lightcurve_plus_with_time_format(self):
        """Test that load_lightcurve_plus correctly handles load_in_jd_time parameter."""
        with patch.object(LightcurveDB, "load_lightcurve") as mock_load_lc:
            with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp_class:
                # Mock lightcurve
                mock_lc = MagicMock()
                mock_load_lc.return_value = mock_lc

                # Mock LightCurvePlus instance
                mock_lcp = MagicMock()
                mock_lcp_class.return_value = mock_lcp

                # Test with load_in_jd_time=False
                result = LightcurveDB.load_lightcurve_plus("test.fits", load_in_jd_time=False)

                # Verify correct parameters were passed
                mock_load_lc.assert_called_once_with("test.fits", load_in_jd_time=False)
                mock_lcp_class.assert_called_once_with(mock_lc)
                assert result == mock_lcp

                # Reset mocks and test with load_in_jd_time=True
                mock_load_lc.reset_mock()
                mock_lcp_class.reset_mock()

                _ = LightcurveDB.load_lightcurve_plus("test.fits", load_in_jd_time=True)

                # Verify correct parameters were passed
                mock_load_lc.assert_called_once_with("test.fits", load_in_jd_time=True)
                mock_lcp_class.assert_called_once_with(mock_lc)

    def test_instance_methods_with_load_in_jd_time(self, lightcurve_db):
        """Test that instance methods correctly pass load_in_jd_time parameter."""
        with patch.object(LightcurveDB, "load_lightcurve") as mock_load_lc:
            with patch("exotools.db.lightcurve_db.LightCurvePlus") as mock_lcp_class:
                # Mock lightcurve and LightCurvePlus
                mock_lc = MagicMock()
                mock_load_lc.return_value = mock_lc
                mock_lcp = MagicMock()
                mock_lcp.time = [Time(2458000.0, format="jd")]
                mock_lcp_class.return_value = mock_lcp

                # Test load_by_obs_id with load_in_jd_time=True
                _ = lightcurve_db.load_by_obs_id(1, load_in_jd_time=True)

                # Verify parameters were passed correctly
                mock_load_lc.assert_called_with("path/to/lc1.fits", load_in_jd_time=True)
                mock_lcp_class.assert_called_with(mock_lc)

                # Verify that to_jd_time() was not called when load_in_jd_time=True
                assert mock_lcp.to_jd_time.call_count == 0

                # Reset mocks and test load_by_tic
                mock_load_lc.reset_mock()
                mock_lcp_class.reset_mock()
                mock_lcp.to_jd_time.reset_mock()

                _ = lightcurve_db.load_by_tic(100, load_in_jd_time=True)

                # Should be called twice for TIC 100 (has 2 observations)
                assert mock_load_lc.call_count == 2
                assert mock_lcp_class.call_count == 2

                # Verify load_in_jd_time=True was passed to load_lightcurve calls
                for call in mock_load_lc.call_args_list:
                    assert call[1]["load_in_jd_time"]
                # Verify LightCurvePlus constructor was called without load_in_jd_time
                for call in mock_lcp_class.call_args_list:
                    assert len(call[1]) == 1  # Only obs_id parameter
                # Verify to_jd_time() was not called, since load_in_jd_time=True
                assert mock_lcp.to_jd_time.call_count == 0

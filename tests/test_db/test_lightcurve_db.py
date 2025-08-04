from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import QTable
from lightkurve import LightCurve

from exotools import LightcurveDB
from exotools.db.lightcurve_plus import LightCurvePlus


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
        mock_lc1.time = [10, 11, 12]
        mock_lc2 = MagicMock(spec=LightCurve)
        mock_lc2.time = [20, 21, 22]
        mock_load_lightcurve.side_effect = [mock_lc1, mock_lc2]

        # Test loading lightcurves for TIC ID 100
        result = lightcurve_db.load_by_tic(100)
        assert len(result) == 2
        assert all(isinstance(lc, LightCurvePlus) for lc in result)
        assert mock_load_lightcurve.call_count == 2

        # Test with start_time_at_zero=True
        mock_load_lightcurve.reset_mock()
        mock_load_lightcurve.side_effect = [mock_lc1, mock_lc2]
        result = lightcurve_db.load_by_tic(100, start_time_at_zero=True)
        assert len(result) == 2
        assert mock_load_lightcurve.call_count == 2

        # Test with non-existent TIC ID
        mock_load_lightcurve.reset_mock()
        result = lightcurve_db.load_by_tic(999)
        assert result is None
        assert mock_load_lightcurve.call_count == 0

    @patch("exotools.db.lightcurve_db.load_lightcurve")
    @patch("lightkurve.LightCurveCollection.stitch")
    def test_load_stitched_by_tic(self, mock_stitch, mock_load_lightcurve, lightcurve_db):
        """Test the load_stitched_by_tic method."""
        # Mock the load_lightcurve function and stitch method
        mock_lc1 = MagicMock(spec=LightCurve)
        mock_lc1.time = [10, 11, 12]
        mock_lc2 = MagicMock(spec=LightCurve)
        mock_lc2.time = [20, 21, 22]
        mock_load_lightcurve.side_effect = [mock_lc1, mock_lc2]

        mock_stitched = MagicMock(spec=LightCurve)
        mock_stitched.time = [10, 11, 12, 20, 21, 22]
        mock_stitch.return_value = mock_stitched

        # Test loading stitched lightcurve for TIC ID 100
        result = lightcurve_db.load_stitched_by_tic(100)
        assert isinstance(result, LightCurvePlus)
        assert mock_load_lightcurve.call_count == 2
        assert mock_stitch.call_count == 1

        # Test with start_time_at_zero=True
        mock_load_lightcurve.reset_mock()
        mock_stitch.reset_mock()
        mock_load_lightcurve.side_effect = [mock_lc1, mock_lc2]
        result = lightcurve_db.load_stitched_by_tic(100, start_time_at_zero=True)
        assert isinstance(result, LightCurvePlus)
        assert mock_load_lightcurve.call_count == 2
        assert mock_stitch.call_count == 1

        # Test with non-existent TIC ID
        mock_load_lightcurve.reset_mock()
        mock_stitch.reset_mock()
        result = lightcurve_db.load_stitched_by_tic(999)
        assert result is None
        assert mock_load_lightcurve.call_count == 0
        assert mock_stitch.call_count == 0

    @patch("exotools.db.lightcurve_db.load_lightcurve")
    def test_load_by_obs_id(self, mock_load_lightcurve, lightcurve_db):
        """Test the load_by_obs_id method."""
        # Mock the load_lightcurve function
        mock_lc = MagicMock(spec=LightCurve)
        mock_lc.time = [10, 11, 12]
        mock_load_lightcurve.return_value = mock_lc

        # Test loading lightcurve for obs_id 1
        result = lightcurve_db.load_by_obs_id(1)
        assert isinstance(result, LightCurvePlus)
        assert mock_load_lightcurve.call_count == 1

        # Test with start_time_at_zero=True
        mock_load_lightcurve.reset_mock()
        mock_load_lightcurve.return_value = mock_lc
        result = lightcurve_db.load_by_obs_id(1, start_time_at_zero=True)
        assert isinstance(result, LightCurvePlus)
        assert mock_load_lightcurve.call_count == 1

        # Test with non-existent obs_id
        mock_load_lightcurve.reset_mock()
        result = lightcurve_db.load_by_obs_id(999)
        assert result is None
        assert mock_load_lightcurve.call_count == 0

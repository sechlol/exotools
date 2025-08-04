import numpy as np
import pytest
from astropy.table import QTable

from exotools import TessMetaDB
from exotools.db.base_db import NAN_VALUE
from exotools.utils.qtable_utils import QTableHeader


class TestTessMetaDB:
    @pytest.fixture
    def tess_meta_db(self, tess_observations_test_data: tuple[QTable, QTableHeader]) -> TessMetaDB:
        return TessMetaDB(meta_dataset=tess_observations_test_data[0])

    def test_init(self, tess_meta_db, tess_observations_test_data):
        """Test initialization of TessMetaDB."""
        # Check that the dataset was properly set
        assert len(tess_meta_db) == len(tess_observations_test_data[0])
        assert tess_meta_db._id_column == "tic_id"

    def test_tic_ids_property(self, tess_meta_db):
        """Test the tic_ids property."""
        tic_ids = tess_meta_db.tic_ids
        assert isinstance(tic_ids, np.ndarray)
        assert len(tic_ids) == len(tess_meta_db)

    def test_obs_id_property(self, tess_meta_db):
        """Test the obs_id property."""
        obs_ids = tess_meta_db.obs_id
        assert isinstance(obs_ids, np.ndarray)
        assert len(obs_ids) == len(tess_meta_db)

    def test_unique_tic_ids_property(self, tess_meta_db):
        """Test the unique_tic_ids property."""
        unique_tic_ids = tess_meta_db.unique_tic_ids
        assert isinstance(unique_tic_ids, np.ndarray)
        assert len(unique_tic_ids) <= len(tess_meta_db)
        # Check that all IDs are unique
        assert len(unique_tic_ids) == len(set(unique_tic_ids))

    def test_unique_obs_ids_property(self, tess_meta_db):
        """Test the unique_obs_ids property."""
        unique_obs_ids = tess_meta_db.unique_obs_ids
        assert isinstance(unique_obs_ids, np.ndarray)
        assert len(unique_obs_ids) <= len(tess_meta_db)
        # Check that all IDs are unique
        assert len(unique_obs_ids) == len(set(unique_obs_ids))

    def test_data_urls_property(self, tess_meta_db):
        """Test the data_urls property."""
        data_urls = tess_meta_db.data_urls
        assert isinstance(data_urls, np.ndarray)
        assert len(data_urls) == len(tess_meta_db)

    def test_select_by_obs_id(self, tess_meta_db):
        """Test the select_by_obs_id method."""
        # Create a dataset with valid obs_id values
        if len(tess_meta_db) > 0:
            # Ensure at least some records have valid obs_id values
            valid_indices = np.arange(len(tess_meta_db) // 2)
            test_obs_ids = np.array([101, 102, 103])[: len(valid_indices)]

            # Set obs_id values for testing
            tess_meta_db.view["obs_id"][valid_indices] = test_obs_ids

            # Test with a single obs_id
            result = tess_meta_db.select_by_obs_id(np.array([101]))
            assert isinstance(result, TessMetaDB)
            if len(result) > 0:
                assert all(obs_id == 101 for obs_id in result.obs_id)

            # Test with multiple obs_ids
            result = tess_meta_db.select_by_obs_id(np.array([101, 102]))
            assert isinstance(result, TessMetaDB)
            if len(result) > 0:
                assert all(obs_id in [101, 102] for obs_id in result.obs_id)

            # Test with non-existent obs_id
            result = tess_meta_db.select_by_obs_id(np.array([999]))
            assert isinstance(result, TessMetaDB)
            assert len(result) == 0

    def test_inherited_methods(self, tess_meta_db):
        """Test that inherited methods from BaseDB work correctly."""
        # Test where method
        if len(tess_meta_db) > 0:
            first_tic_id = tess_meta_db.tic_ids[0]
            filtered = tess_meta_db.where(tic_id=first_tic_id)
            assert isinstance(filtered, TessMetaDB)
            assert all(row["tic_id"] == first_tic_id for row in filtered.view)

        # Test to_pandas method
        df = tess_meta_db.to_pandas()
        assert len(df) == len(tess_meta_db)

        # Test with_valid_ids method
        valid_ids = tess_meta_db.with_valid_ids()
        assert isinstance(valid_ids, TessMetaDB)
        if len(valid_ids) > 0:
            assert all(tic_id != NAN_VALUE for tic_id in valid_ids.view["tic_id"])

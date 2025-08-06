import numpy as np
import pytest

from exotools import TicObsDB
from exotools.constants import NAN_VALUE


class TestTessMetaDB:
    @pytest.fixture
    def tess_meta_db(self, tic_observations_test_data) -> TicObsDB:
        return TicObsDB(meta_dataset=tic_observations_test_data[0].copy())

    def test_init(self, tess_meta_db, tic_observations_test_data):
        """Test initialization of TessMetaDB."""
        # Check that the dataset was properly set
        assert len(tess_meta_db) == len(tic_observations_test_data[0])
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
            valid_indices = np.arange(3)  # Just use first 3 indices
            test_obs_ids = np.array([101, 102, 103])

            # Make a copy of the view to avoid modifying the original
            test_db = tess_meta_db._factory(tess_meta_db.view.copy())

            # Set obs_id values for testing (only for the first 3 records)
            for i, obs_id in zip(valid_indices, test_obs_ids):
                if i < len(test_db.view):
                    test_db.view["obs_id"][i] = obs_id

            # Test with a single obs_id
            result = test_db.select_by_obs_id(np.array([101]))
            assert isinstance(result, TicObsDB)
            if len(result) > 0:
                assert all(obs_id == 101 for obs_id in result.obs_id)

            # Test with multiple obs_ids
            result = test_db.select_by_obs_id(np.array([101, 102]))
            assert isinstance(result, TicObsDB)
            if len(result) > 0:
                assert all(obs_id in [101, 102] for obs_id in result.obs_id)

            # Test with non-existent obs_id
            result = test_db.select_by_obs_id(np.array([999]))
            assert isinstance(result, TicObsDB)
            assert len(result) == 0

    def test_select_by_tic_id(self, tess_meta_db):
        """Test the select_by_tic_id method."""
        # Create a dataset with valid tic_id values
        if len(tess_meta_db) > 0:
            # Ensure at least some records have valid obs_id values
            valid_indices = np.arange(3)  # Just use first 3 indices
            test_tic_ids = np.array([101, 102, 103])

            # Make a copy of the view to avoid modifying the original
            test_db = tess_meta_db._factory(tess_meta_db.view.copy())

            # Set tic_id values for testing (only for the first 3 records)
            for i, tic_id in zip(valid_indices, test_tic_ids):
                if i < len(test_db.view):
                    test_db.view["tic_id"][i] = tic_id

            # Test with a single tic_id
            result = test_db.select_by_tic_id(np.array([101]))
            assert isinstance(result, TicObsDB)
            if len(result) > 0:
                assert all(tic_id == 101 for tic_id in result.tic_ids)

            # Test with multiple tic_ids
            result = test_db.select_by_tic_id(np.array([101, 102]))
            assert isinstance(result, TicObsDB)
            if len(result) > 0:
                assert all(tic_id in [101, 102] for tic_id in result.tic_ids)

            # Test with non-existent tic_id
            result = test_db.select_by_tic_id(np.array([999]))
            assert isinstance(result, TicObsDB)
            assert len(result) == 0

    def test_inherited_methods(self, tess_meta_db):
        """Test that inherited methods from BaseDB work correctly."""
        # Test where method
        if len(tess_meta_db) > 0:
            first_tic_id = tess_meta_db.tic_ids[0]
            filtered = tess_meta_db.where(tic_id=first_tic_id)
            assert isinstance(filtered, TicObsDB)
            assert all(row["tic_id"] == first_tic_id for row in filtered.view)

        # Test to_pandas method
        df = tess_meta_db.to_pandas()
        assert len(df) == len(tess_meta_db)

        # Test with_valid_ids method
        valid_ids = tess_meta_db.with_valid_ids()
        assert isinstance(valid_ids, TicObsDB)
        if len(valid_ids) > 0:
            assert all(tic_id != NAN_VALUE for tic_id in valid_ids.view["tic_id"])

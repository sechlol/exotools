import numpy as np
import pytest
from astropy.table import QTable

from exotools import TicDB
from exotools.utils.qtable_utils import QTableHeader


class TestTicDB:
    @pytest.fixture
    def tic_db(self, tess_tic_test_data: tuple[QTable, QTableHeader]) -> TicDB:
        return TicDB(dataset=tess_tic_test_data[0])

    @pytest.fixture
    def tic_by_id_db(self, tess_tic_by_id_test_data: tuple[QTable, QTableHeader]) -> TicDB:
        return TicDB(dataset=tess_tic_by_id_test_data[0])

    def test_init(self, tic_db, tess_tic_test_data):
        """Test initialization of TicDB."""
        # Check that the dataset was properly set
        assert len(tic_db) == len(tess_tic_test_data[0])
        assert tic_db._id_column == "tic_id"

    def test_tic_ids_property(self, tic_db):
        """Test the tic_ids property."""
        tic_ids = tic_db.tic_ids
        assert isinstance(tic_ids, np.ndarray)
        assert len(tic_ids) == len(tic_db)

    def test_gaia_ids_property(self, tic_db):
        """Test the gaia_ids property."""
        gaia_ids = tic_db.gaia_ids
        assert isinstance(gaia_ids, np.ndarray)
        assert len(gaia_ids) == len(tic_db)

    def test_unique_tic_ids_property(self, tic_db):
        """Test the unique_tic_ids property."""
        unique_tic_ids = tic_db.unique_tic_ids
        assert isinstance(unique_tic_ids, np.ndarray)
        assert len(unique_tic_ids) <= len(tic_db)
        # Check that all IDs are unique
        assert len(unique_tic_ids) == len(set(unique_tic_ids))

    def test_unique_gaia_ids_property(self, tic_db):
        """Test the unique_gaia_ids property."""
        unique_gaia_ids = tic_db.unique_gaia_ids
        assert isinstance(unique_gaia_ids, np.ndarray)
        assert len(unique_gaia_ids) <= len(tic_db)
        # Check that all IDs are unique
        assert len(unique_gaia_ids) == len(set(unique_gaia_ids))

    def test_inherited_methods(self, tic_db):
        """Test that inherited methods from BaseDB work correctly."""
        # Test where method
        if len(tic_db) > 0:
            first_tic_id = tic_db.tic_ids[0]
            filtered = tic_db.where(tic_id=first_tic_id)
            assert isinstance(filtered, TicDB)
            assert all(row["tic_id"] == first_tic_id for row in filtered.view)

        # Test where_true method
        if len(tic_db) > 0:
            mask = np.zeros(len(tic_db), dtype=bool)
            mask[0] = True  # Select only the first record
            filtered = tic_db.where_true(mask)
            assert isinstance(filtered, TicDB)
            assert len(filtered) == 1
            assert filtered.view[0]["tic_id"] == tic_db.view[0]["tic_id"]

        # Test to_pandas method
        df = tic_db.to_pandas()
        assert len(df) == len(tic_db)

        # Test select_random_sample method
        if len(tic_db) >= 2:
            sample = tic_db.select_random_sample(2)
            assert isinstance(sample, TicDB)
            assert len(sample) == 2

        # Test with_valid_ids method
        valid_ids = tic_db.with_valid_ids()
        assert isinstance(valid_ids, TicDB)

    def test_tic_by_id_fixture(self, tic_by_id_db, tess_tic_by_id_test_data):
        """Test the tic_by_id_db fixture."""
        assert len(tic_by_id_db) == len(tess_tic_by_id_test_data[0])
        assert tic_by_id_db._id_column == "tic_id"

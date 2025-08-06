import numpy as np
import pytest
from astropy.table import QTable

from exotools import CandidateDB
from exotools.utils.qtable_utils import QTableHeader


class TestCandidateDb:
    @pytest.fixture
    def candidate_db(self, candidate_exoplanets_test_data: tuple[QTable, QTableHeader]) -> CandidateDB:
        return CandidateDB(toi_dataset=candidate_exoplanets_test_data[0])

    def test_init(self, candidate_db, candidate_exoplanets_test_data):
        """Test initialization of CandidateDB."""
        # Check that the dataset was properly set
        assert len(candidate_db) == len(candidate_exoplanets_test_data[0])
        assert candidate_db._id_column == "tic_id"

    def test_tic_ids_property(self, candidate_db):
        """Test the tic_ids property."""
        tic_ids = candidate_db.tic_ids
        assert isinstance(tic_ids, np.ndarray)
        assert len(tic_ids) == len(candidate_db)

    def test_unique_tic_ids_property(self, candidate_db):
        """Test the unique_tic_ids property."""
        unique_tic_ids = candidate_db.unique_tic_ids
        assert isinstance(unique_tic_ids, np.ndarray)
        assert len(unique_tic_ids) <= len(candidate_db)
        # Check that all IDs are unique
        assert len(unique_tic_ids) == len(set(unique_tic_ids))

    def test_inherited_methods(self, candidate_db):
        """Test that inherited methods from BaseDB work correctly."""
        # Test where method
        if len(candidate_db) > 0:
            first_tic_id = candidate_db.tic_ids[0]
            filtered = candidate_db.where(tic_id=first_tic_id)
            assert isinstance(filtered, CandidateDB)
            assert all(row["tic_id"] == first_tic_id for row in filtered.view)

        # Test to_pandas method
        df = candidate_db.to_pandas()
        assert len(df) == len(candidate_db)

        # Test with_valid_ids method
        valid_ids = candidate_db.with_valid_ids()
        assert isinstance(valid_ids, CandidateDB)

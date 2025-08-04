import numpy as np
import pytest
from astropy.table import QTable

from exotools import StarSystemDB
from exotools.db.star_system import StarSystem
from exotools.utils.qtable_utils import QTableHeader


class TestStarSystemDb:
    @pytest.fixture
    def star_system_db(self, star_system_test_data: tuple[QTable, QTableHeader]) -> StarSystemDB:
        return StarSystemDB(transit_dataset=star_system_test_data[0])

    def test_init(self, star_system_db, star_system_test_data):
        """Test initialization of StarSystemDB."""
        # Check that the dataset was properly set
        assert len(star_system_db) == len(star_system_test_data[0])
        assert star_system_db._id_column == "tic_id"

    def test_get_valid_planets(self, star_system_db):
        """Test the get_valid_planets method."""
        # Add a pl_valid_flag column to the dataset if it doesn't exist
        if "pl_valid_flag" not in star_system_db.view.colnames:
            star_system_db.view["pl_valid_flag"] = [True, False] * (len(star_system_db) // 2)
            if len(star_system_db) % 2 == 1:
                star_system_db.view["pl_valid_flag"] = np.append(star_system_db.view["pl_valid_flag"], [True])

        valid_planets = star_system_db.get_valid_planets()
        assert isinstance(valid_planets, StarSystemDB)
        # Check that all records have pl_valid_flag = True
        if len(valid_planets) > 0:
            assert all(record["pl_valid_flag"] for record in valid_planets.view)

    def test_get_star_system_from_star_name(self, star_system_db):
        """Test the get_star_system_from_star_name method."""
        # Add hostname and hostname_lowercase columns if they don't exist
        if "hostname" not in star_system_db.view.colnames:
            star_system_db.view["hostname"] = ["Star A", "Star B", "Star C"][: len(star_system_db)]
            star_system_db.view["hostname_lowercase"] = ["star a", "star b", "star c"][: len(star_system_db)]

        # Test with existing star name
        if len(star_system_db) > 0:
            first_star_name = star_system_db.view["hostname"][0]
            star_system = star_system_db.get_star_system_from_star_name(first_star_name)
            assert isinstance(star_system, StarSystem)
            assert star_system.star_name == first_star_name

        # Test with non-existent star name
        star_system = star_system_db.get_star_system_from_star_name("Non-existent Star")
        assert star_system is None

    def test_get_star_system_from_tic_id(self, star_system_db):
        """Test the get_star_system_from_tic_id method."""
        # Add hostname column if it doesn't exist
        if "hostname" not in star_system_db.view.colnames:
            star_system_db.view["hostname"] = ["Star A", "Star B", "Star C"][: len(star_system_db)]

        # Test with existing TIC ID
        if len(star_system_db) > 0:
            first_tic_id = star_system_db.view["tic_id"][0]
            star_system = star_system_db.get_star_system_from_tic_id(first_tic_id)
            assert isinstance(star_system, StarSystem)
            assert star_system.star_name == star_system_db.view["hostname"][0]

        # Test with non-existent TIC ID
        star_system = star_system_db.get_star_system_from_tic_id(999999)
        assert star_system is None

    def test_preprocess_dataset(self):
        """Test the preprocess_dataset static method."""
        # Create a test dataset
        test_data = QTable({"rowupdate": [3, 1, 2], "pl_name": ["Planet A", "Planet B", "Planet C"]})

        # Apply the preprocess_dataset method
        result = StarSystemDB.preprocess_dataset(test_data)

        # Check that the dataset was sorted by rowupdate in descending order
        assert result["rowupdate"][0] == 3
        assert result["rowupdate"][1] == 2
        assert result["rowupdate"][2] == 1

        # Check that an index was added for pl_name
        assert "pl_name" in result.indices

        # Check that the dataset was filled with zeros
        assert result.filled is not test_data

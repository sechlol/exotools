from astropy.table import QTable

from exotools import StarSystemDB
from exotools.db.star_system import StarSystem


class TestStarSystemDb:
    def test_init(self, star_system_test_db):
        """Test initialization of StarSystemDB."""
        # Check that the dataset was properly set
        assert len(star_system_test_db) > 0
        assert star_system_test_db._id_column == "tic_id"

    def test_get_valid_planets(self, star_system_test_db):
        """Test the get_valid_planets method."""
        # pl_valid_flag column is now added in the fixture
        valid_planets = star_system_test_db.get_valid_planets()
        assert isinstance(valid_planets, StarSystemDB)
        assert len(valid_planets) > 0
        assert all(valid_planets.view["pl_valid_flag"])

    def test_get_star_system_from_star_name(self, star_system_test_db):
        """Test the get_star_system_from_star_name method."""
        # Get the first star name from the dataset
        if len(star_system_test_db) > 0:
            star_name = star_system_test_db.view["hostname"][0]
            star_system = star_system_test_db.get_star_system_from_star_name(star_name)
            assert isinstance(star_system, StarSystem)
            assert star_system.star_name == star_name

            # Test with non-existent star name
            non_existent_star = "NonExistentStar123456789"
            assert star_system_test_db.get_star_system_from_star_name(non_existent_star) is None

    def test_get_star_system_from_tic_id(self, star_system_test_db):
        """Test the get_star_system_from_tic_id method."""
        # Get the first TIC ID from the dataset
        if len(star_system_test_db) > 0:
            tic_id = star_system_test_db.view["tic_id"][0]
            star_system = star_system_test_db.get_star_system_from_tic_id(tic_id)
            assert isinstance(star_system, StarSystem)

            # Test with non-existent TIC ID
            non_existent_tic = 999999999
            assert star_system_test_db.get_star_system_from_tic_id(non_existent_tic) is None

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
        assert "pl_name" in result.colnames

        # Check that the dataset was filled with zeros
        assert result.filled is not test_data

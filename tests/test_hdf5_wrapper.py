from astropy import units as u
from astropy.table import QTable

from exotools.io.hdf5_storage import Hdf5Wrapper
from exotools.utils.qtable_utils import get_header_from_table
from tests.conftest import TEST_TMP_DIR, TEST_ASSETS_QTABLES
from tests.utils.comparison import compare_qtables

_TEST_FILE = TEST_TMP_DIR / "test.hdf5"


class TestHdf5Wrapper:
    def setup_method(self):
        """Create test file before each test method"""
        _TEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test file after each test method"""
        if _TEST_FILE.exists():
            _TEST_FILE.unlink()

    def test_read_and_write_json(self):
        w = Hdf5Wrapper(_TEST_FILE)

        # Test data
        test_data = {
            "name": "test_dataset",
            "version": "1.0",
            "metadata": {"created_by": "test", "items": [1, 2, 3, 4, 5]},
        }

        w.write_json(test_data, "test_config")
        read_data = w.read_json("test_config")

        assert read_data == test_data
        assert read_data["name"] == "test_dataset"
        assert read_data["metadata"]["items"] == [1, 2, 3, 4, 5]

    def test_write_json_same_name(self):
        w = Hdf5Wrapper(_TEST_FILE)
        name = "duplicate_name"

        # Write initial data
        initial_data = {"value": "first"}
        w.write_json(initial_data, name)

        # Try to write with same name - should raise ValueError
        duplicate_data = {"value": "second"}
        try:
            w.write_json(duplicate_data, name)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert "already exists" in str(e)

        # Verify original data is still there
        read_data = w.read_json(name)
        assert read_data == initial_data

    def test_read_and_write_qtable_simple(self):
        w = Hdf5Wrapper(_TEST_FILE)

        # Create test QTable with units
        test_table = QTable(
            {
                "name": ["star1", "star2", "star3"],
                "ra": [10.5, 20.3, 30.1] * u.deg,
                "dec": [-5.2, 15.7, -25.8] * u.deg,
                "magnitude": [12.5, 14.2, 11.8] * u.mag,
            }
        )

        # Add descriptions
        test_table["name"].description = "Star identifier"
        test_table["ra"].description = "Right ascension"
        test_table["dec"].description = "Declination"
        test_table["magnitude"].description = "Apparent magnitude"

        # Get header and write table
        header = get_header_from_table(test_table)
        w.write_qtable(test_table, header, "test_stars")

        # Read it back
        read_table = w.read_qtable("test_stars")

        # Verify the data matches using our comparison function
        assert compare_qtables(test_table, read_table)

    def test_write_read_qtable_full(self, all_test_qtables: dict[str, QTable]):
        w = Hdf5Wrapper(_TEST_FILE)
        for name, qtable in all_test_qtables.items():
            header = get_header_from_table(qtable)
            w.write_qtable(qtable, header, name)
            read_qtable = w.read_qtable(name)
            assert compare_qtables(qtable, read_qtable)

    def test_write_qtable_same_name(self):
        w = Hdf5Wrapper(_TEST_FILE)

        # Create test table
        test_path_1 = TEST_ASSETS_QTABLES / "exo_db.ecsv"
        test_path_2 = TEST_ASSETS_QTABLES / "candidate_db.ecsv"
        test_table_1 = QTable.read(test_path_1)
        test_table_2 = QTable.read(test_path_2)

        header = get_header_from_table(test_table_1)

        # Write initial table
        w.write_qtable(test_table_1, header, "duplicate_table")

        try:
            w.write_qtable(test_table_2, header, "duplicate_table")
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert "already exists" in str(e)
            assert "duplicate_table" in str(e)

    def test_get_structure(self):
        w = Hdf5Wrapper(_TEST_FILE)

        # Initially the file doesn't exist, so structure should be empty
        assert w.get_structure() == []

        # Add some data
        w.write_json({"test": "data"}, "json_data")

        # Create test QTable
        test_table = QTable(
            {
                "name": ["star1", "star2", "star3"],
                "ra": [10.5, 20.3, 30.1] * u.deg,
            }
        )
        header = get_header_from_table(test_table)
        w.write_qtable(test_table, header, "test_stars")

        # Get structure
        structure = w.get_structure()

        # Check that our data is in the structure
        assert len(structure) > 0

        # Check for the JSON dataset
        json_found = False
        for item in structure:
            if "json_data" in item and "DATASET" in item:
                json_found = True
                break
        assert json_found, "JSON dataset not found in structure"

        # Check for the QTable data
        qtable_found = False
        for item in structure:
            if "test_stars" in item:
                qtable_found = True
                break
        assert qtable_found, "QTable data not found in structure"

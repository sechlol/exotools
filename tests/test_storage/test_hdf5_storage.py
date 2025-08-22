import pytest
from astropy import units as u
from astropy.table import QTable

from exotools.io.hdf5_storage import Hdf5Storage
from exotools.utils.qtable_utils import QTableHeader, get_header_from_table
from tests.conftest import TEST_TMP_DIR
from tests.utils.table_comparison import compare_qtables

_TEST_FILE = TEST_TMP_DIR / "test.hdf5"


@pytest.fixture
def storage_wrapper():
    """Fixture that creates a storage wrapper instance for testing."""
    return Hdf5Storage(_TEST_FILE)


class TestHdf5Storage:
    def setup_method(self):
        """Create test file before each test method"""
        _TEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test file after each test method"""
        if _TEST_FILE.exists():
            _TEST_FILE.unlink()

    def test_read_and_write_json(self, storage_wrapper):
        w = storage_wrapper

        # Test data
        test_data = {
            "name": "test_dataset",
            "version": "1.0",
            "metadata": {"created_by": "test", "items": [1, 2, 3, 4, 5]},
        }

        # Write JSON data
        w.write_json(test_data, "test_config")

        # Read it back
        read_data = w.read_json("test_config")

        # Verify the data matches
        assert read_data == test_data
        assert read_data["name"] == "test_dataset"
        assert read_data["metadata"]["items"] == [1, 2, 3, 4, 5]

    def test_write_json_same_name_without_override(self, storage_wrapper):
        w = storage_wrapper

        # Write initial data
        initial_data = {"value": "first"}
        w.write_json(initial_data, "duplicate_name")

        # Try to write with same name without override - should raise ValueError
        duplicate_data = {"value": "second"}
        with pytest.raises(ValueError) as excinfo:
            w.write_json(duplicate_data, "duplicate_name")

        # Check error message
        assert "already exists" in str(excinfo.value)
        assert "duplicate_name" in str(excinfo.value)

        # Verify original data is still there
        read_data = w.read_json("duplicate_name")
        assert read_data == initial_data

    def test_write_json_same_name_with_override(self, storage_wrapper):
        w = storage_wrapper

        # Write initial data
        initial_data = {"value": "first"}
        w.write_json(initial_data, "override_test")

        # Write with same name using override=True
        new_data = {"value": "second", "extra": "field"}
        w.write_json(new_data, "override_test", override=True)

        # Verify new data overwrote the old
        read_data = w.read_json("override_test")
        assert read_data == new_data
        assert read_data["value"] == "second"
        assert read_data["extra"] == "field"

    def test_read_json_nonexistent_file(self, storage_wrapper):
        w = storage_wrapper

        # Try to read non-existent file
        with pytest.raises(Exception):
            w.read_json("nonexistent")

        # Check that an appropriate error is raised
        # Note: The exact error type might differ from fs_wrapper

    def test_read_and_write_qtable_simple(self, storage_wrapper):
        w = storage_wrapper

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
        assert compare_qtables(test_table, read_table)

    def test_write_read_qtable_full(
        self, storage_wrapper, all_test_qtables_and_headers: dict[str, tuple[QTable, QTableHeader]]
    ):
        w = storage_wrapper
        for name, (test_qtable, test_header) in all_test_qtables_and_headers.items():
            try:
                w.write_qtable(test_qtable, test_header, name)
            except Exception as e:
                print(e)
                raise
            read_qtable = w.read_qtable(name)
            read_header = w.read_qtable_header(name)
            assert test_header == read_header
            assert compare_qtables(test_qtable, read_qtable)

    def test_read_qtable_nonexistent(self, storage_wrapper):
        w = storage_wrapper

        # Try to read non-existent table
        try:
            w.read_qtable("nonexistent_table")
            assert False, "Expected FileNotFoundError was not raised"
        except FileNotFoundError:
            assert True

    def test_read_qtable_header(self, storage_wrapper):
        w = storage_wrapper

        # Create test QTable with units
        test_table = QTable(
            {
                "name": ["star1", "star2", "star3"],
                "ra": [10.5, 20.3, 30.1] * u.deg,
                "dec": [-5.2, 15.7, -25.8] * u.deg,
            }
        )

        # Add descriptions
        test_table["name"].description = "Star identifier"
        test_table["ra"].description = "Right ascension"
        test_table["dec"].description = "Declination"

        # Get header and write table
        header = get_header_from_table(test_table)
        w.write_qtable(test_table, header, "test_stars_header")

        # Read header back
        read_header = w.read_qtable_header("test_stars_header")

        # Verify header matches
        assert read_header["name"].description == "Star identifier"
        assert read_header["ra"].description == "Right ascension"
        assert read_header["ra"].unit == "deg"
        assert read_header["dec"].description == "Declination"
        assert read_header["dec"].unit == "deg"

    def test_write_qtable_same_name_without_override(self, storage_wrapper):
        w = storage_wrapper

        # Create test QTable
        test_table = QTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        header = get_header_from_table(test_table)

        # Write initial table
        w.write_qtable(test_table, header, "duplicate_table")

        # Try to write with same name without override - should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            w.write_qtable(test_table, header, "duplicate_table")

        # Check error message
        assert "already exists" in str(excinfo.value)

    def test_write_qtable_same_name_with_override(self, storage_wrapper):
        w = storage_wrapper

        # Create initial test QTable
        initial_table = QTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        initial_header = get_header_from_table(initial_table)

        # Write initial table
        w.write_qtable(initial_table, initial_header, "override_table")

        # Create new test QTable with different data
        new_table = QTable({"a": [10, 20, 30], "c": [40, 50, 60]})
        new_header = get_header_from_table(new_table)

        # Write with same name using override=True
        w.write_qtable(new_table, new_header, "override_table", override=True)

        # Read it back and verify it's the new table
        read_table = w.read_qtable("override_table")
        assert compare_qtables(new_table, read_table)

        # Verify it's not the initial table by checking specific values
        # Since compare_qtables raises an exception for different schemas, we check values directly
        assert "b" not in read_table.colnames  # Initial table had 'b', new one doesn't
        assert "c" in read_table.colnames  # New table has 'c', initial one didn't
        assert read_table["a"][0] == 10  # Value from new table, not initial table

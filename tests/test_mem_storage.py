import pytest
from astropy import units as u
from astropy.table import QTable

from exotools.io.memory_storage import MemoryStorage
from exotools.utils.qtable_utils import get_header_from_table, QTableHeader
from tests.utils.comparison import compare_qtables


@pytest.fixture
def storage_wrapper():
    """Fixture that creates a memory storage wrapper instance for testing."""
    return MemoryStorage(name="test_instance")


@pytest.fixture
def another_storage_wrapper():
    """Fixture that creates another memory storage wrapper instance for testing shared memory."""
    return MemoryStorage(name="another_test_instance")


class TestMemoryStorage:
    def setup_method(self):
        """Clear class memory before each test method"""
        MemoryStorage.clear()

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
        with pytest.raises(ValueError) as excinfo:
            w.read_json("nonexistent")

        # Check error message
        assert "does not exist" in str(excinfo.value)
        assert "nonexistent" in str(excinfo.value)

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
            w.write_qtable(test_qtable, test_header, name)
            read_qtable = w.read_qtable(name)
            read_header = w.read_qtable_header(name)
            assert compare_qtables(test_qtable, read_qtable)
            assert test_header == read_header

    def test_read_qtable_nonexistent(self, storage_wrapper):
        w = storage_wrapper

        # Try to read non-existent table
        with pytest.raises(ValueError) as excinfo:
            w.read_qtable("nonexistent_table")

        # Check error message
        assert "does not exist" in str(excinfo.value)
        assert "nonexistent_table" in str(excinfo.value)

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
        assert "b" not in read_table.colnames  # Initial table had 'b', new one doesn't
        assert "c" in read_table.colnames  # New table has 'c', initial one didn't
        assert read_table["a"][0] == 10  # Value from new table, not initial table

    def test_shared_memory_between_instances(self, storage_wrapper, another_storage_wrapper):
        """Test that memory is shared between different instances but namespaced by instance name."""
        w1 = storage_wrapper
        w2 = another_storage_wrapper

        # Create data in first instance
        data1 = {"value": "from_instance_1"}
        w1.write_json(data1, "shared_test")

        # Create data with same name in second instance
        data2 = {"value": "from_instance_2"}
        w2.write_json(data2, "shared_test")

        # Verify each instance can read its own data
        read1 = w1.read_json("shared_test")
        read2 = w2.read_json("shared_test")

        assert read1 == data1
        assert read2 == data2
        assert read1 != read2

        # Verify data is stored with namespaced keys in the shared memory
        assert len(MemoryStorage._memory) == 2

    def test_data_independence(self, storage_wrapper):
        """Test that modifying returned data doesn't affect stored data."""
        w = storage_wrapper

        # Create original data
        original_data = {"items": [1, 2, 3], "nested": {"value": "test"}}
        w.write_json(original_data, "independence_test")

        # Get data and modify it
        read_data = w.read_json("independence_test")
        read_data["items"].append(4)
        read_data["nested"]["value"] = "modified"

        # Read again and verify original data is unchanged
        reread_data = w.read_json("independence_test")
        assert reread_data["items"] == [1, 2, 3]
        assert reread_data["nested"]["value"] == "test"

        # Same test with QTable
        test_table = QTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        header = get_header_from_table(test_table)
        w.write_qtable(test_table, header, "table_independence")

        # Get table and modify it
        read_table = w.read_qtable("table_independence")
        read_table["a"][0] = 999

        # Read again and verify original data is unchanged
        reread_table = w.read_qtable("table_independence")
        assert reread_table["a"][0] == 1

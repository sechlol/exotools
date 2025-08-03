import shutil

import pytest
from astropy import units as u
from astropy.table import QTable

from exotools.io.fs_storage import EcsvStorage, FeatherStorage
from exotools.utils.qtable_utils import QTableHeader, get_header_from_table

from .conftest import TEST_TMP_DIR
from .utils.comparison import compare_qtables

_TEST_DIR = TEST_TMP_DIR / "fs_test"
_TEST_HDF5 = _TEST_DIR / "test.hdf5"


@pytest.fixture(params=[FeatherStorage, EcsvStorage])
def storage_class(request):
    """Fixture that provides both storage classes for testing."""
    return request.param


@pytest.fixture
def storage_wrapper(storage_class):
    """Fixture that creates a storage wrapper instance for testing."""

    return storage_class(_TEST_DIR)


class TestFsStorage:
    def setup_method(self):
        """Create test directory before each test method"""
        _TEST_DIR.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test directory after each test method"""
        if _TEST_DIR.exists():
            shutil.rmtree(_TEST_DIR)

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

        # Verify file was created
        json_file = _TEST_DIR / "test_config.json"
        assert json_file.exists()

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
        try:
            w.write_json(duplicate_data, "duplicate_name")
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert "already exists" in str(e)
            assert "duplicate_name" in str(e)

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
        try:
            w.read_json("nonexistent")
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert "does not exist" in str(e)
            assert "nonexistent" in str(e)

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

        # Verify files were created
        data_file = _TEST_DIR / f"test_stars{w._suffix}"
        header_file = _TEST_DIR / "test_stars_header.json"
        assert data_file.exists()
        assert header_file.exists()

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
        try:
            w.read_qtable("nonexistent_table")
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert "does not exist" in str(e)

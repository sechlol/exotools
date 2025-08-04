import numpy as np
import pandas as pd
import pytest
from astropy.table import QTable
from typing_extensions import Self

from exotools.db.base_db import NAN_VALUE, BaseDB


class TestDB(BaseDB):
    """Concrete implementation of BaseDB for testing purposes."""

    def _factory(self, dataset: QTable) -> Self:
        return TestDB(dataset, id_field="id")


class TestBaseDb:
    @pytest.fixture
    def sample_qtable(self):
        """Create a sample QTable for testing."""
        data = {
            "id": [1, 2, 3, 4, 5, NAN_VALUE, 1],  # Note the duplicate ID and NAN_VALUE
            "name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Invalid", "Alpha Duplicate"],
            "value": [10.5, 20.3, 15.7, 8.2, 30.1, 0.0, 11.0],
            "category": ["A", "B", "A", "C", "B", "D", "A"],
            "flag": [True, False, True, True, False, False, True],
        }

        # Create QTable
        qtable = QTable(data)
        return qtable

    @pytest.fixture
    def test_db(self, sample_qtable):
        """Create a TestDB instance with the sample data."""
        return TestDB(sample_qtable, id_field="id")

    @pytest.fixture
    def empty_qtable_with_columns(self):
        """Create an empty QTable with columns but no data."""
        data = {
            "id": [],
            "name": [],
            "value": [],
            "category": [],
            "flag": [],
        }
        return QTable(data)

    @pytest.fixture
    def empty_qtable_no_columns(self):
        """Create a completely empty QTable with no columns."""
        return QTable()

    def test_init(self, sample_qtable):
        """Test initialization of BaseDB."""
        db = TestDB(sample_qtable, id_field="id")

        # Check internal state
        assert db._ds is sample_qtable
        assert db._id_column == "id"

    def test_init_empty_with_columns(self, empty_qtable_with_columns):
        """Test initialization with empty dataset that has columns."""
        db = TestDB(empty_qtable_with_columns, id_field="id")
        assert len(db) == 0
        assert db._id_column == "id"

    def test_init_empty_no_columns(self, empty_qtable_no_columns):
        """Test initialization with completely empty dataset raises error."""
        with pytest.raises(ValueError, match="empty column set"):
            TestDB(empty_qtable_no_columns, id_field="id")

    def test_len(self, test_db, sample_qtable):
        """Test __len__ method."""
        assert len(test_db) == len(sample_qtable)

    def test_view(self, test_db, sample_qtable):
        """Test view property."""
        assert test_db.view is sample_qtable

    def test_dataset_copy(self, test_db, sample_qtable):
        """Test dataset_copy property."""
        copy = test_db.dataset_copy
        assert copy is not sample_qtable  # Should be a different object
        assert len(copy) == len(sample_qtable)

        # Modify the copy and ensure original is unchanged
        copy.add_row([99, "New", 50.0, "X", False])
        assert len(copy) == len(sample_qtable) + 1

    def test_where_single_value(self, test_db):
        """Test where method with a single value filter."""
        filtered = test_db.where(category="A")
        assert len(filtered) == 3
        assert all(row["category"] == "A" for row in filtered.view)

    def test_where_multiple_values(self, test_db):
        """Test where method with multiple values filter."""
        filtered = test_db.where(category=["A", "B"])
        assert len(filtered) == 5
        assert all(row["category"] in ["A", "B"] for row in filtered.view)

    def test_where_numpy_array(self, test_db):
        """Test where method with a numpy array of values."""
        # Create a numpy array of IDs to filter by
        ids_to_filter = np.array([1, 3, 5])

        # Filter using the numpy array
        filtered = test_db.where(id=ids_to_filter)

        # Should return rows with IDs 1, 3, and 5 (note that ID 1 appears twice)
        assert len(filtered) == 4

        # Check that all returned IDs are in the filter array
        assert all(row["id"] in ids_to_filter for row in filtered.view)

        # Check that the specific names we expect are present
        expected_names = {"Alpha", "Gamma", "Epsilon", "Alpha Duplicate"}
        actual_names = set(filtered.view["name"])
        assert actual_names == expected_names

    def test_where_multiple_conditions(self, test_db):
        """Test where method with multiple conditions."""
        filtered = test_db.where(category="A", flag=True)
        assert len(filtered) == 3
        assert all(row["category"] == "A" and row["flag"] for row in filtered.view)

    def test_where_multiple_conditions_mixed_values(self, test_db):
        """Test where method with multiple conditions using lists and single values."""
        filtered = test_db.where(category=["A", "C"], flag=True)
        assert len(filtered) == 4
        assert all((row["category"] in ["A", "C"]) and row["flag"] for row in filtered.view)

    def test_where_no_match(self, test_db):
        """Test where method with no matching records."""
        filtered = test_db.where(category="Z")
        assert len(filtered) == 0

    def test_where_true(self, test_db):
        """Test where_true method."""
        mask = test_db.view["value"] > 15.0
        filtered = test_db.where_true(mask)
        assert len(filtered) == 3
        assert all(row["value"] > 15.0 for row in filtered.view)

    def test_with_valid_ids(self, test_db):
        """Test with_valid_ids method."""
        # Original dataset has 7 rows, one with NAN_VALUE as ID
        assert len(test_db) == 7

        # Get only records with valid IDs
        filtered = test_db.with_valid_ids()

        # Should have 6 rows (all except the one with NAN_VALUE)
        assert len(filtered) == 6

        # All IDs should be valid (not NAN_VALUE)
        assert all(row["id"] != NAN_VALUE for row in filtered.view)

        # Check that the row with name "Invalid" is excluded
        assert "Invalid" not in filtered.view["name"]

    def test_filter_chain(self, test_db):
        """Test chaining multiple filters together."""
        # Start with the full dataset (7 rows)
        assert len(test_db) == 7

        # Chain 1: Filter to only valid IDs (6 rows)
        chain1 = test_db.with_valid_ids()
        assert len(chain1) == 6

        # Chain 2: Filter to only category A or B (5 rows in filtered)
        chain2 = chain1.where(category=["A", "B"])
        assert len(chain2) == 5
        assert all(row["category"] in ["A", "B"] and row["id"] != NAN_VALUE for row in chain2.view)

        # Chain 3: Filter to only rows with value > 15 (3 rows)
        chain3 = chain2.where_true(chain2.view["value"] > 15.0)
        assert len(chain3) == 3
        assert all(
            row["category"] in ["A", "B"] and row["id"] != NAN_VALUE and row["value"] > 15.0 for row in chain3.view
        )

        # Verify the specific rows that should remain
        names_in_result = set(chain3.view["name"])
        assert names_in_result == {"Beta", "Epsilon", "Gamma"}

        # Alternative chaining syntax in a single expression
        value_threshold = 15.0
        result = test_db.where_true(test_db.view["value"] > value_threshold).with_valid_ids().where(category=["A", "B"])

        # Should get the same result as the step-by-step chain
        assert len(result) == len(chain3)
        assert set(result.view["name"]) == names_in_result

    def test_select_random_sample(self, test_db):
        """Test select_random_sample method."""
        # Set seed for reproducibility
        np.random.seed(42)

        # Select 3 random samples
        sample = test_db.select_random_sample(n=3)

        # Should return a TestDB instance
        assert isinstance(sample, TestDB)

        # Should have 3 rows
        assert len(sample.view) == 3

    def test_to_pandas(self, test_db):
        """Test to_pandas method."""
        df = test_db.to_pandas()

        # Should be a pandas DataFrame
        assert isinstance(df, pd.DataFrame)

        # Should have same number of rows as the original QTable
        assert len(df) == len(test_db.view)

        # Should have the same columns, and no index column
        assert set(df.columns) == {"id", "name", "value", "category", "flag"}

    def test_to_pandas_empty(self, empty_qtable_with_columns):
        """Test to_pandas method with an empty dataset."""
        empty_db = TestDB(empty_qtable_with_columns, id_field="id")
        df = empty_db.to_pandas()

        # Should be an empty DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

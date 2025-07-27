from astropy.table import QTable

from tests.utils import compare_qtables


class TestUtils:
    def test_compare_tables(self, all_test_qtables: dict[str, QTable]):
        for qtable_name, qtable in all_test_qtables.items():
            assert compare_qtables(expected_table=qtable, test_table=qtable)

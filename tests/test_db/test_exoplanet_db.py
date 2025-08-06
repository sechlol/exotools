import numpy as np
import pytest
from astropy.table import QTable

from exotools import ExoDB
from exotools.utils.qtable_utils import QTableHeader


class TestExoplanetDb:
    @pytest.fixture
    def exo_db(self, known_exoplanets_test_data: tuple[QTable, QTableHeader]) -> ExoDB:
        return ExoDB(exoplanets_dataset=known_exoplanets_test_data[0])

    def test_init(self, exo_db, known_exoplanets_test_data):
        """Test initialization of ExoDB."""
        # Check that the dataset was properly set
        assert len(exo_db) == len(known_exoplanets_test_data[0])
        assert exo_db._id_column == "tic_id"

    def test_tic_ids_property(self, exo_db):
        """Test the tic_ids property."""
        tic_ids = exo_db.tic_ids
        assert isinstance(tic_ids, np.ndarray)
        assert len(tic_ids) == len(exo_db)

    def test_gaia_ids_property(self, exo_db):
        """Test the gaia_ids property."""
        gaia_ids = exo_db.gaia_ids
        assert isinstance(gaia_ids, np.ndarray)
        assert len(gaia_ids) == len(exo_db)

    def test_unique_tic_ids_property(self, exo_db):
        """Test the unique_tic_ids property."""
        unique_tic_ids = exo_db.unique_tic_ids
        assert isinstance(unique_tic_ids, np.ndarray)
        assert len(unique_tic_ids) <= len(exo_db)
        # Check that all IDs are unique
        assert len(unique_tic_ids) == len(set(unique_tic_ids))

    def test_unique_gaia_ids_property(self, exo_db):
        """Test the unique_gaia_ids property."""
        unique_gaia_ids = exo_db.unique_gaia_ids
        assert isinstance(unique_gaia_ids, np.ndarray)
        assert len(unique_gaia_ids) <= len(exo_db)
        # Check that all IDs are unique
        assert len(unique_gaia_ids) == len(set(unique_gaia_ids))

    def test_get_star_names(self, exo_db):
        """Test the get_star_names method."""
        star_names = exo_db.get_star_names()
        assert isinstance(star_names, list)
        assert all(isinstance(name, str) for name in star_names)
        # Check that all names are unique
        assert len(star_names) == len(set(star_names))

    def test_get_default_records(self, exo_db):
        """Test the get_default_records method."""
        default_records = exo_db.get_default_records()
        assert isinstance(default_records, ExoDB)
        # Check that all records have default_flag = 1
        if len(default_records) > 0:
            assert all(record["default_flag"] == 1 for record in default_records.view)

    def test_get_tess_planets(self, exo_db):
        """Test the get_tess_planets method."""
        tess_planets = exo_db.get_tess_planets()
        assert isinstance(tess_planets, ExoDB)
        # Check that all records have TESS in disc_telescope
        if len(tess_planets) > 0:
            assert all("TESS" in record["disc_telescope"] for record in tess_planets.view)

    def test_get_kepler_planets(self, exo_db):
        """Test the get_kepler_planets method."""
        kepler_planets = exo_db.get_kepler_planets()
        assert isinstance(kepler_planets, ExoDB)
        # Check that all records have Kepler in disc_telescope
        if len(kepler_planets) > 0:
            assert all("Kepler" in record["disc_telescope"] for record in kepler_planets.view)

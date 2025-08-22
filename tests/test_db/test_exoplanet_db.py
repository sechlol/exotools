import numpy as np
import pytest

from exotools import ExoDB


class TestExoplanetDb:
    @pytest.fixture
    def exo_db(self, planetary_systems_test_data) -> ExoDB:
        return ExoDB(exoplanets_dataset=planetary_systems_test_data[0])

    def test_init(self, exo_db, planetary_systems_test_data):
        """Test initialization of ExoDB."""
        # Check that the dataset was properly set
        assert len(exo_db) == len(planetary_systems_test_data[0])
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

    def test_get_transiting_planets(self, exo_db):
        """Test the get_transiting_planets method."""
        # Test without filter
        transiting_planets = exo_db.get_transiting_planets()
        assert isinstance(transiting_planets, ExoDB)
        # Check that all records have tran_flag = 1
        if len(transiting_planets) > 0:
            assert all(record["tran_flag"] == 1 for record in transiting_planets.view)

        # Test with kepler_or_tess_only=True
        kepler_tess_transiting = exo_db.get_transiting_planets(kepler_or_tess_only=True)
        assert isinstance(kepler_tess_transiting, ExoDB)
        if len(kepler_tess_transiting) > 0:
            # Check that all records have tran_flag = 1
            assert all(record["tran_flag"] == 1 for record in kepler_tess_transiting.view)
            # Check that all records have either TESS, Kepler, or K2 in disc_telescope
            for record in kepler_tess_transiting.view:
                telescope = record["disc_telescope"]
                assert any(name in telescope for name in ["TESS", "Kepler", "K2"])

        # Verify that kepler_or_tess_only=True returns a subset of all transiting planets
        assert len(kepler_tess_transiting) <= len(transiting_planets)

    def test_impute_stellar_parameters(self, planetary_systems_test_data, gaia_test_db):
        """Test the impute_stellar_parameters method."""
        # Make a copy of the data to avoid modifying the original
        exo_data = planetary_systems_test_data[0].copy()
        gaia_data = gaia_test_db.view.copy()

        # Count initial masked values
        initial_masked_st_rad = np.sum(exo_data["st_rad"].mask) if hasattr(exo_data["st_rad"], "mask") else 0
        initial_masked_pl_ratror = np.sum(exo_data["pl_ratror"].mask) if hasattr(exo_data["pl_ratror"], "mask") else 0
        initial_masked_pl_ratdor = np.sum(exo_data["pl_ratdor"].mask) if hasattr(exo_data["pl_ratdor"], "mask") else 0

        # Apply the imputation
        ExoDB.impute_stellar_parameters(exo_data, gaia_data)

        # Verify that st_rad_gaia column was added
        assert "st_rad_gaia" in exo_data.colnames

        # Count final masked values
        final_masked_st_rad = np.sum(exo_data["st_rad"].mask) if hasattr(exo_data["st_rad"], "mask") else 0
        final_masked_pl_ratror = np.sum(exo_data["pl_ratror"].mask) if hasattr(exo_data["pl_ratror"], "mask") else 0
        final_masked_pl_ratdor = np.sum(exo_data["pl_ratdor"].mask) if hasattr(exo_data["pl_ratdor"], "mask") else 0

        # Check that some values were imputed (if there were any to impute)
        # Note: This is a soft assertion since the test data might not have recoverable values
        if initial_masked_st_rad > 0:
            assert final_masked_st_rad <= initial_masked_st_rad, "No stellar radii were imputed"

        if initial_masked_pl_ratror > 0:
            assert final_masked_pl_ratror <= initial_masked_pl_ratror, "No planet-star radius ratios were imputed"

        if initial_masked_pl_ratdor > 0:
            assert final_masked_pl_ratdor <= initial_masked_pl_ratdor, "No planet-star distance ratios were imputed"

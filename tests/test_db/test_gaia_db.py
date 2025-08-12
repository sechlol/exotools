import numpy as np
from astropy.table import MaskedColumn, QTable

from exotools import GaiaDB


class TestGaiaDb:
    def test_init(self, gaia_test_db):
        """Test initialization of GaiaDB."""
        # Check that the dataset was properly set
        assert len(gaia_test_db) > 0
        assert gaia_test_db._id_column == "gaia_id"

    def test_gaia_ids_property(self, gaia_test_db):
        """Test the gaia_ids property."""
        gaia_ids = gaia_test_db.gaia_ids
        assert isinstance(gaia_ids, np.ndarray)
        assert len(gaia_ids) == len(gaia_test_db)

    def test_unique_gaia_ids_property(self, gaia_test_db):
        """Test the unique_gaia_ids property."""
        unique_gaia_ids = gaia_test_db.unique_gaia_ids
        assert isinstance(unique_gaia_ids, np.ndarray)
        assert len(unique_gaia_ids) <= len(gaia_test_db)
        # Check that all IDs are unique
        assert len(unique_gaia_ids) == len(set(unique_gaia_ids))

    def test_impute_radius(self, gaia_test_db):
        """Test the impute_radius static method."""
        # Create a test dataset with masked values
        test_data = QTable()
        test_data["radius_flame"] = MaskedColumn([1.0, 2.0, -1, 4.0], mask=[False, False, True, False])
        test_data["radius_gspphot"] = MaskedColumn([1.5, -1, 3.0, 4.5], mask=[False, True, False, False])

        # Apply the impute_radius method
        result = GaiaDB.impute_radius(test_data)

        # Check that a new 'radius' column was added
        assert "radius" in result.colnames

        # Check that values were imputed correctly
        assert result["radius"][0] == 1.25  # Average of 1.0 and 1.5
        assert result["radius"][1] == 2.0  # Only radius_flame available
        assert result["radius"][2] == 3.0  # Only radius_gspphot available
        assert result["radius"][3] == 4.25  # Average of 4.0 and 4.5

    def test_compute_mean_temperature(self, gaia_test_db):
        """Test the compute_mean_temperature static method."""
        # Create a test dataset
        test_data = QTable()
        test_data["teff_gspphot"] = [5000, 6000, 7000]
        test_data["teff_gspspec"] = [5100, 6100, 7100]
        test_data["teff_esphs"] = [5200, 6200, 7200]
        test_data["teff_espucd"] = [5300, 6300, 7300]
        test_data["teff_msc1"] = [5400, 6400, 7400]
        test_data["teff_msc2"] = [5500, 6500, 7500]

        # Apply the compute_mean_temperature method
        result = GaiaDB.compute_mean_temperature(test_data)

        # Check that a new 'teff_mean' column was added
        assert "teff_mean" in result.colnames

        # Check that values were computed correctly
        assert result["teff_mean"][0] == 5250.0  # Average of all temperatures for first row
        assert result["teff_mean"][1] == 6250.0  # Average of all temperatures for second row
        assert result["teff_mean"][2] == 7250.0  # Average of all temperatures for third row

    def test_compute_habitable_zone(self, gaia_test_db):
        """Test the compute_habitable_zone static method."""
        # Create a test dataset
        test_data = QTable()
        test_data["lum_flame"] = [1.0, 4.0, 0.0, -1.0]  # Include invalid values

        # Apply the compute_habitable_zone method
        result = GaiaDB.compute_habitable_zone(test_data)

        # Check that new columns were added
        assert "inner_hz" in result.colnames
        assert "outer_hz" in result.colnames

        # Check that values were computed correctly
        assert np.isclose(result["inner_hz"][0], 0.9534, rtol=1e-4)  # sqrt(1.0/1.1)
        assert np.isclose(result["outer_hz"][0], 1.3736, rtol=1e-4)  # sqrt(1.0/0.53)
        assert np.isclose(result["inner_hz"][1], 1.9069, rtol=1e-4)  # sqrt(4.0/1.1)
        assert np.isclose(result["outer_hz"][1], 2.7472, rtol=1e-4)  # sqrt(4.0/0.53)
        assert np.isnan(result["inner_hz"][2])  # 0.0 luminosity
        assert np.isnan(result["outer_hz"][2])  # 0.0 luminosity
        assert np.isnan(result["inner_hz"][3])  # -1.0 luminosity
        assert np.isnan(result["outer_hz"][3])  # -1.0 luminosity

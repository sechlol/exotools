from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from exotools.datasets import (
    CandidateExoplanetsDataset,
    GaiaParametersDataset,
    LightcurveDataset,
    PlanetarySystemsDataset,
    TicCatalogDataset,
)
from exotools.datasets.tic_observations import TicObservationsDataset
from exotools.io import MemoryStorage

from .conftest import TEST_ASSETS_LC


class TestDatasets:
    @staticmethod
    def teardown_method():
        TicCatalogDataset._catalog_downloader = None

    def test_planetary_systems_dataset(self, planetary_systems_test_data, gaia_parameters_test_data):
        """Test PlanetarySystemsDataset with mocked downloader"""
        qtable, header = planetary_systems_test_data
        gaia_qtable, gaia_header = gaia_parameters_test_data
        storage = MemoryStorage()

        # Mock the downloader
        with patch("exotools.datasets.planetary_systems.PlanetarySystemsDownloader") as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader.download.return_value = (qtable, header)
            mock_downloader_class.return_value = mock_downloader

            # Mock GaiaDownloader as well
            with patch("exotools.datasets.gaia_parameters.GaiaDownloader") as mock_gaia_downloader_class:
                mock_gaia_downloader = MagicMock()
                mock_gaia_downloader.download_by_id.return_value = (gaia_qtable, gaia_header)
                mock_gaia_downloader_class.return_value = mock_gaia_downloader

                # Test dataset
                dataset = PlanetarySystemsDataset(storage=storage)

                # Test download without Gaia data
                exo_db = dataset.download_known_exoplanets(with_gaia_star_data=False, store=True, limit=10)

                # Verify downloader was called correctly
                mock_downloader.download.assert_called_once_with(limit=10, columns=None, where=None)

                # Verify data was stored in memory
                data_key = storage._get_prefixed_key(dataset.name, ".qtable")
                assert data_key in storage._memory
                stored_qtable = storage._memory[data_key]
                assert len(stored_qtable) == len(qtable)

                # Verify ExoDB was created
                assert exo_db is not None
                assert len(exo_db._ds) == len(qtable)

                # Test loading stored dataset
                loaded_db = dataset.load_known_exoplanets_dataset(with_gaia_star_data=False)
                assert loaded_db is not None
                assert len(loaded_db._ds) == len(qtable)

    def test_candidate_exoplanets_dataset(self, candidate_exoplanets_test_data):
        """Test CandidateExoplanetsDataset with mocked downloader"""
        qtable, header = candidate_exoplanets_test_data
        storage = MemoryStorage()

        # Mock the downloader
        with patch("exotools.datasets.candidate_exoplanets.CandidateExoplanetsDownloader") as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader.download.return_value = (qtable, header)
            mock_downloader_class.return_value = mock_downloader

            # Test dataset
            dataset = CandidateExoplanetsDataset(storage=storage)

            # Test download
            candidate_db = dataset.download_candidate_exoplanets(limit=50, store=True)

            # Verify downloader was called correctly
            mock_downloader.download.assert_called_once_with(limit=50)

            # Verify data was stored in memory
            data_key = storage._get_prefixed_key(dataset.name, ".qtable")
            assert data_key in storage._memory
            stored_qtable = storage._memory[data_key]
            assert len(stored_qtable) == len(qtable)

            # Verify CandidateDB was created
            assert candidate_db is not None
            assert len(candidate_db._ds) == len(qtable)

            # Test loading stored dataset
            loaded_db = dataset.load_candidate_exoplanets_dataset()
            assert loaded_db is not None
            assert len(loaded_db._ds) == len(qtable)

    def test_gaia_dataset(self, gaia_parameters_test_data):
        """Test GaiaParametersDataset with mocked downloader"""
        qtable, header = gaia_parameters_test_data
        storage = MemoryStorage()

        # Mock the downloader
        with patch("exotools.datasets.gaia_parameters.GaiaDownloader") as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader.download_by_id.return_value = (qtable, header)
            mock_downloader_class.return_value = mock_downloader

            # Test dataset
            dataset = GaiaParametersDataset(storage=storage)

            # Test download with sample Gaia IDs
            test_gaia_ids = [123456789, 987654321, 555666777]
            gaia_db = dataset.download_gaia_parameters(gaia_ids=test_gaia_ids, store=True)

            # Verify downloader was called correctly (with ids= keyword argument)
            mock_downloader.download_by_id.assert_called_once_with(ids=test_gaia_ids)

            # Verify data was stored in memory
            data_key = storage._get_prefixed_key(dataset.name, ".qtable")
            assert data_key in storage._memory
            stored_qtable = storage._memory[data_key]
            assert len(stored_qtable) == len(qtable)

            # Verify GaiaDB was created
            assert gaia_db is not None
            assert len(gaia_db._ds) == len(qtable)

            # Test loading stored dataset
            loaded_db = dataset.load_gaia_parameters_dataset()
            assert loaded_db is not None
            assert len(loaded_db._ds) == len(qtable)

    def test_tess_authentication(self):
        """Test that authentication is required for TIC queries but not for metadata queries"""
        storage = MemoryStorage()

        # Create mock QTables with proper structure
        import numpy as np
        from astropy.table import QTable

        # Create a simple mock QTable for observations with required columns
        mock_obs_qtable = QTable(
            {"tic_id": np.array([123456, 789012], dtype=int), "sector": np.array([1, 2], dtype=int)}
        )
        mock_obs_header = MagicMock()

        # Create a simple mock QTable for TIC data with required columns
        mock_tic_qtable = QTable({"tic_id": np.array([123456], dtype=int), "mass": np.array([1.0], dtype=float)})
        mock_tic_header = MagicMock()

        # Mock the downloaders
        with patch("exotools.datasets.tic_observations.TessObservationsDownloader") as mock_obs_downloader_class:
            mock_obs_downloader = MagicMock()
            mock_obs_downloader.download_by_id.return_value = (mock_obs_qtable, mock_obs_header)
            mock_obs_downloader_class.return_value = mock_obs_downloader

            with patch("exotools.datasets.tic_catalog.TessCatalogDownloader") as mock_cat_downloader_class:
                mock_cat_downloader = MagicMock()
                mock_cat_downloader.download_by_id.return_value = (mock_tic_qtable, mock_tic_header)
                mock_cat_downloader.download.return_value = (mock_tic_qtable, mock_tic_header)
                mock_cat_downloader_class.return_value = mock_cat_downloader

                # Create dataset
                catalog_dataset = TicCatalogDataset(storage=storage)
                observation_dataset = TicObservationsDataset(storage=storage)

                # 1. Test that metadata can be queried without authentication
                test_tic_ids = [123456, 789012]
                observation_dataset.download_observation_metadata(targets_tic_id=test_tic_ids)
                mock_obs_downloader.download_by_id.assert_called_once_with(test_tic_ids)

                # 2. Test that TIC queries fail without authentication
                with pytest.raises(ValueError, match="You need to call TicCatalogDataset.authenticate()"):
                    catalog_dataset.download_tic_targets(limit=10)

                with pytest.raises(ValueError, match="You need to call TicCatalogDataset.authenticate()"):
                    catalog_dataset.download_tic_targets_by_ids(tic_ids=[123456])

                # 3. Test that authentication enables TIC queries
                TicCatalogDataset.authenticate_casjobs(username="test_user", password="test_pass")

                # Verify that the catalog downloader was created with the right credentials
                mock_cat_downloader_class.assert_called_once_with(username="test_user", password="test_pass")

                # Now TIC queries should work
                catalog_dataset.download_tic_targets(limit=10)
                mock_cat_downloader.download.assert_called_once_with(limit=10)

                catalog_dataset.download_tic_targets_by_ids(tic_ids=[123456])
                mock_cat_downloader.download_by_id.assert_called_once_with([123456])

    def test_tess_dataset(self, tic_observations_test_data):
        """Test TessDataset with mocked downloaders"""
        qtable, header = tic_observations_test_data
        storage = MemoryStorage()

        # Mock the observations downloader
        with patch("exotools.datasets.tic_observations.TessObservationsDownloader") as mock_obs_downloader_class:
            mock_obs_downloader = MagicMock()
            mock_obs_downloader.download_by_id.return_value = (qtable, header)
            mock_obs_downloader_class.return_value = mock_obs_downloader

            # Mock the catalog downloader (optional)
            with patch("exotools.datasets.tic_catalog.TessCatalogDownloader") as mock_cat_downloader_class:
                mock_cat_downloader = MagicMock()
                mock_cat_downloader.download_by_id.return_value = (qtable, header)
                mock_cat_downloader.download.return_value = (qtable, header)
                mock_cat_downloader_class.return_value = mock_cat_downloader

                # Test dataset
                catalog_dataset = TicCatalogDataset(storage=storage)
                observation_dataset = TicObservationsDataset(storage=storage)

                # Test download observation metadata
                test_tic_ids = [123456, 789012, 345678]
                tess_meta_db = observation_dataset.download_observation_metadata(
                    targets_tic_id=test_tic_ids, store=True
                )

                # Verify observations downloader was called correctly
                mock_obs_downloader.download_by_id.assert_called_once_with(test_tic_ids)

                # Verify data was stored in memory
                observations_name = observation_dataset.name
                data_key = storage._get_prefixed_key(observations_name, ".qtable")
                assert data_key in storage._memory
                stored_qtable = storage._memory[data_key]
                assert len(stored_qtable) == len(qtable)

                # Verify TessMetaDB was created (uses dataset attribute, not meta_dataset)
                assert tess_meta_db is not None
                assert len(tess_meta_db._ds) == len(qtable)

                # Test loading stored dataset
                loaded_db = observation_dataset.load_observation_metadata()
                assert loaded_db is not None
                assert len(loaded_db._ds) == len(qtable)

                # Test download_tic_targets
                TicCatalogDataset.authenticate_casjobs("username", "password")

                test_limit = 50
                test_mass_range = (0.8, 1.2)
                test_priority = 0.5
                tic_db = catalog_dataset.download_tic_targets(
                    limit=test_limit,
                    star_mass_range=test_mass_range,
                    priority_threshold=test_priority,
                    store=True,
                )

                # Verify catalog downloader was configured and called correctly
                assert mock_cat_downloader.star_mass_range == test_mass_range
                assert mock_cat_downloader.priority_threshold == test_priority
                mock_cat_downloader.download.assert_called_once_with(limit=test_limit)

                # Verify TIC data was stored in memory
                tic_name = catalog_dataset.name
                tic_data_key = storage._get_prefixed_key(tic_name, ".qtable")
                assert tic_data_key in storage._memory
                stored_tic_qtable = storage._memory[tic_data_key]
                assert len(stored_tic_qtable) == len(qtable)

                # Verify TicDB was created
                assert tic_db is not None
                assert len(tic_db._ds) == len(qtable)

                # Test load_tic_target_dataset
                loaded_tic_db = catalog_dataset.load_tic_target_dataset()
                assert loaded_tic_db is not None
                assert len(loaded_tic_db._ds) == len(qtable)

                # Test download_tic_targets_by_ids with a different name
                test_name = "custom_dataset"
                tic_by_id_db = catalog_dataset.download_tic_targets_by_ids(
                    tic_ids=test_tic_ids, store=True, with_name=test_name
                )

                # Verify catalog downloader was called correctly
                mock_cat_downloader.download_by_id.assert_called_with(test_tic_ids)

                # Verify TIC data was stored in memory with custom name
                custom_tic_name = f"{catalog_dataset.name}_{test_name}"
                custom_tic_data_key = storage._get_prefixed_key(custom_tic_name, ".qtable")
                assert custom_tic_data_key in storage._memory
                stored_custom_tic_qtable = storage._memory[custom_tic_data_key]
                assert len(stored_custom_tic_qtable) == len(qtable)

                # Verify TicDB was created
                assert tic_by_id_db is not None
                assert len(tic_by_id_db._ds) == len(qtable)

                # Test loading custom named dataset
                loaded_custom_tic_db = catalog_dataset.load_tic_target_dataset(with_name=test_name)
                assert loaded_custom_tic_db is not None
                assert len(loaded_custom_tic_db._ds) == len(qtable)

    def test_lightcurve_dataset(self, tic_observations_test_data, lightcurve_test_paths):
        """Test LightcurveDataset with mocked downloader"""
        # Setup test data
        lc_obs_ids = list(lightcurve_test_paths.keys())
        tess_qtable, tess_header = tic_observations_test_data

        # Limit tess_qtable to only test IDs
        tess_qtable = tess_qtable[np.isin(tess_qtable["obs_id"], lc_obs_ids)]

        # Mock the LightcurveDownloader
        with patch("exotools.datasets.lightcurves.LightcurveDownloader") as mock_downloader_class:
            mock_downloader = MagicMock()

            # Mock download_fits_parallel to return our test paths
            mock_downloader.download_fits_parallel.return_value = list(lightcurve_test_paths.values())
            mock_downloader_class.return_value = mock_downloader

            # Create the dataset with our temporary path
            dataset = LightcurveDataset(lc_storage_path=TEST_ASSETS_LC)

            # Create a mock TessMetaDB with test data
            from exotools.db import TicObsDB

            tess_meta_db = TicObsDB(tess_qtable)

            # Test downloading lightcurves from TessMetaDB
            lc_db = dataset.download_lightcurves_from_tic_db(tic_obs_db=tess_meta_db)

            # Verify downloader was called correctly
            mock_downloader.download_fits_parallel.assert_called_once()

            # Verify LightcurveDB was created
            assert lc_db is not None
            assert len(lc_db.view) > 0

            # Test loading lightcurve dataset
            loaded_db = dataset.load_lightcurve_dataset()
            assert loaded_db is not None
            assert len(loaded_db.view) > 0

            # Test that we can load a lightcurve by TIC ID
            tic_ids = lc_db.unique_tic_ids
            if len(tic_ids) > 0:
                first_tic_id = tic_ids[0]
                lightcurves = lc_db.load_by_tic(first_tic_id)
                assert lightcurves is not None
                assert len(lightcurves) > 0

                # Test stitched lightcurve
                stitched_lc = lc_db.load_stitched_by_tic(first_tic_id)
                assert stitched_lc is not None

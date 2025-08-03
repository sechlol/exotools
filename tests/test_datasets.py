from unittest.mock import patch, MagicMock

import numpy as np

from exotools.datasets import (
    KnownExoplanetsDataset,
    CandidateExoplanetsDataset,
    TessDataset,
    GaiaParametersDataset,
    LightcurveDataset,
)
from exotools.io import MemoryStorage
from tests.conftest import TEST_ASSETS_LC


class TestDatasets:
    def test_known_exoplanets_dataset(self, known_exoplanets_test_data, gaia_parameters_test_data):
        """Test KnownExoplanetsDataset with mocked downloader"""
        qtable, header = known_exoplanets_test_data
        gaia_qtable, gaia_header = gaia_parameters_test_data
        storage = MemoryStorage()

        # Mock the downloader
        with patch("exotools.datasets.known_exoplanets.KnownExoplanetsDownloader") as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader.download.return_value = (qtable, header)
            mock_downloader_class.return_value = mock_downloader

            # Mock GaiaDownloader as well
            with patch("exotools.datasets.gaia_parameters.GaiaDownloader") as mock_gaia_downloader_class:
                mock_gaia_downloader = MagicMock()
                mock_gaia_downloader.download_by_id.return_value = (gaia_qtable, gaia_header)
                mock_gaia_downloader_class.return_value = mock_gaia_downloader

                # Test dataset
                dataset = KnownExoplanetsDataset(storage=storage)

                # Test download without Gaia data
                exo_db = dataset.download_known_exoplanets(with_gaia_star_data=False, store=True, limit=10)

                # Verify downloader was called correctly
                mock_downloader.download.assert_called_once_with(limit=10, columns=None)

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

    def test_tess_dataset(self, tess_observations_test_data):
        """Test TessDataset with mocked downloaders"""
        qtable, header = tess_observations_test_data
        storage = MemoryStorage()

        # Mock the observations downloader
        with patch("exotools.datasets.tess.TessObservationsDownloader") as mock_obs_downloader_class:
            mock_obs_downloader = MagicMock()
            mock_obs_downloader.download_by_id.return_value = (qtable, header)
            mock_obs_downloader_class.return_value = mock_obs_downloader

            # Mock the catalog downloader (optional)
            with patch("exotools.datasets.tess.TessCatalogDownloader") as mock_cat_downloader_class:
                mock_cat_downloader = MagicMock()
                mock_cat_downloader.download_by_id.return_value = (qtable, header)
                mock_cat_downloader_class.return_value = mock_cat_downloader

                # Test dataset
                dataset = TessDataset(storage=storage, username="test_user", password="test_pass")

                # Test download observation metadata
                test_tic_ids = [123456, 789012, 345678]
                tess_meta_db = dataset.download_observation_metadata(targets_tic_id=test_tic_ids, store=True)

                # Verify observations downloader was called correctly
                mock_obs_downloader.download_by_id.assert_called_once_with(test_tic_ids)

                # Verify data was stored in memory
                observations_name = dataset._observations_name
                data_key = storage._get_prefixed_key(observations_name, ".qtable")
                assert data_key in storage._memory
                stored_qtable = storage._memory[data_key]
                assert len(stored_qtable) == len(qtable)

                # Verify TessMetaDB was created (uses dataset attribute, not meta_dataset)
                assert tess_meta_db is not None
                assert len(tess_meta_db._ds) == len(qtable)

                # Test loading stored dataset
                loaded_db = dataset.load_observation_metadata()
                assert loaded_db is not None
                assert len(loaded_db._ds) == len(qtable)

                # Test TIC catalog download by IDs
                tic_db = dataset.download_tic_targets_by_ids(tic_ids=test_tic_ids, store=True)

                # Verify catalog downloader was called correctly
                mock_cat_downloader.download_by_id.assert_called_once_with(test_tic_ids)

                # Verify TIC data was stored in memory
                tic_name = dataset._tic_by_id_name
                tic_data_key = storage._get_prefixed_key(tic_name, ".qtable")
                assert tic_data_key in storage._memory
                stored_tic_qtable = storage._memory[tic_data_key]
                assert len(stored_tic_qtable) == len(qtable)

                # Verify TicDB was created
                assert tic_db is not None
                assert len(tic_db._ds) == len(qtable)

    def test_lightcurve_dataset(self, tess_observations_test_data, lightcurve_test_paths):
        """Test LightcurveDataset with mocked downloader"""
        # Setup test data
        lc_obs_ids = list(lightcurve_test_paths.keys())
        tess_qtable, tess_header = tess_observations_test_data

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
            from exotools.db import TessMetaDB

            tess_meta_db = TessMetaDB(tess_qtable)

            # Test downloading lightcurves from TessMetaDB
            lc_db = dataset.download_lightcurves_from_tess_db(tess_db=tess_meta_db)

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

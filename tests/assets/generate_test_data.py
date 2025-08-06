import logging
import os

import numpy as np

from exotools import CandidateExoplanetsDataset, KnownExoplanetsDataset, LightcurveDataset, TessDataset
from exotools.datasets import GaiaParametersDataset
from exotools.io.fs_storage import EcsvStorage
from tests.conftest import _TEST_ASSETS_DIR, TEST_ASSETS_LC, TEST_ASSETS_QTABLES
from tests.utils.table_comparison import compare_qtables

logger = logging.getLogger(__name__)

_HOSTNAME_FILE = _TEST_ASSETS_DIR / "planet_hostnames.txt"


def generate_test_qtables():
    storage = EcsvStorage(TEST_ASSETS_QTABLES)
    known_ds = KnownExoplanetsDataset(storage=storage)
    candidates_ds = CandidateExoplanetsDataset(storage=storage)
    gaia_dataset = GaiaParametersDataset(storage=storage)
    tess_ds = TessDataset(
        storage=storage,
        username=os.environ.get("CASJOB_USER"),
        password=os.environ.get("CASJOB_PASSWORD"),
    )

    planet_hostnames = _HOSTNAME_FILE.read_text().splitlines()

    # Download planet datasets, from a selected list of star names
    known = known_ds.download_known_exoplanets(
        where={"hostname": planet_hostnames},
        with_gaia_star_data=True,
        store=True,
    )
    candidates = candidates_ds.download_candidate_exoplanets(limit=150, store=True)
    all_tic_ids = np.concatenate([known.unique_tic_ids[:150], candidates.unique_tic_ids])

    # Download TESS datasets
    tess_meta = tess_ds.download_observation_metadata(targets_tic_id=all_tic_ids, store=True)
    tess_tic = tess_ds.download_tic_targets(limit=150, store=True)
    tess_tic_by_id = tess_ds.download_tic_targets_by_ids(tic_ids=all_tic_ids, store=True)

    # Gaia dataset
    daia_db = gaia_dataset.download_gaia_parameters(tess_tic.gaia_ids, store=True)

    originals = [known, candidates, tess_meta, tess_tic, tess_tic_by_id, daia_db]
    # Reload datasets to compare them
    loaded = [
        known_ds.load_known_exoplanets_dataset(),
        candidates_ds.load_candidate_exoplanets_dataset(),
        tess_ds.load_observation_metadata(),
        tess_ds.load_tic_target_dataset(),
        tess_ds.load_tic_target_dataset_by_id(),
        gaia_dataset.load_gaia_parameters_dataset(),
    ]

    # Need to make sure the qtables are the same
    for i, (original, loaded) in enumerate(zip(originals, loaded)):
        try:
            assert compare_qtables(original.view, loaded.view)
        except AssertionError as e:
            logger.error(f"Failed to compare qtables {i}: {repr(e)}")


def generate_test_lightcurves():
    tess_meta = TessDataset(storage=EcsvStorage(TEST_ASSETS_QTABLES)).load_observation_metadata()
    lc_dataset = LightcurveDataset(lc_storage_path=TEST_ASSETS_LC, override_existing=True)

    small_meta = tess_meta.select_random_sample(n=10)
    lc_dataset.download_lightcurves_from_tess_db(small_meta)


def ensure_credentials():
    required_credentials = ["CASJOB_USER", "CASJOB_PASSWORD", "MAST_TOKEN"]
    for cred in required_credentials:
        if os.environ.get(cred) is None:
            raise ValueError(f"Missing required environment variable '{cred}', please set it and try again.")


def set_credentials():
    os.environ["CASJOB_USER"] = "******"
    os.environ["CASJOB_PASSWORD"] = "******"
    os.environ["MAST_TOKEN"] = "******"


def main():
    # Example way to set credentials
    set_credentials()

    ensure_credentials()
    generate_test_qtables()
    generate_test_lightcurves()
    logger.info("Done!")


if __name__ == "__main__":
    main()

import logging

import numpy as np

from exotools import CandidateExoplanetsDataset, KnownExoplanetsDataset, LightcurveDataset, TessDataset
from exotools.io.fs_storage import EcsvStorage
from tests.conftest import TEST_ASSETS_LC, TEST_ASSETS_QTABLES
from tests.utils.comparison import compare_qtables

logger = logging.getLogger(__name__)


def generate_test_qtables():
    storage = EcsvStorage(TEST_ASSETS_QTABLES)
    known_ds = KnownExoplanetsDataset(storage=storage)
    candidates_ds = CandidateExoplanetsDataset(storage=storage)
    tess_ds = TessDataset(storage=storage)

    # Download datasets
    known = known_ds.download_known_exoplanets(limit=150, with_gaia_star_data=True)
    candidates = candidates_ds.download_candidate_exoplanets(limit=150, store=True)
    all_ids = np.concatenate([known.unique_ids, candidates.unique_ids])
    tess_meta = tess_ds.download_observation_metadata(targets_tic_id=all_ids, store=True)

    # TODO: need to authenticate to download the TIC dataset
    # tess_tic = tess_ds.search_tic_targets(limit=150, store=True)

    originals = [known, candidates, tess_meta]

    # Load datasets
    loaded = [
        known_ds.load_known_exoplanets_dataset(),
        candidates_ds.load_candidate_exoplanets_dataset(),
        tess_ds.load_observation_metadata(),
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

    small_meta = tess_meta.select_random_sample(n=10, unique_ids=False)
    lc_dataset.download_lightcurves_from_tess_db(small_meta)


def main():
    generate_test_qtables()
    generate_test_lightcurves()
    logger.info("Done!")


if __name__ == "__main__":
    main()

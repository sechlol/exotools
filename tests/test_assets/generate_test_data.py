import numpy as np

from exotools import KnownExoplanetsDataset, CandidateExoplanetsDataset, TessDataset
from exotools.io.fs_storage import EcsvStorage
from tests.paths import TEST_ASSETS_QTABLES
from tests.utils import compare_qtables


def generate_test_qtables():
    storage = EcsvStorage(TEST_ASSETS_QTABLES)
    known_ds = KnownExoplanetsDataset(storage)
    candidates_ds = CandidateExoplanetsDataset(storage)
    tess_ds = TessDataset(storage)

    # Download datasets
    known = known_ds.download_known_exoplanets(limit=150, with_gaia_star_data=True)
    candidates = candidates_ds.download_candidate_exoplanets(limit=150, store=True)
    all_ids = np.concatenate([known.unique_ids, candidates.unique_ids])
    tess = tess_ds.download_observation_metadata(targets_tic_id=all_ids, store=True)
    originals = [known, candidates, tess]

    # Load datasets
    loaded = [
        known_ds.load_known_exoplanets_dataset(),
        candidates_ds.load_candidate_exoplanets_dataset(),
        tess_ds.load_observation_metadata(),
    ]

    # Need to make sure the qtables are the same
    for original, loaded in zip(originals, loaded):
        assert compare_qtables(original.view, loaded.view)


def main():
    generate_test_qtables()
    print("Done!")


if __name__ == "__main__":
    main()

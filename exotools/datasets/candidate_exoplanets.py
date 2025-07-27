from typing import Optional

from astropy.table import QTable

from exotools.db.exo_db import ExoDB
from exotools.db.toi_db import CandidateDB
from exotools.downloaders.toi_exoplanets_downloader import CandidateExoplanetsDownloader
from exotools.io.base_storage_wrapper import StorageWrapper


class CandidateExoplanetsDataset:
    _DATASET_NAME_CANDIDATES = "candidate_exoplanets"

    def __init__(self, storage: StorageWrapper):
        self._storage = storage

    def load_candidate_exoplanets_dataset(self) -> Optional[CandidateDB]:
        try:
            candidate_qtable = self._storage.read_qtable(table_name=self._DATASET_NAME_CANDIDATES)
        except ValueError:
            print(
                "Candidate Exoplanets dataset not found. "
                "You need to download it first by calling download_candidate_exoplanets(store=True)."
            )
            return None

        return _create_candidate_db(candidate_dataset=candidate_qtable)

    def download_candidate_exoplanets(self, limit: Optional[int] = None, store: bool = True) -> CandidateDB:
        print("Preparing to download candidate exoplanets dataset...")
        candidate_qtable, candidate_header = CandidateExoplanetsDownloader().download(limit=limit)

        if store:
            self._storage.write_qtable(
                table=candidate_qtable,
                header=candidate_header,
                table_name=self._DATASET_NAME_CANDIDATES,
            )

        return _create_candidate_db(candidate_qtable)


def _create_candidate_db(candidate_dataset: QTable) -> CandidateDB:
    ExoDB.compute_bounds(candidate_dataset)
    return CandidateDB(candidate_dataset)

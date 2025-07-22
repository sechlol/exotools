from pathlib import Path
from typing import Optional

from astropy.table import QTable

from ._downloaders.toi_exoplanets_downloader import CandidateExoplanetsDownloader
from .db.exo_db import ExoDB
from .db.toi_db import CandidateDB
from .utils.qtable_utils import read_qtable


class CandidateExoplanetsDataset:
    _DATASET_NAME_CANDIDATES = "candidate_exoplanets"

    def __init__(self, storage_folder_path: Path):
        self._folder_path = storage_folder_path

    def load_candidate_exoplanets_dataset(self) -> Optional[CandidateDB]:
        try:
            candidate_qtable = read_qtable(file_path=self._folder_path, file_name=self._DATASET_NAME_CANDIDATES)
        except ValueError:
            print(
                "Candidate Exoplanets dataset not found. "
                "You need to download it first by calling download_candidate_exoplanets()."
            )
            return None

        return _create_candidate_db(candidate_dataset=candidate_qtable)

    def download_candidate_exoplanets(self, limit: Optional[int] = None) -> CandidateDB:
        self._folder_path.mkdir(parents=True, exist_ok=True)

        print("Preparing to download candidate exoplanets dataset...")
        candidate_qtable = CandidateExoplanetsDownloader().download(
            limit=limit,
            out_folder_path=self._folder_path,
            out_file_name=self._DATASET_NAME_CANDIDATES,
        )

        return _create_candidate_db(candidate_qtable)


def _create_candidate_db(candidate_dataset: QTable) -> CandidateDB:
    ExoDB.compute_bounds(candidate_dataset)
    return CandidateDB(candidate_dataset)

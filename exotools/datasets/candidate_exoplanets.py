import logging
from typing import Optional

from astropy.table import QTable

from exotools.datasets.base_dataset import BaseDataset
from exotools.db import CandidateDB, ExoDB
from exotools.downloaders import CandidateExoplanetsDownloader
from exotools.io import BaseStorage

logger = logging.getLogger(__name__)


class CandidateExoplanetsDataset(BaseDataset):
    _DATASET_NAME_CANDIDATES = "candidate_exoplanets"

    def __init__(self, dataset_tag: Optional[str] = None, storage: Optional[BaseStorage] = None):
        super().__init__(dataset_name=self._DATASET_NAME_CANDIDATES, dataset_tag=dataset_tag, storage=storage)

    def load_candidate_exoplanets_dataset(self) -> Optional[CandidateDB]:
        try:
            candidate_qtable = self._storage.read_qtable(table_name=self.name)
        except ValueError:
            logger.error(
                "Candidate Exoplanets dataset not found. "
                "You need to download it first by calling download_candidate_exoplanets(store=True)."
            )
            return None

        return _create_candidate_db(candidate_dataset=candidate_qtable)

    def download_candidate_exoplanets(self, limit: Optional[int] = None, store: bool = True) -> CandidateDB:
        logger.info("Preparing to download candidate exoplanets dataset...")
        candidate_qtable, candidate_header = CandidateExoplanetsDownloader().download(limit=limit)

        if store:
            self._storage.write_qtable(
                table=candidate_qtable,
                header=candidate_header,
                table_name=self.name,
            )

        return _create_candidate_db(candidate_qtable)


def _create_candidate_db(candidate_dataset: QTable) -> CandidateDB:
    ExoDB.compute_bounds(candidate_dataset)
    return CandidateDB(candidate_dataset)

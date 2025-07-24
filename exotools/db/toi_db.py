from astropy.table import QTable

from .base_db import BaseDB

_ID_FIELD = "tic_id"


class CandidateDB(BaseDB):
    def __init__(self, toi_dataset: QTable):
        super().__init__(toi_dataset, id_field=_ID_FIELD)

    def _factory(self, dataset: QTable) -> "CandidateDB":
        return CandidateDB(dataset)

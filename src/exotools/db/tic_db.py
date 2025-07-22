from astropy.table import QTable
from .base_db import BaseDB

_ID_FIELD = "tic_id"


class TicDB(BaseDB):
    def __init__(self, dataset: QTable):
        super().__init__(dataset, id_field="tic_id")

    def _factory(self, dataset: QTable) -> "TicDB":
        return TicDB(dataset)

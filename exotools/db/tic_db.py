from typing import Self

from astropy.table import QTable
from .base_db import BaseDB

_ID_FIELD = "tic_id"


class TicDB(BaseDB):
    """
    Dtypes:
    -------------------
    tic_id        int64
    gaia_id       int64
    priority    float64
    ra          float64
    dec         float64
    -------------------
    """

    def __init__(self, dataset: QTable):
        super().__init__(dataset, id_field="tic_id")

    def _factory(self, dataset: QTable) -> Self:
        return TicDB(dataset)

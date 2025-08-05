import numpy as np
from astropy.table import QTable
from typing_extensions import Self

from .base_db import BaseDB

_ID_FIELD = "tic_id"


class CandidateDB(BaseDB):
    def __init__(self, toi_dataset: QTable):
        super().__init__(toi_dataset, id_field=_ID_FIELD)

    def _factory(self, dataset: QTable) -> Self:
        return CandidateDB(dataset)

    @property
    def tic_ids(self) -> np.ndarray:
        return self._masked_ds["tic_id"].value

    @property
    def unique_tic_ids(self) -> np.ndarray:
        return np.unique(self.tic_ids)

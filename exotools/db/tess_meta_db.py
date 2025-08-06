import numpy as np
from astropy.table import QTable
from typing_extensions import Self

from exotools.constants import NAN_VALUE

from .base_db import BaseDB

_ID_FIELD = "tic_id"


class TicObsDB(BaseDB):
    """
    Dtypes:
    ---------------------------
    tic_id               int64
    sequence_number       int8
    dataURL             object
    t_obs_release      float64
    t_min              float64
    t_max              float64
    obs_id               int64
    ---------------------------
    """

    def __init__(self, meta_dataset: QTable):
        super().__init__(meta_dataset, id_field=_ID_FIELD)

    @property
    def tic_ids(self) -> np.ndarray:
        return self.view["tic_id"].value

    @property
    def obs_id(self) -> np.ndarray:
        return self.view["obs_id"].value

    @property
    def unique_tic_ids(self) -> np.ndarray:
        return np.unique(self.tic_ids)

    @property
    def unique_obs_ids(self) -> np.ndarray:
        return np.unique(self.obs_id)

    @property
    def data_urls(self):
        return self.view["dataURL"].value

    def _factory(self, dataset: QTable) -> Self:
        return TicObsDB(dataset)

    def select_by_obs_id(self, other_obs_id: np.ndarray) -> Self:
        masked_ds = self.view[self.view["obs_id"] != NAN_VALUE]
        selection = np.isin(masked_ds["obs_id"], other_obs_id)
        return self._factory(masked_ds[selection])

    def select_by_tic_id(self, other_tic_ids: np.ndarray) -> Self:
        masked_ds = self.view[self.view["tic_id"] != NAN_VALUE]
        selection = np.isin(masked_ds["tic_id"], other_tic_ids)
        return self._factory(masked_ds[selection])

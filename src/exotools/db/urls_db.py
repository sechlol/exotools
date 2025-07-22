import numpy as np
from astropy.table import QTable

from .base_db import BaseDB, NAN_VALUE

_ID_FIELD = "tic_id"


class UrlsDB(BaseDB):
    def __init__(self, urls_dataset: QTable):
        super().__init__(urls_dataset, id_field=_ID_FIELD)

    @property
    def obs_id(self):
        return self.view["obs_id"].value

    def _factory(self, dataset: QTable) -> "UrlsDB":
        return UrlsDB(dataset)

    def select_by_obs_id(self, other_obs_id: np.ndarray) -> "UrlsDB":
        masked_ds = self.view[self.view["obs_id"] != NAN_VALUE]
        selection = np.isin(masked_ds["obs_id"], other_obs_id)
        return self._factory(masked_ds[selection])

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from astropy.table import QTable

NAN_VALUE = -1


class BaseDB(ABC):
    def __init__(self, dataset: QTable, id_field: str):
        dataset.add_index(id_field)
        self._ds = dataset
        self._id_column = id_field
        self._id_mask = self._ds[self._id_column] != NAN_VALUE
        self._masked_ds = self._ds[self._id_mask]

    def __len__(self):
        return len(self.view)

    @abstractmethod
    def _factory(self, dataset: QTable) -> "BaseDB":
        pass

    @property
    def view(self) -> QTable:
        return self._ds

    @property
    def dataset_copy(self) -> QTable:
        return self._ds.copy()

    @property
    def ids(self) -> np.ndarray:
        return self._masked_ds[self._id_column].value

    @property
    def unique_ids(self) -> np.ndarray:
        return np.unique(self.ids)

    def match_ids(self, other_ids: np.ndarray) -> np.ndarray:
        """
        Get matching IDs from another id set
        """
        return np.isin(self.ids, other_ids)

    def match_field(self, field_name: str, other_values: np.ndarray) -> np.ndarray:
        """
        Get matching IDs from another id set
        """
        return np.isin(self.view[field_name], other_values)

    def select_by_id(self, other_ids: np.ndarray) -> "BaseDB":
        """
        Match ids and returns data
        """
        matching_ids = self.match_ids(other_ids)
        return self._factory(self._masked_ds[matching_ids])

    def select_by_mask(self, bit_mask: np.ndarray) -> "BaseDB":
        """
        Returns data that matches the mask
        """
        return self._factory(self._ds[bit_mask])

    def select_random_sample(self, n: int, unique_ids: bool = True) -> "BaseDB":
        if unique_ids:
            return self.select_by_id(np.random.choice(self.unique_ids, size=n, replace=False))

        random_indices = np.random.choice(len(self.view), size=n, replace=False)
        return self._factory(self.view[random_indices])

    def to_pandas(self) -> pd.DataFrame:
        return self.view.to_pandas().reset_index()

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd
from astropy.table import QTable
from typing_extensions import Self

logger = logging.getLogger(__name__)

NAN_VALUE = -1


class BaseDB(ABC):
    def __init__(self, dataset: QTable, id_field: str):
        if len(dataset.columns) == 0:
            raise ValueError("Attempting to create BaseDB with empty column set.")

        # An empty dataset with columns but no data is a valid dataset
        if len(dataset) != 0:
            dataset.add_index(id_field)

        self._ds = dataset
        self._id_column = id_field

    def __len__(self):
        return len(self.view)

    @abstractmethod
    def _factory(self, dataset: QTable) -> Self:
        pass

    @property
    def view(self) -> QTable:
        return self._ds

    @property
    def dataset_copy(self) -> QTable:
        return self._ds.copy()

    def where(self, **kwargs) -> Self:
        """
        Filters the data by the given fields.
        """
        conditions = np.ones(len(self._ds), dtype=bool)

        for field_name, value in kwargs.items():
            if field_name in self._ds.colnames:
                # Check if value is a sequence-like object (list, tuple, numpy array, etc.)
                if isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (str, bytes)):
                    conditions &= np.isin(self._ds[field_name], value)
                else:
                    conditions &= self._ds[field_name] == value
        return self._factory(self._ds[conditions])

    def where_true(self, bit_mask: np.ndarray) -> Self:
        """
        Returns data that matches the mask
        """
        return self._factory(self._ds[bit_mask])

    def with_valid_ids(self) -> Self:
        valid_id_mask = self._ds[self._id_column] != NAN_VALUE
        return self.where_true(valid_id_mask)

    def select_random_sample(self, n: int) -> Self:
        random_indices = np.random.choice(len(self._ds), size=n, replace=False)
        return self._factory(self._ds[random_indices])

    def to_pandas(self) -> pd.DataFrame:
        if len(self._ds) == 0:
            return pd.DataFrame()
        return self._ds.to_pandas().reset_index()

import abc
from typing import Optional

from exotools.io import BaseStorage, MemoryStorage


class BaseDataset(abc.ABC):
    def __init__(self, dataset_name: str, dataset_tag: Optional[str] = None, storage: Optional[BaseStorage] = None):
        self._dataset_name = dataset_name + (f"_{dataset_tag}" if dataset_tag else "")
        self._storage = storage or MemoryStorage()

    @property
    def name(self) -> str:
        return self._dataset_name

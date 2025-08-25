from astropy.table import QTable
from typing_extensions import Self

from .ps_db import PsDB


class ExoCompDB(PsDB):
    def _factory(self, dataset: QTable) -> Self:
        return ExoCompDB(dataset)

from abc import ABCMeta

import numpy as np
from astropy.table import QTable
from typing_extensions import Self

from exotools.db.base_db import BaseDB


class PsDB(BaseDB, metaclass=ABCMeta):
    def __init__(self, exoplanets_dataset: QTable):
        super().__init__(exoplanets_dataset, id_field="tic_id")

    @property
    def tic_ids(self) -> np.ndarray:
        return self.view["tic_id"].value

    @property
    def gaia_ids(self) -> np.ndarray:
        return self.view["gaia_id"].value

    @property
    def unique_tic_ids(self) -> np.ndarray:
        return np.unique(self.tic_ids)

    @property
    def unique_gaia_ids(self) -> np.ndarray:
        return np.unique(self.gaia_ids)

    def get_star_names(self) -> list[str]:
        return np.unique(self.view["hostname"]).tolist()

    def get_tess_planets(self) -> Self:
        # Create a boolean mask for rows where disc_telescope contains "TESS"
        mask_tess = np.char.find(self.view["disc_telescope"].value.astype("U"), "TESS") != -1
        return self._factory(self.view[mask_tess])

    def get_kepler_planets(self) -> Self:
        # Create a boolean mask for rows where disc_telescope contains "Kepler"
        mask_kepler = np.char.find(self.view["disc_telescope"].value.astype("U"), "Kepler") != -1
        mask_k2 = np.char.find(self.view["disc_telescope"].value.astype("U"), "K2") != -1
        return self._factory(self.view[mask_kepler | mask_k2])

    def get_transiting_planets(self, kepler_or_tess_only: bool = False) -> Self:
        condition = self.view["tran_flag"] == 1
        if kepler_or_tess_only:
            # Create a boolean mask for rows where disc_telescope contains "TESS" or "Kepler"
            mask_tess = np.char.find(self.view["disc_telescope"].value.astype("U"), "TESS") != -1
            mask_kepler = np.char.find(self.view["disc_telescope"].value.astype("U"), "Kepler") != -1
            mask_k2 = np.char.find(self.view["disc_telescope"].value.astype("U"), "K2") != -1
            telescope_mask = mask_tess | mask_kepler | mask_k2

            # Apply the mask
            condition &= telescope_mask
        return self._factory(self.view[condition])

    @staticmethod
    def preprocess_dataset(dataset: QTable):
        # Set lowercase hostname for faster retrieval
        dataset["hostname_lowercase"] = np.char.lower(dataset["hostname"].tolist())

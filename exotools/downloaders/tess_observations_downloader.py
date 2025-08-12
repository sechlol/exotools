import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from astropy.table import QTable
from tqdm import tqdm

from exotools.utils.observations_fix import Observations
from exotools.utils.qtable_utils import QTableHeader, get_empty_table_header

from .dataset_downloader import DatasetDownloader, iterate_chunks

logger = logging.getLogger(__name__)


class TessObservationsDownloader(DatasetDownloader):
    _mandatory_columns = {"target_name", "sequence_number", "dataURL", "t_obs_release", "t_min", "t_max", "obsid"}

    def _initialize_services(self):
        return

    def _download(self, limit: Optional[int] = None, **kwargs) -> QTable:
        raise NotImplementedError("TessObservationsDownloader does not implement this download method")

    def _download_by_id(self, ids: list[int], columns: Optional[Sequence[str]] = None, **kwargs) -> QTable:
        all_data = []
        chunk_ids = list(iterate_chunks(ids, chunk_size=2000))
        n = len(chunk_ids)
        columns = self._mandatory_columns | set(columns or [])
        for i, tic_ids in tqdm(enumerate(chunk_ids), desc="Querying TESS observations", total=n):
            try:
                chunk_data: pd.DataFrame = Observations.query_criteria_columns_async(
                    columns=list(columns),
                    provenance_name="SPOC",
                    filters="TESS",
                    project="TESS",
                    dataproduct_type="timeseries",
                    target_name=tic_ids,
                    t_exptime=[119, 121],
                )
                all_data.append(chunk_data)
            except Exception as e:
                logger.error(f"Exception generated while downloading URLs data for index i={i}")
                raise e

        all_data = (
            pd.concat(all_data, axis=0)
            .astype(
                {
                    "obsid": np.int64,
                    "target_name": np.int64,
                    "sequence_number": np.int8,
                    "dataURL": str,
                    "t_obs_release": float,
                    "t_min": float,
                    "t_max": float,
                }
            )
            .rename(columns={"target_name": "tic_id", "obsid": "obs_id"})
        )
        just_lc = all_data[all_data["dataURL"].str.endswith("s_lc.fits")]

        return QTable.from_pandas(just_lc)

    def _clean_and_fix(self, table: QTable) -> QTable:
        return table

    def _get_table_header(self, table: QTable) -> QTableHeader:
        return get_empty_table_header(table)

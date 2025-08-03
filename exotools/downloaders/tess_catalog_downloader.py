import logging
from io import BytesIO
from typing import Optional, Sequence

import astropy.units as u
from astropy.table import QTable, vstack
from casjobs import CasJobs
from tqdm import tqdm

from exotools.utils.qtable_utils import QTableHeader, get_empty_table_header

from ._utils import override_units
from .dataset_downloader import DatasetDownloader, iterate_chunks
from .tap_service import TicService

logger = logging.getLogger(__name__)


class TessCatalogDownloader(DatasetDownloader):
    """
    This class uses CasJobs interface to query the service at https://mastweb.stsci.edu/mcasjobs/services/jobs.asmx
    A Web interface is available at https://mastweb.stsci.edu/mcasjobs/SubmitJob.aspx
    Create an account at https://mastweb.stsci.edu/mcasjobs/CreateAccount.aspx
    # See also: https://tess.mit.edu/science/tess-input-catalogue/
    """

    _catalog = "TESS_v82"
    _units_override = {"ra": u.deg, "dec": u.deg}
    _mandatory_fields = {"ra", "dec", "priority"}

    def __init__(
        self,
        username: str,
        password: str,
        star_mass_range: tuple[float, float] = (0.7, 1.3),
        priority_threshold: float = 0.001,
        verbose_log: bool = False,
    ):
        """
        TODO: write docs for parmeters
        This class uses CasJobs interface to query the service, you need to create an account at
        https://mastweb.stsci.edu/mcasjobs/CreateAccount.aspx and provide the username and password
        """
        self._priority_threshold = priority_threshold
        self._star_mass_range = star_mass_range
        self._tic_service = TicService()
        self._verbose_log = verbose_log
        self._casjob_api = CasJobs(
            userid=username,
            password=password,
            base_url="https://mastweb.stsci.edu/mcasjobs/services/jobs.asmx",
            context=self._catalog,
        )

    @property
    def star_mass_range(self) -> tuple[float, float]:
        return self._star_mass_range

    @star_mass_range.setter
    def star_mass_range(self, value: tuple[float, float]):
        self._star_mass_range = value

    @property
    def priority_threshold(self) -> float:
        return self._priority_threshold

    @priority_threshold.setter
    def priority_threshold(self, value: float):
        self._priority_threshold = value

    def _clean_and_fix(self, table: QTable) -> QTable:
        override_units(table, self._units_override)
        # Force gaia_id to be integer
        table["gaia_id"] = table.to_pandas()["gaia_id"].astype(int).to_numpy()
        return table

    def _get_table_header(self, table: QTable) -> QTableHeader:
        header = get_empty_table_header(table)

        header["tic_id"].description = "Star TIC id [int64]"
        header["gaia_id"].description = "Star GAIA id [int64]"
        header["ra"].description = "Star right ascension [deg]"
        header["dec"].description = "Star declination [deg]"
        header[
            "priority"
        ].description = "Higher priority is assigned to the TESS team to targets that are likely to host planets"
        return header

    def _download_by_id(self, ids: Sequence[int], columns: Optional[Sequence[str]] = None, **kwargs) -> QTable:
        # Bigger chunk size will cause a server error, because the query becomes too big
        chunk_size = 400
        all_tables = []
        chunks = list(iterate_chunks(ids=ids, chunk_size=chunk_size))

        fields = ",".join(list(self._mandatory_fields | set(columns or [])))
        for i, ids in tqdm(enumerate(chunks), desc="Querying TIC chunks", total=len(chunks)):
            formatted_ids = ",".join([f"'{tid}'" for tid in ids])

            query = f"""select id as tic_id, gaia as gaia_id, {fields}
                        from dbo.CatalogRecord
                        where gaia is not null and id IN ({formatted_ids})"""
            table = self._tic_service.query(query_string=query)
            all_tables.append(table)

        return QTable(vstack(all_tables))

    def _download(self, limit: Optional[int] = None, columns: Optional[Sequence[str]] = None, **kwargs) -> QTable:
        limit_clause = f"top {limit}" if limit else ""
        fields = ",".join(list(self._mandatory_fields | set(columns or [])))
        query = f"""select {limit_clause} id as tic_id, gaia as gaia_id, {fields}
                from dbo.CatalogRecord
                where gaia is not null
                    and priority > {self._priority_threshold}
                    and mass between {self._star_mass_range[0]} and {self._star_mass_range[1]}"""
        result = self._query_ctl_casjob(catalog=self._catalog, query=query, estimated_minutes=1)
        logger.info(f"DONE. Collected {len(result)} stars.")

        return result

    def _query_ctl_casjob(
        self, catalog: str, query: str, estimated_minutes: int = 5, quick: bool = False
    ) -> Optional[QTable]:
        if quick:
            return self._casjob_api.quick(q=query, context=catalog)

        temp_table = "temp_table"
        self._log("Preparing remote DB")
        try:
            self._casjob_api.drop_table(temp_table)
        except Exception:
            # Table might not exist
            pass

        cas_query = f"select * into mydb.{temp_table} from ({query}) as subquery"
        self._log(cas_query)
        job_id = self._casjob_api.submit(cas_query, context=catalog, task_name="python_api", estimate=estimated_minutes)
        logger.info(f"Started Casjob {job_id}")
        status_code, status_name = self._casjob_api.monitor(job_id)
        self._log(f"Job {job_id} ended with status {status_code}: {status_name}")
        if status_code != 5:
            return None

        buffer_csv = BytesIO()
        logger.info("Downloading output")
        self._casjob_api.request_and_get_output(temp_table, outtype="CSV", outfn=buffer_csv)
        logger.info(f"Downloaded {(buffer_csv.getbuffer().nbytes / 10**6):.2f} MB of data")

        # Cleanup result table
        try:
            self._log("Cleanup remote data")
            self._casjob_api.drop_table(temp_table)
        except Exception as e:
            logger.error("Exception raised in query_ctl_casjob():", repr(e))
            # We already got the result, and don't want to block here
            pass

        buffer_csv.seek(0)
        return QTable.read(buffer_csv, format="csv")

    def _log(self, message: str):
        if self._verbose_log:
            logger.info(message)
        else:
            logger.trace(message)

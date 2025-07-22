from io import BytesIO
from typing import Optional, Sequence

import astropy.units as u
from astropy.table import QTable, vstack
from casjobs import CasJobs

from .dataset_downloader import DatasetDownloader, override_units, iterate_chunks
from src.exotools.utils.qtable_utils import QTableHeader, get_empty_table_header
from .tap_service import TicService


class TessCatalogDownloader(DatasetDownloader):
    """
    This class uses CasJobs interface to query the service at https://mastweb.stsci.edu/mcasjobs/services/jobs.asmx
    A Web interface is available at https://mastweb.stsci.edu/mcasjobs/SubmitJob.aspx
    Create an account at https://mastweb.stsci.edu/mcasjobs/CreateAccount.aspx
    # See also: https://tess.mit.edu/science/tess-input-catalogue/
    """

    _priority_threshold = 0.001
    _star_mass_range = [0.7, 1.3]
    _catalog = "TESS_v82"
    _units_override = {"ra": u.deg, "dec": u.deg}

    def __init__(self, username: str, password: str):
        """
        This class uses CasJobs interface to query the service, you need to create an account at
        https://mastweb.stsci.edu/mcasjobs/CreateAccount.aspx and provide the username and password
        """
        self._tic_service = TicService()
        self._casjob_api = CasJobs(
            userid=username,
            password=password,
            base_url="https://mastweb.stsci.edu/mcasjobs/services/jobs.asmx",
            context=self._catalog,
        )

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
        header["priority"].description = (
            "Higher priority is assigned to the TESS team to targets that are " "likely to host planets"
        )
        return header

    def _download_by_id(self, ids: Sequence[int]) -> QTable:
        # Bigger chunk size will cause a server error, because the query becomes too big
        chunk_size = 400
        all_tables = []
        chunks = list(iterate_chunks(ids=ids, chunk_size=chunk_size))

        for i, ids in enumerate(chunks):
            print(f"Query chunk {i + 1}/{len(chunks)}")
            ids = [f"'{tid}'" for tid in ids]
            formatted_ids = f"({','.join(ids)})"

            query = f"""select id as tic_id, gaia as gaia_id, priority, ra, dec 
                        from dbo.CatalogRecord 
                        where gaia is not null and id IN {formatted_ids}"""
            table = self._tic_service.query(query_string=query, sync=True)
            all_tables.append(table)

        return QTable(vstack(all_tables))

    def _download(self, limit: Optional[int] = None) -> QTable:
        limit_clause = f"top {limit}" if limit else ""
        query = f"""select {limit_clause} id as tic_id, gaia as gaia_id, priority, ra, dec 
                from dbo.CatalogRecord 
                where gaia is not null 
                    and priority > {self._priority_threshold} 
                    and mass between {self._star_mass_range[0]} and {self._star_mass_range[1]}"""
        result = self._query_ctl_casjob(catalog=self._catalog, query=query, estimated_minutes=1)

        print(f"DONE. Collected {len(result)} stars.")

        return result

    def _query_ctl_casjob(
        self, catalog: str, query: str, estimated_minutes: int = 5, quick: bool = False
    ) -> Optional[QTable]:
        if quick:
            return self._casjob_api.quick(q=query, context=catalog)

        temp_table = "temp_table"
        print("Preparing remote DB")
        try:
            self._casjob_api.drop_table(temp_table)
        except Exception:
            # Table might not exist
            pass

        cas_query = f"select * into mydb.{temp_table} from ({query}) as subquery"
        print(cas_query)
        job_id = self._casjob_api.submit(cas_query, context=catalog, task_name="python_api", estimate=estimated_minutes)
        print(f"Started Casjob {job_id}")
        status_code, status_name = self._casjob_api.monitor(job_id)
        print(f"Job {job_id} ended with status {status_code}: {status_name}")
        if status_code != 5:
            return None

        buffer_csv = BytesIO()
        print(f"Downloading output")
        self._casjob_api.request_and_get_output(temp_table, outtype="CSV", outfn=buffer_csv)

        print(f"Downloaded {(buffer_csv.getbuffer().nbytes / 10 ** 6):.2f} MB of data")

        # Cleanup result table
        try:
            print(f"Cleanup remote data")
            self._casjob_api.drop_table(temp_table)
        except Exception as e:
            print("Exception raised in query_ctl_casjob():", repr(e))
            # We already got the result, and don't want to block here
            pass

        buffer_csv.seek(0)
        return QTable.read(buffer_csv, format="csv")

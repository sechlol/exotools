from typing import Optional, Iterable

import astropy.units as u
from astropy.table import QTable, vstack
from astroquery.gaia import Gaia
from tqdm import tqdm

from exotools.utils.qtable_utils import QTableHeader
from .dataset_downloader import DatasetDownloader, iterate_chunks
from .tap_service import GaiaService


class GaiaDownloader(DatasetDownloader):
    """
    Archive: https://gea.esac.esa.int/archive/
    gaia_source table documentation:
        https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html#gaia_source-non_single_star
    astrophysical_parameters table documentation:
        https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html
    """

    _parameters_table = "gaiadr3.astrophysical_parameters"

    def __init__(self):
        self._gaia_service = GaiaService()

    def _download_by_id(self, ids: list[str]) -> QTable:
        # This is a hard limit imposed by the server. Synchronous queries are allowed up to 2000 results
        all_tables = []
        chunk_ids = list(iterate_chunks(ids, chunk_size=1000))

        for i, gaia_ids in tqdm(enumerate(chunk_ids), desc="Querying Gaia chunks", total=len(chunk_ids)):
            try:
                query = _get_gaia_targets_data_query(gaia_object_ids=gaia_ids, from_dr2=True)
                table = Gaia.launch_job(query).get_results()
                all_tables.append(table)
            except Exception as e:
                print(f"Exception generated while downloading Gaia data for index i={i}")
                raise e
        return QTable(vstack(all_tables))

    def _clean_and_fix(self, table: QTable) -> QTable:
        # Rename columns to lowercase
        table.rename_columns(table.colnames, [col.lower() for col in table.colnames])

        # Drop duplicates based on the "source_id" column
        # _, unique_indices = np.unique(table['source_id'], return_index=True)
        # unique_table = table[sorted(unique_indices)]

        # Rename source_id -> gaia_id
        table.rename_column("source_id", "gaia_id")

        return table

    def _get_table_header(self, table: QTable) -> QTableHeader:
        """
        Fetch column information from the TAP service, and select only the necessary ones
        """
        all_info = self._gaia_service.get_field_info(self._parameters_table)

        fields_info = {k: v for k, v in all_info.items() if k in table.columns}

        # There is an annoying unit that is saved as "'dex'", with quotes
        fields_info["mh_gspphot"].unit = u.dex.name

        return fields_info

    def _download(self, limit: Optional[int] = None) -> QTable:
        raise NotImplementedError("GaiaDownloader doesn't support this download method")


def _get_gaia_targets_data_query(
    gaia_object_ids: Iterable[str],
    from_dr2: bool = True,
    limit: Optional[int] = None,
    must_have_photometry_data: bool = False,
    select_all_fields: bool = False,
) -> str:
    ids = [f"'{oid}'" for oid in gaia_object_ids]
    formatted_ids = f"({','.join(ids)})"

    # I'm looking for single stars
    extra_conditions = "AND dr3.non_single_star = 0 " "AND dr3_astro.classprob_dsc_combmod_star > 0.99"

    if must_have_photometry_data:
        extra_conditions += " AND dr3.has_epoch_photometry = 'True'"

    limit_clause = f"top {limit}" if limit else ""

    if select_all_fields:
        selection = f"SELECT {limit_clause} dr3.*, dr3_astro.* "
    else:
        selection = f"""SELECT {limit_clause} dr3.source_id,
                  dr3.phot_g_mean_mag, dr3.phot_bp_mean_mag, dr3.phot_rp_mean_mag,  
                  dr3.phot_g_mean_flux_over_error, dr3.phot_bp_mean_flux_over_error, dr3.phot_rp_mean_flux_over_error, 
                  dr3.phot_variable_flag, 
                  dr3_astro.teff_gspphot, teff_gspspec, teff_esphs, teff_espucd, teff_msc1, teff_msc2, 
                  dr3_astro.mh_gspphot,
                  dr3_astro.logg_gspphot,  
                  dr3_astro.distance_gspphot, distance_msc,  
                  mg_gspphot,  
                  spectraltype_esphs,  
                  age_flame,  
                  mass_flame,  
                  lum_flame,  
                  radius_flame, radius_gspphot """
    if from_dr2:
        query = (
            selection
            + f"""
                FROM gaiadr2.gaia_source AS dr2 
                JOIN gaiadr3.dr2_neighbourhood AS dr3_n ON dr2.source_id = dr3_n.dr2_source_id 
                JOIN gaiadr3.gaia_source_lite AS dr3 ON dr3.source_id = dr3_n.dr3_source_id 
                JOIN gaiadr3.astrophysical_parameters AS dr3_astro ON dr3.source_id = dr3_astro.source_id 
                WHERE dr2.source_id IN {formatted_ids} {extra_conditions}"""
        )
    else:
        query = (
            selection
            + f"""
                 FROM gaiadr3.gaia_source_lite as dr3 
                 JOIN gaiadr3.astrophysical_parameters AS dr3_astro ON dr3.source_id = dr3_astro.source_id 
                 WHERE source_id in {formatted_ids} {extra_conditions}"""
        )

    return query

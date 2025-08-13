import logging
from typing import Any, Optional, Sequence

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import MaskedColumn, QTable

from exotools.utils.qtable_utils import QTableHeader, TableColumnInfo
from exotools.utils.unit_mapper import UNIT_MAPPER

from ._utils import fix_unrecognized_units, override_units
from .dataset_downloader import DatasetDownloader
from .tap_service import ExoService, TapService

logger = logging.getLogger(__name__)


def _get_error_parameters(parameters: list[str], include_original: Optional[bool] = False) -> list[str]:
    errs = [f"{p}err{i}" for p in parameters for i in [1, 2]]
    if include_original:
        return parameters + errs
    return errs


class KnownExoplanetsDownloader(DatasetDownloader):
    """
    Data source: Nasa Exoplanet Archive (table: Planetary Systems)
    https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
    https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html

    Some notes on ps table quickness:
    https://decovar.dev/blog/2022/02/26/astronomy-databases-tap-adql/
    """

    _table_name = "ps"
    _mandatory_fields = ["tic_id", "gaia_id", "hostname", "pl_name", "default_flag"]

    # NOTE: the units of pl_tranmid and its boundaries are mistakenly labelled as "day". It should be "hours"
    #   see https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
    _unit_overrides = {p: u.hour for p in _get_error_parameters(["pl_trandur"], True)}

    _exo_service: Optional[TapService] = None

    def _initialize_services(self):
        if self._exo_service is None:
            self._exo_service = ExoService()

    def _download(
        self,
        limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        where: Optional[dict[str, Any | list[Any]]] = None,
        **kwargs,
    ) -> QTable:
        """
        Download exoplanet data from the NASA Exoplanet Archive.

        Args:
            limit: Maximum number of exoplanets to retrieve
            columns: Specific columns to retrieve
            where: Dictionary of field-value pairs to filter the data
            **kwargs: Additional parameters for future extensibility

        Returns:
            QTable containing the downloaded data
        """
        limit_clause = f"top {limit}" if limit else ""
        where_clause = _get_where_clause(where=where)
        fields = self._get_fields_to_query(columns=columns, use_cached_fields=True)

        query_str = f"select {limit_clause} {fields} from {self._table_name} {where_clause}"

        logger.info(f"Querying {self._exo_service.url}...")
        dataset = self._exo_service.query(query_str)
        n_planets = len(pd.unique(dataset["pl_name"]))
        n_records = len(dataset)

        logger.info(f"DONE! Collected {n_planets} unique planets, for a total of {n_records} records.")
        return dataset

    def _clean_and_fix(self, table: QTable) -> QTable:
        _parse_ids(table)
        fix_unrecognized_units(table=table, units_map=UNIT_MAPPER)
        override_units(table=table, unit_overrides=self._unit_overrides)

        return table

    def _get_table_header(self, table: QTable) -> QTableHeader:
        return _get_fixed_table_header(table=table, table_name=self._table_name, tap_service=self._exo_service)

    def _download_by_id(self, ids: Sequence[int], **kwargs) -> QTable:
        raise NotImplementedError("KnownExoplanetsDownloader does not support download_by_id() method")

    def _get_fields_to_query(self, use_cached_fields: bool = False, columns: Optional[Sequence[str]] = None) -> str:
        if columns:
            col_set = set(columns)
            col_set.update(self._mandatory_fields)
            fields = list(col_set)
        elif use_cached_fields:
            fields = _ALL_FIELDS
        else:
            # There are some unnecessary fields that cause troubles when saving the DF, we should remove them
            fields = [f for f in self._exo_service.get_field_names(self._table_name) if not f.endswith("str")]

        return ",".join(fields)


def _get_fixed_table_header(table: QTable, table_name: str, tap_service: TapService) -> QTableHeader:
    descriptions = tap_service.get_field_descriptions(table_name)
    return {
        c: TableColumnInfo(
            unit=str(table[c].unit) if hasattr(table[c], "unit") and table[c].unit is not None else None,
            dtype=table[c].dtype.str if hasattr(table[c], "dtype") and table[c].dtype is not None else None,
            description=descriptions[c],
        )
        for c in table.columns
        if c in descriptions
    }


def _parse_ids(table: QTable):
    """
    Parses some relevant IDs to their numerical representation, from string to masked integer
    """
    fill_value = -1
    tic_ids = np.char.split(table["tic_id"].astype(str), " ")
    gaia_ids = np.char.split(table["gaia_id"].astype(str), " ")

    # Extract the desired elements and convert to pandas nullable integer type
    tic_id_extracted = [int(item[1]) if len(item) > 1 else fill_value for item in tic_ids]
    gaia_id_extracted = [int(item[2]) if len(item) > 2 else fill_value for item in gaia_ids]

    table["tic_id"] = MaskedColumn(tic_id_extracted, fill_value=fill_value, dtype="int64")
    table["gaia_id"] = MaskedColumn(gaia_id_extracted, fill_value=fill_value, dtype="int64")


def _get_where_clause(where: Optional[dict[str, Any | list[Any]]]) -> str:
    if not where:
        return ""

    conditions = []
    for field, condition in where.items():
        if isinstance(condition, list):
            # For list values, format each item appropriately
            formatted_items = []
            for item in condition:
                if isinstance(item, str):
                    formatted_items.append(f"'{item}'")
                else:
                    formatted_items.append(str(item))
            conditions.append(f"{field} in ({','.join(formatted_items)})")
        else:
            # For single values
            if isinstance(condition, str):
                conditions.append(f"{field} = '{condition}'")
            else:
                conditions.append(f"{field} = {str(condition)}")

    condition_str = " and ".join(conditions)
    return f"where {condition_str}"


# TODO: Slim down the field list to only the ones we need
_ALL_FIELDS = [
    "tic_id",
    "gaia_id",
    "pl_name",
    "pl_letter",
    "hostname",
    "hd_name",
    "hip_name",
    "default_flag",
    "pl_refname",
    "st_refname",
    "sy_refname",
    "disc_pubdate",
    "disc_year",
    "discoverymethod",
    "disc_locale",
    "disc_facility",
    "disc_instrument",
    "disc_telescope",
    "disc_refname",
    "ra",
    "dec",
    "glon",
    "glat",
    "elon",
    "elat",
    "pl_orbper",
    "pl_orbpererr1",
    "pl_orbpererr2",
    "pl_orbperlim",
    "pl_orblpererr1",
    "pl_orblper",
    "pl_orblpererr2",
    "pl_orblperlim",
    "pl_orbsmax",
    "pl_orbsmaxerr1",
    "pl_orbsmaxerr2",
    "pl_orbsmaxlim",
    "pl_orbincl",
    "pl_orbinclerr1",
    "pl_orbinclerr2",
    "pl_orbincllim",
    "pl_orbtper",
    "pl_orbtpererr1",
    "pl_orbtpererr2",
    "pl_orbtperlim",
    "pl_orbeccen",
    "pl_orbeccenerr1",
    "pl_orbeccenerr2",
    "pl_orbeccenlim",
    "pl_eqt",
    "pl_eqterr1",
    "pl_eqterr2",
    "pl_eqtlim",
    "pl_occdep",
    "pl_occdeperr1",
    "pl_occdeperr2",
    "pl_occdeplim",
    "pl_insol",
    "pl_insolerr1",
    "pl_insolerr2",
    "pl_insollim",
    "pl_dens",
    "pl_denserr1",
    "pl_denserr2",
    "pl_denslim",
    "pl_trandep",
    "pl_trandeperr1",
    "pl_trandeperr2",
    "pl_trandeplim",
    "pl_tranmid",
    "pl_tranmiderr1",
    "pl_tranmiderr2",
    "pl_tranmidlim",
    "pl_trandur",
    "pl_trandurerr1",
    "pl_trandurerr2",
    "pl_trandurlim",
    "pl_rvamp",
    "pl_rvamperr1",
    "pl_rvamperr2",
    "pl_rvamplim",
    "pl_radj",
    "pl_radjerr1",
    "pl_radjerr2",
    "pl_radjlim",
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_radelim",
    "pl_ratror",
    "pl_ratrorerr1",
    "pl_ratrorerr2",
    "pl_ratrorlim",
    "pl_ratdor",
    "pl_ratdorerr1",
    "pl_ratdorerr2",
    "pl_ratdorlim",
    "pl_imppar",
    "pl_impparerr1",
    "pl_impparerr2",
    "pl_impparlim",
    "pl_cmassj",
    "pl_cmassjerr1",
    "pl_cmassjerr2",
    "pl_cmassjlim",
    "pl_cmasse",
    "pl_cmasseerr1",
    "pl_cmasseerr2",
    "pl_cmasselim",
    "pl_massj",
    "pl_massjerr1",
    "pl_massjerr2",
    "pl_massjlim",
    "pl_masse",
    "pl_masseerr1",
    "pl_masseerr2",
    "pl_masselim",
    "pl_bmassj",
    "pl_bmassjerr1",
    "pl_bmassjerr2",
    "pl_bmassjlim",
    "pl_bmasse",
    "pl_bmasseerr1",
    "pl_bmasseerr2",
    "pl_bmasselim",
    "pl_bmassprov",
    "pl_msinij",
    "pl_msinijerr1",
    "pl_msinijerr2",
    "pl_msinijlim",
    "pl_msinie",
    "pl_msinieerr1",
    "pl_msinieerr2",
    "pl_msinielim",
    "st_teff",
    "st_tefferr1",
    "st_tefferr2",
    "st_tefflim",
    "st_met",
    "st_meterr1",
    "st_meterr2",
    "st_metlim",
    "st_radv",
    "st_radverr1",
    "st_radverr2",
    "st_radvlim",
    "st_vsin",
    "st_vsinerr1",
    "st_vsinerr2",
    "st_vsinlim",
    "st_lum",
    "st_lumerr1",
    "st_lumerr2",
    "st_lumlim",
    "st_logg",
    "st_loggerr1",
    "st_loggerr2",
    "st_logglim",
    "st_age",
    "st_ageerr1",
    "st_ageerr2",
    "st_agelim",
    "st_mass",
    "st_masserr1",
    "st_masserr2",
    "st_masslim",
    "st_dens",
    "st_denserr1",
    "st_denserr2",
    "st_denslim",
    "st_rad",
    "st_raderr1",
    "st_raderr2",
    "st_radlim",
    "ttv_flag",
    "ptv_flag",
    "tran_flag",
    "rv_flag",
    "ast_flag",
    "obm_flag",
    "micro_flag",
    "etv_flag",
    "ima_flag",
    "pul_flag",
    "soltype",
    "sy_snum",
    "sy_pnum",
    "sy_mnum",
    "st_nphot",
    "st_nrvc",
    "st_nspec",
    "pl_nnotes",
    "pl_ntranspec",
    "pl_nespec",
    "pl_ndispec",
    "sy_pm",
    "sy_pmerr1",
    "sy_pmerr2",
    "sy_pmra",
    "sy_pmraerr1",
    "sy_pmraerr2",
    "sy_pmdec",
    "sy_pmdecerr1",
    "sy_pmdecerr2",
    "sy_plx",
    "sy_plxerr1",
    "sy_plxerr2",
    "sy_dist",
    "sy_disterr1",
    "sy_disterr2",
    "sy_bmag",
    "sy_bmagerr1",
    "sy_bmagerr2",
    "sy_vmag",
    "sy_vmagerr1",
    "sy_vmagerr2",
    "sy_jmag",
    "sy_jmagerr1",
    "sy_jmagerr2",
    "sy_hmag",
    "sy_hmagerr1",
    "sy_hmagerr2",
    "sy_kmag",
    "sy_kmagerr1",
    "sy_kmagerr2",
    "sy_umag",
    "sy_umagerr1",
    "sy_umagerr2",
    "sy_rmag",
    "sy_rmagerr1",
    "sy_rmagerr2",
    "sy_imag",
    "sy_imagerr1",
    "sy_imagerr2",
    "sy_zmag",
    "sy_zmagerr1",
    "sy_zmagerr2",
    "sy_w1mag",
    "sy_w1magerr1",
    "sy_w1magerr2",
    "sy_w2mag",
    "sy_w2magerr1",
    "sy_w2magerr2",
    "sy_w3mag",
    "sy_w3magerr1",
    "sy_w3magerr2",
    "sy_w4mag",
    "sy_w4magerr1",
    "sy_w4magerr2",
    "sy_gmag",
    "sy_gmagerr1",
    "sy_gmagerr2",
    "sy_gaiamag",
    "sy_gaiamagerr1",
    "sy_gaiamagerr2",
    "sy_tmag",
    "sy_tmagerr1",
    "sy_tmagerr2",
    "pl_controv_flag",
    "pl_tsystemref",
    "st_metratio",
    "st_spectype",
    "sy_kepmag",
    "sy_kepmagerr1",
    "sy_kepmagerr2",
    "st_rotp",
    "st_rotperr1",
    "st_rotperr2",
    "st_rotplim",
    "pl_projobliq",
    "pl_projobliqerr1",
    "pl_projobliqerr2",
    "pl_projobliqlim",
    "x",
    "pl_trueobliq",
    "pl_trueobliqerr1",
    "y",
    "z",
    "pl_trueobliqerr2",
    "htm20",
    "pl_trueobliqlim",
    "cb_flag",
    "sy_icmag",
    "sy_icmagerr1",
    "sy_icmagerr2",
    "rowupdate",
    "pl_pubdate",
    "releasedate",
    "dkin_flag",
]

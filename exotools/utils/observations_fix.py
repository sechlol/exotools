import json
from typing import Optional

import pandas as pd
from astroquery.exceptions import InvalidQueryError
from astroquery.mast import ObservationsClass


class ObservationFix(ObservationsClass):
    def query_criteria_columns_async(self, *, columns: Optional[list[str]] = None, **criteria) -> pd.DataFrame:
        """
        Given an set of criteria, returns a list of MAST observations.
        Valid criteria are returned by ``get_metadata("observations")``

        Parameters
        ----------
        columns : List[str], optional
            Used to select a subset of columns from the data
        pagesize : int, optional
            Can be used to override the default pagesize.
            E.g. when using a slow internet connection.
        page : int, optional
            Can be used to override the default behavior of all results being returned to obtain
            one sepcific page of results.
        **criteria
            Criteria to apply. At least one non-positional criteria must be supplied.
            Valid criteria are coordinates, objectname, radius (as in `query_region` and `query_object`),
            and all observation fields returned by the ``get_metadata("observations")``.
            The Column Name is the keyword, with the argument being one or more acceptable values for that parameter,
            except for fields with a float datatype where the argument should be in the form [minVal, maxVal].
            For non-float type criteria wildcards maybe used (both * and % are considered wildcards), however
            only one wildcarded value can be processed per criterion.
            RA and Dec must be given in decimal degrees, and datetimes in MJD.
            For example: filters=["FUV","NUV"],proposal_pi="Ost*",t_max=[52264.4586,54452.8914]
        """

        position, mashup_filters = self._parse_caom_criteria(**criteria)

        if not mashup_filters:
            raise InvalidQueryError("At least one non-positional criterion must be supplied.")

        selector = ", ".join(columns) if columns else "*"
        if position:
            service = self._caom_filtered_position
            params = {"columns": selector, "filters": mashup_filters, "position": position}
        else:
            service = self._caom_filtered
            params = {"columns": selector, "filters": mashup_filters}

        result = self._portal_api_connection.service_request_async(service, params)

        if columns:
            return pd.DataFrame(data=json.loads(result[0].text)["data"])
        return result


Observations = ObservationFix()

"""ExoTools - Tools for working with exoplanet data."""

__version__ = "0.0.2"


from .datasets import (
    CandidateExoplanetsDataset,
    GaiaParametersDataset,
    KnownExoplanetsDataset,
    LightcurveDataset,
    TicCatalogDataset,
    TicObservationsDataset,
)
from .db import CandidateDB, ExoDB, GaiaDB, LightcurveDB, LightCurvePlus, StarSystemDB, TicDB, TicObsDB
from .db.star_system import Planet, Star, StarSystem, UncertainDataSource, UncertainValue
from .utils.download import DownloadParams

__all__ = [
    # Main dataset classes
    "KnownExoplanetsDataset",
    "CandidateExoplanetsDataset",
    "TicCatalogDataset",
    "GaiaParametersDataset",
    "TicObservationsDataset",
    "LightcurveDataset",
    # Database classes
    "CandidateDB",
    "ExoDB",
    "GaiaDB",
    "StarSystemDB",
    "LightcurveDB",
    "LightCurvePlus",
    "TicObsDB",
    "TicDB",
    # Star system types
    "Star",
    "Planet",
    "StarSystem",
    "UncertainValue",
    "UncertainDataSource",
    # Utility types
    "DownloadParams",
]

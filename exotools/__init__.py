"""ExoTools - Tools for working with exoplanet data."""

__version__ = "0.1.0"

from .db.star_system import Star, Planet, StarSystem, UncertainValue, UncertainDataSource
from exotools.datasets.known_exoplanets import KnownExoplanetsDataset
from exotools.datasets.candidate_exoplanets import CandidateExoplanetsDataset
from exotools.datasets.tess import TessDataset
from exotools.datasets.lightcurves import LightcurveDataset

from .db import (
    CandidateDB,
    ExoDB,
    GaiaDB,
    StarSystemDB,
    LightcurveDB,
    LightCurvePlus,
    TessMetaDB,
    TicDB,
)

from .utils.download import DownloadParams

__all__ = [
    # Main dataset classes
    "KnownExoplanetsDataset",
    "CandidateExoplanetsDataset",
    "TessDataset",
    "LightcurveDataset",
    # Database classes
    "CandidateDB",
    "ExoDB",
    "GaiaDB",
    "StarSystemDB",
    "LightcurveDB",
    "LightCurvePlus",
    "TessMetaDB",
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

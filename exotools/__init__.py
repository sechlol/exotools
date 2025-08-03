"""ExoTools - Tools for working with exoplanet data."""

__version__ = "0.0.2"

from exotools.datasets.candidate_exoplanets import CandidateExoplanetsDataset
from exotools.datasets.known_exoplanets import KnownExoplanetsDataset
from exotools.datasets.lightcurves import LightcurveDataset
from exotools.datasets.tess import TessDataset

from .db import CandidateDB, ExoDB, GaiaDB, LightcurveDB, LightCurvePlus, StarSystemDB, TessMetaDB, TicDB
from .db.star_system import Planet, Star, StarSystem, UncertainDataSource, UncertainValue
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

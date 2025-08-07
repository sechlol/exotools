"""ExoTools - Tools for working with exoplanet data."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("exotools")
except importlib.metadata.PackageNotFoundError:
    # Package is not installed, try to read from pyproject.toml
    import os
    from pathlib import Path

    import tomli

    pyproject_path = Path(os.path.realpath(__file__)).parent / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as f:
            __version__ = tomli.load(f)["project"]["version"]
    except (FileNotFoundError, KeyError, ImportError):
        __version__ = "0.0.0"


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

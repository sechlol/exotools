# ExoTools


<div style="text-align:center">
  <img src="img/exotools_horizontal.png" alt="exotools logo" width="500"/>
</div>

[![Tests](https://github.com/sechlol/exotools/actions/workflows/tests.yml/badge.svg)](https://github.com/sechlol/exotools/actions/workflows/tests.yml)
[![Lint](https://github.com/sechlol/exotools/actions/workflows/lint.yml/badge.svg)](https://github.com/sechlol/exotools/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/sechlol/exotools/branch/main/graph/badge.svg)](https://codecov.io/gh/sechlol/exotools)


Python tools for working with exoplanet data from various sources including NASA's Exoplanet Archive, TESS, and Gaia.

## Installation

```bash
pip install exotools
```

## Features

- **Comprehensive Exoplanet Datasets**:
  - Known exoplanets from NASA's Exoplanet Archive
  - Candidate exoplanets (TOIs)
  - TESS observations
  - Gaia DR3 stellar parameters
  - Lightcurve data
- **Automatic Data Management**:
  - Cross-matching between NASA Exoplanet Archive and Gaia DR3
  - Automatic data cleaning and preprocessing
  - Handling of physical quantities with astropy QTable
- **Flexible Storage Options**:
  - HDF5 (single file, hierarchical storage)
  - ECSV (human-readable, portable)
  - Feather (fast and efficient)
- **High-Level APIs**:
  - Unified interface for accessing different data sources
  - Object-oriented representation of star systems
  - Parallel lightcurve downloading and processing

## Quick Start

```python
from exotools import KnownExoplanetsDataset
from exotools.io import EcsvStorage

# Download and store exoplanets from NASA Exoplanet Archive
storage = EcsvStorage(root_path="path/to/your/dataset")
exo_dataset = KnownExoplanetsDataset(storage=storage)
exo_db = exo_dataset.download_known_exoplanets(limit=10, store=True)

# Load existing dataset from disk
exo_db = exo_dataset.load_known_exoplanets_dataset()

# Access data via high-level API
star_names = exo_db.get_star_names()

# Or directly via the QTable view
star_names = exo_db.view["hostname"].tolist()

# Transform to pandas DataFrame for analysis
df = exo_db.to_pandas()
```

## Available Datasets

### Known Exoplanets

Access confirmed exoplanets from NASA's Exoplanet Archive with optional Gaia DR3 cross-matching:

```python
from exotools import KnownExoplanetsDataset, GaiaParametersDataset
from exotools.io import HDF5Storage

storage = HDF5Storage("exoplanet_data.h5")
dataset = KnownExoplanetsDataset(storage)

# Download with Gaia cross-matching
exo_db = dataset.download_known_exoplanets(with_gaia_star_data=True, store=True)
gaia_db = GaiaParametersDataset(storage).load_gaia_parameters_dataset()

# Get planets found by TESS, one record per planet
tess_planets = exo_db.get_tess_planets().get_default_records()

# Orbital periods of planets detected by TESS
orbital_periods = tess_planets.view["pl_orbper"]

# Distances of TESS planets from Earth, from the Gaia DR3 catalog
distances = gaia_db.get_by_tic_id(tess_planets.tic_ids).view["distance_gspphot"]
```

### Candidate Exoplanets (TOIs)

Work with TESS Objects of Interest (TOIs):

```python
from exotools import CandidateExoplanetsDataset

dataset = CandidateExoplanetsDataset(storage)
candidates_db = dataset.download_candidate_exoplanets(store=True)
```

### TESS Data

Access TESS observations and metadata:

```python
from exotools import TessDataset

# Download specific records (by tic_id) from TIC dataset to access TESS data products
tess_dataset = TessDataset(storage=storage)
tess_meta = tess_dataset.download_observation_metadata(targets_tic_id=exo_db.unique_ids)

```

### Lightcurve Data

Download and process lightcurves with parallel processing:

```python
from exotools import LightcurveDataset

# Download light curve .fits files
lc_dataset = LightcurveDataset(storage, verbose=True)
lc_db = lc_dataset.download_lightcurves_from_tess_db(tess_meta)

print(f"Downloaded {len(lc_db)} lightcurves:", lc_db.unique_obs_ids)

```

## Object-Oriented Star System Interface

Access exoplanet data through an intuitive object-oriented interface:

```python
from exotools import KnownExoplanetsDataset

dataset = KnownExoplanetsDataset(storage)
star_systems = dataset.load_star_system_dataset()

# Access star systems by name
kepler_system = star_systems.get_star_system_from_star_name("Kepler-90")

# Access star properties
star = kepler_system.star
print(f"Star: {star.name}")
print(f"  Radius: {star.radius.central}")  # UncertainValue with central, lower, upper
print(f"  Mass: {star.mass.central}")

# Access planets in the system, access their properties with units and confidence intervals
for planet in kepler_system.planets:
    print(f"Planet: {planet.name}")
    print(f"  Radius: {planet.radius.central}")
    print(f"  Mass: {planet.mass.central}")
    print(f"  Orbital Period: {planet.orbital_period.central}")
    print(f"  Transit Duration: {planet.transit_duration.central}")
```

## Storage Options

Choose the storage backend that fits your needs:

```python
# HDF5: Single file, hierarchical storage
from exotools.io import HDF5Storage
hdf5_storage = HDF5Storage("exoplanet_data.h5")

# ECSV: Human-readable, portable
from exotools.io import EcsvStorage
ecsv_storage = EcsvStorage("data_directory")

# Feather: Fast and efficient
from exotools.io import FeatherStorage
feather_storage = FeatherStorage("data_directory")
```

## License

MIT License

## Citation

If you use exotools in your research, please cite it as follows:

```bibtex
@misc{cardin2025exotools,
  author       = {Christian Cardin},
  title        = {ExoTools: Astronomical data access and analysis toolkit},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/sechlol/exotools}},
  note         = {Version 0.1.2},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

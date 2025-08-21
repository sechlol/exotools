# ExoTools

<div style="text-align:center">
  <img src="img/exotools_horizontal.png" alt="exotools logo" width="500"/>
</div>

[![Tests](https://github.com/sechlol/exotools/actions/workflows/tests.yml/badge.svg)](https://github.com/sechlol/exotools/actions/workflows/tests.yml)
[![Lint](https://github.com/sechlol/exotools/actions/workflows/lint.yml/badge.svg)](https://github.com/sechlol/exotools/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/sechlol/exotools/graph/badge.svg?token=M9PKWIJ25Z)](https://codecov.io/gh/sechlol/exotools)

[![PyPI version](https://badge.fury.io/py/exotools.svg)](https://badge.fury.io/py/exotools)
[![Python Versions](https://img.shields.io/pypi/pyversions/exotools.svg)](https://pypi.org/project/exotools/)
[![Powered by AstroPy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](https://www.astropy.org/)

Python tools for working with exoplanet data from various sources including NASA's Exoplanet Archive, TESS, and Gaia.

## Installation

Install exotools with pip:

```bash
pip install exotools
```

For development or the latest features, install from source:

```bash
git clone https://github.com/sechlol/exotools.git
cd exotools
pip install -e .
```

## Getting Started

### Basic Usage with Default Storage

The simplest way to get started is using the default in-memory storage:

```python
from exotools import KnownExoplanetsDataset

# Create dataset with default in-memory storage
dataset = KnownExoplanetsDataset()

# Download a small sample of known exoplanets
exo_db = dataset.download_known_exoplanets(limit=10, store=True)

# Access the data
print(f"Downloaded {len(exo_db)} exoplanet records")
print("First few host star names:", exo_db.view["hostname"][:5])

# Convert to pandas for analysis
df = exo_db.to_pandas()
print(df.head())
```

### Using Persistent Storage

For larger datasets or persistent storage, use one of the available storage backends:

```python
from exotools import KnownExoplanetsDataset
from exotools.io import Hdf5Storage

# Create HDF5 storage backend
storage = Hdf5Storage("exoplanet_data.h5")
dataset = KnownExoplanetsDataset(storage=storage)

# Download and store data persistently
exo_db = dataset.download_known_exoplanets(limit=100, store=True)

# Later, load the same data from storage
exo_db = dataset.load_known_exoplanets_dataset()
```

## Storage Options

ExoTools supports multiple storage backends to fit different use cases:

### Memory Storage (Default)
Fast but temporary storage - data is lost when the program ends:

```python
from exotools import KnownExoplanetsDataset

# Uses MemoryStorage by default
dataset = KnownExoplanetsDataset()
```

### Feather Storage (Recommended for Local Use)
**Recommended** for local development and analysis - fastest read access for the "download once, reuse multiple times" workflow:

```python
from exotools.io import FeatherStorage
from exotools import KnownExoplanetsDataset

storage = FeatherStorage("data_directory")
dataset = KnownExoplanetsDataset(storage=storage)
```

### HDF5 Storage (Recommended for Portability)
**Recommended** for managing multiple datasets in a single file or when transferring between machines (e.g., in HPC environments):

```python
from exotools.io import Hdf5Storage
from exotools import KnownExoplanetsDataset

storage = Hdf5Storage("my_data.h5")
dataset = KnownExoplanetsDataset(storage=storage)
```

### ECSV Storage (Maximum Compatibility)
Human-readable CSV format with metadata - slower but offers maximum portability and inspection capabilities:

```python
from exotools.io import EcsvStorage
from exotools import KnownExoplanetsDataset

storage = EcsvStorage("data_directory")
dataset = KnownExoplanetsDataset(storage=storage)
```

## Available Datasets

### Known Exoplanets

Access confirmed exoplanets from NASA's Exoplanet Archive:

```python
from exotools import KnownExoplanetsDataset
from exotools.io import Hdf5Storage

storage = Hdf5Storage("exoplanet_data.h5")
dataset = KnownExoplanetsDataset(storage=storage)

# Download confirmed exoplanets
exo_db = dataset.download_known_exoplanets(store=True)

# Load from storage
exo_db = dataset.load_known_exoplanets_dataset()

# Access specific data
tess_planets = exo_db.get_tess_planets()
print(f"Found {len(tess_planets)} planets discovered by TESS")
```

### Known Exoplanets with Gaia Data

Cross-match with Gaia DR3 stellar parameters:

```python
# Download with Gaia stellar data
exo_db = dataset.download_known_exoplanets(with_gaia_star_data=True, store=True)

# Load Gaia data separately
gaia_db = dataset.load_gaia_dataset_of_known_exoplanets()

# Get stellar distances from Gaia
distances = gaia_db.view["distance_gspphot"]
print(f"Stellar distances: {distances[:5]}")
```

### Candidate Exoplanets (TOIs)

Work with TESS Objects of Interest:

```python
from exotools import CandidateExoplanetsDataset

dataset = CandidateExoplanetsDataset(storage=storage)

# Download candidate exoplanets
candidates_db = dataset.download_candidate_exoplanets(store=True)

# Load from storage
candidates_db = dataset.load_candidate_exoplanets_dataset()

print(f"Found {len(candidates_db)} candidate exoplanets")
```

### Gaia Stellar Parameters

Access Gaia DR3 stellar parameters for specific stars:

```python
from exotools import GaiaParametersDataset

# Optional: authenticate for higher query limits
GaiaParametersDataset.authenticate("your_username", "your_password")

dataset = GaiaParametersDataset(storage=storage)

# Download Gaia data for specific Gaia source IDs
gaia_ids = [1234567890123456789, 2345678901234567890]
gaia_db = dataset.download_gaia_parameters(gaia_ids, store=True)

# Load from storage
gaia_db = dataset.load_gaia_parameters_dataset()

# Access stellar parameters
print(f"Downloaded {len(gaia_db)} stellar records")
print("Stellar masses:", gaia_db.view["mass_flame"][:5])
print("Stellar distances:", gaia_db.view["distance_gspphot"][:5])
print("Effective temperatures:", gaia_db.view["teff_gspphot"][:5])

# Access computed properties
print("Habitable zone inner edges:", gaia_db.view["hz_inner"][:5])
print("Habitable zone outer edges:", gaia_db.view["hz_outer"][:5])
```

### TESS Catalog Data

Access TESS Input Catalog (requires MAST authentication):

```python
from exotools import TicCatalogDataset

# Authenticate with MAST (required for TIC queries)
TicCatalogDataset.authenticate_casjobs("your_username", "your_password")

dataset = TicCatalogDataset(storage=storage)

# Search for targets by stellar mass
tic_db = dataset.download_tic_targets(
    star_mass_range=(0.8, 1.2),  # Solar masses
    limit=50,
    store=True
)

# Download specific TIC targets by ID
tic_ids = [1234567, 2345678, 3456789]
tic_db = dataset.download_tic_targets_by_ids(tic_ids, store=True)
```

### TESS Observation Metadata

Get observation metadata for lightcurve downloads:

```python
from exotools import TicObservationsDataset

# Optional: authenticate for faster downloads
TicObservationsDataset.authenticate_mast("your_mast_token")

dataset = TicObservationsDataset(storage=storage)

# Get observation metadata for specific TIC IDs
tic_ids = exo_db.unique_tic_ids[:10]  # First 10 TIC IDs
obs_db = dataset.download_observation_metadata(tic_ids, store=True)

print(f"Found {len(obs_db)} TESS observations")
```

### Lightcurve Data

Download TESS lightcurve FITS files:

```python
from exotools import LightcurveDataset
from pathlib import Path

# Lightcurves are stored as FITS files in the filesystem
lc_dataset = LightcurveDataset(
    lc_storage_path=Path("lightcurves"),
    verbose=True
)

# Download lightcurves for the observations
lc_db = lc_dataset.download_lightcurves_from_tic_db(obs_db)

print(f"Downloaded {len(lc_db)} lightcurves")

# Load previously downloaded lightcurves
lc_db = lc_dataset.load_lightcurve_dataset()
```

## Object-Oriented Star System Interface

Access exoplanet data through an intuitive object-oriented interface:

```python
from exotools import KnownExoplanetsDataset

dataset = KnownExoplanetsDataset(storage=storage)

# Download data with Gaia cross-matching (required for star systems)
exo_db = dataset.download_known_exoplanets(with_gaia_star_data=True, store=True)

# Load star system representation
star_systems = dataset.load_star_system_dataset()

# Access star systems by name
kepler_system = star_systems.get_star_system_from_star_name("Kepler-90")

# Access star properties with uncertainties
star = kepler_system.star
print(f"Star: {star.name}")
print(f"  Radius: {star.radius.central} ± {star.radius.upper - star.radius.central}")
print(f"  Mass: {star.mass.central} ± {star.mass.upper - star.mass.central}")

# Access planets in the system
for planet in kepler_system.planets:
    print(f"Planet: {planet.name}")
    print(f"  Radius: {planet.radius.central}")
    print(f"  Mass: {planet.mass.central}")
    print(f"  Orbital Period: {planet.orbital_period.central}")
```

## Advanced Usage

### Working with Multiple Datasets

```python
from exotools import (
    KnownExoplanetsDataset,
    CandidateExoplanetsDataset,
    TicObservationsDataset,
    LightcurveDataset
)
from exotools.io import Hdf5Storage
from pathlib import Path

# Shared storage for metadata
storage = Hdf5Storage("exoplanet_data.h5")

# Download known exoplanets
known_dataset = KnownExoplanetsDataset(storage=storage)
exo_db = known_dataset.download_known_exoplanets(limit=50, store=True)

# Download candidates
candidate_dataset = CandidateExoplanetsDataset(storage=storage)
candidates_db = candidate_dataset.download_candidate_exoplanets(limit=50, store=True)

# Get TESS observations for known exoplanets
obs_dataset = TicObservationsDataset(storage=storage)
obs_db = obs_dataset.download_observation_metadata(
    exo_db.unique_tic_ids[:10],
    store=True
)

# Download lightcurves
lc_dataset = LightcurveDataset(Path("lightcurves"))
lc_db = lc_dataset.download_lightcurves_from_tic_db(obs_db)
```

### Custom Storage Configuration

```python
from exotools.io import Hdf5Storage, FeatherStorage
from exotools import KnownExoplanetsDataset

# Configure HDF5 storage with custom path
hdf5_storage = Hdf5Storage("custom/path/exodata.h5")

# Configure Feather storage with custom directory
feather_storage = FeatherStorage("custom/data/directory")

# Use different storage for different datasets
exo_dataset = KnownExoplanetsDataset(storage=hdf5_storage)
candidate_dataset = CandidateExoplanetsDataset(storage=feather_storage)
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

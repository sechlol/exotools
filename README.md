# exotools

Python tools for working with exoplanet data from various sources including NASA's Exoplanet Archive, TESS, and Gaia.

## Installation

```bash
pip install exotools
```

## Features

- Access to multiple exoplanet datasets:
  - Known exoplanets from NASA's Exoplanet Archive
  - Candidate exoplanets
  - TESS observations
  - Gaia parameters
  - Lightcurve data
- Efficient data storage options (HDF5, ECSV, Feather)
- Unified API for accessing different data sources
- Comprehensive database management for exoplanet data

## Usage

```python
import exotools

# Get known exoplanets dataset
from exotools import KnownExoplanetsDataset
dataset = KnownExoplanetsDataset()
exoplanets = dataset.get_data()
print(f"Found {len(exoplanets)} exoplanets")

# Access candidate exoplanets
from exotools import CandidateExoplanetsDataset
candidates = CandidateExoplanetsDataset().get_data()

# Work with TESS data
from exotools import TessDataset
tess_data = TessDataset().get_data()
```

## Storage Options

exotools provides multiple storage backends for efficient data handling:

```python
from exotools.io import Hdf5Storage, EcsvStorage, FeatherStorage

# Use HDF5 for unified storage of tables and metadata
storage = Hdf5Storage("path/to/data.h5")

# Or use ECSV for human-readable storage
storage = EcsvStorage("path/to/data_dir")

# Or use Feather for high-performance storage
storage = FeatherStorage("path/to/data_dir")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

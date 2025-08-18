Installation
============

Requirements
------------

exotools requires Python 3.10 or later.

From PyPI
---------

The recommended way to install exotools is from PyPI:

.. code-block:: bash

    pip install exotools

This will install the latest stable release along with all required dependencies.

From Source
------------

You can also install the development version directly from GitHub:

.. code-block:: bash

    git clone https://github.com/sechlol/exotools.git
    cd exotools
    pip install -e ".[dev]"

Dependencies
------------

exotools depends on the following packages:

* numpy
* pandas
* pyarrow
* tables
* tabulate
* h5py
* pydantic
* oktopus
* lightkurve
* astropy
* astroquery
* casjobs
* tqdm
* pyvo
* tomli

Development dependencies:

* pytest
* pytest-cov
* python-dotenv
* ruff

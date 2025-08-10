Usage
=====

Getting Started
--------------

To use exotools in a project:

.. code-block:: python

    import exotools

Working with Exoplanet Datasets
------------------------------

exotools provides several dataset classes for working with exoplanet data:

.. code-block:: python

    from exotools import KnownExoplanetsDataset, CandidateExoplanetsDataset, TessDataset, LightcurveDataset

    # Load known exoplanets
    known = KnownExoplanetsDataset()

    # Access candidate exoplanets
    candidates = CandidateExoplanetsDataset()

    # Work with TESS data
    tess = TessDataset()

    # Analyze lightcurves
    lc = LightcurveDataset()

Database Utilities
-----------------

exotools provides database classes for efficient data management:

.. code-block:: python

    from exotools import CandidateDB, ExoDB, GaiaDB, StarSystemDB, LightcurveDB, TessMetaDB, TicDB

    # Access exoplanet database
    exo_db = ExoDB()

    # Work with Gaia data
    gaia_db = GaiaDB()

    # Manage lightcurve data
    lc_db = LightcurveDB()

Download Utilities
-----------------

For downloading data:

.. code-block:: python

    from exotools import DownloadParams

    params = DownloadParams(...)
    # Use params with appropriate dataset classes

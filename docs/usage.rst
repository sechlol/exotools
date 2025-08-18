Usage
=====

Getting Started
---------------

To use exotools in a project:

.. code-block:: python

    import exotools

Working with Exoplanet Datasets
-------------------------------

exotools provides several dataset classes for working with exoplanet data:

.. code-block:: python

    from exotools import KnownExoplanetsDataset, CandidateExoplanetsDataset, LightcurveDataset

    # Load known exoplanets
    known = KnownExoplanetsDataset()

    # Access candidate exoplanets
    candidates = CandidateExoplanetsDataset()

    # Analyze lightcurves
    lc = LightcurveDataset()

Database Utilities
------------------

exotools provides database classes for efficient data management:

.. code-block:: python

    from exotools import CandidateDB, ExoDB, GaiaDB, StarSystemDB, LightcurveDB, TicDB

    # Access exoplanet database
    exo_db = ExoDB()

    # Work with Gaia data
    gaia_db = GaiaDB()

    # Manage lightcurve data
    lc_db = LightcurveDB()

    # Access TIC catalog data
    tic_db = TicDB()

Star System Components
----------------------

exotools provides classes for working with star systems:

.. code-block:: python

    from exotools import Star, Planet, StarSystem, UncertainValue

    # Create a star
    star = Star(name="Sun", mass=1.0)

    # Create a planet
    planet = Planet(name="Earth", radius=1.0)

    # Create a star system
    system = StarSystem(star=star, planets=[planet])

Download Utilities
------------------

For downloading data:

.. code-block:: python

    from exotools import DownloadParams

    params = DownloadParams(...)
    # Use params with appropriate dataset classes

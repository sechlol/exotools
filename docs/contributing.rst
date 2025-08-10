Contributing
============

Contributions to exotools are welcome! Here's how you can help:

Development Setup
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/your-username/exotools.git
       cd exotools

3. Install development dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

4. Create a branch for your feature:

   .. code-block:: bash

       git checkout -b feature-name

Coding Standards
--------------

* We use ruff for code formatting and linting
* All code should include appropriate type hints
* New features should include tests

Testing
------

Run the test suite with pytest:

.. code-block:: bash

    pytest

Pull Requests
-----------

1. Update the documentation to reflect any changes
2. Update the CHANGELOG.md file
3. Make sure all tests pass
4. Submit a pull request to the main repository

Releasing
--------

For maintainers, to release a new version:

1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Build the package:

   .. code-block:: bash

       python -m build

4. Upload to PyPI:

   .. code-block:: bash

       python -m twine upload dist/*

# Changelog

All notable changes to the exotools package will be documented in this file.

## [0.4.0] - 2026-03-21

### Changed
- Replaced ad-hoc `.env` loading with `ExotoolsSecrets` (`pydantic_settings.BaseSettings`) and `load_secrets(env_file: Path)`.
- Added runtime dependency on `pydantic-settings`. CasJobs credentials now use `CASJOB_WSID` (integer) instead of `CASJOB_USER`; see `.env.example`.

## [0.3.0] - 2026-03-12

### Fixed
- Fixed a breaking change with the PS table, removing `gaia_id` and replacing it with `gaia_dr2_id` and `gaia_dr3_id`
- Add version check workflow


## [0.2.0] - 2026-03-11

### Changed
- Start using `uv` for dependency tracking.
- Deprecate Python 3.10 and 3.11


## [0.1.0] - 2025-07-28

### Added
- Initial release of exotools
- Support for accessing exoplanet data from NASA's Exoplanet Archive, TESS, and Gaia
- Download Light curves given TIC IDs
- Multiple storage backends: HDF5, ECSV, and Feather
- Database management for exoplanet data
- Public API with clean organization using __all__ lists

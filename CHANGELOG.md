# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Local caching for GHCN station data in `data/ghcn/` to improve performance and offline capability.
- Verification of ERA5 NetCDF data coverage (confirmed 1981-2024).

### Changed
- Modified `IranianDataFetcher` to check for local CSV files before fetching from NOAA.

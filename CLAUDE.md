# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Weather station data analysis package for extreme value statistics, long-term climate trends, and drought analysis. Features 133 years of data from Potsdam Säkularstation (Germany) and comprehensive Iran megadrought analysis tools (1950-2025).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for development
pip install -e .                      # editable install

# Run tests
pytest                                # all tests with coverage
pytest -m "not slow"                  # skip slow tests
pytest -m unit                        # unit tests only
pytest -m integration                 # integration tests only
pytest tests/test_data_fetcher.py    # single test file

# Code quality
black src/ tests/                     # format code
isort src/ tests/                     # sort imports
mypy src/                             # type checking
flake8 src/ tests/                    # linting
pre-commit run --all-files            # all quality checks

# Run analysis scripts
python potsdam_extreme_values.py      # complete extreme value analysis
python hottest_temperature_plot.py    # temperature analysis
python real_precipitation_plot.py     # precipitation analysis
python iran_drought_analysis.py       # Basic Iran drought analysis (2018-2025)
python iran_megadrought_analysis.py   # COMPREHENSIVE Iran analysis (1950-2025, all 10 stations)
```

## Architecture

### Core Package (`src/weatherstation_analysis/`)

- **data_fetcher.py**: `PotsdamDataFetcher` - fetches data from Meteostat API for Potsdam station
- **weather_fetcher.py**: `WeatherDataFetcher` - generic weather data fetcher
- **extreme_analyzer.py**: `ExtremeValueAnalyzer` - statistical analysis (Gumbel distribution, return periods, threshold exceedances)
- **visualization.py**: `WeatherPlotter` - publication-quality matplotlib plots
- **city_manager.py**: `CityManager` - manages city/station metadata

### Iran Drought Analysis (`src/weatherstation_analysis/`)

**Basic modules:**
- **iran_data_fetcher.py**: `IranianDataFetcher` - fetches NOAA GHCN-Daily data for 10 Iranian stations
- **drought_analyzer.py**: `DroughtAnalyzer` - calculates SPI, precipitation deficits, anomalies
- **drought_plotter.py**: `DroughtPlotter` - drought-specific visualizations

**Advanced modules (for publication-quality research):**
- **advanced_drought_analyzer.py**:
  - `DroughtReturnPeriodAnalyzer` - Gumbel/GEV extreme value analysis for drought return periods
  - `CompoundEventAnalyzer` - concurrent drought-heat event analysis
  - `DroughtDSAAnalyzer` - Duration-Severity-Area curve analysis
  - `DroughtRegimeAnalyzer` - CUSUM change point detection, decadal trends
  - `WaveletDroughtAnalyzer` - periodicity detection (ENSO, PDO connections)
  - `MegadroughtAnalyzer` - unified comprehensive analysis
- **advanced_drought_plotter.py**: `AdvancedDroughtPlotter` - publication-quality multi-panel figures

### Top-Level Analysis Scripts

- **iran_megadrought_analysis.py**: Comprehensive scientific analysis of Iran's 2018-2025 megadrought with full 1950-2025 historical context, all 10 stations, return period analysis, compound events, and regime shifts. Output to `results/iran_megadrought_analysis/`.

## Code Style

- Python 3.8+
- Black formatter (88 char line length)
- isort for imports (black profile)
- Type hints required (mypy strict mode)
- Google-style docstrings

## Data Sources

- **Primary**: Meteostat API (meteostat.net) - backed by DWD
- **Iran**: NOAA GHCN-Daily (10 major cities: Tehran, Mashhad, Isfahan, Tabriz, Shiraz, Ahvaz, Kerman, Rasht, Zahedan, Bandar Abbas)
- Quality filtering: only years with ≥80% data availability

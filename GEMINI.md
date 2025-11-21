# Gemini Code Companion Context

This document provides a comprehensive overview of the `weatherstation_data_analysis` project, designed to serve as a persistent context for the Gemini AI code assistant. Its purpose is to detail the project's architecture, conventions, and operational procedures, enabling the AI to provide informed and contextually-aware support.

## Project Overview

The `weatherstation_data_analysis` project is a Python-based scientific computing package for analyzing meteorological data. It focuses on extreme value analysis, long-term climate trends, and specialized drought assessment. The package is designed for both research and practical applications, providing reproducible and publication-quality outputs.

### Core Technologies

- **Programming Language:** Python 3.8+
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Data Acquisition:** `meteostat` API, `requests`
- **Statistical Analysis:** `scipy`

### Architectural Style

The project follows a modular, object-oriented architecture. Key functionalities are separated into distinct modules:

#### Core Modules
- `src/weatherstation_analysis/data_fetcher.py`: Handles data retrieval from meteorological APIs.
- `src/weatherstation_analysis/extreme_analyzer.py`: Implements statistical analysis of climate data.
- `src/weatherstation_analysis/visualization.py`: Manages the creation of plots and charts.

#### Iran Drought Analysis Modules
- `src/weatherstation_analysis/iran_data_fetcher.py`: Fetches NOAA GHCN-Daily data for 12 verified Iranian stations.
- `src/weatherstation_analysis/drought_analyzer.py`: Calculates SPI, precipitation deficits, anomalies.
- `src/weatherstation_analysis/drought_plotter.py`: Drought-specific visualizations.

#### Advanced Drought Analysis (Publication-Quality Research)
- `src/weatherstation_analysis/advanced_drought_analyzer.py`:
  - `DroughtReturnPeriodAnalyzer`: Gumbel/GEV extreme value theory for drought return periods
  - `CompoundEventAnalyzer`: Concurrent drought-heat event analysis with joint probabilities
  - `DroughtDSAAnalyzer`: Duration-Severity-Area curve analysis
  - `DroughtRegimeAnalyzer`: CUSUM change point detection, decadal trends
  - `WaveletDroughtAnalyzer`: Spectral analysis for climate oscillation connections
  - `MegadroughtAnalyzer`: Unified comprehensive analysis combining all methods
- `src/weatherstation_analysis/advanced_drought_plotter.py`: Publication-quality multi-panel figures.

This separation of concerns promotes maintainability and scalability.

## Building and Running

### Environment Setup

To create a consistent development environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bijanf/weatherstation_data_analysis.git
    cd weatherstation_data_analysis
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **For development, install additional tools:**
    ```bash
    pip install -r requirements-dev.txt
    pip install -e .
    ```

### Running Analysis Scripts

The project includes several high-level scripts to perform specific analyses. These scripts can be executed directly from the command line:

- **Potsdam Extreme Value Analysis:**
  ```bash
  python potsdam_extreme_values.py
  ```
- **Hottest Temperature Analysis:**
  ```bash
  python hottest_temperature_plot.py
  ```
- **Precipitation Analysis:**
  ```bash
  python real_precipitation_plot.py
  ```
- **Iran Basic Drought Analysis (2018-2025):**
  ```bash
  python iran_drought_analysis.py
  ```
- **Iran Comprehensive Megadrought Analysis (1950-2025, all 12 stations):**
  ```bash
  python iran_megadrought_analysis.py
  ```
- **Iran Simple Precipitation Plot (all stations, full period):**
  ```bash
  python iran_simple_precipitation_plot.py
  ```

### Iran Weather Stations (Verified GHCN IDs)

The following 12 Iranian stations have verified GHCN-Daily IDs with precipitation data:

| City | GHCN ID | Data Period | Baseline Mean |
|------|---------|-------------|---------------|
| Tehran | IR000407540 | 1943-2025 | 219 mm |
| Mashhad | IR000040745 | 1949-2025 | 99 mm |
| Isfahan | IRM00040800 | 1948-2025 | 124 mm |
| Tabriz | IR000040706 | 1951-2025 | 155 mm |
| Shiraz | IR000040848 | 1951-2025 | 148 mm |
| Kerman | IR000040841 | 1951-2025 | 66 mm |
| Kermanshah | IR000407660 | 1951-2025 | 188 mm |
| Zahedan | IR000408560 | 1951-2025 | 48 mm |
| Ahvaz | IRM00040811 | 1973-2025 | 185 mm |
| Bandar Abbas | IRM00040875 | 1967-2025 | 188 mm |
| Yazd | IRM00040821 | 1957-2025 | 75 mm |
| Bushehr | IRM00040858 | 1975-2025 | 41 mm |

### Data Sources

- **Primary (Potsdam):** Meteostat API (meteostat.net) - backed by DWD
- **Iran Stations:** NOAA GHCN-Daily (https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
- **Alternative (Iran):** CHIRPS satellite data (1981-present) for complete spatial coverage

## Development Conventions

The project adheres to a strict set of development conventions to ensure code quality, consistency, and maintainability.

### Code Style and Formatting

- **Style Guide:** Code follows the PEP 8 style guide.
- **Formatting:** `black` is used for automated code formatting, and `isort` for import sorting.
- **Line Length:** Maximum line length is 88 characters.

### Type Checking and Linting

- **Type Checking:** `mypy` is used for static type analysis. All function signatures must include type hints.
- **Linting:** `flake8` is used to enforce code quality and identify potential errors.

### Testing

- **Framework:** `pytest` is the testing framework.
- **Test Location:** Tests are located in the `tests/` directory.
- **Test Naming:** Test files are named `test_*.py` and test functions `test_*`.
- **Test Markers:** Tests are organized with markers (`slow`, `integration`, `unit`) to allow for selective execution.
- **Coverage:** `pytest-cov` is used to measure test coverage. A high level of coverage is expected.

### Pre-Commit Hooks

The project uses `pre-commit` to automate code quality checks before each commit. The hooks are configured in `.pre-commit-config.yaml` and include:

- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Linting
- `mypy`: Static type checking

To install the hooks, run: `pre-commit install`

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

## Proposed Enhancements for Publication-Quality Analysis

To elevate the `iran_megadrought_analysis.py` script to the standard of a top-tier scientific publication, the following enhancements should be implemented. The goal is to move beyond *describing* the drought to *attributing* its causes and impacts, specifically disentangling the roles of global warming, natural variability, and human water demand.

### 1. Make the Analysis "Bulletproof" (Methodological Rigor)

- **Data Homogenization**: Before analysis, apply a data homogenization process to all station records (e.g., using `pyhomog`) to correct for non-climatic shifts (e.g., station moves, instrument changes). This ensures that detected trends are purely climatic.
- **Integrate Satellite Precipitation Data**: Integrate a gridded satellite dataset like **CHIRPS** into the analysis. This will validate the ground station data and provide a more complete spatial picture of precipitation, especially in the vital mountainous headwaters where stations are sparse. This is a critical step.
- **Use Multiple Drought Indices**: The analysis currently relies on the Standardized Precipitation Index (SPI). It should be expanded to include the **Standardized Precipitation-Evapotranspiration Index (SPEI)**. SPEI incorporates temperature's effect on water demand, making it a more robust indicator of drought under a warming climate.
- **Quantify Uncertainty**: All statistical analyses must include uncertainty quantification.
    - For the return period analysis, calculate and plot **confidence intervals**.
    - For all trend analyses, report **p-values** to demonstrate statistical significance.

### 2. Attribute Precipitation Changes (Climate Signal vs. Natural Cycles)

- **Enhance Periodicity Analysis**: The existing `WaveletDroughtAnalyzer` should be enhanced to not just find cycles, but to statistically link them to major climate oscillations (e.g., ENSO, NAO).
- **Method**:
    1. Obtain time series data for relevant climate indices (e.g., Southern Oscillation Index).
    2. Use wavelet analysis to show shared periodicity between the precipitation data and these indices.
    3. Use a statistical model (e.g., multiple linear regression) to model precipitation based on these natural cycles.
    4. Analyze the *residual* of this model. This residual represents the precipitation variability *not* explained by natural cycles, and can be analyzed for a long-term trend more attributable to anthropogenic climate change.

### 3. Disentangle Climate Change from Population and Mismanagement

This is the most novel part of the proposed research. The goal is to separate the climatic drivers (less rain, more heat) from the societal drivers (water demand, policy).

- **Action**: Use a hydrological model to simulate a key river basin (e.g., the Zayandeh-Rud or Karun basins, fed by the Zagros mountains).
- **Method**:
    1.  **Input Climate Data**: Use the historical precipitation and temperature data (from GHCN-D and CHIRPS) to "force" a hydrological model (e.g., a simple conceptual model like HBV).
    2.  **Input Demand Data**: Use historical population data or agricultural land use statistics as a proxy for water demand.
    3.  **Run Two Scenarios**:
        *   **Scenario A (Climate Impact Only)**: Run the model with changing climate data but keep water demand constant at a baseline level (e.g., 1980s levels).
        *   **Scenario B (Combined Impact)**: Run the model with both changing climate data and changing water demand.
- **Outcome**: By comparing the simulated river flow and water storage between Scenario A and Scenario B, you can **quantitatively separate the impact of climate change (drought) from the impact of increased population and mismanagement (demand)**. This will provide a powerful and unique finding.

### 4. Strengthen the Global Warming Connection

- **Action**: Perform a robust trend analysis on the temperature data from all stations (1950-2025).
- **Method**: Calculate the rate of warming (e.g., °C per decade) and its statistical significance for each station and regionally.
- **Outcome**: This will provide direct, quantitative evidence of the local impact of global warming in Iran. This warming trend can then be directly linked to the increased drought severity through the SPEI analysis and the `CompoundEventAnalyzer`, creating a clear and compelling narrative for the paper.

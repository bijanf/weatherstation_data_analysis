# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Weather station data analysis package for extreme value statistics, long-term climate trends, and drought analysis. Features 130+ years of data from the Potsdam Säkularstation (Germany) and a comprehensive Iran megadrought analysis suite (1950–2025). Output is split between a reusable library (`src/weatherstation_analysis/`) and many standalone, run-once analysis/figure scripts at the repo root.

## ⚠️ Critical environment caveat: Meteostat 1.x vs 2.x

This is the first thing to check when anything that fetches weather data fails.

- `requirements.txt` pins `meteostat>=1.6.0` (unbounded), so a fresh install pulls **Meteostat 2.x**, which is a **breaking rewrite**. The 1.x class API (`from meteostat import Stations, Daily`; `Stations().nearby(...).fetch()`; `Daily(id, start, end).fetch()`) **no longer exists**.
- As a result, under the currently-installed `meteostat` (2.1.3) **`import weatherstation_analysis` raises `ImportError: cannot import name 'Stations'`**, and **`pytest` collects 0 tests with 3 collection errors** — the package and most scripts are written against the removed 1.x API.
- Files still on the broken 1.x API: package modules `data_fetcher.py`, `weather_fetcher.py`, `city_manager.py`; scripts `real_precipitation_plot.py`, `germany_centennial_stations.py`, `berlin_autumn_2024_infographic.py`, `berlin_autumn_decadal_analysis.py`, and the root `test_meteostat_api.py` / `test_german_stations.py` probes.
- **`hottest_temperature_plot.py` is the one file already migrated to the 2.x functional API** — use it as the reference for porting the rest.

### Meteostat 2.x API (the working pattern, from `hottest_temperature_plot.py`)

```python
import meteostat
from meteostat import Parameter, Point, daily

meteostat.config.block_large_requests = False   # required for daily ranges > 30 years
near = meteostat.stations.nearby(Point(52.3833, 13.0667, 81), radius=50000, limit=1)
sid = near.index[0]                              # DataFrame indexed by station id; Potsdam = "10379"
df = daily(sid, start, end, parameters=[Parameter.TMAX, Parameter.TMIN]).fetch()  # date-indexed DataFrame
```

Notes: `daily()/hourly()/monthly()` are module-level functions returning a `TimeSeries`; call `.fetch()` for a DataFrame. `meteostat.stations.nearby/meta/query` replace the old `Stations` class (`from meteostat.stations import ...` does NOT work — access via `meteostat.stations.<fn>`). Prefer **one wide date-range request** over per-year loops. Meteostat/DWD serves Potsdam (`10379`) data with ~0-day latency, so the current (incomplete) day is available but preliminary.

## Common Commands

```bash
# Install
pip install -r requirements.txt          # NOTE: pulls meteostat 2.x — see caveat above
pip install -r requirements-dev.txt      # dev tools (black, isort, mypy, flake8, pytest, pre-commit)
pip install -e .                         # editable install (src layout)

# Tests (currently failing to collect until the meteostat-2.x migration is finished)
pytest                                   # all tests + coverage (config in pyproject.toml)
pytest -m "not slow"                     # skip slow tests
pytest -m unit                           # markers: unit, integration, slow
pytest tests/test_extreme_analyzer.py    # single file
pytest tests/test_extreme_analyzer.py::TestExtremeValueAnalyzer::test_x   # single test

# Code quality (CI enforces black + flake8; mypy/isort are advisory there)
black src/ tests/                        # 88-char line length, target py38
isort src/ tests/                        # black profile
flake8 src/ tests/                       # config lives in .flake8 (see note below)
mypy src/                                # strict mode (see pyproject.toml)
pre-commit run --all-files

# Representative analysis scripts (run from the repo root; they write into plots/ or results/)
python hottest_temperature_plot.py       # Potsdam tmax/tmin extremes + threshold-day counts (MIGRATED to 2.x)
python potsdam_extreme_values.py         # Gumbel return periods / threshold exceedances
python iran_megadrought_analysis.py      # comprehensive Iran 1950–2025 analysis -> results/iran_megadrought_analysis/
```

## Architecture

Three layers: a **library** under `src/weatherstation_analysis/`, and at the repo root a set of **standalone analysis scripts** plus **ad-hoc API probes**.

### Core data flow

The consistent pipeline across the codebase is **fetcher → analyzer → plotter**:

- **Fetchers** return per-year dicts/DataFrames after a quality gate (only years with **≥80% daily coverage** are kept). `PotsdamDataFetcher` (`data_fetcher.py`) is the canonical example.
- **Analyzers** consume that data: `ExtremeValueAnalyzer` (Gumbel distribution, return periods, threshold exceedances) for the German stations; the drought analyzers (below) for Iran.
- **Plotters** (`WeatherPlotter`, `DroughtPlotter`, `AdvancedDroughtPlotter`) produce publication-/social-quality matplotlib figures.

### Library modules (`src/weatherstation_analysis/`)

`__init__.py` re-exports the public classes — read it for the authoritative surface. Beyond the core fetcher/analyzer/plotter trio and `CityManager`:

- **Iran drought (basic)**: `iran_data_fetcher.py` (`IranianDataFetcher`, `IranianStationRegistry`, `MultiStationFetcher` — NOAA GHCN-Daily, 10 cities), `drought_analyzer.py` (`DroughtAnalyzer`, `MultiStationDroughtAnalyzer` — SPI, deficits, anomalies), `drought_plotter.py`.
- **Iran drought (advanced, publication-grade)**: `advanced_drought_analyzer.py` bundles `DroughtReturnPeriodAnalyzer` (Gumbel/GEV), `CompoundEventAnalyzer` (concurrent drought–heat), `DroughtDSAAnalyzer` (Duration-Severity-Area), `DroughtRegimeAnalyzer` (CUSUM change points, decadal trends), `WaveletDroughtAnalyzer` (ENSO/PDO periodicity), `MegadroughtAnalyzer` (unified); plotted by `advanced_drought_plotter.py`.
- **Multi-source / gridded fetchers** (not mentioned in older docs): `era5_fetcher.py` (`ERA5Fetcher`, uses `cdsapi`), `chirps_fetcher.py` (`CHIRPSFetcher`, exported), and a **separate, un-exported** `chirps_data_fetcher.py` (`CHIRPSDataFetcher`, uses the `climateserv` API). When touching CHIRPS, confirm which of the two you mean. CHIRPS access has been historically painful — see `CHIRPS_Data_Access_Efforts.md`. `era5_fetcher.py`/`chirps_fetcher.py` guard `xarray` as an optional dependency via `TYPE_CHECKING`.

### Root-level scripts (not importable; run directly)

Grouped by topic — each writes PNGs into `plots/` unless noted:

- **Potsdam/Germany centennial**: `hottest_temperature_plot.py`, `potsdam_yearly_cycle.py` (annual daily-max temperature-cycle chart: min–max envelope + percentile bands + daily warm/cool bars + monthly anomaly table, English labels; Meteostat 2.x; renders Potsdam + Berlin-Dahlem in landscape plus Instagram 4:5 portrait and 9:16 story formats via the `LAYOUTS` config), `potsdam_extreme_values.py`, `real_precipitation_plot.py`, `real_precipitation_plot_dwd.py`, `update_potsdam_plot.py`, `germany_centennial_*.py`.
- **Berlin**: `berlin_autumn_2024_infographic.py`, `berlin_autumn_decadal_analysis.py` (also emit Bluesky captions into `plots/`).
- **Texas**: `texas_flash_flood_cumulative_improved.py`.
- **Iran**: `iran_megadrought_analysis.py` (canonical, → `results/iran_megadrought_analysis/`), `iran_drought_analysis.py`, `iran_hydrological_drought_analysis.py`, `iran_simple_precipitation_plot.py`.
- **Ad-hoc API probes (NOT pytest tests)**: root `test_meteostat_api.py`, `test_german_stations.py`, `test_wetterdienst_correct_api.py`. `pytest` only collects `tests/` (per `testpaths`), so these are scratch scripts despite the `test_` prefix.

## Conventions & config gotchas

- **Output dirs**: top-level scripts write figures to `plots/` (these PNGs ARE version-controlled — `.gitignore` deliberately keeps them) and structured results to `results/<analysis_name>/`.
- **flake8 config duplication**: there is both a `.flake8` file and a `[tool.flake8]` table in `pyproject.toml`. flake8 does **not** read `pyproject.toml` by default, so **`.flake8` is the effective config** (max-line-length 88, ignore E203/W503, max-complexity 10, google docstrings).
- **Dead entry point**: `setup.py` declares console_script `weatherstation-analysis=weatherstation_analysis.cli:main`, but there is **no `cli.py`** — installing then running the command will fail until a CLI is added or the entry point removed.
- **CI**: `.github/workflows/simple-ci.yml` is active (push/PR to `main`, **Python 3.9**): pytest+coverage, `black --check`, `flake8` (all hard-fail); `mypy` and `isort --check` run with `continue-on-error`. The richer `.github/workflows/ci.yml.disabled` is intentionally disabled. Note CI's 3.9 vs black/mypy's `py38` target.
- **`GEMINI.md`** is a parallel AI-assistant instruction file (for Gemini); keep substantive guidance changes in sync with this file when relevant.
- Style: Python 3.8+ (classifiers 3.8–3.11), Black (88 cols), isort (black profile), mypy strict, Google-style docstrings.

## Data Sources

- **Germany**: Meteostat (meteostat.net, backed by DWD). Potsdam Säkularstation ≈ Meteostat station `10379` at (52.3833°N, 13.0667°E).
- **Iran**: NOAA GHCN-Daily — Tehran, Mashhad, Isfahan, Tabriz, Shiraz, Ahvaz, Kerman, Rasht, Zahedan, Bandar Abbas.
- **Gridded (in progress)**: ERA5 via `cdsapi`, CHIRPS via `climateserv`.
- Quality filter throughout: keep only years with **≥80% daily availability**.

## Roadmap (aspirational — not yet implemented)

The repo carries a research roadmap to lift `iran_megadrought_analysis.py` to publication standard; treat as future work, not existing capability:

1. **Rigor**: data homogenization (`pyhomog`), integrate CHIRPS satellite precip, add SPEI alongside SPI, and quantify uncertainty (confidence intervals on return periods, p-values on trends).
2. **Attribution of precip change**: link `WaveletDroughtAnalyzer` cycles to climate indices (ENSO/NAO), regress precip on natural cycles, analyze the residual for an anthropogenic trend.
3. **Climate vs. demand**: force a conceptual hydrological model (e.g. HBV) on a Zagros-fed basin (Zayandeh-Rud/Karun); compare a climate-only scenario vs. a climate+demand scenario to separate drought from water mismanagement.
4. **Warming signal**: robust per-station/regional temperature trends (°C/decade + significance), tied back to SPEI and compound heat–drought events.

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

- `src/weatherstation_analysis/data_fetcher.py`: Handles data retrieval from meteorological APIs.
- `src/weatherstation_analysis/extreme_analyzer.py`: Implements statistical analysis of climate data.
- `src/weatherstation_analysis/visualization.py`: Manages the creation of plots and charts.
- `src/weatherstation_analysis/drought_analyzer.py`: Contains specialized tools for drought analysis.
- `src/weatherstation_analysis/iran_data_fetcher.py`: Manages data retrieval for Iranian weather stations.

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
- **Iran Drought Analysis:**
  ```bash
  python iran_drought_analysis.py
  ```

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

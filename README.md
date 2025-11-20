# Weather Station Data Analysis

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Simple CI](https://github.com/bijanf/weatherstation_data_analysis/workflows/Simple%20CI/badge.svg)](https://github.com/bijanf/weatherstation_data_analysis/actions)
[![Coverage Status](https://img.shields.io/badge/coverage-73%25-brightgreen)](https://github.com/bijanf/weatherstation_data_analysis/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)
[![Linting: flake8](https://img.shields.io/badge/linting-flake8-yellowgreen)](https://flake8.pycqa.org/)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green)](https://docs.pytest.org/)

![Extreme Value Analysis](plots/extreme_statistics_summary.png)

## 🌟 Overview

A comprehensive Python package for analyzing weather station data with a focus on **extreme value statistics**, **long-term climate trends**, and **drought analysis**. This project provides professional-grade tools for climate data analysis, featuring 133 years of real meteorological data from the Potsdam Säkularstation, Germany, and comprehensive drought analysis tools for Iranian weather stations.

### 🎯 Key Features

- **📊 Extreme Value Analysis**: Annual maxima, return periods, threshold exceedances
- **🌡️ Temperature Extremes**: Comprehensive temperature trend analysis
- **🌧️ Precipitation Analysis**: Cumulative precipitation and drought analysis
- **🇮🇷 Iran Drought Analysis**: Specialized tools for analyzing Iran's severe 2018-2025 drought
- **📉 Drought Indices**: Standardized Precipitation Index (SPI), precipitation deficits, anomalies
- **📈 Statistical Modeling**: Gumbel distribution fitting, trend analysis, multi-station comparisons
- **🎨 Publication-Quality Plots**: Professional visualizations ready for scientific publication
- **🔬 Reproducible Science**: Fully documented, tested, and version-controlled analysis
- **⚡ Modular Architecture**: Clean, maintainable code with proper separation of concerns

## 📈 Analysis Results

### 🌪️ Extreme Value Statistics (1893-2024)

| **Metric** | **Value** | **Year** |
|------------|-----------|----------|
| Maximum daily precipitation | **104.8 mm** | 1978 |
| Highest temperature | **38.9°C** | 2022 |
| Lowest temperature | **-26.8°C** | 1929 |
| Largest temperature range | **61.5°C** | 1929 |

## 📊 Visualization Gallery

Our analysis generates 9 comprehensive plots that tell the story of 133 years of climate data:

### 🌧️ **Precipitation Analysis**

![Extreme Precipitation Analysis](plots/extreme_precipitation_analysis.png)
**Extreme Precipitation Analysis** - Shows annual maximum daily precipitation with return period analysis using Gumbel distribution. The plot reveals increasing variability in extreme precipitation events and helps identify 100-year return periods.

![Cumulative Precipitation](plots/cumulative_precipitation_plot.png)
**Cumulative Precipitation** - Displays cumulative daily precipitation throughout the year for 130+ years of measurements. Highlights the exceptional drought year 2018 (red line) and current year 2025 (blue line) against historical variability.

### 🌡️ **Temperature Extremes**

![Temperature Extremes Analysis](plots/temperature_extremes_analysis.png)
**Temperature Extremes Analysis** - Four-panel comprehensive analysis showing: (1) annual maximum temperatures, (2) annual minimum temperatures, (3) temperature range (max-min), and (4) correlation between temperature extremes over time.

![Hottest Temperature Plot](plots/hottest_temperature_plot.png)
**Hottest Temperature Each Year** - Time series of annual maximum temperatures with polynomial trend line. Shows warming trend with the record high of 38.9°C in 2022.

![Coldest Temperature Plot](plots/coldest_temperature_plot.png)
**Coldest Temperature Each Year** - Time series of annual minimum temperatures with trend analysis. Reveals winter warming trend with fewer extreme cold events in recent decades.

### 🔥❄️ **Extreme Day Frequency**

![Days Above 30°C](plots/days_above_30C_plot.png)
**Hot Days Frequency** - Annual count of days with maximum temperature >30°C. Shows increasing frequency of hot days, indicating intensifying heat events.

![Days Below 0°C](plots/days_below_0C_plot.png)
**Cold Days Frequency** - Annual count of days with minimum temperature <0°C. Demonstrates decreasing frequency of freezing days, consistent with warming trends.

### 📈 **Statistical Summary**

![Extreme Statistics Summary](plots/extreme_statistics_summary.png)
**Extreme Statistics Summary** - Four-panel dashboard showing: (1) precipitation extremes distribution, (2) temperature extremes box plots, (3) return period analysis, and (4) threshold exceedance trends.

![Threshold Exceedance Analysis](plots/threshold_exceedance_analysis.png)
**Threshold Exceedance Analysis** - Analyzes frequency of extreme weather events exceeding various thresholds over time. Shows changing patterns in extreme event occurrence across decades.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bijanf/weatherstation_data_analysis.git
cd weatherstation_data_analysis

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Basic Usage

```python
from weatherstation_analysis import PotsdamDataFetcher, ExtremeValueAnalyzer, WeatherPlotter

# Fetch weather data
fetcher = PotsdamDataFetcher()
data = fetcher.fetch_comprehensive_data()

# Analyze extremes
analyzer = ExtremeValueAnalyzer(data)
extremes = analyzer.analyze_annual_extremes()

# Create visualizations
plotter = WeatherPlotter()
plotter.plot_annual_precipitation_extremes(extremes, 'precipitation_extremes.png')
```

### Command Line Interface

```bash
# Run complete extreme value analysis
python potsdam_extreme_values.py

# Run temperature analysis
python hottest_temperature_plot.py

# Run precipitation analysis
python real_precipitation_plot.py
```

## 🏗️ Project Structure

```
weatherstation_data_analysis/
├── src/
│   └── weatherstation_analysis/
│       ├── __init__.py
│       ├── data_fetcher.py      # Data acquisition module
│       ├── extreme_analyzer.py  # Statistical analysis module
│       └── visualization.py     # Plotting and visualization
├── tests/
│   ├── test_data_fetcher.py
│   ├── test_extreme_analyzer.py
│   └── test_visualization.py
├── docs/
│   └── ...                     # Documentation
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── scripts/
│   ├── potsdam_extreme_values.py
│   ├── hottest_temperature_plot.py
│   └── real_precipitation_plot.py
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml             # Project configuration
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 🔬 Scientific Methodology

### Data Source
- **Primary**: Meteostat API (meteostat.net)
- **Original**: Deutscher Wetterdienst (DWD)
- **Station**: Potsdam Säkularstation Telegraphenberg
- **Coordinates**: 52.3833°N, 13.0667°E
- **Elevation**: 81m above sea level
- **Period**: 1893-2025 (133 years)

### Quality Control
- **Coverage filtering**: Only includes years with ≥80% data availability
- **Missing value handling**: Proper NaN handling and interpolation
- **Leap year awareness**: Correct handling of 366-day years
- **Data validation**: Comprehensive error checking and logging

### Statistical Methods
- **Extreme Value Theory**: Gumbel distribution fitting
- **Trend Analysis**: Linear and polynomial regression
- **Return Period Analysis**: Empirical and theoretical return periods
- **Threshold Analysis**: Peak-over-threshold methods
- **Correlation Analysis**: Multivariate extreme dependencies

## 📊 API Reference

### Data Fetcher
```python
class PotsdamDataFetcher:
    def fetch_comprehensive_data(start_year=1890, end_year=2026) -> Dict[int, pd.DataFrame]
    def fetch_temperature_data(start_year=1890, end_year=2026) -> Dict[int, pd.DataFrame]
    def fetch_precipitation_data(start_year=1890, end_year=2026) -> Dict[int, pd.DataFrame]
```

### Extreme Analyzer
```python
class ExtremeValueAnalyzer:
    def analyze_annual_extremes(exclude_current_year=True) -> pd.DataFrame
    def calculate_return_periods(values, distribution='gumbel') -> Tuple[np.ndarray, np.ndarray]
    def analyze_threshold_exceedances(thresholds) -> pd.DataFrame
    def calculate_trends(variable) -> Dict[str, float]
```

### Visualization
```python
class WeatherPlotter:
    def plot_annual_precipitation_extremes(extremes_df, save_path) -> plt.Figure
    def plot_temperature_extremes_analysis(extremes_df, save_path) -> plt.Figure
    def plot_threshold_exceedance_analysis(all_data, save_path) -> plt.Figure
    def plot_statistics_summary(extremes_df, save_path) -> plt.Figure
```

## 🇮🇷 Iran Drought Analysis (2018-2025)

### Overview

This package includes comprehensive tools for analyzing **Iran's severe drought conditions** over the past 6-7 years. The drought, described as one of the worst in 50+ years, has resulted in:
- **89% decrease** in fall 2025 rainfall (driest autumn in 50 years)
- **45% below normal** annual precipitation (2024-25)
- **19 provinces** experiencing significant drought

### Quick Start

```bash
# Run comprehensive Iran drought analysis
python iran_drought_analysis.py
```

This will:
1. Fetch precipitation data from 10 major Iranian weather stations (NOAA GHCN-Daily)
2. Calculate precipitation deficits vs. baseline (1981-2010)
3. Compute Standardized Precipitation Index (SPI)
4. Generate publication-quality visualizations
5. Export statistical summaries for scientific papers

### Station Coverage

10 major Iranian cities analyzed:
- **Tehran** (Capital, Central Iran)
- **Mashhad** (Northeast)
- **Isfahan** (Central)
- **Tabriz** (Northwest)
- **Shiraz** (South-Central)
- **Ahvaz** (Southwest)
- **Kerman** (Southeast)
- **Rasht** (Caspian Sea)
- **Zahedan** (Southeast)
- **Bandar Abbas** (Persian Gulf)

### Example Usage

```python
from weatherstation_analysis.iran_data_fetcher import IranianDataFetcher
from weatherstation_analysis.drought_analyzer import DroughtAnalyzer
from weatherstation_analysis.drought_plotter import DroughtPlotter

# Fetch precipitation data
fetcher = IranianDataFetcher("Tehran")
prcp_data = fetcher.fetch_precipitation_data(start_year=1950, end_year=2025)

# Analyze drought conditions
analyzer = DroughtAnalyzer(
    precipitation_data=prcp_data,
    station_name="Tehran",
    baseline_start=1981,
    baseline_end=2010
)

# Perform comprehensive analysis
results = analyzer.analyze_drought_period(start_year=2018, end_year=2025)

# Calculate Standardized Precipitation Index
spi_data = analyzer.calculate_spi(scale_months=12)

# Generate visualizations
plotter = DroughtPlotter()
plotter.plot_comprehensive_drought_dashboard(
    drought_results=results,
    station_name="Tehran",
    output_file="results/tehran_drought_dashboard.png"
)
```

### Key Metrics Calculated

- **Precipitation Deficit**: Absolute (mm) and percentage deficit vs. baseline
- **Standardized Precipitation Index (SPI-12)**: International drought severity index
- **Cumulative Deficit**: Total water shortfall over drought period
- **Percent of Normal**: Annual precipitation as % of 1981-2010 baseline
- **Regional Comparison**: Multi-station drought severity patterns

### Output Files

```
results/iran_drought_analysis/
├── plots/
│   ├── tehran_drought_dashboard.png          # Comprehensive 4-panel view
│   ├── tehran_deficit_timeseries.png         # Annual deficit trends
│   ├── tehran_spi_timeseries.png             # SPI drought index
│   └── regional_drought_comparison.png       # Multi-station comparison
└── data/
    ├── tehran_annual_deficit.csv             # Annual statistics
    ├── tehran_spi12.csv                      # SPI time series
    └── regional_drought_summary.csv          # Regional summary
```

### Scientific Paper Support

The analysis generates all data and visualizations needed for scientific publication:
- **Quantitative metrics** for drought severity assessment
- **Statistical summaries** comparing to baseline period
- **Publication-quality plots** ready for journal submission
- **Scientific paper template** with complete structure and guidance

See detailed documentation:
- **[IRAN_DROUGHT_README.md](IRAN_DROUGHT_README.md)** - Comprehensive analysis guide
- **[scientific_paper_template.md](scientific_paper_template.md)** - Paper writing template

### Data Source

**NOAA GHCN-Daily** (Global Historical Climatology Network):
- URL: https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/
- Coverage: 1950-2025
- Quality: International standard, quality-controlled dataset

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "unit"      # Run only unit tests
pytest -m "integration"  # Run only integration tests
```

## 🔧 Development

### Code Quality Tools

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run all quality checks
pre-commit run --all-files
```

### Setting up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write comprehensive docstrings (Google style)
- Maintain test coverage above 90%
- Use meaningful variable and function names

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Deutscher Wetterdienst (DWD)** for maintaining long-term climate records
- **Meteostat project** for providing accessible climate data APIs
- **Potsdam Säkularstation** for 133 years of continuous measurements
- **Open Source Community** for the amazing tools and libraries used

## 📞 Contact

**Bijan Fallah** - Climate Data Scientist
- GitHub: [@bijanf](https://github.com/bijanf)

## 🔗 Links

- [Documentation](https://bijanf.github.io/weatherstation_data_analysis/)
- [PyPI Package](https://pypi.org/project/weatherstation-analysis/)
- [Issues](https://github.com/bijanf/weatherstation_data_analysis/issues)
- [Discussions](https://github.com/bijanf/weatherstation_data_analysis/discussions)

---

*For questions about the analysis or data, please open an issue or start a discussion.*

**⭐ If you find this project useful, please give it a star! ⭐**
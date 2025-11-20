# Iran Drought Analysis - Comprehensive Study (2018-2025)

## Overview

This analysis framework is specifically designed to study **Iran's severe drought conditions** over the past 6-7 years using historical precipitation data from the NOAA Global Historical Climatology Network (GHCN-Daily).

## Background

Iran is experiencing one of its most severe droughts in recorded history:
- **5th consecutive year of drought** (as of 2025)
- Fall 2025: **89% decrease** in rainfall - driest autumn in **50 years**
- November 2025: Only **2.3mm** precipitation (**81% below normal**)
- 2024-25 water year: **45% below normal** rainfall
- **19 provinces** experiencing significant drought

This drought has profound implications for:
- Water resource management
- Agricultural productivity
- Food security
- Economic stability
- Population displacement
- Regional climate patterns

## Scientific Approach

### Data Source
- **NOAA GHCN-Daily**: Global Historical Climatology Network - Daily dataset
- **Coverage**: 10 major Iranian weather stations with long-term records
- **Time Period**: 1950-2025 (where available)
- **Variables**: Daily precipitation, temperature

### Methodology

#### 1. **Baseline Comparison**
- Reference period: **1981-2010** (WMO standard climatological normal period)
- Calculate annual precipitation totals
- Compare recent years (2018-2025) against baseline statistics

#### 2. **Drought Indices**
- **Precipitation Deficit**: Absolute (mm) and percentage deficit
- **Precipitation Anomaly**: Standardized deviation from normal
- **SPI (Standardized Precipitation Index)**:
  - 12-month scale (SPI-12) for long-term drought assessment
  - Based on gamma distribution fitting
  - Classification: Extreme/Severe/Moderate drought

#### 3. **Regional Analysis**
- Multi-station comparison across Iran
- Spatial patterns of drought severity
- Identification of most affected regions

## Station Coverage

### Major Iranian Cities Analyzed

| Station | Coordinates | Elevation | Region | GHCN ID |
|---------|-------------|-----------|--------|---------|
| **Tehran (Mehrabad)** | 35.69°N, 51.31°E | 1191m | Capital, Central | IR000040754 |
| **Mashhad** | 36.27°N, 59.63°E | 999m | Northeast | IR000040745 |
| **Isfahan** | 32.75°N, 51.67°E | 1550m | Central | IR000040800 |
| **Tabriz** | 38.13°N, 46.30°E | 1361m | Northwest | IR000040708 |
| **Shiraz** | 29.53°N, 52.60°E | 1484m | South-Central | IR000040848 |
| **Ahvaz** | 31.33°N, 48.67°E | 23m | Southwest | IR000040831 |
| **Kerman** | 30.25°N, 56.97°E | 1754m | Southeast | IR000040856 |
| **Rasht** | 37.32°N, 49.62°E | -7m | North (Caspian) | IR000040719 |
| **Zahedan** | 29.47°N, 60.88°E | 1370m | Southeast | IR000040869 |
| **Bandar Abbas** | 27.22°N, 56.37°E | 10m | South (Persian Gulf) | IR000040885 |

## Installation & Setup

### Requirements

```bash
# Core dependencies already in requirements.txt
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.6.0
seaborn>=0.11.0
scipy>=1.9.0
requests>=2.28.0
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd weatherstation_data_analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start - Single Station Analysis

```python
from weatherstation_analysis.iran_data_fetcher import IranianDataFetcher
from weatherstation_analysis.drought_analyzer import DroughtAnalyzer
from weatherstation_analysis.drought_plotter import DroughtPlotter

# 1. Fetch precipitation data
fetcher = IranianDataFetcher("Tehran")
prcp_data = fetcher.fetch_precipitation_data(start_year=1950, end_year=2025)

# 2. Analyze drought conditions
analyzer = DroughtAnalyzer(
    precipitation_data=prcp_data,
    station_name="Tehran",
    baseline_start=1981,
    baseline_end=2010
)

# 3. Perform comprehensive analysis
results = analyzer.analyze_drought_period(start_year=2018, end_year=2025)

# 4. Calculate SPI
spi_data = analyzer.calculate_spi(scale_months=12)

# 5. Generate visualizations
plotter = DroughtPlotter()
plotter.plot_comprehensive_drought_dashboard(
    drought_results=results,
    station_name="Tehran",
    output_file="tehran_drought_dashboard.png"
)
```

### Comprehensive Multi-Station Analysis

```bash
# Run the complete analysis script
python iran_drought_analysis.py
```

This will:
1. Fetch data from all major Iranian stations
2. Calculate precipitation deficits and drought indices
3. Generate publication-quality visualizations
4. Export statistical summaries to CSV
5. Create comprehensive reports

### Output Structure

```
results/iran_drought_analysis/
├── plots/
│   ├── tehran_drought_dashboard.png          # 4-panel comprehensive view
│   ├── tehran_deficit_timeseries.png         # Annual deficit trends
│   ├── tehran_spi_timeseries.png             # SPI drought index
│   └── regional_drought_comparison.png       # Multi-station comparison
└── data/
    ├── tehran_annual_deficit.csv             # Annual statistics
    ├── tehran_spi12.csv                      # SPI time series
    └── regional_drought_summary.csv          # Regional summary
```

## Key Metrics Calculated

### 1. Precipitation Deficit
- **Absolute Deficit**: `baseline_mean - actual_precipitation` (mm)
- **Percentage Deficit**: `(baseline - actual) / baseline × 100` (%)
- **Percent of Normal**: `actual / baseline × 100` (%)

### 2. Standardized Precipitation Index (SPI)
| SPI Value | Category | Interpretation |
|-----------|----------|----------------|
| ≥ 2.0 | Extremely Wet | 🟦 Extremely wet conditions |
| 1.5 to 2.0 | Very Wet | 🔵 Very wet conditions |
| 1.0 to 1.5 | Moderately Wet | 🟢 Moderately wet |
| -1.0 to 1.0 | Normal | ⚪ Near normal conditions |
| -1.5 to -1.0 | Moderate Drought | 🟡 Moderate drought |
| -2.0 to -1.5 | Severe Drought | 🟠 Severe drought |
| ≤ -2.0 | Extreme Drought | 🔴 Extreme drought |

### 3. Cumulative Deficit
- Total water deficit over the drought period
- Indicates magnitude of precipitation shortfall
- Critical for water resource planning

## Visualizations

### 1. Comprehensive Drought Dashboard (4-panel)
- **Panel 1**: Annual precipitation vs baseline
- **Panel 2**: Percent of normal precipitation
- **Panel 3**: Cumulative deficit over time
- **Panel 4**: Statistical summary table

### 2. Precipitation Deficit Time Series
- Annual precipitation bars
- Baseline reference line
- Deficit/surplus annotations
- Cumulative deficit trends

### 3. SPI Time Series
- Color-coded drought severity
- Drought period highlighting
- Threshold reference lines
- Long-term trends

### 4. Regional Comparison
- Multi-station deficit comparison
- Worst drought year identification
- Spatial variation patterns

## Scientific Paper Preparation

### Suggested Paper Structure

#### 1. **Title**
"Quantifying Iran's Unprecedented Drought (2018-2025): A Comprehensive Analysis of Precipitation Deficits Across Major Urban Centers"

#### 2. **Abstract** (250 words)
- Background: Iran's water crisis context
- Objective: Quantify drought severity using GHCN-Daily data
- Methods: SPI calculation, baseline comparison (1981-2010)
- Results: Key findings (e.g., "45% mean deficit, 89% reduction in fall 2025")
- Conclusion: Implications for water management

#### 3. **Introduction**
- Iran's water scarcity context
- Recent drought observations (2018-2025)
- Potential anthropogenic factors (climate change, water mismanagement)
- Study objectives and significance

#### 4. **Data and Methods**
- **Data Source**: NOAA GHCN-Daily
- **Station Network**: 10 major urban centers
- **Baseline Period**: 1981-2010 (WMO standard)
- **Drought Metrics**:
  - Precipitation deficit (absolute and percentage)
  - Standardized Precipitation Index (SPI-12)
  - Cumulative deficit
- **Statistical Methods**: Gamma distribution fitting for SPI

#### 5. **Results**
- **5.1 Single Station Analysis (Tehran)**
  - Annual precipitation trends
  - Deficit quantification
  - SPI trends
  - Worst drought years

- **5.2 Regional Analysis**
  - Spatial patterns across Iran
  - Most/least affected regions
  - Regional variation in drought severity

- **5.3 Temporal Patterns**
  - Year-to-year variability
  - Persistent vs episodic drought
  - Worst year identification

#### 6. **Discussion**
- **6.1 Drought Severity Assessment**
  - Comparison to historical droughts
  - Context within global drought patterns

- **6.2 Potential Causes**
  - Climate change impacts
  - Large-scale climate oscillations (ENSO, NAO)
  - Anthropogenic water use
  - Land-use changes

- **6.3 Implications**
  - Water resource management
  - Agricultural impacts
  - Economic consequences
  - Policy recommendations

- **6.4 Limitations**
  - Data availability constraints
  - Station coverage gaps
  - Baseline period selection

#### 7. **Conclusions**
- Summary of key findings
- Severity assessment
- Future research directions
- Policy implications

#### 8. **References**
Key papers to cite:
- GHCN-Daily methodology papers
- SPI methodology (McKee et al., 1993)
- Iran climate studies
- IPCC reports on regional climate change

### Key Statistics to Report

From the analysis output, report:
1. **Baseline statistics** (1981-2010 mean annual precipitation per station)
2. **Drought period statistics**:
   - Mean annual deficit (%)
   - Total cumulative deficit (mm)
   - Years with deficit / total years
   - Worst year and its deficit
3. **SPI trends**:
   - Mean SPI-12 during drought period
   - Minimum SPI-12 observed
   - Frequency of extreme/severe drought months
4. **Regional patterns**:
   - Most affected station
   - Least affected station
   - Spatial gradient (if present)

## Data Availability Notes

### Current Status (November 2025)
- NOAA GHCN-Daily data is accessible via their API
- Some stations may have recent data gaps
- Fall-back to manual CSV downloads if API unavailable

### Alternative Data Sources

If NOAA GHCN is unavailable:

1. **Iran Meteorological Organization (IRIMO)**
   - Official Iranian weather data
   - URL: http://www.irimo.ir/eng/index.php
   - Contact: data@irimo.ir

2. **CHIRPS (Satellite-based precipitation)**
   - URL: https://data.chc.ucsb.edu/products/CHIRPS-2.0/
   - 1981-present, gridded data
   - Resolution: 0.05° (~5km)

3. **ERA5 Reanalysis**
   - URL: https://cds.climate.copernicus.eu/
   - Global coverage, hourly data
   - Resolution: 0.25° (~30km)

4. **Open Data Bay**
   - Historical Iranian precipitation (1901-2022)
   - 31 cities, monthly data
   - URL: Listed in research references

## Troubleshooting

### Issue: "Could not fetch data from NOAA"

**Solutions:**
1. Check internet connectivity
2. Wait and retry (API may be temporarily down)
3. Download CSV files manually from: https://www.ncei.noaa.gov/cdo-web/
4. Use alternative data sources (see above)
5. Contact NOAA support: ncei.orders@noaa.gov

### Issue: "No data available for baseline period"

**Solutions:**
1. Adjust baseline period (e.g., 1991-2020 instead of 1981-2010)
2. Use a different station with longer records
3. Check station ID in GHCN inventory

### Issue: "SPI calculation failed"

**Solutions:**
1. Ensure sufficient data points (need at least 30 years for baseline)
2. Check for excessive missing data
3. Try different distribution (normal vs gamma)

## Citation

If you use this analysis framework in your research, please cite:

```bibtex
@software{iran_drought_analysis_2025,
  title = {Iran Drought Analysis Framework (2018-2025)},
  author = {Weather Station Data Analysis Project},
  year = {2025},
  url = {https://github.com/[your-repo]},
  note = {Comprehensive drought analysis toolkit for Iranian weather stations}
}
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional Iranian weather stations
- Temperature-based drought indices (SPEI)
- Groundwater level integration
- Satellite data integration (GRACE, NDVI)
- Machine learning drought prediction

## License

[Specify your license here]

## Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [your email]

## Acknowledgments

- **NOAA NCEI** for GHCN-Daily data
- **Iran Meteorological Organization** for maintaining weather stations
- **Scientific community** researching Iran's water crisis

---

**Last Updated**: November 2025

**Status**: ✅ Production Ready

**Note**: This is a critical analysis for understanding Iran's water crisis. The drought is ongoing and has significant humanitarian implications. Results should be shared with relevant authorities and used to inform evidence-based water management policies.

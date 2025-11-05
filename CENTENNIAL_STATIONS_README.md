# Germany's Centennial Weather Stations Analysis

## Overview

This analysis focuses on Germany's longest-running weather stations with 100+ years of continuous records, visualizing warming trends and precipitation patterns with special color coding for extreme years.

## Concept

The visualization shows:
1. **Temperature Warming Trend**: Annual temperature anomalies (relative to 1961-1990 baseline)
2. **Cumulative Precipitation**: Annual total precipitation for each year
3. **Color Coding**:
   - 🔴 **Red**: Extreme hot years (>1.5 std dev above normal)
   - 🔵 **Blue**: Extreme cold years (>1.5 std dev below normal)
   - 🟢 **Green**: Extreme wet years (>1.5 std dev above normal precipitation)
   - 🟠 **Orange**: Extreme dry years (>1.5 std dev below normal precipitation)
   - 🟣 **Purple**: Recent years (1995+) - highlighting recent warming
   - ⚫ **Grey**: Normal years

## German Weather Stations with 100+ Years of Data

Based on DWD (Deutscher Wetterdienst) Climate Data Center records:

| Station ID | Name | Start Year | Years of Data | Location |
|------------|------|------------|---------------|----------|
| 00433 | **Hohenpeißenberg** | 1781 | 244+ years | 47.80°N, 11.01°E |
| 02925 | **Potsdam** | 1893 | 132+ years | 52.38°N, 13.07°E |
| 02014 | **Leck** | 1887 | 138+ years | 54.77°N, 8.97°E |
| 02290 | **Hamburg-Fuhlsbüttel** | 1891 | 134+ years | 53.63°N, 10.00°E |
| 02559 | **München-Nymphenburg** | 1879 | 146+ years | 48.16°N, 11.50°E |
| 02834 | **Nürnberg** | 1879 | 146+ years | 49.50°N, 11.05°E |
| 04887 | **Würzburg** | 1879 | 146+ years | 49.77°N, 9.96°E |
| 01048 | **Dresden-Klotzsche** | 1886 | 139+ years | 51.13°N, 13.75°E |

**Hohenpeißenberg** is particularly notable as it's one of the longest continuously operating meteorological observatories in the world with over 240 years of data!

## Scripts

### 1. `germany_centennial_climate_dwd.py`
Complete implementation using wetterdienst library to access DWD data directly.

**Features:**
- Automatically finds all German stations with 100+ years of records
- Fetches temperature and precipitation data from DWD CDC
- Calculates annual statistics and temperature anomalies
- Identifies extreme years statistically
- Creates publication-quality visualizations
- Generates comprehensive summary reports

**Usage:**
```bash
python germany_centennial_climate_dwd.py
```

### 2. `germany_centennial_simple.py`
Simplified version with predefined station list for faster execution.

**Usage:**
```bash
python germany_centennial_simple.py
```

### 3. `germany_centennial_stations.py`
Alternative implementation using Meteostat API (backup option).

## Data Sources

### Primary: DWD Climate Data Center
- **Provider**: Deutscher Wetterdienst (German Weather Service)
- **Access**: wetterdienst Python library
- **URL**: https://opendata.dwd.de/climate_environment/CDC/
- **Data**: Historical daily temperature and precipitation records
- **Quality**: Official government data with high quality control

### Alternative: Meteostat
- **Provider**: Meteostat.net (aggregates multiple sources including DWD)
- **Access**: meteostat Python library
- **Data**: Historical weather data from multiple sources

## Installation

### Requirements
```bash
pip install -r requirements.txt

# For DWD access (recommended):
pip install wetterdienst

# For Meteostat (alternative):
pip install meteostat
```

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.6.0
- scipy >= 1.9.0
- wetterdienst >= 0.100.0 (for DWD data)
- meteostat >= 1.6.0 (for Meteostat data)

## Methodology

### Temperature Anomaly Calculation
1. Calculate annual mean temperature for each year
2. Establish baseline period: 1961-1990 (standard WMO reference period)
3. Compute anomaly: `anomaly = annual_mean - baseline_mean`
4. Positive anomalies indicate warming, negative indicate cooling

### Extreme Year Identification
Uses statistical approach based on standard deviations:
- Calculate z-score for each year: `z = (value - mean) / std_dev`
- Classify as extreme if |z| > 1.5 (roughly top/bottom 7% of years)
- Priority: Temperature extremes first, then precipitation if not temperature extreme

### Trend Analysis
- Linear regression on temperature anomalies
- Comparison of pre-1950 vs. post-1990 means
- Quantification of warming rate per century

## Expected Results

Based on historical analysis of German centennial stations:

### Temperature Trends
- **Warming**: ~1-1.5°C increase from pre-1950 to post-1990
- **Trend**: +0.8 to +1.2°C per century (accelerating in recent decades)
- **Extreme Years**: More frequent hot years in recent decades
- **Coldest Years**: Typically found in 19th/early 20th century
- **Warmest Years**: Concentrated in 21st century (2018, 2019, 2020, 2022)

### Precipitation Patterns
- **Variability**: High year-to-year variation
- **Trends**: Less clear than temperature (varies by region)
- **Extreme Years**: Both wet and dry extremes present throughout record
- **Recent Changes**: Some evidence of increased variability

## Scientific Context

### Climate Change Evidence
The centennial station data provides clear evidence of:
1. **Warming trend**: Consistent across all long-term German stations
2. **Acceleration**: Warming rate increased post-1980
3. **Recent extremes**: Recent decades show unprecedented warmth
4. **Baseline shift**: The "normal" temperature has shifted upward

### Why This Matters
- **Historical context**: 100+ years provides pre-industrial baseline
- **Local relevance**: Shows climate change in specific German locations
- **Public communication**: Visual and intuitive presentation
- **Policy**: Supports climate action and adaptation planning

## Visualization Design

### Design Principles
- **Public-friendly**: Clear, not overly technical
- **Intuitive colors**: Hot=red, cold=blue, wet=green, dry=orange
- **Emphasis**: Recent years stand out to show current trends
- **Context**: Trend lines and statistical summaries aid interpretation

### Publication Quality
- High resolution (300 DPI)
- Professional color scheme
- Clear labels and legends
- Statistical rigor with accessible presentation

## Troubleshooting

### Network Issues
Both Met eostat and DWD CDC require internet access. If you encounter connection issues:
1. Check firewall settings
2. Try alternative data source
3. Use cached data if available
4. Download data manually from DWD CDC website

### API Changes
Weather data APIs may change. Check:
- wetterdienst documentation: https://wetterdienst.readthedocs.io/
- meteostat documentation: https://dev.meteostat.net/

## Future Enhancements

Potential improvements:
1. **Multi-station analysis**: Combine multiple centennial stations
2. **Regional patterns**: Compare northern vs. southern Germany
3. **Seasonal analysis**: Break down warming by season
4. **Extreme events**: Analyze heatwaves, cold snaps, droughts
5. **Interactive plots**: Web-based interactive visualizations
6. **Automated updates**: Regular data refreshes
7. **Additional variables**: Wind, sunshine, snow depth

## References

- Deutscher Wetterdienst (DWD): https://www.dwd.de/
- DWD Climate Data Center: https://opendata.dwd.de/
- Meteostat: https://meteostat.net/
- WMO Normals: https://public.wmo.int/en/our-mandate/climate/wmo-climate-normals

## Contact

For questions about this analysis, please open an issue on the GitHub repository.

---

**Note**: This analysis is based on observational data from official meteorological services. The warming trends shown are consistent with global climate change patterns documented by IPCC and other scientific bodies.

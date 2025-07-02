# Potsdam Precipitation Analysis: 133 Years of Climate Data

A comprehensive analysis of precipitation patterns at Potsdam Säkularstation, Germany, using 133 years of real meteorological data (1893-2025). This project creates compelling visualizations showing historical drought patterns and current climate trends.

## 🌧️ Project Overview

This project analyzes cumulative daily precipitation data from one of the world's oldest weather stations - the Potsdam Säkularstation Telegraphenberg. The analysis covers:

- **133 years of real data** (1893-2025) from Meteostat/DWD
- **Data quality verification** with 80%+ coverage filtering
- **Instagram-ready visualizations** highlighting climate extremes
- **Scientific accuracy** with no synthetic data generation

## 📊 Key Findings

- **2018**: Driest year on record (346mm total precipitation)
- **2025**: Currently tracking below the 2018 drought year through July
- **Historical context**: Complete precipitation records from 1893-2024
- **Data integrity**: 100% coverage for nearly all years in the dataset

## 🎯 Main Script: `real_precipitation_plot.py`

The main analysis script creates a powerful visualization showing:
- All historical years (1893-2024) in gray
- 2018 drought year highlighted in red
- 2025 current year highlighted in blue
- Professional Instagram-ready formatting

### Key Features:
- **Real data only**: No synthetic or interpolated values
- **Quality control**: Filters years with <80% data coverage
- **Clear visualization**: Simple color scheme focusing on key comparisons
- **Scientific accuracy**: Proper attribution and data sources

## 🚀 Getting Started

### Prerequisites

```python
import pandas as pd
import matplotlib.pyplot as plt
from meteostat import Stations, Daily
from datetime import datetime, date
import numpy as np
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/weatherstation_data_analysis.git
cd weatherstation_data_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python real_precipitation_plot.py
```

## 📁 Project Structure

```
weatherstation_data_analysis/
├── real_precipitation_plot.py          # Main analysis script
├── real_cumulative_precipitation_plot.png  # Generated visualization
├── Saekularstation_Potsdam_Telegraphenberg.ipynb  # Jupyter notebook
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
└── potsdam_data/                       # Data directory
    └── potsdam_dwd.zip                 # Raw data archive
```

## 🎨 Visualization Output

The script generates `real_cumulative_precipitation_plot.png` featuring:
- **12x10 inch format** optimized for Instagram
- **Bold, large fonts** (18pt axes, 16pt ticks) for social media readability
- **Clear legend** distinguishing historical data from extreme years
- **Professional attribution** with data source and creator credit

## 🔬 Data Sources

- **Primary**: Meteostat API (meteostat.net)
- **Original**: Deutscher Wetterdienst (DWD)
- **Station**: Potsdam Säkularstation Telegraphenberg
- **Coordinates**: 52.3833°N, 13.0667°E
- **Elevation**: 81m above sea level
- **Period**: 1893-2025 (133 years)

## 📱 Social Media Ready

The project includes Instagram-optimized content:
- **Kid-friendly explanation** suitable for science education
- **Comprehensive hashtags** for climate and data science communities
- **High-resolution output** (300 DPI) for professional sharing
- **Clear messaging** about climate patterns and data integrity

## 🛠️ Technical Details

### Data Quality Control
- **Coverage filtering**: Only includes years with ≥80% data availability
- **Missing value handling**: NaN values filled with 0.0 (no precipitation)
- **Leap year awareness**: Proper handling of 366-day years
- **Current year handling**: 2025 data through July 2nd only

### Performance
- **Efficient data retrieval**: Year-by-year API calls with error handling
- **Memory optimization**: Processes data incrementally
- **Error resilience**: Graceful handling of missing data periods

## 🔄 Version History

- **v2.0**: Complete rewrite focusing on 2018 drought vs 2025 comparison
- **v1.5**: Added 5 driest years visualization with color coding
- **v1.0**: Initial multi-color analysis with period-based grouping

## 👥 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with clear description

## 📄 License

This project is open source. Please cite appropriately when using the code or visualizations.

## 👨‍🔬 Author

**Bijan Fallah** - Climate Data Scientist
- GitHub: [@bijanf](https://github.com/bijanf)
- Data visualization and climate analysis specialist

## 🙏 Acknowledgments

- **Deutscher Wetterdienst (DWD)** for maintaining long-term climate records
- **Meteostat project** for providing accessible climate data APIs
- **Potsdam Säkularstation** for 133 years of continuous measurements

---

*For questions about the analysis or data, please open an issue or contact the author.*

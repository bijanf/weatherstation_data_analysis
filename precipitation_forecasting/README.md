# ERA5 Precipitation Forecasting Project

## 🌧️ Project Overview

This project aims to develop a sophisticated precipitation forecasting system using ERA5 reanalysis data and cutting-edge machine learning techniques. The goal is to predict cumulative precipitation for various forecast horizons (30-180 days) for Potsdam, Germany, using historical weather patterns and multiple AI/ML models.

## 🎯 Project Objectives

- **Primary Goal**: Forecast cumulative precipitation at the end of the year (blue line in current plots)
- **Secondary Goals**: 
  - Develop ensemble forecasting methods
  - Compare classical vs. modern ML approaches
  - Implement uncertainty quantification
  - Create automated daily forecast updates

## 🏗️ Project Structure

```
precipitation_forecasting/
├── data_pipeline/           # Data acquisition and preprocessing
│   ├── era5_downloader.py   # ERA5 data download from CDS
│   ├── data_preprocessor.py # Feature engineering & preprocessing
│   └── data_merger.py       # Combine multiple data sources
├── models/                  # Forecasting models
│   ├── baseline_models.py   # Climatology, trends, seasonal naive
│   ├── classical_ml.py      # ARIMA, Prophet, exponential smoothing
│   ├── ensemble_models.py   # Random Forest, XGBoost, LightGBM
│   ├── deep_learning.py     # LSTM, GRU, Transformers
│   └── model_factory.py     # Model management and selection
├── evaluation/              # Model evaluation and validation
│   ├── metrics.py          # Evaluation metrics and scoring
│   ├── backtesting.py      # Historical forecast validation
│   └── uncertainty.py      # Uncertainty quantification
├── visualization/           # Plotting and dashboards
│   ├── forecast_plots.py   # Time series and forecast visualization
│   ├── model_comparison.py # Model performance comparison
│   └── dashboard.py        # Interactive web dashboard
├── config/                  # Configuration files
│   ├── settings.yaml       # Main configuration
│   └── model_configs/      # Model-specific configurations
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── data/                   # Data storage
│   ├── raw/               # Raw ERA5 data
│   ├── processed/         # Preprocessed features
│   └── forecasts/         # Model predictions
└── results/               # Model outputs and reports
    ├── plots/             # Generated plots
    ├── reports/           # Analysis reports
    └── model_outputs/     # Trained models
```

## 📊 Data Sources

### Primary Data: ERA5 Reanalysis
- **Provider**: Copernicus Climate Data Store (CDS)
- **Temporal Coverage**: 1940-present (updated with ~5-day delay)
- **Spatial Resolution**: 0.25° × 0.25° (~25km)
- **Variables**:
  - Total precipitation
  - 2m temperature
  - Mean sea level pressure
  - 10m wind components (u, v)
  - 2m dewpoint temperature

### Secondary Data: Station Observations
- **Provider**: Meteostat/DWD
- **Station**: Potsdam (10381)
- **Purpose**: Validation and bias correction

## 🤖 Modeling Approach

### Phase 1: Baseline Models
- **Climatology**: Historical averages by day/month
- **Linear Trends**: Simple trend extrapolation
- **Seasonal Naive**: Last year's values
- **Moving Average**: Rolling average forecasts

### Phase 2: Classical Time Series Models
- **ARIMA**: Auto-regressive integrated moving average
- **Seasonal ARIMA**: SARIMA with seasonal components
- **Prophet**: Facebook's forecasting tool
- **Exponential Smoothing**: Holt-Winters methods

### Phase 3: Machine Learning Ensemble
- **Random Forest**: Tree-based ensemble
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Fast gradient boosting
- **Support Vector Regression**: Non-linear regression

### Phase 4: Deep Learning Models
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units
- **Transformers**: Attention-based models
- **CNN-LSTM**: Convolutional + recurrent hybrid
- **Attention-LSTM**: LSTM with attention mechanisms

### Phase 5: Advanced Ensemble Methods
- **Stacking**: Meta-learning approach
- **Bayesian Model Averaging**: Probabilistic ensemble
- **Dynamic Ensemble**: Adaptive model weights
- **Multi-Model Superensemble**: Weather-inspired approach

## 📈 Feature Engineering

### Temporal Features
- Lag features (1, 7, 14, 30, 90, 365 days)
- Rolling statistics (mean, std, min, max)
- Seasonal indicators (month, season, day of year)
- Cyclical encoding (sin/cos transformations)

### Meteorological Features
- Temperature anomalies and degree days
- Pressure tendencies and anomalies
- Wind speed and direction
- Humidity indices
- Precipitation indices (dry/wet spells)

### Climate Indices (Future Enhancement)
- North Atlantic Oscillation (NAO)
- Arctic Oscillation (AO)
- Atlantic Multidecadal Oscillation (AMO)
- El Niño Southern Oscillation (ENSO)

## 🎯 Evaluation Framework

### Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination
- **Skill Score**: Relative to climatology benchmark

### Validation Strategy
- **Time Series Cross-Validation**: Expanding window approach
- **Backtesting**: Historical forecast validation
- **Seasonal Evaluation**: Performance by season
- **Extreme Event Analysis**: Heavy precipitation events

### Uncertainty Quantification
- **Quantile Regression**: Prediction intervals
- **Bootstrap Ensembles**: Resampling-based uncertainty
- **Bayesian Methods**: Posterior distributions
- **Conformal Prediction**: Distribution-free intervals

## 🚀 Development Roadmap

### Week 1-2: Foundation
- [x] Set up project structure
- [x] Implement ERA5 data pipeline
- [x] Create feature engineering framework
- [ ] Develop baseline models
- [ ] Set up evaluation system

### Week 3-4: Classical Methods
- [ ] Implement ARIMA/SARIMA models
- [ ] Add Prophet forecasting
- [ ] Develop exponential smoothing
- [ ] Create model comparison framework

### Week 5-6: Machine Learning
- [ ] Implement ensemble methods (RF, XGB, LGB)
- [ ] Add support vector regression
- [ ] Develop hyperparameter optimization
- [ ] Create model selection pipeline

### Week 7-8: Deep Learning
- [ ] Implement LSTM/GRU models
- [ ] Add Transformer architecture
- [ ] Develop CNN-LSTM hybrid
- [ ] Create attention mechanisms

### Week 9-10: Advanced Ensemble
- [ ] Implement stacking ensemble
- [ ] Add Bayesian model averaging
- [ ] Develop dynamic ensemble weights
- [ ] Create multi-model superensemble

### Week 11-12: Production System
- [ ] Implement automated data updates
- [ ] Create forecast generation pipeline
- [ ] Develop web dashboard
- [ ] Add monitoring and alerting

## 📊 Daily Workflow

### Data Pipeline (Automated)
1. **06:00 UTC**: Check for new ERA5 data
2. **06:30 UTC**: Download and preprocess new data
3. **07:00 UTC**: Generate daily forecasts
4. **07:30 UTC**: Update visualizations and dashboard
5. **08:00 UTC**: Send forecast summary (optional)

### Model Management
- **Weekly**: Performance monitoring and evaluation
- **Monthly**: Model retraining and optimization
- **Quarterly**: Model architecture updates
- **Annually**: Comprehensive model review

## 🔧 Technical Requirements

### Dependencies
- **Python 3.8+**
- **Core Libraries**: pandas, numpy, scipy, scikit-learn
- **Deep Learning**: PyTorch, TensorFlow
- **Time Series**: statsmodels, prophet, sktime
- **Meteorology**: cdsapi, xarray, cfgrib
- **Visualization**: matplotlib, seaborn, plotly

### Infrastructure
- **Storage**: ~10GB for historical ERA5 data
- **Compute**: GPU recommended for deep learning models
- **API Access**: CDS API account for ERA5 data
- **Deployment**: Docker containers for production

## 📋 Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd precipitation_forecasting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure CDS API
Create `~/.cdsapirc` with your CDS API credentials:
```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-api-key>
```

### 4. Download Historical Data
```bash
python scripts/download_historical_data.py
```

### 5. Train Initial Models
```bash
python scripts/train_baseline_models.py
```

## 🎨 Visualization Examples

### Forecast Plots
- Time series with prediction intervals
- Seasonal decomposition
- Residual analysis
- Model comparison charts

### Performance Dashboards
- Real-time forecast accuracy
- Model performance metrics
- Seasonal skill scores
- Uncertainty quantification

## 📊 Expected Outcomes

### Scientific Impact
- **Methodology**: Novel ensemble approaches for precipitation forecasting
- **Insights**: Understanding of seasonal precipitation predictability
- **Benchmarks**: Comprehensive model comparison framework

### Technical Achievements
- **Accuracy**: Improved forecast skill over climatology
- **Reliability**: Robust uncertainty quantification
- **Automation**: Fully automated forecast system
- **Scalability**: Extensible to other locations

### GitHub Portfolio Value
- **Complexity**: Advanced ML/AI techniques
- **Completeness**: End-to-end ML project
- **Innovation**: Novel ensemble methods
- **Documentation**: Professional documentation and visualization

## 🤝 Contributing

### Code Style
- Follow PEP 8 conventions
- Use type hints
- Add comprehensive docstrings
- Write unit tests for new features

### Git Workflow
- Feature branches for development
- Descriptive commit messages
- Pull request reviews
- Continuous integration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Copernicus Climate Data Store** for ERA5 data
- **Meteostat** for station observations
- **Open Source Community** for ML libraries and tools

---

*This project represents a comprehensive approach to precipitation forecasting, combining traditional meteorological knowledge with modern machine learning techniques. The goal is to create a robust, automated system that can provide valuable insights into precipitation patterns and improve forecast accuracy.*
# PIK Cluster Setup Instructions

## Data Available on PIK Cluster

### ERA5 Data Location
- **Base path**: `/p/projects/climate_data_central/reanalysis/ERA5/`
- **ERA5-Land path**: `/p/projects/climate_data_central/reanalysis/ERA5-Land/`

### Available Variables
**ERA5 Standard:**
- `total_precipitation` ✅
- `2m_temperature` ✅
- `10m_u_component_of_wind` ✅
- `10m_v_component_of_wind` ✅
- `surface_pressure` ✅
- `mean_sea_level_pressure` ✅
- `maximum_2m_temperature_since_previous_post_processing` ✅
- `minimum_2m_temperature_since_previous_post_processing` ✅
- `surface_solar_radiation_downwards` ✅
- `surface_thermal_radiation_downwards` ✅

**ERA5-Land (Higher Resolution):**
- `total_precipitation` ✅
- `2m_temperature` ✅
- `surface_pressure` ✅
- `2m_dewpoint_temperature` ✅

## Setup Instructions

### 1. Clone Repository on Cluster
```bash
# SSH to cluster
ssh fallah@hpc.pik-potsdam.de

# Clone the repository
git clone https://github.com/bijanf/weatherstation_data_analysis.git
cd weatherstation_data_analysis

# Switch to the ERA5 forecasting branch
git checkout era5-precipitation-forecasting

# Navigate to the project
cd precipitation_forecasting
```

### 2. Environment Setup
```bash
# Load Python module (if needed)
module load python/3.9

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Exploration
```bash
# Check precipitation data structure
ls /p/projects/climate_data_central/reanalysis/ERA5/total_precipitation/ | head -10

# Check available years
ls /p/projects/climate_data_central/reanalysis/ERA5/total_precipitation/ | grep -E "19|20" | head -10

# Check file format and naming
ls /p/projects/climate_data_central/reanalysis/ERA5/total_precipitation/*2024* | head -5
```

### 4. Test Data Loading
```bash
# Test cluster data loader
python -c "from data_pipeline.cluster_data_loader import ClusterDataLoader; loader = ClusterDataLoader(); print('Data loader initialized successfully')"
```

## Next Steps

1. **Create cluster-optimized data loader** that reads from existing files
2. **Implement data preprocessing** for cluster NetCDF/GRIB files
3. **Test baseline models** with real ERA5 data
4. **Set up automated processing** on cluster resources

## Advantages of Cluster Approach

- ✅ **No download time** - data already available
- ✅ **High performance** - cluster computing resources
- ✅ **Large datasets** - full ERA5 archive access
- ✅ **Parallel processing** - utilize cluster nodes
- ✅ **Storage efficiency** - no data duplication
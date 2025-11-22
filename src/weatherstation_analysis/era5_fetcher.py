"""
ERA5 Data Fetcher for Iran Drought Analysis.

Fetches ERA5 reanalysis data from Copernicus Climate Data Store (CDS).

Key variables:
- Snow Water Equivalent (SWE): Critical for Alborz mountains
- Temperature: For SPEI calculation and trend analysis
- Total Precipitation: For comparison with CHIRPS
- Evaporation: For water balance

Data source: https://cds.climate.copernicus.eu/
Requires CDS API key in ~/.cdsapirc
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

try:
    import xarray
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import cdsapi

    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

logger = logging.getLogger(__name__)


class ERA5Fetcher:
    """
    Fetch ERA5 reanalysis data for Iran.

    Provides access to:
    - Snow Water Equivalent (SWE) for mountain analysis
    - Temperature for SPEI and warming trends
    - Precipitation for validation
    - Evaporation for water balance
    """

    # Iran bounding box [N, W, S, E]
    IRAN_BOUNDS = [40, 44, 25, 64]

    # Variable mappings
    VARIABLES = {
        "snow": "snow_depth_water_equivalent",
        "temperature": "2m_temperature",
        "precipitation": "total_precipitation",
        "evaporation": "total_evaporation",
    }

    # Alborz region for snow analysis
    ALBORZ_BOUNDS = {"lat_min": 35.5, "lat_max": 37.0, "lon_min": 50.0, "lon_max": 54.0}

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize ERA5 fetcher.

        Args:
            data_dir: Directory to cache downloaded data
        """
        if not HAS_XARRAY:
            raise ImportError("xarray required. Install: pip install xarray netCDF4")
        if not HAS_CDSAPI:
            raise ImportError("cdsapi required. Install: pip install cdsapi")

        self.data_dir = Path(data_dir) if data_dir else Path("data/era5")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CDS client
        try:
            self.client = cdsapi.Client()
            logger.info("CDS API client initialized")
        except Exception as e:
            logger.error(f"CDS API initialization failed: {e}")
            raise

        self._datasets = {}

    def fetch_monthly_data(
        self, variables: List[str], start_year: int = 1981, end_year: int = 2025
    ) -> Optional["xr.Dataset"]:
        """
        Fetch monthly ERA5 data for Iran.

        Args:
            variables: List of variable names ('snow', 'temperature', 'precipitation', 'evaporation')
            start_year: First year
            end_year: Last year

        Returns:
            xarray Dataset with requested variables
        """
        # Map variable names
        era5_vars = [self.VARIABLES.get(v, v) for v in variables]

        # Create filename based on variables
        var_str = "_".join(sorted(variables))
        filename = f"era5_iran_{var_str}_{start_year}_{end_year}.nc"
        local_file = self.data_dir / filename

        if local_file.exists():
            logger.info(f"Loading cached ERA5 data from {local_file}")
            return xr.open_dataset(local_file)

        logger.info(
            f"Downloading ERA5 data for {variables} ({start_year}-{end_year})..."
        )

        try:
            self.client.retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                {
                    "product_type": "monthly_averaged_reanalysis",
                    "variable": era5_vars,
                    "year": [str(y) for y in range(start_year, end_year + 1)],
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    "time": "00:00",
                    "area": self.IRAN_BOUNDS,
                    "format": "netcdf",
                },
                str(local_file),
            )

            # Check if file is a ZIP archive (CDS sometimes returns ZIP)
            import zipfile

            if zipfile.is_zipfile(local_file):
                logger.info("Extracting ZIP archive...")
                with zipfile.ZipFile(local_file, "r") as zf:
                    zf.extractall(self.data_dir)
                local_file.unlink()  # Remove ZIP

                # Find and merge extracted NetCDF files
                nc_files = list(self.data_dir.glob("data_stream-*.nc"))
                if nc_files:
                    datasets = [xr.open_dataset(f) for f in nc_files]
                    ds = xr.merge(datasets)
                    # Clean up individual files
                    for f in nc_files:
                        f.unlink()
                    # Save merged dataset
                    merged_file = self.data_dir / filename
                    ds.to_netcdf(merged_file)
                    logger.info(f"Downloaded ERA5 data: {list(ds.data_vars)}")
                    return ds
            else:
                ds = xr.open_dataset(local_file)
                logger.info(f"Downloaded ERA5 data: {list(ds.data_vars)}")
                return ds

        except Exception as e:
            logger.error(f"ERA5 download failed: {e}")
            return None

    def fetch_snow_data(
        self, start_year: int = 1981, end_year: int = 2025
    ) -> Optional["xr.Dataset"]:
        """
        Fetch Snow Water Equivalent data for Iran.

        Returns:
            xarray Dataset with SWE
        """
        return self.fetch_monthly_data(["snow"], start_year, end_year)

    def fetch_temperature_data(
        self, start_year: int = 1981, end_year: int = 2025
    ) -> Optional["xr.Dataset"]:
        """
        Fetch 2m temperature data for Iran.

        Returns:
            xarray Dataset with temperature (in Kelvin)
        """
        return self.fetch_monthly_data(["temperature"], start_year, end_year)

    def calculate_alborz_snowpack(
        self,
        ds: Optional["xr.Dataset"] = None,
        start_year: int = 1981,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Calculate seasonal snowpack statistics for Alborz region.

        Args:
            ds: Snow dataset (fetched if None)
            start_year: First year
            end_year: Last year

        Returns:
            DataFrame with annual snowpack statistics
        """
        if ds is None:
            ds = self.fetch_snow_data(start_year, end_year)

        if ds is None:
            raise ValueError("Could not fetch snow data")

        # Find SWE variable
        swe_var = None
        for var in ["sd", "sde", "snow_depth_water_equivalent"]:
            if var in ds.data_vars:
                swe_var = var
                break

        if swe_var is None:
            raise ValueError(f"SWE variable not found. Available: {list(ds.data_vars)}")

        # Subset to Alborz
        alborz = ds.sel(
            latitude=slice(
                self.ALBORZ_BOUNDS["lat_max"], self.ALBORZ_BOUNDS["lat_min"]
            ),
            longitude=slice(
                self.ALBORZ_BOUNDS["lon_min"], self.ALBORZ_BOUNDS["lon_max"]
            ),
        )

        # Calculate regional mean SWE
        swe_mean = alborz[swe_var].mean(dim=["latitude", "longitude"])

        # Convert to DataFrame
        df = swe_mean.to_dataframe().reset_index()
        df = df.rename(columns={swe_var: "swe_m"})
        df["swe_mm"] = df["swe_m"] * 1000  # Convert m to mm
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        # Calculate annual statistics
        annual_stats = []
        for year in df["year"].unique():
            year_data = df[df["year"] == year]

            # Peak SWE (usually in March)
            peak_swe = year_data["swe_mm"].max()
            peak_month = year_data.loc[year_data["swe_mm"].idxmax(), "month"]

            # Winter average (Dec-Mar)
            winter_months = [12, 1, 2, 3]
            winter_data = year_data[year_data["month"].isin(winter_months)]
            winter_mean = (
                winter_data["swe_mm"].mean() if len(winter_data) > 0 else np.nan
            )

            annual_stats.append(
                {
                    "year": year,
                    "peak_swe_mm": peak_swe,
                    "peak_month": peak_month,
                    "winter_mean_swe_mm": winter_mean,
                }
            )

        return pd.DataFrame(annual_stats)

    def calculate_temperature_trend(
        self,
        ds: Optional["xr.Dataset"] = None,
        start_year: int = 1981,
        end_year: int = 2025,
    ) -> Dict:
        """
        Calculate temperature trend for Iran.

        Args:
            ds: Temperature dataset (fetched if None)
            start_year: First year
            end_year: Last year

        Returns:
            Dictionary with trend statistics
        """
        from scipy import stats

        if ds is None:
            ds = self.fetch_temperature_data(start_year, end_year)

        if ds is None:
            raise ValueError("Could not fetch temperature data")

        # Find temperature variable
        temp_var = None
        for var in ["t2m", "2m_temperature", "temperature"]:
            if var in ds.data_vars:
                temp_var = var
                break

        if temp_var is None:
            raise ValueError(
                f"Temperature variable not found. Available: {list(ds.data_vars)}"
            )

        # Calculate national mean temperature
        temp_mean = ds[temp_var].mean(dim=["latitude", "longitude"])

        # Convert to Celsius
        temp_c = temp_mean - 273.15

        # Annual mean
        annual = temp_c.resample(time="YE").mean()
        df = annual.to_dataframe().reset_index()
        df = df.rename(columns={temp_var: "temp_c"})
        df["year"] = df["time"].dt.year

        # Calculate linear trend
        years = df["year"].values
        temps = df["temp_c"].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, temps)

        return {
            "trend_c_per_decade": slope * 10,
            "r_squared": r_value**2,
            "p_value": p_value,
            "mean_temp_c": temps.mean(),
            "annual_data": df,
        }

    def get_iran_national_timeseries(
        self, variable: str, start_year: int = 1981, end_year: int = 2025
    ) -> pd.DataFrame:
        """
        Get national average time series for a variable.

        Args:
            variable: Variable name ('snow', 'temperature', 'precipitation')
            start_year: First year
            end_year: Last year

        Returns:
            DataFrame with monthly national average
        """
        ds = self.fetch_monthly_data([variable], start_year, end_year)
        if ds is None:
            raise ValueError(f"Could not fetch {variable} data")

        # Get the variable (may have different name in file)
        var_name = list(ds.data_vars)[0]

        # National mean
        national_mean = ds[var_name].mean(dim=["latitude", "longitude"])

        df = national_mean.to_dataframe().reset_index()
        df = df.rename(columns={var_name: variable})
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        return df


def test_era5_connection() -> bool:
    """Test if ERA5 CDS API is configured and working."""
    try:
        client = cdsapi.Client()
        logger.info("CDS API connection successful")
        return True
    except Exception as e:
        logger.error(f"CDS API connection failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing ERA5 fetcher...")

    if test_era5_connection():
        try:
            fetcher = ERA5Fetcher()

            # Test temperature data (small request)
            print("\nFetching temperature data for 2023...")
            ds = fetcher.fetch_temperature_data(2023, 2023)

            if ds is not None:
                print(f"Success! Variables: {list(ds.data_vars)}")
                print(f"Dimensions: {dict(ds.dims)}")

                # Calculate trend
                trend = fetcher.calculate_temperature_trend(ds, 2023, 2023)
                print(f"Mean temperature: {trend['mean_temp_c']:.1f}°C")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("CDS API not configured. Create ~/.cdsapirc with your credentials.")

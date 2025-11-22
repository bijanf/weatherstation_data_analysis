"""
CHIRPS Data Fetcher for Iran Drought Analysis.

Fetches gridded precipitation data from CHIRPS (Climate Hazards Group
InfraRed Precipitation with Station data) for Iran.

Advantages over station data:
- Complete spatial coverage (no gaps)
- Covers mountains (Alborz, Zagros) where no stations exist
- 0.05° resolution (~5km)
- 1981-present

Data source: https://data.chc.ucsb.edu/products/CHIRPS-2.0/
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


class CHIRPSFetcher:
    """
    Fetch CHIRPS gridded precipitation data for Iran.

    CHIRPS provides satellite + station merged gridded precipitation
    at 0.05° resolution from 1981-present.
    """

    # Iran bounding box
    IRAN_BOUNDS = {"lat_min": 25.0, "lat_max": 40.0, "lon_min": 44.0, "lon_max": 64.0}

    # CHIRPS data URLs
    BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"
    MONTHLY_TIF_URL = f"{BASE_URL}/global_monthly/tifs"
    MONTHLY_NC_URL = f"{BASE_URL}/global_monthly/netcdf"

    # Iran geographic regions for analysis
    REGIONS = {
        "alborz": {
            "name": "Alborz Mountains",
            "lat_min": 35.5,
            "lat_max": 37.0,
            "lon_min": 50.0,
            "lon_max": 54.0,
            "description": "Northern mountain range, Tehran water source",
        },
        "zagros": {
            "name": "Zagros Mountains",
            "lat_min": 32.0,
            "lat_max": 35.0,
            "lon_min": 46.0,
            "lon_max": 50.0,
            "description": "Western mountain range",
        },
        "central_plateau": {
            "name": "Central Plateau",
            "lat_min": 31.0,
            "lat_max": 35.0,
            "lon_min": 51.0,
            "lon_max": 56.0,
            "description": "Semi-arid interior (Tehran, Isfahan, Yazd)",
        },
        "caspian": {
            "name": "Caspian Coast",
            "lat_min": 36.5,
            "lat_max": 38.5,
            "lon_min": 49.0,
            "lon_max": 54.0,
            "description": "Humid northern coast",
        },
        "persian_gulf": {
            "name": "Persian Gulf Coast",
            "lat_min": 25.0,
            "lat_max": 28.0,
            "lon_min": 50.0,
            "lon_max": 58.0,
            "description": "Hot desert coast (Bandar Abbas, Bushehr)",
        },
        "southeast_desert": {
            "name": "Southeast Desert",
            "lat_min": 28.0,
            "lat_max": 32.0,
            "lon_min": 58.0,
            "lon_max": 62.0,
            "description": "Arid region (Zahedan)",
        },
    }

    # Station locations for validation
    STATION_COORDS = {
        "Tehran": (35.683, 51.317),
        "Mashhad": (36.270, 59.633),
        "Isfahan": (32.617, 51.667),
        "Tabriz": (38.083, 46.283),
        "Shiraz": (29.550, 52.600),
        "Kerman": (30.250, 56.967),
        "Kermanshah": (34.350, 47.117),
        "Zahedan": (29.467, 60.883),
        "Ahvaz": (31.333, 48.667),
        "Bandar Abbas": (27.217, 56.367),
        "Yazd": (31.900, 54.283),
        "Bushehr": (28.967, 50.833),
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize CHIRPS fetcher.

        Args:
            data_dir: Directory to cache downloaded data
        """
        if not HAS_XARRAY:
            raise ImportError(
                "xarray is required. Install with: pip install xarray netCDF4"
            )

        self.data_dir = Path(data_dir) if data_dir else Path("data/chirps")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._dataset = None

    def fetch_monthly(
        self, start_year: int = 1981, end_year: int = 2025
    ) -> Optional[xr.Dataset]:
        """
        Fetch monthly CHIRPS data for Iran by downloading GeoTIFF files.

        Args:
            start_year: First year to fetch
            end_year: Last year to fetch

        Returns:
            xarray Dataset with monthly precipitation for Iran
        """
        logger.info(f"Fetching CHIRPS monthly data for Iran ({start_year}-{end_year})")

        try:
            import rioxarray  # For reading GeoTIFFs
            import gzip
            import tempfile
        except ImportError:
            logger.error("rioxarray required. Install: pip install rioxarray")
            return None

        datasets = []
        times = []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Skip future months
                from datetime import datetime

                if year == 2025 and month > datetime.now().month:
                    break

                filename = f"chirps-v2.0.{year}.{month:02d}.tif.gz"
                url = f"{self.MONTHLY_TIF_URL}/{filename}"
                local_gz = self.data_dir / filename
                local_tif = self.data_dir / filename.replace(".gz", "")

                try:
                    # Download if not cached
                    if not local_tif.exists():
                        if not local_gz.exists():
                            logger.debug(f"Downloading {filename}...")
                            response = requests.get(url, timeout=60)
                            if response.status_code != 200:
                                logger.warning(
                                    f"Could not download {filename}: {response.status_code}"
                                )
                                continue
                            with open(local_gz, "wb") as f:
                                f.write(response.content)

                        # Decompress
                        with gzip.open(local_gz, "rb") as f_in:
                            with open(local_tif, "wb") as f_out:
                                f_out.write(f_in.read())
                        local_gz.unlink()  # Remove .gz after extraction

                    # Read and subset to Iran
                    da = rioxarray.open_rasterio(local_tif).squeeze()

                    # Subset to Iran (note: y is descending in GeoTIFFs)
                    da_iran = da.sel(
                        y=slice(
                            self.IRAN_BOUNDS["lat_max"], self.IRAN_BOUNDS["lat_min"]
                        ),
                        x=slice(
                            self.IRAN_BOUNDS["lon_min"], self.IRAN_BOUNDS["lon_max"]
                        ),
                    )

                    datasets.append(da_iran.values)
                    times.append(pd.Timestamp(year=year, month=month, day=1))

                except Exception as e:
                    logger.warning(f"Error processing {filename}: {e}")
                    continue

            logger.info(f"Processed year {year}")

        if not datasets:
            logger.error("No CHIRPS data could be fetched")
            return None

        # Stack into xarray Dataset
        # Get coordinates from last successful read
        da_template = rioxarray.open_rasterio(local_tif).squeeze()
        da_iran = da_template.sel(
            y=slice(self.IRAN_BOUNDS["lat_max"], self.IRAN_BOUNDS["lat_min"]),
            x=slice(self.IRAN_BOUNDS["lon_min"], self.IRAN_BOUNDS["lon_max"]),
        )

        data_array = xr.DataArray(
            data=np.stack(datasets),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": times,
                "latitude": da_iran.y.values,
                "longitude": da_iran.x.values,
            },
            name="precip",
        )

        self._dataset = data_array.to_dataset()
        logger.info(f"Loaded CHIRPS data: {len(times)} months for Iran")

        return self._dataset

    def extract_regional_means(
        self, dataset: Optional[xr.Dataset] = None
    ) -> pd.DataFrame:
        """
        Extract mean precipitation for each defined region.

        Args:
            dataset: CHIRPS dataset (uses cached if None)

        Returns:
            DataFrame with columns: date, region, precipitation_mm
        """
        ds = dataset if dataset is not None else self._dataset
        if ds is None:
            raise ValueError("No dataset available. Call fetch_monthly() first.")

        results = []
        precip_var = "precip" if "precip" in ds.data_vars else list(ds.data_vars)[0]

        for region_id, region_info in self.REGIONS.items():
            # Subset to region
            region_ds = ds.sel(
                latitude=slice(region_info["lat_max"], region_info["lat_min"]),
                longitude=slice(region_info["lon_min"], region_info["lon_max"]),
            )

            # Mask no-data values (CHIRPS uses -9999)
            region_data = region_ds[precip_var].where(region_ds[precip_var] >= 0)

            # Calculate spatial mean (ignoring NaN)
            regional_mean = region_data.mean(dim=["latitude", "longitude"], skipna=True)

            # Convert to DataFrame
            df = regional_mean.to_dataframe().reset_index()
            df["region"] = region_id
            df["region_name"] = region_info["name"]
            df = df.rename(columns={precip_var: "precipitation_mm"})

            results.append(df[["time", "region", "region_name", "precipitation_mm"]])

        return pd.concat(results, ignore_index=True)

    def extract_station_pixels(
        self, dataset: Optional[xr.Dataset] = None
    ) -> pd.DataFrame:
        """
        Extract CHIRPS values at station locations for validation.

        Args:
            dataset: CHIRPS dataset (uses cached if None)

        Returns:
            DataFrame with CHIRPS values at each station location
        """
        ds = dataset if dataset is not None else self._dataset
        if ds is None:
            raise ValueError("No dataset available. Call fetch_monthly() first.")

        results = []
        precip_var = "precip" if "precip" in ds.data_vars else list(ds.data_vars)[0]

        for station, (lat, lon) in self.STATION_COORDS.items():
            try:
                # Extract nearest pixel
                station_data = ds[precip_var].sel(
                    latitude=lat, longitude=lon, method="nearest"
                )

                df = station_data.to_dataframe().reset_index()
                df["station"] = station
                df = df.rename(columns={precip_var: "chirps_mm"})
                results.append(df[["time", "station", "chirps_mm"]])

            except Exception as e:
                logger.warning(f"Could not extract {station}: {e}")
                continue

        return pd.concat(results, ignore_index=True)

    def calculate_annual_totals(
        self, dataset: Optional[xr.Dataset] = None
    ) -> xr.Dataset:
        """
        Calculate annual precipitation totals.

        Args:
            dataset: CHIRPS dataset (uses cached if None)

        Returns:
            Dataset with annual totals
        """
        ds = dataset if dataset is not None else self._dataset
        if ds is None:
            raise ValueError("No dataset available. Call fetch_monthly() first.")

        precip_var = "precip" if "precip" in ds.data_vars else list(ds.data_vars)[0]

        # Resample to annual and sum
        annual = ds[precip_var].resample(time="YE").sum()

        return annual.to_dataset(name="annual_precip_mm")

    def calculate_anomalies(
        self,
        dataset: Optional[xr.Dataset] = None,
        baseline_start: int = 1981,
        baseline_end: int = 2010,
    ) -> xr.Dataset:
        """
        Calculate precipitation anomalies relative to baseline period.

        Args:
            dataset: CHIRPS dataset
            baseline_start: Start year for baseline
            baseline_end: End year for baseline

        Returns:
            Dataset with anomalies (mm and percentage)
        """
        ds = dataset if dataset is not None else self._dataset
        if ds is None:
            raise ValueError("No dataset available. Call fetch_monthly() first.")

        precip_var = "precip" if "precip" in ds.data_vars else list(ds.data_vars)[0]

        # Calculate baseline climatology
        baseline = ds[precip_var].sel(
            time=slice(f"{baseline_start}-01-01", f"{baseline_end}-12-31")
        )
        climatology = baseline.groupby("time.month").mean()

        # Calculate anomalies
        anomaly_mm = ds[precip_var].groupby("time.month") - climatology
        anomaly_pct = (ds[precip_var].groupby("time.month") / climatology - 1) * 100

        return xr.Dataset(
            {
                "anomaly_mm": anomaly_mm,
                "anomaly_pct": anomaly_pct,
                "climatology": climatology,
            }
        )

    def get_iran_national_timeseries(
        self, dataset: Optional[xr.Dataset] = None
    ) -> pd.DataFrame:
        """
        Get national average precipitation time series.

        Args:
            dataset: CHIRPS dataset

        Returns:
            DataFrame with monthly national average
        """
        ds = dataset if dataset is not None else self._dataset
        if ds is None:
            raise ValueError("No dataset available. Call fetch_monthly() first.")

        precip_var = "precip" if "precip" in ds.data_vars else list(ds.data_vars)[0]

        # National mean
        national_mean = ds[precip_var].mean(dim=["latitude", "longitude"])

        df = national_mean.to_dataframe().reset_index()
        df = df.rename(columns={precip_var: "precipitation_mm"})
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        return df


def download_chirps_for_iran(
    start_year: int = 1981, end_year: int = 2025, data_dir: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convenience function to download and process CHIRPS data for Iran.

    Args:
        start_year: First year
        end_year: Last year
        data_dir: Directory for cached data

    Returns:
        Tuple of (regional_means, station_validation) DataFrames
    """
    fetcher = CHIRPSFetcher(data_dir=Path(data_dir) if data_dir else None)

    # Fetch data
    ds = fetcher.fetch_monthly(start_year, end_year)
    if ds is None:
        return None, None

    # Extract regional means
    regional = fetcher.extract_regional_means()

    # Extract station pixels for validation
    station = fetcher.extract_station_pixels()

    return regional, station


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO)

    print("Testing CHIRPS fetcher for Iran...")

    try:
        fetcher = CHIRPSFetcher()

        # Try to fetch just 2020 as a test
        ds = fetcher.fetch_monthly(2020, 2020)

        if ds is not None:
            print(f"\nDataset dimensions: {ds.dims}")
            print(f"Variables: {list(ds.data_vars)}")
            print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")

            # Test regional extraction
            regional = fetcher.extract_regional_means()
            print(f"\nRegional data shape: {regional.shape}")
            print(regional.groupby("region")["precipitation_mm"].mean())

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: CHIRPS requires internet access and xarray/netCDF4 packages.")

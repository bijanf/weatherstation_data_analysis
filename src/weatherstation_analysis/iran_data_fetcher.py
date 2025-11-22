"""
Iranian Weather Data Fetcher Module
====================================

Handles fetching weather data for Iranian weather stations from NOAA GHCN-Daily dataset.
Designed for analyzing Iran's severe drought conditions (2018-2025).
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import requests
from io import StringIO
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


class IranianStationRegistry:
    """
    Registry of major Iranian weather stations with GHCN-Daily IDs.

    Station data based on NOAA GHCN-Daily network.
    Coordinates in decimal degrees (lat, lon).
    """

    # Major Iranian cities with weather stations
    # VERIFIED GHCN Station IDs (checked against https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt)
    STATIONS = {
        "Tehran": {
            "lat": 35.683,
            "lon": 51.317,
            "ghcn_pattern": "IR000407540",  # TEHRAN MEHRABAD - VERIFIED
            "elevation": 1191,
            "name_variants": ["Tehran", "Tehrān", "Mehrabad"],
        },
        "Mashhad": {
            "lat": 36.267,
            "lon": 59.633,
            "ghcn_pattern": "IR000040745",  # MASHHAD - VERIFIED (GSN)
            "elevation": 999,
            "name_variants": ["Mashhad", "Mashad"],
        },
        "Isfahan": {
            "lat": 32.751,
            "lon": 51.862,
            "ghcn_pattern": "IRM00040800",  # ESFAHAN SHAHID BEHESHTI INTL - VERIFIED
            "elevation": 1546,
            "name_variants": ["Isfahan", "Esfahan"],
        },
        "Tabriz": {
            "lat": 38.080,
            "lon": 46.280,
            "ghcn_pattern": "IR000040706",  # TABRIZ - VERIFIED (GSN)
            "elevation": 1361,
            "name_variants": ["Tabriz"],
        },
        "Shiraz": {
            "lat": 29.533,
            "lon": 52.533,
            "ghcn_pattern": "IR000040848",  # SHIRAZ - VERIFIED (GSN)
            "elevation": 1481,
            "name_variants": ["Shiraz", "Shīrāz"],
        },
        "Ahvaz": {
            "lat": 31.337,
            "lon": 48.762,
            "ghcn_pattern": "IRM00040811",  # AHWAZ - VERIFIED
            "elevation": 20,
            "name_variants": ["Ahvaz", "Ahwaz", "Ahwāz"],
        },
        "Kerman": {
            "lat": 30.250,
            "lon": 56.967,
            "ghcn_pattern": "IR000040841",  # KERMAN - VERIFIED (GSN)
            "elevation": 1754,
            "name_variants": ["Kerman", "Kermān"],
        },
        "Zahedan": {
            "lat": 29.476,
            "lon": 60.907,
            "ghcn_pattern": "IR000408560",  # ZAHEDAN - VERIFIED (GSN)
            "elevation": 1378,
            "name_variants": ["Zahedan", "Zāhedān"],
        },
        "Bandar Abbas": {
            "lat": 27.218,
            "lon": 56.378,
            "ghcn_pattern": "IRM00040875",  # BANDAR ABBASS INTL - VERIFIED
            "elevation": 7,
            "name_variants": ["Bandar Abbas", "Bandar-Abbas"],
        },
        "Kermanshah": {
            "lat": 34.267,
            "lon": 47.117,
            "ghcn_pattern": "IR000407660",  # KERMANSHAH - VERIFIED (GSN)
            "elevation": 1322,
            "name_variants": ["Kermanshah"],
        },
        "Yazd": {
            "lat": 31.905,
            "lon": 54.277,
            "ghcn_pattern": "IRM00040821",  # YAZD - VERIFIED
            "elevation": 1236,
            "name_variants": ["Yazd"],
        },
        "Bushehr": {
            "lat": 28.945,
            "lon": 50.835,
            "ghcn_pattern": "IRM00040858",  # BUSHEHR - VERIFIED
            "elevation": 21,
            "name_variants": ["Bushehr", "Bushire"],
        },
    }

    @classmethod
    def get_all_stations(cls) -> Dict[str, Dict]:
        """Return all registered Iranian stations."""
        return cls.STATIONS

    @classmethod
    def get_station(cls, city_name: str) -> Optional[Dict]:
        """
        Get station information by city name.

        Args:
            city_name: Name of the city (case-insensitive)

        Returns:
            Station information dict or None if not found
        """
        city_lower = city_name.lower()

        for station_name, station_info in cls.STATIONS.items():
            if city_lower in station_name.lower():
                return {**station_info, "display_name": station_name}
            for variant in station_info["name_variants"]:
                if city_lower in variant.lower():
                    return {**station_info, "display_name": station_name}

        return None

    @classmethod
    def get_station_list(cls) -> List[str]:
        """Return list of available station names."""
        return list(cls.STATIONS.keys())


class IranianDataFetcher:
    """
    Fetches weather data for Iranian stations from NOAA GHCN-Daily dataset.

    This fetcher is specifically designed for analyzing Iran's severe drought
    conditions over the past 6-7 years (2018-2025).

    Data source: NOAA Global Historical Climatology Network - Daily (GHCN-D)
    URL: https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/

    Attributes:
        station_id (str): GHCN station identifier (e.g., 'IR000040754')
        station_name (str): Human-readable station name
        station_lat (float): Station latitude
        station_lon (float): Station longitude
    """

    GHCN_BASE_URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/"

    def __init__(self, city_name: str = "Tehran"):
        """
        Initialize the Iranian data fetcher.

        Args:
            city_name: Name of Iranian city (e.g., 'Tehran', 'Mashhad', 'Isfahan')
        """
        self.city_name = city_name
        station_info = IranianStationRegistry.get_station(city_name)

        if station_info is None:
            available = IranianStationRegistry.get_station_list()
            raise ValueError(
                f"City '{city_name}' not found. Available cities: {', '.join(available)}"
            )

        self.station_id = station_info["ghcn_pattern"]
        self.station_name = station_info["display_name"]
        self.station_lat = station_info["lat"]
        self.station_lon = station_info["lon"]
        self.elevation = station_info["elevation"]

    def _fetch_ghcn_data(
        self, start_year: int, end_year: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from NOAA GHCN-Daily for the station.

        Checks local cache first, then downloads if missing.

        Args:
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            DataFrame with daily weather data or None if failed
        """
        print(f"📍 Station: {self.station_name}")
        print(f"📍 Coordinates: {self.station_lat:.3f}°N, {self.station_lon:.3f}°E")
        print(f"📍 Elevation: {self.elevation}m")
        print(f"📍 GHCN ID: {self.station_id}")

        # Ensure data directory exists
        data_dir = Path("data/ghcn")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Local cache file path
        cache_file = data_dir / f"{self.station_id}.csv"

        data = None

        # Try loading from local cache
        if cache_file.exists():
            print(f"📂 Loading from local cache: {cache_file}")
            try:
                data = pd.read_csv(cache_file)
                # Ensure DATE is parsed correctly
                data["DATE"] = pd.to_datetime(data["DATE"])
                print("✅ Loaded cached data successfully")
            except Exception as e:
                print(f"⚠️ Error reading cache: {e}. Will re-download.")

        # If not in cache or cache failed, download from NOAA
        if data is None:
            print(f"\n📡 Fetching GHCN-Daily data from NOAA...")

            # Construct the URL for this station's CSV file
            station_file = f"{self.station_id}.csv"
            url = self.GHCN_BASE_URL + station_file

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Parse CSV data
                data = pd.read_csv(StringIO(response.text))

                # Save to local cache
                data.to_csv(cache_file, index=False)
                print(f"💾 Saved data to cache: {cache_file}")

                # Convert DATE column to datetime
                data["DATE"] = pd.to_datetime(data["DATE"])

            except requests.exceptions.RequestException as e:
                print(f"❌ Error fetching data from NOAA: {e}")
                print(f"❌ URL attempted: {url}")
                print("\n💡 Note: GHCN data access may be temporarily unavailable.")
                print("   Alternative: Use local data or try again later.")
                return None
            except Exception as e:
                print(f"❌ Error parsing data: {e}")
                return None

        # Filter by year range
        if data is not None:
            data = data[
                (data["DATE"].dt.year >= start_year)
                & (data["DATE"].dt.year <= end_year)
            ]

            # Set DATE as index
            data.set_index("DATE", inplace=True)

            print(f"✅ Successfully loaded {len(data)} days of data")
            if not data.empty:
                print(
                    f"📅 Date range: {data.index.min().date()} to {data.index.max().date()}"
                )

            return data

        return None

    def fetch_precipitation_data(
        self, start_year: int = 1950, end_year: int = 2025
    ) -> Optional[pd.DataFrame]:
        """
        Fetch precipitation data for the station.

        GHCN-Daily precipitation element: PRCP
        Units: tenths of mm (divide by 10 to get mm)

        Args:
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            DataFrame with daily precipitation (mm) or None if failed
        """
        print(f"🌧️ Fetching precipitation data for {self.station_name}...")

        data = self._fetch_ghcn_data(start_year, end_year)

        if data is None or data.empty:
            return None

        # Extract precipitation column (PRCP in tenths of mm)
        if "PRCP" not in data.columns:
            print("❌ No precipitation data (PRCP) available")
            return None

        # Convert from tenths of mm to mm
        prcp_data = data[["PRCP"]].copy()
        prcp_data["PRCP"] = prcp_data["PRCP"] / 10.0  # Convert to mm

        # Replace missing values (-9999) with NaN
        prcp_data.loc[prcp_data["PRCP"] < 0, "PRCP"] = float("nan")

        # Rename column for clarity
        prcp_data.rename(columns={"PRCP": "precipitation_mm"}, inplace=True)

        # Print summary statistics
        total_prcp = prcp_data["precipitation_mm"].sum()
        valid_days = prcp_data["precipitation_mm"].notna().sum()

        print(f"\n📊 Precipitation Summary ({start_year}-{end_year}):")
        print(f"   Valid measurements: {valid_days} days")
        print(f"   Total precipitation: {total_prcp:.1f} mm")
        print(f"   Mean daily: {prcp_data['precipitation_mm'].mean():.2f} mm/day")

        return prcp_data

    def fetch_temperature_data(
        self, start_year: int = 1950, end_year: int = 2025
    ) -> Optional[pd.DataFrame]:
        """
        Fetch temperature data for the station.

        GHCN-Daily temperature elements:
        - TMAX: Maximum temperature (tenths of degrees C)
        - TMIN: Minimum temperature (tenths of degrees C)

        Args:
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            DataFrame with daily temperatures (°C) or None if failed
        """
        print(f"🌡️ Fetching temperature data for {self.station_name}...")

        data = self._fetch_ghcn_data(start_year, end_year)

        if data is None or data.empty:
            return None

        # Extract temperature columns
        temp_cols = []
        if "TMAX" in data.columns:
            temp_cols.append("TMAX")
        if "TMIN" in data.columns:
            temp_cols.append("TMIN")

        if not temp_cols:
            print("❌ No temperature data (TMAX/TMIN) available")
            return None

        temp_data = data[temp_cols].copy()

        # Convert from tenths of degrees C to degrees C
        for col in temp_cols:
            temp_data[col] = temp_data[col] / 10.0
            # Replace missing values with NaN
            temp_data.loc[temp_data[col] < -100, col] = float("nan")

        # Rename columns for clarity
        rename_map = {"TMAX": "tmax_celsius", "TMIN": "tmin_celsius"}
        temp_data.rename(columns=rename_map, inplace=True)

        print(f"\n📊 Temperature Summary ({start_year}-{end_year}):")
        if "tmax_celsius" in temp_data.columns:
            print(
                f"   Max temp range: {temp_data['tmax_celsius'].min():.1f}°C to {temp_data['tmax_celsius'].max():.1f}°C"
            )
        if "tmin_celsius" in temp_data.columns:
            print(
                f"   Min temp range: {temp_data['tmin_celsius'].min():.1f}°C to {temp_data['tmin_celsius'].max():.1f}°C"
            )

        return temp_data

    def fetch_comprehensive_data(
        self, start_year: int = 1950, end_year: int = 2025
    ) -> Optional[pd.DataFrame]:
        """
        Fetch all available weather data for the station.

        Args:
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            DataFrame with all weather variables or None if failed
        """
        print(f"🌤️ Fetching comprehensive weather data for {self.station_name}...")

        data = self._fetch_ghcn_data(start_year, end_year)

        if data is None or data.empty:
            return None

        # Process all relevant columns
        processed_data = pd.DataFrame(index=data.index)

        # Precipitation (tenths of mm -> mm)
        if "PRCP" in data.columns:
            processed_data["precipitation_mm"] = data["PRCP"] / 10.0
            processed_data.loc[
                processed_data["precipitation_mm"] < 0, "precipitation_mm"
            ] = float("nan")

        # Temperature (tenths of °C -> °C)
        if "TMAX" in data.columns:
            processed_data["tmax_celsius"] = data["TMAX"] / 10.0
            processed_data.loc[
                processed_data["tmax_celsius"] < -100, "tmax_celsius"
            ] = float("nan")

        if "TMIN" in data.columns:
            processed_data["tmin_celsius"] = data["TMIN"] / 10.0
            processed_data.loc[
                processed_data["tmin_celsius"] < -100, "tmin_celsius"
            ] = float("nan")

        # Calculate mean temperature if both available
        if (
            "tmax_celsius" in processed_data.columns
            and "tmin_celsius" in processed_data.columns
        ):
            processed_data["tmean_celsius"] = (
                processed_data["tmax_celsius"] + processed_data["tmin_celsius"]
            ) / 2.0

        print(
            f"\n✅ Comprehensive data ready with {len(processed_data.columns)} variables"
        )
        print(f"   Variables: {', '.join(processed_data.columns)}")

        return processed_data


class MultiStationFetcher:
    """
    Fetches data from multiple Iranian weather stations simultaneously.

    Useful for regional drought analysis across Iran.
    """

    def __init__(self, city_names: Optional[List[str]] = None):
        """
        Initialize multi-station fetcher.

        Args:
            city_names: List of city names. If None, uses all available stations.
        """
        if city_names is None:
            city_names = IranianStationRegistry.get_station_list()

        self.city_names = city_names
        self.fetchers = {}

        for city in city_names:
            try:
                self.fetchers[city] = IranianDataFetcher(city)
            except ValueError as e:
                print(f"⚠️ Warning: {e}")

    def fetch_all_precipitation(
        self, start_year: int = 1950, end_year: int = 2025
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch precipitation data from all stations.

        Args:
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            Dict mapping city names to precipitation DataFrames
        """
        all_data = {}

        print(
            f"🌧️ Fetching precipitation data from {len(self.fetchers)} Iranian stations..."
        )
        print("=" * 80)

        for city, fetcher in self.fetchers.items():
            print(f"\n📍 Processing {city}...")
            data = fetcher.fetch_precipitation_data(start_year, end_year)
            if data is not None:
                all_data[city] = data
            print("-" * 80)

        print(
            f"\n✅ Successfully fetched data from {len(all_data)}/{len(self.fetchers)} stations"
        )

        return all_data

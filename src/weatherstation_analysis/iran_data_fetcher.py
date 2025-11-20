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

warnings.filterwarnings("ignore")


class IranianStationRegistry:
    """
    Registry of major Iranian weather stations with GHCN-Daily IDs.

    Station data based on NOAA GHCN-Daily network.
    Coordinates in decimal degrees (lat, lon).
    """

    # Major Iranian cities with weather stations
    STATIONS = {
        "Tehran (Mehrabad)": {
            "lat": 35.689,
            "lon": 51.313,
            "ghcn_pattern": "IR000040754",  # Mehrabad International Airport
            "elevation": 1191,
            "name_variants": ["Tehran", "Tehrān", "Mehrabad"]
        },
        "Mashhad": {
            "lat": 36.267,
            "lon": 59.633,
            "ghcn_pattern": "IR000040745",
            "elevation": 999,
            "name_variants": ["Mashhad", "Mashad"]
        },
        "Isfahan": {
            "lat": 32.750,
            "lon": 51.667,
            "ghcn_pattern": "IR000040800",
            "elevation": 1550,
            "name_variants": ["Isfahan", "Esfahan"]
        },
        "Tabriz": {
            "lat": 38.133,
            "lon": 46.300,
            "ghcn_pattern": "IR000040708",
            "elevation": 1361,
            "name_variants": ["Tabriz"]
        },
        "Shiraz": {
            "lat": 29.533,
            "lon": 52.600,
            "ghcn_pattern": "IR000040848",
            "elevation": 1484,
            "name_variants": ["Shiraz", "Shīrāz"]
        },
        "Ahvaz": {
            "lat": 31.333,
            "lon": 48.667,
            "ghcn_pattern": "IR000040831",
            "elevation": 23,
            "name_variants": ["Ahvaz", "Ahwaz", "Ahwāz"]
        },
        "Kerman": {
            "lat": 30.250,
            "lon": 56.967,
            "ghcn_pattern": "IR000040856",
            "elevation": 1754,
            "name_variants": ["Kerman", "Kermān"]
        },
        "Rasht": {
            "lat": 37.317,
            "lon": 49.617,
            "ghcn_pattern": "IR000040719",
            "elevation": -7,
            "name_variants": ["Rasht"]
        },
        "Zahedan": {
            "lat": 29.467,
            "lon": 60.883,
            "ghcn_pattern": "IR000040869",
            "elevation": 1370,
            "name_variants": ["Zahedan", "Zāhedān"]
        },
        "Bandar Abbas": {
            "lat": 27.217,
            "lon": 56.367,
            "ghcn_pattern": "IR000040885",
            "elevation": 10,
            "name_variants": ["Bandar Abbas", "Bandar-Abbas"]
        }
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

    def _fetch_ghcn_data(self, start_year: int, end_year: int) -> Optional[pd.DataFrame]:
        """
        Fetch data from NOAA GHCN-Daily for the station.

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
        print(f"\n📡 Fetching GHCN-Daily data from NOAA...")

        # Construct the URL for this station's CSV file
        station_file = f"{self.station_id}.csv"
        url = self.GHCN_BASE_URL + station_file

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV data
            data = pd.read_csv(StringIO(response.text))

            # Convert DATE column to datetime
            data['DATE'] = pd.to_datetime(data['DATE'])

            # Filter by year range
            data = data[
                (data['DATE'].dt.year >= start_year) &
                (data['DATE'].dt.year <= end_year)
            ]

            # Set DATE as index
            data.set_index('DATE', inplace=True)

            print(f"✅ Successfully downloaded {len(data)} days of data")
            print(f"📅 Date range: {data.index.min().date()} to {data.index.max().date()}")

            return data

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data from NOAA: {e}")
            print(f"❌ URL attempted: {url}")
            print("\n💡 Note: GHCN data access may be temporarily unavailable.")
            print("   Alternative: Use local data or try again later.")
            return None
        except Exception as e:
            print(f"❌ Error parsing data: {e}")
            return None

    def fetch_precipitation_data(
        self,
        start_year: int = 1950,
        end_year: int = 2025
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
        if 'PRCP' not in data.columns:
            print("❌ No precipitation data (PRCP) available")
            return None

        # Convert from tenths of mm to mm
        prcp_data = data[['PRCP']].copy()
        prcp_data['PRCP'] = prcp_data['PRCP'] / 10.0  # Convert to mm

        # Replace missing values (-9999) with NaN
        prcp_data.loc[prcp_data['PRCP'] < 0, 'PRCP'] = float('nan')

        # Rename column for clarity
        prcp_data.rename(columns={'PRCP': 'precipitation_mm'}, inplace=True)

        # Print summary statistics
        total_prcp = prcp_data['precipitation_mm'].sum()
        valid_days = prcp_data['precipitation_mm'].notna().sum()

        print(f"\n📊 Precipitation Summary ({start_year}-{end_year}):")
        print(f"   Valid measurements: {valid_days} days")
        print(f"   Total precipitation: {total_prcp:.1f} mm")
        print(f"   Mean daily: {prcp_data['precipitation_mm'].mean():.2f} mm/day")

        return prcp_data

    def fetch_temperature_data(
        self,
        start_year: int = 1950,
        end_year: int = 2025
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
        if 'TMAX' in data.columns:
            temp_cols.append('TMAX')
        if 'TMIN' in data.columns:
            temp_cols.append('TMIN')

        if not temp_cols:
            print("❌ No temperature data (TMAX/TMIN) available")
            return None

        temp_data = data[temp_cols].copy()

        # Convert from tenths of degrees C to degrees C
        for col in temp_cols:
            temp_data[col] = temp_data[col] / 10.0
            # Replace missing values with NaN
            temp_data.loc[temp_data[col] < -100, col] = float('nan')

        # Rename columns for clarity
        rename_map = {'TMAX': 'tmax_celsius', 'TMIN': 'tmin_celsius'}
        temp_data.rename(columns=rename_map, inplace=True)

        print(f"\n📊 Temperature Summary ({start_year}-{end_year}):")
        if 'tmax_celsius' in temp_data.columns:
            print(f"   Max temp range: {temp_data['tmax_celsius'].min():.1f}°C to {temp_data['tmax_celsius'].max():.1f}°C")
        if 'tmin_celsius' in temp_data.columns:
            print(f"   Min temp range: {temp_data['tmin_celsius'].min():.1f}°C to {temp_data['tmin_celsius'].max():.1f}°C")

        return temp_data

    def fetch_comprehensive_data(
        self,
        start_year: int = 1950,
        end_year: int = 2025
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
        if 'PRCP' in data.columns:
            processed_data['precipitation_mm'] = data['PRCP'] / 10.0
            processed_data.loc[processed_data['precipitation_mm'] < 0, 'precipitation_mm'] = float('nan')

        # Temperature (tenths of °C -> °C)
        if 'TMAX' in data.columns:
            processed_data['tmax_celsius'] = data['TMAX'] / 10.0
            processed_data.loc[processed_data['tmax_celsius'] < -100, 'tmax_celsius'] = float('nan')

        if 'TMIN' in data.columns:
            processed_data['tmin_celsius'] = data['TMIN'] / 10.0
            processed_data.loc[processed_data['tmin_celsius'] < -100, 'tmin_celsius'] = float('nan')

        # Calculate mean temperature if both available
        if 'tmax_celsius' in processed_data.columns and 'tmin_celsius' in processed_data.columns:
            processed_data['tmean_celsius'] = (
                processed_data['tmax_celsius'] + processed_data['tmin_celsius']
            ) / 2.0

        print(f"\n✅ Comprehensive data ready with {len(processed_data.columns)} variables")
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
        self,
        start_year: int = 1950,
        end_year: int = 2025
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

        print(f"🌧️ Fetching precipitation data from {len(self.fetchers)} Iranian stations...")
        print("=" * 80)

        for city, fetcher in self.fetchers.items():
            print(f"\n📍 Processing {city}...")
            data = fetcher.fetch_precipitation_data(start_year, end_year)
            if data is not None:
                all_data[city] = data
            print("-" * 80)

        print(f"\n✅ Successfully fetched data from {len(all_data)}/{len(self.fetchers)} stations")

        return all_data

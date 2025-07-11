"""
Weather Data Fetcher Module
===========================

Generic weather data fetcher that can work with any city in Germany.
"""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from meteostat import Daily
import warnings
from .city_manager import CityManager

warnings.filterwarnings("ignore")


class WeatherDataFetcher:
    """
    Generic weather data fetcher for German cities.

    This class can fetch weather data for any German city by finding
    the nearest weather stations and retrieving historical data.
    """

    def __init__(self, city_name: str = "Potsdam", min_coverage: float = 80.0):
        """
        Initialize the weather data fetcher.

        Args:
            city_name: Name of the city (default: "Potsdam")
            min_coverage: Minimum data coverage percentage (default: 80.0)
        """
        self.city_name = city_name
        self.min_coverage = min_coverage
        self.city_manager = CityManager()

        # Try to resolve city name
        self.city_info = self.city_manager.find_city_match(city_name)
        if not self.city_info:
            suggestions = self.city_manager.get_suggestions(city_name)
            if suggestions:
                print(
                    f"❌ City '{city_name}' not found. "
                    f"Did you mean: {', '.join(suggestions)}?"
                )
            else:
                print(f"❌ City '{city_name}' not found. Available cities:")
                cities = self.city_manager.list_available_cities()
                print(f"   {', '.join(cities[:10])}...")
            raise ValueError(f"City '{city_name}' not found")

        # Find stations near the city
        self.stations_df = self.city_manager.find_stations_near_city(city_name)
        if self.stations_df is None or self.stations_df.empty:
            raise ValueError(f"No weather stations found near {city_name}")

        # Use the closest station with good data
        self.station_id = self.stations_df.index[0]
        self.station_name = self.stations_df.loc[self.station_id, "name"]
        self.station_lat = self.stations_df.loc[self.station_id, "latitude"]
        self.station_lon = self.stations_df.loc[self.station_id, "longitude"]
        self.distance_km = self.stations_df.loc[self.station_id, "distance_km"]

        print(f"📍 Using station: {self.station_name}")
        print(f"📍 Distance from {self.city_info['name']}: {self.distance_km:.1f} km")
        print(f"📍 Coordinates: {self.station_lat:.4f}°N, {self.station_lon:.4f}°E")

    def _calculate_expected_days(self, year: int) -> int:
        """Calculate expected days for a given year."""
        if year == 2025:
            return 183  # Up to July 2nd

        # Check if leap year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        return 366 if is_leap else 365

    def _validate_data_coverage(
        self, data: pd.DataFrame, year: int, required_columns: List[str]
    ) -> bool:
        """Validate data coverage meets minimum requirements."""
        if data.empty:
            return False

        expected_days = self._calculate_expected_days(year)

        # Check coverage for required columns
        for col in required_columns:
            if col not in data.columns:
                print(f"❌ {year}: Missing column '{col}'")
                return False

            actual_days = len(data[col].dropna())
            coverage_percentage = (actual_days / expected_days) * 100

            if year != 2025 and coverage_percentage < self.min_coverage:
                print(
                    f"❌ {year}: Insufficient {col} coverage "
                    f"({coverage_percentage:.1f}%)"
                )
                return False

        # If we get here, coverage is acceptable
        actual_days = len(data.dropna())
        coverage_percentage = (actual_days / expected_days) * 100

        # 2025 is always included regardless of coverage
        if year == 2025 or coverage_percentage >= self.min_coverage:
            status = "(partial year)" if year == 2025 else ""
            print(
                f"✅ {year}: {actual_days}/{expected_days} days "
                f"({coverage_percentage:.1f}%) {status}"
            )
            return True
        else:
            print(f"❌ {year}: Insufficient coverage ({coverage_percentage:.1f}%)")
            return False

    def fetch_comprehensive_data(
        self, start_year: int = 1890, end_year: int = 2026
    ) -> Optional[Dict[int, pd.DataFrame]]:
        """
        Fetch comprehensive weather data for the city.

        Args:
            start_year: Starting year for data collection
            end_year: Ending year for data collection

        Returns:
            Dictionary mapping years to DataFrames with weather data
        """
        print(f"🌡️ Fetching comprehensive weather data for {self.city_info['name']}...")
        print(f"📊 Station: {self.station_name}")

        all_data = {}
        years_to_analyze = list(range(start_year, end_year))

        for year in years_to_analyze:
            print(f"📡 Downloading data for {year}...")

            start_date = datetime(year, 1, 1)
            if year == 2025:
                end_date = datetime(2025, 7, 2)
            else:
                end_date = datetime(year, 12, 31)

            try:
                data = Daily(self.station_id, start_date, end_date)
                data = data.fetch()

                if not data.empty:
                    # Check if we have the required columns
                    required_columns = ["prcp", "tmax", "tmin"]
                    available_columns = [
                        col for col in required_columns if col in data.columns
                    ]

                    if available_columns and self._validate_data_coverage(
                        data, year, available_columns
                    ):
                        # Fill missing precipitation with 0, leave temperature as NaN
                        if "prcp" in data.columns:
                            data["prcp"] = data["prcp"].fillna(0.0)

                        all_data[year] = data[available_columns].copy()
                else:
                    print(f"❌ {year}: No data available")

            except Exception as e:
                print(f"❌ {year}: Error - {e}")

        if not all_data:
            print("❌ No data could be retrieved")
            return None

        print(f"\n✅ Successfully retrieved data for {len(all_data)} years")
        return all_data

    def fetch_temperature_data(
        self, start_year: int = 1890, end_year: int = 2026
    ) -> Optional[Dict[int, pd.DataFrame]]:
        """Fetch temperature-focused data."""
        print(f"🌡️ Fetching temperature data for {self.city_info['name']}...")

        all_data = {}
        years_to_analyze = list(range(start_year, end_year))

        for year in years_to_analyze:
            print(f"📡 Downloading temperature data for {year}...")

            start_date = datetime(year, 1, 1)
            if year == 2025:
                end_date = datetime(2025, 7, 2)
            else:
                end_date = datetime(year, 12, 31)

            try:
                data = Daily(self.station_id, start_date, end_date)
                data = data.fetch()

                if not data.empty and {"tmax", "tmin"}.intersection(data.columns):
                    expected_days = self._calculate_expected_days(year)

                    # Check temperature data coverage
                    temp_columns = [
                        col for col in ["tmax", "tmin"] if col in data.columns
                    ]
                    non_null_data = data[temp_columns].dropna()
                    actual_days = len(non_null_data)
                    coverage_percentage = (actual_days / expected_days) * 100

                    if year == 2025 or coverage_percentage >= self.min_coverage:
                        all_data[year] = data[["tmax", "tmin"]].copy()
                        max_temp = data["tmax"].max()
                        status = "(partial year)" if year == 2025 else ""
                        print(
                            f"✅ {year}: {actual_days}/{expected_days} days, "
                            f"max: {max_temp:.1f}°C ({coverage_percentage:.1f}%) "
                            f"{status}"
                        )
                    else:
                        print(
                            f"❌ {year}: Insufficient coverage "
                            f"({coverage_percentage:.1f}%)"
                        )
                else:
                    print(f"❌ {year}: No temperature data available")

            except Exception as e:
                print(f"❌ {year}: Error - {e}")

        if not all_data:
            print("❌ No temperature data could be retrieved")
            return None

        print(f"\n✅ Successfully retrieved temperature data for {len(all_data)} years")
        return all_data

    def fetch_precipitation_data(
        self, start_year: int = 1890, end_year: int = 2026
    ) -> Optional[Dict[int, pd.DataFrame]]:
        """Fetch precipitation-focused data."""
        print(f"🌧️ Fetching precipitation data for {self.city_info['name']}...")

        all_data = {}
        years_to_analyze = list(range(start_year, end_year))

        for year in years_to_analyze:
            print(f"📡 Downloading precipitation data for {year}...")

            start_date = datetime(year, 1, 1)
            if year == 2025:
                end_date = datetime(2025, 7, 2)
            else:
                end_date = datetime(year, 12, 31)

            try:
                data = Daily(self.station_id, start_date, end_date)
                data = data.fetch()

                if not data.empty and "prcp" in data.columns:
                    expected_days = self._calculate_expected_days(year)

                    non_null_data = data["prcp"].dropna()
                    actual_days = len(non_null_data)
                    coverage_percentage = (actual_days / expected_days) * 100

                    if year == 2025 or coverage_percentage >= self.min_coverage:
                        # Fill missing precipitation with 0
                        daily_prcp = data["prcp"].fillna(0.0)
                        all_data[year] = daily_prcp
                        total = daily_prcp.sum()
                        status = "(partial year)" if year == 2025 else ""
                        print(
                            f"✅ {year}: {actual_days}/{expected_days} days, "
                            f"total: {total:.1f}mm ({coverage_percentage:.1f}%) "
                            f"{status}"
                        )
                    else:
                        print(
                            f"❌ {year}: Insufficient coverage "
                            f"({coverage_percentage:.1f}%)"
                        )
                else:
                    print(f"❌ {year}: No precipitation data available")

            except Exception as e:
                print(f"❌ {year}: Error - {e}")

        if not all_data:
            print("❌ No precipitation data could be retrieved")
            return None

        print(
            f"\n✅ Successfully retrieved precipitation data for {len(all_data)} years"
        )
        return all_data

    def get_station_info(self) -> Dict:
        """Get information about the weather station being used."""
        return {
            "city": self.city_info["name"],
            "station_id": self.station_id,
            "station_name": self.station_name,
            "latitude": self.station_lat,
            "longitude": self.station_lon,
            "distance_km": self.distance_km,
        }


# Backward compatibility alias
class PotsdamDataFetcher(WeatherDataFetcher):
    """Backward compatibility class for Potsdam-specific data fetching."""

    def __init__(
        self,
        station_lat: float = 52.3833,
        station_lon: float = 13.0667,
        min_coverage: float = 80.0,
    ):
        """Initialize with Potsdam coordinates (for backward compatibility)."""
        super().__init__(city_name="Potsdam", min_coverage=min_coverage)

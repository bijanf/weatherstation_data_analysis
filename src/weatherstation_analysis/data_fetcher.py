"""
Data Fetcher Module
===================

Handles fetching weather data from various sources, primarily Meteostat API.
"""

from typing import Dict, Optional, Union, List
import pandas as pd
import numpy as np
from datetime import datetime, date
from meteostat import Stations, Daily
import warnings

warnings.filterwarnings("ignore")


class PotsdamDataFetcher:
    """
    Fetches weather data for Potsdam Säkularstation from Meteostat API.

    This class provides methods to retrieve comprehensive weather data
    including precipitation, temperature, and other meteorological variables.

    Attributes:
        station_lat (float): Latitude of Potsdam station
        station_lon (float): Longitude of Potsdam station
        min_coverage (float): Minimum data coverage percentage required
        station_id (str): Meteostat station identifier
        station_name (str): Station name
    """

    def __init__(
        self,
        station_lat: float = 52.3833,
        station_lon: float = 13.0667,
        min_coverage: float = 80.0,
    ):
        """
        Initialize the data fetcher.

        Args:
            station_lat: Latitude of the weather station
            station_lon: Longitude of the weather station
            min_coverage: Minimum data coverage percentage (0-100)
        """
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.min_coverage = min_coverage
        self.station_id: Optional[str] = None
        self.station_name: Optional[str] = None

    def _get_station_info(self) -> bool:
        """
        Retrieve station information from Meteostat.

        Returns:
            bool: True if station found, False otherwise
        """
        try:
            stations = Stations()
            stations = stations.nearby(self.station_lat, self.station_lon)
            station = stations.fetch(1)

            if station.empty:
                print("❌ No station found")
                return False

            self.station_id = station.index[0]
            self.station_name = station.loc[self.station_id, "name"]

            print(f"📍 Station: {self.station_name}")
            print(f"📍 Coordinates: {self.station_lat:.4f}°N, {self.station_lon:.4f}°E")

            return True

        except Exception as e:
            print(f"❌ Error getting station info: {e}")
            return False

    def _calculate_expected_days(self, year: int) -> int:
        """
        Calculate expected number of days for a given year.

        Args:
            year: Year to calculate days for

        Returns:
            int: Expected number of days (365 or 366)
        """
        if year == 2025:
            return 183  # Days from Jan 1 to July 2

        # Check if leap year
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 366 if is_leap else 365

    def _validate_data_coverage(
        self, data: pd.DataFrame, year: int, required_columns: List[str]
    ) -> bool:
        """
        Validate data coverage meets minimum requirements.

        Args:
            data: DataFrame with weather data
            year: Year being validated
            required_columns: List of required column names

        Returns:
            bool: True if coverage is sufficient
        """
        if data.empty:
            return False

        # Check if all required columns exist
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            print(f"❌ {year}: Missing columns: {missing_cols}")
            return False

        # Calculate coverage
        expected_days = self._calculate_expected_days(year)
        actual_days = len(data.dropna(subset=required_columns))
        coverage_percentage = (actual_days / expected_days) * 100

        # 2025 is always included regardless of coverage
        if year == 2025 or coverage_percentage >= self.min_coverage:
            status = "(partial year)" if year == 2025 else ""
            print(
                f"✅ {year}: {actual_days}/{expected_days} days ({coverage_percentage:.1f}%) {status}"
            )
            return True
        else:
            print(f"❌ {year}: Insufficient coverage ({coverage_percentage:.1f}%)")
            return False

    def fetch_comprehensive_data(
        self, start_year: int = 1890, end_year: int = 2026
    ) -> Optional[Dict[int, pd.DataFrame]]:
        """
        Fetch comprehensive weather data for specified year range.

        Args:
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            Dict mapping years to DataFrames with weather data, or None if failed
        """
        print("🌡️ Fetching comprehensive weather data for Potsdam...")

        if not self._get_station_info():
            return None

        years_to_analyze = list(range(start_year, end_year))
        all_data = {}
        required_columns = ["prcp", "tmax", "tmin"]

        for year in years_to_analyze:
            print(f"📡 Downloading data for {year}...")

            start_date = datetime(year, 1, 1)
            end_date = datetime(2025, 7, 2) if year == 2025 else datetime(year, 12, 31)

            try:
                data = Daily(self.station_id, start_date, end_date).fetch()

                if self._validate_data_coverage(data, year, required_columns):
                    all_data[year] = data[required_columns].copy()

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
        """
        Fetch temperature data specifically.

        Args:
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            Dict mapping years to DataFrames with temperature data, or None if failed
        """
        print("🌡️ Fetching temperature data for Potsdam...")

        if not self._get_station_info():
            return None

        years_to_analyze = list(range(start_year, end_year))
        all_data = {}

        for year in years_to_analyze:
            print(f"📡 Downloading temperature data for {year}...")

            start_date = datetime(year, 1, 1)
            end_date = datetime(2025, 7, 2) if year == 2025 else datetime(year, 12, 31)

            try:
                data = Daily(self.station_id, start_date, end_date).fetch()

                if not data.empty and "tmax" in data.columns:
                    expected_days = self._calculate_expected_days(year)
                    actual_days = len(data["tmax"].dropna())
                    coverage_percentage = (actual_days / expected_days) * 100

                    if year == 2025 or coverage_percentage >= self.min_coverage:
                        all_data[year] = data[["tmax", "tmin"]].copy()
                        max_temp = data["tmax"].max()
                        status = "(partial year)" if year == 2025 else ""
                        print(
                            f"✅ {year}: {actual_days}/{expected_days} days, max: {max_temp:.1f}°C ({coverage_percentage:.1f}%) {status}"
                        )
                    else:
                        print(
                            f"❌ {year}: Insufficient coverage ({coverage_percentage:.1f}%)"
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
        """
        Fetch precipitation data specifically.

        Args:
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            Dict mapping years to DataFrames with precipitation data, or None if failed
        """
        print("🌧️ Fetching precipitation data for Potsdam...")

        if not self._get_station_info():
            return None

        years_to_analyze = list(range(start_year, end_year))
        all_data = {}

        for year in years_to_analyze:
            print(f"📡 Downloading precipitation data for {year}...")

            start_date = datetime(year, 1, 1)
            end_date = datetime(2025, 7, 2) if year == 2025 else datetime(year, 12, 31)

            try:
                data = Daily(self.station_id, start_date, end_date).fetch()

                if not data.empty and "prcp" in data.columns:
                    expected_days = self._calculate_expected_days(year)
                    actual_days = len(data["prcp"].dropna())
                    coverage_percentage = (actual_days / expected_days) * 100

                    if year == 2025 or coverage_percentage >= self.min_coverage:
                        daily_prcp = data["prcp"].fillna(0.0)
                        all_data[year] = daily_prcp
                        total = daily_prcp.sum()
                        status = "(partial year)" if year == 2025 else ""
                        print(
                            f"✅ {year}: {actual_days}/{expected_days} days, total: {total:.1f}mm ({coverage_percentage:.1f}%) {status}"
                        )
                    else:
                        print(
                            f"❌ {year}: Insufficient coverage ({coverage_percentage:.1f}%)"
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

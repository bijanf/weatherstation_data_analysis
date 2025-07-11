"""
Tests for data fetcher module.
"""

import pandas as pd
from unittest.mock import Mock, patch
from src.weatherstation_analysis.data_fetcher import PotsdamDataFetcher


class TestPotsdamDataFetcher:
    """Test cases for PotsdamDataFetcher class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        fetcher = PotsdamDataFetcher()
        assert fetcher.station_lat == 52.3833
        assert fetcher.station_lon == 13.0667
        assert fetcher.min_coverage == 80.0
        assert fetcher.station_id is None
        assert fetcher.station_name is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        fetcher = PotsdamDataFetcher(
            station_lat=50.0, station_lon=10.0, min_coverage=90.0
        )
        assert fetcher.station_lat == 50.0
        assert fetcher.station_lon == 10.0
        assert fetcher.min_coverage == 90.0

    def test_calculate_expected_days_regular_year(self):
        """Test expected days calculation for regular year."""
        fetcher = PotsdamDataFetcher()
        assert fetcher._calculate_expected_days(2023) == 365

    def test_calculate_expected_days_leap_year(self):
        """Test expected days calculation for leap year."""
        fetcher = PotsdamDataFetcher()
        assert fetcher._calculate_expected_days(2024) == 366

    def test_calculate_expected_days_current_year(self):
        """Test expected days calculation for current year (2025)."""
        fetcher = PotsdamDataFetcher()
        assert fetcher._calculate_expected_days(2025) == 183

    def test_validate_data_coverage_sufficient(self):
        """Test data coverage validation with sufficient data."""
        fetcher = PotsdamDataFetcher()

        # Create mock data with sufficient coverage
        mock_data = pd.DataFrame(
            {
                "prcp": [1.0] * 300,  # 300 days of data
                "tmax": [20.0] * 300,
                "tmin": [10.0] * 300,
            }
        )

        result = fetcher._validate_data_coverage(
            mock_data, 2023, ["prcp", "tmax", "tmin"]
        )
        assert result is True

    def test_validate_data_coverage_insufficient(self):
        """Test data coverage validation with insufficient data."""
        fetcher = PotsdamDataFetcher()

        # Create mock data with insufficient coverage
        mock_data = pd.DataFrame(
            {
                "prcp": [1.0] * 100,  # Only 100 days of data
                "tmax": [20.0] * 100,
                "tmin": [10.0] * 100,
            }
        )

        result = fetcher._validate_data_coverage(
            mock_data, 2023, ["prcp", "tmax", "tmin"]
        )
        assert result is False

    def test_validate_data_coverage_missing_columns(self):
        """Test data coverage validation with missing columns."""
        fetcher = PotsdamDataFetcher()

        # Create mock data missing required columns
        mock_data = pd.DataFrame(
            {
                "prcp": [1.0] * 300,
                "tmax": [20.0] * 300,
                # Missing 'tmin' column
            }
        )

        result = fetcher._validate_data_coverage(
            mock_data, 2023, ["prcp", "tmax", "tmin"]
        )
        assert result is False

    def test_validate_data_coverage_empty_data(self):
        """Test data coverage validation with empty data."""
        fetcher = PotsdamDataFetcher()

        mock_data = pd.DataFrame()

        result = fetcher._validate_data_coverage(
            mock_data, 2023, ["prcp", "tmax", "tmin"]
        )
        assert result is False

    @patch("src.weatherstation_analysis.data_fetcher.Stations")
    def test_get_station_info_success(self, mock_stations):
        """Test successful station info retrieval."""
        # Mock the stations response
        mock_station_data = pd.DataFrame(
            {"name": ["Test Station"], "latitude": [52.3833], "longitude": [13.0667]},
            index=["TEST_ID"],
        )

        mock_stations_instance = Mock()
        mock_stations_instance.nearby.return_value = mock_stations_instance
        mock_stations_instance.fetch.return_value = mock_station_data
        mock_stations.return_value = mock_stations_instance

        fetcher = PotsdamDataFetcher()
        result = fetcher._get_station_info()

        assert result is True
        assert fetcher.station_id == "TEST_ID"
        assert fetcher.station_name == "Test Station"

    @patch("src.weatherstation_analysis.data_fetcher.Stations")
    def test_get_station_info_no_station_found(self, mock_stations):
        """Test station info retrieval when no station found."""
        # Mock empty response
        mock_stations_instance = Mock()
        mock_stations_instance.nearby.return_value = mock_stations_instance
        mock_stations_instance.fetch.return_value = pd.DataFrame()
        mock_stations.return_value = mock_stations_instance

        fetcher = PotsdamDataFetcher()
        result = fetcher._get_station_info()

        assert result is False
        assert fetcher.station_id is None
        assert fetcher.station_name is None

    @patch("src.weatherstation_analysis.data_fetcher.Stations")
    def test_get_station_info_exception(self, mock_stations):
        """Test station info retrieval with exception."""
        # Mock exception
        mock_stations.side_effect = Exception("API Error")

        fetcher = PotsdamDataFetcher()
        result = fetcher._get_station_info()

        assert result is False

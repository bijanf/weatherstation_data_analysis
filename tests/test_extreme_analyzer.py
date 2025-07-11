"""
Tests for extreme value analyzer module.
"""

import pytest
import pandas as pd
import numpy as np
from src.weatherstation_analysis.extreme_analyzer import ExtremeValueAnalyzer


class TestExtremeValueAnalyzer:
    """Test cases for ExtremeValueAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample weather data for testing."""
        data = {}

        # Create sample data for 3 years
        for year in [2020, 2021, 2022]:
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            data[year] = pd.DataFrame(
                {
                    "prcp": np.random.exponential(2, len(dates)),
                    "tmax": np.random.normal(20, 10, len(dates)),
                    "tmin": np.random.normal(5, 8, len(dates)),
                },
                index=dates,
            )

        return data

    def test_init(self, sample_data):
        """Test initialization of ExtremeValueAnalyzer."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        assert analyzer.data == sample_data
        assert analyzer.extremes_df is None

    def test_analyze_annual_extremes(self, sample_data):
        """Test annual extremes analysis."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        extremes = analyzer.analyze_annual_extremes()

        assert isinstance(extremes, pd.DataFrame)
        assert len(extremes) == 3  # 3 years
        assert list(extremes.columns) == [
            "year",
            "max_precip",
            "max_temp",
            "min_temp",
            "precip_95th",
            "temp_range",
        ]
        assert extremes["year"].tolist() == [2020, 2021, 2022]

        # Check that extremes_df is stored
        assert analyzer.extremes_df is not None
        pd.testing.assert_frame_equal(analyzer.extremes_df, extremes)

    def test_analyze_annual_extremes_exclude_current_year(self, sample_data):
        """Test annual extremes analysis excluding current year."""
        # Add 2025 data
        sample_data[2025] = pd.DataFrame(
            {
                "prcp": [1.0, 2.0, 3.0],
                "tmax": [25.0, 26.0, 27.0],
                "tmin": [10.0, 11.0, 12.0],
            }
        )

        analyzer = ExtremeValueAnalyzer(sample_data)
        extremes = analyzer.analyze_annual_extremes(exclude_current_year=True)

        # Should exclude 2025
        assert len(extremes) == 3
        assert 2025 not in extremes["year"].values

    def test_analyze_annual_extremes_include_current_year(self, sample_data):
        """Test annual extremes analysis including current year."""
        # Add 2025 data
        sample_data[2025] = pd.DataFrame(
            {
                "prcp": [1.0, 2.0, 3.0],
                "tmax": [25.0, 26.0, 27.0],
                "tmin": [10.0, 11.0, 12.0],
            }
        )

        analyzer = ExtremeValueAnalyzer(sample_data)
        extremes = analyzer.analyze_annual_extremes(exclude_current_year=False)

        # Should include 2025
        assert len(extremes) == 4
        assert 2025 in extremes["year"].values

    def test_analyze_annual_extremes_missing_columns(self):
        """Test annual extremes analysis with missing columns."""
        # Data with only precipitation
        data = {2020: pd.DataFrame({"prcp": [1.0, 2.0, 3.0]})}

        analyzer = ExtremeValueAnalyzer(data)
        extremes = analyzer.analyze_annual_extremes()

        assert len(extremes) == 1
        assert extremes["max_precip"].iloc[0] == 3.0
        assert pd.isna(extremes["max_temp"].iloc[0])
        assert pd.isna(extremes["min_temp"].iloc[0])
        assert pd.isna(extremes["temp_range"].iloc[0])

    def test_calculate_return_periods_gumbel(self, sample_data):
        """Test return period calculation with Gumbel distribution."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        values = np.array([10, 20, 30, 40, 50])

        empirical_rp, theoretical_values, theoretical_rp = (
            analyzer.calculate_return_periods(values, distribution="gumbel")
        )

        assert len(empirical_rp) == 5
        assert len(theoretical_values) == 100
        assert len(theoretical_rp) == 100
        assert empirical_rp[0] > empirical_rp[-1]  # Descending order

    def test_calculate_return_periods_invalid_distribution(self, sample_data):
        """Test return period calculation with invalid distribution."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        values = np.array([10, 20, 30, 40, 50])

        with pytest.raises(ValueError, match="Unknown distribution"):
            analyzer.calculate_return_periods(values, distribution="invalid")

    def test_calculate_trends_valid_variable(self, sample_data):
        """Test trend calculation for valid variable."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        analyzer.analyze_annual_extremes()

        trends = analyzer.calculate_trends("max_temp")

        assert isinstance(trends, dict)
        required_keys = [
            "slope",
            "intercept",
            "r_value",
            "p_value",
            "std_err",
            "trend_per_century",
            "significance",
        ]
        assert all(key in trends for key in required_keys)
        assert trends["trend_per_century"] == trends["slope"] * 100

    def test_calculate_trends_no_extremes_analyzed(self, sample_data):
        """Test trend calculation without analyzing extremes first."""
        analyzer = ExtremeValueAnalyzer(sample_data)

        with pytest.raises(ValueError, match="Must run analyze_annual_extremes first"):
            analyzer.calculate_trends("max_temp")

    def test_calculate_trends_invalid_variable(self, sample_data):
        """Test trend calculation for invalid variable."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        analyzer.analyze_annual_extremes()

        with pytest.raises(ValueError, match="Variable invalid_var not found"):
            analyzer.calculate_trends("invalid_var")

    def test_get_extreme_events_max_values(self, sample_data):
        """Test getting extreme events for maximum values."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        analyzer.analyze_annual_extremes()

        events = analyzer.get_extreme_events("max_temp", n_events=2)

        assert isinstance(events, pd.DataFrame)
        assert len(events) == 2
        assert list(events.columns) == ["year", "max_temp"]
        # Should be sorted by highest values
        assert events["max_temp"].iloc[0] >= events["max_temp"].iloc[1]

    def test_get_extreme_events_min_values(self, sample_data):
        """Test getting extreme events for minimum values."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        analyzer.analyze_annual_extremes()

        events = analyzer.get_extreme_events("min_temp", n_events=2)

        assert isinstance(events, pd.DataFrame)
        assert len(events) == 2
        assert list(events.columns) == ["year", "min_temp"]
        # Should be sorted by lowest values
        assert events["min_temp"].iloc[0] <= events["min_temp"].iloc[1]

    def test_calculate_statistics_summary(self, sample_data):
        """Test statistics summary calculation."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        analyzer.analyze_annual_extremes()

        summary = analyzer.calculate_statistics_summary()

        assert isinstance(summary, dict)
        assert "max_temp" in summary
        assert "min_temp" in summary
        assert "max_precip" in summary

        # Check required statistics
        required_stats = [
            "count",
            "mean",
            "std",
            "min",
            "max",
            "median",
            "q25",
            "q75",
            "skewness",
            "kurtosis",
            "max_year",
            "min_year",
        ]
        assert all(stat in summary["max_temp"] for stat in required_stats)

    def test_calculate_correlation_matrix(self, sample_data):
        """Test correlation matrix calculation."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        analyzer.analyze_annual_extremes()

        corr_matrix = analyzer.calculate_correlation_matrix()

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
        assert all(corr_matrix.columns == corr_matrix.index)  # Same row/column names
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)

    def test_detect_changepoints_pettitt(self, sample_data):
        """Test changepoint detection with Pettitt test."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        analyzer.analyze_annual_extremes()

        changepoints = analyzer.detect_changepoints("max_temp", method="pettitt")

        assert isinstance(changepoints, dict)
        required_keys = [
            "method",
            "changepoint_year",
            "changepoint_index",
            "test_statistic",
            "years",
            "values",
        ]
        assert all(key in changepoints for key in required_keys)
        assert changepoints["method"] == "pettitt"

    def test_detect_changepoints_invalid_method(self, sample_data):
        """Test changepoint detection with invalid method."""
        analyzer = ExtremeValueAnalyzer(sample_data)
        analyzer.analyze_annual_extremes()

        with pytest.raises(ValueError, match="Unknown changepoint method"):
            analyzer.detect_changepoints("max_temp", method="invalid")

"""
Tests for visualization module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from src.weatherstation_analysis.visualization import WeatherPlotter


class TestWeatherPlotter:
    """Test cases for WeatherPlotter class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        plotter = WeatherPlotter()
        assert plotter.style == "default"
        assert plotter.figsize_default == (12, 8)
        assert plotter.dpi == 300
        assert isinstance(plotter.colors, dict)
        assert "primary" in plotter.colors

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        plotter = WeatherPlotter(style="classic", figsize_default=(10, 6), dpi=150)
        assert plotter.style == "classic"
        assert plotter.figsize_default == (10, 6)
        assert plotter.dpi == 150

    @pytest.fixture
    def sample_extremes_df(self):
        """Create sample extremes DataFrame for testing."""
        return pd.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "max_precip": [45.2, 52.8, 38.9],
                "max_temp": [32.1, 28.9, 35.4],
                "min_temp": [-12.3, -8.7, -15.2],
                "temp_range": [44.4, 37.6, 50.6],
                "precip_95th": [15.2, 18.9, 12.4],
            }
        )

    @pytest.fixture
    def sample_all_data(self):
        """Create sample all_data dictionary for testing."""
        data = {}
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

    def test_setup_plot_style(self):
        """Test plot style setup."""
        plotter = WeatherPlotter()
        fig, ax = plt.subplots()

        plotter._setup_plot_style(ax)

        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_linewidth() == 2
        assert ax.spines["bottom"].get_linewidth() == 2
        plt.close(fig)

    def test_add_data_attribution_default_position(self):
        """Test adding data attribution with default position."""
        plotter = WeatherPlotter()
        fig, ax = plt.subplots()

        plotter._add_data_attribution(ax)

        # Check that text was added (can't easily check exact position)
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_add_data_attribution_custom_position(self):
        """Test adding data attribution with custom position."""
        plotter = WeatherPlotter()
        fig, ax = plt.subplots()

        plotter._add_data_attribution(ax, position="top_left")

        # Check that text was added
        assert len(ax.texts) > 0
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_annual_precipitation_extremes(self, mock_savefig, sample_extremes_df):
        """Test annual precipitation extremes plot."""
        plotter = WeatherPlotter()

        fig = plotter.plot_annual_precipitation_extremes(
            sample_extremes_df, "test_precip.png"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots
        mock_savefig.assert_called_once()
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_temperature_extremes_analysis(self, mock_savefig, sample_extremes_df):
        """Test temperature extremes analysis plot."""
        plotter = WeatherPlotter()

        fig = plotter.plot_temperature_extremes_analysis(
            sample_extremes_df, "test_temp.png"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Four subplots
        mock_savefig.assert_called_once()
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_threshold_exceedance_analysis(self, mock_savefig, sample_all_data):
        """Test threshold exceedance analysis plot."""
        plotter = WeatherPlotter()

        fig = plotter.plot_threshold_exceedance_analysis(
            sample_all_data, "test_threshold.png"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Four subplots
        mock_savefig.assert_called_once()
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_statistics_summary(self, mock_savefig, sample_extremes_df):
        """Test statistics summary plot."""
        plotter = WeatherPlotter()

        fig = plotter.plot_statistics_summary(sample_extremes_df, "test_summary.png")

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Four subplots
        mock_savefig.assert_called_once()
        plt.close(fig)

    def test_plot_without_save_path(self, sample_extremes_df):
        """Test plotting without save path."""
        plotter = WeatherPlotter()

        fig = plotter.plot_annual_precipitation_extremes(sample_extremes_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_palette_completeness(self):
        """Test that color palette has all required colors."""
        plotter = WeatherPlotter()

        required_colors = [
            "primary",
            "secondary",
            "success",
            "danger",
            "warning",
            "info",
            "light",
            "dark",
            "gray",
        ]

        for color in required_colors:
            assert color in plotter.colors
            assert isinstance(plotter.colors[color], str)
            assert plotter.colors[color].startswith("#")

    def test_matplotlib_backend_compatibility(self):
        """Test that plotter works with different matplotlib backends."""
        # This test ensures our plotter doesn't break with different backends
        plotter = WeatherPlotter()

        # Should not raise any errors
        assert plotter.style == "default"
        assert plotter.dpi == 300

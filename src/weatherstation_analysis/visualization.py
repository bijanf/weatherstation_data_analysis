"""
Visualization Module
===================

Creates publication-quality plots and visualizations for weather data analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from scipy.stats import gumbel_r
import warnings

warnings.filterwarnings("ignore")


class WeatherPlotter:
    """
    Creates professional weather data visualizations.

    This class provides methods to create various types of plots for weather
    data analysis including time series, extreme value analysis, and statistical
    summaries.

    Attributes:
        style (str): Matplotlib style to use
        figsize_default (tuple): Default figure size
        dpi (int): Resolution for saved plots
    """

    def __init__(
        self,
        style: str = "default",
        figsize_default: Tuple[int, int] = (12, 8),
        dpi: int = 300,
    ):
        """
        Initialize the weather plotter.

        Args:
            style: Matplotlib style to use
            figsize_default: Default figure size (width, height)
            dpi: Resolution for saved plots
        """
        self.style = style
        self.figsize_default = figsize_default
        self.dpi = dpi

        # Set up plotting style
        plt.style.use(self.style)

        # Define color palette
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff7f0e",
            "info": "#17a2b8",
            "light": "#f8f9fa",
            "dark": "#343a40",
            "gray": "#888888",
        }

    def _setup_plot_style(self, ax: plt.Axes) -> None:
        """
        Apply consistent styling to plot axes.

        Args:
            ax: Matplotlib axes object
        """
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12, length=6, width=2)

    def _add_data_attribution(
        self, ax: plt.Axes, position: str = "bottom_right"
    ) -> None:
        """
        Add data source attribution to plot.

        Args:
            ax: Matplotlib axes object
            position: Position for attribution text
        """
        today = date.today()
        attribution_text = (
            f'Data: Meteostat/DWD\\nUpdated: {today.strftime("%d.%m.%Y")}'
        )

        positions = {
            "bottom_right": (0.99, 0.01, "right", "bottom"),
            "bottom_left": (0.01, 0.01, "left", "bottom"),
            "top_right": (0.99, 0.99, "right", "top"),
            "top_left": (0.01, 0.99, "left", "top"),
        }

        x, y, ha, va = positions.get(position, positions["bottom_right"])

        ax.text(
            x,
            y,
            attribution_text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=10,
            color="#444444",
            alpha=0.8,
        )

    def plot_annual_precipitation_extremes(
        self, extremes_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create annual precipitation extremes plot with return period analysis.

        Args:
            extremes_df: DataFrame with extreme values
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        years = np.array(extremes_df["year"])
        max_precip = np.array(extremes_df["max_precip"])

        # Top plot: Time series of annual maxima
        ax1.scatter(
            years,
            max_precip,
            color=self.colors["primary"],
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Trend line
        z = np.polyfit(years, max_precip, 2)
        p = np.poly1d(z)
        trend_years = np.linspace(years.min(), years.max(), 300)
        trend_line = p(trend_years)
        ax1.plot(
            trend_years, trend_line, color=self.colors["danger"], linewidth=2, alpha=0.8
        )

        # Highlight extreme years
        top_5_idx = np.argsort(max_precip)[-5:]
        ax1.scatter(
            years[top_5_idx],
            max_precip[top_5_idx],
            color=self.colors["danger"],
            s=100,
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )

        # Annotate highest
        max_idx = np.argmax(max_precip)
        ax1.annotate(
            f"{max_precip[max_idx]:.1f}mm\\n({years[max_idx]})",
            xy=(years[max_idx], max_precip[max_idx]),
            xytext=(years[max_idx] + 5, max_precip[max_idx] + 2),
            fontsize=12,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=self.colors["danger"], lw=2),
        )

        ax1.set_xlabel("Year", fontsize=14, fontweight="bold")
        ax1.set_ylabel(
            "Maximum Daily Precipitation (mm)", fontsize=14, fontweight="bold"
        )
        ax1.set_title(
            "Annual Maximum Daily Precipitation\\nPotsdam Säkularstation (1893-2024)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        self._setup_plot_style(ax1)

        # Bottom plot: Return period analysis
        sorted_precip = np.sort(max_precip)[::-1]
        n = len(sorted_precip)
        return_periods = (n + 1) / np.arange(1, n + 1)

        ax2.scatter(
            return_periods,
            sorted_precip,
            color=self.colors["warning"],
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Fit Gumbel distribution
        params = gumbel_r.fit(max_precip)
        theoretical_rp = np.logspace(0, 2, 100)
        theoretical_precip = gumbel_r.ppf(1 - 1 / theoretical_rp, *params)

        ax2.plot(
            theoretical_rp,
            theoretical_precip,
            color=self.colors["danger"],
            linewidth=2,
            label="Gumbel Distribution Fit",
            alpha=0.8,
        )

        ax2.set_xscale("log")
        ax2.set_xlabel("Return Period (years)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Daily Precipitation (mm)", fontsize=14, fontweight="bold")
        ax2.set_title(
            "Return Period Analysis - Maximum Daily Precipitation",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax2.legend(fontsize=12)
        self._setup_plot_style(ax2)

        # Add reference lines
        for rp in [2, 5, 10, 25, 50, 100]:
            if rp <= return_periods.max():
                precip_val = gumbel_r.ppf(1 - 1 / rp, *params)
                ax2.axvline(x=rp, color="gray", linestyle="--", alpha=0.5)
                ax2.text(
                    rp,
                    precip_val + 2,
                    f"{rp}y",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="gray",
                )

        self._add_data_attribution(ax2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"📊 Plot saved as '{save_path}'")

        return fig

    def plot_temperature_extremes_analysis(
        self, extremes_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive temperature extremes analysis plot.

        Args:
            extremes_df: DataFrame with extreme values
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        years = np.array(extremes_df["year"])
        max_temp = np.array(extremes_df["max_temp"])
        min_temp = np.array(extremes_df["min_temp"])
        temp_range = np.array(extremes_df["temp_range"])

        trend_years = np.linspace(years.min(), years.max(), 300)

        # Top left: Annual maximum temperatures
        ax1.scatter(years, max_temp, color=self.colors["danger"], s=40, alpha=0.7)
        z1 = np.polyfit(years, max_temp, 2)
        p1 = np.poly1d(z1)
        ax1.plot(trend_years, p1(trend_years), color="black", linewidth=2, alpha=0.8)
        ax1.set_ylabel("Maximum Temperature (°C)", fontsize=12, fontweight="bold")
        ax1.set_title("Annual Maximum Temperature", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax1)

        # Top right: Annual minimum temperatures
        ax2.scatter(years, min_temp, color=self.colors["success"], s=40, alpha=0.7)
        z2 = np.polyfit(years, min_temp, 2)
        p2 = np.poly1d(z2)
        ax2.plot(trend_years, p2(trend_years), color="black", linewidth=2, alpha=0.8)
        ax2.set_ylabel("Minimum Temperature (°C)", fontsize=12, fontweight="bold")
        ax2.set_title("Annual Minimum Temperature", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax2)

        # Bottom left: Temperature range
        ax3.scatter(years, temp_range, color=self.colors["warning"], s=40, alpha=0.7)
        z3 = np.polyfit(years, temp_range, 2)
        p3 = np.poly1d(z3)
        ax3.plot(trend_years, p3(trend_years), color="black", linewidth=2, alpha=0.8)
        ax3.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Temperature Range (°C)", fontsize=12, fontweight="bold")
        ax3.set_title("Annual Temperature Range", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax3)

        # Bottom right: Distribution of maximum temperatures
        ax4.hist(
            max_temp, bins=20, alpha=0.7, color=self.colors["danger"], edgecolor="black"
        )
        ax4.axvline(x=35, color="red", linestyle="--", linewidth=2, alpha=0.8)
        ax4.set_xlabel("Maximum Temperature (°C)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax4.set_title(
            "Distribution of Annual Maximum Temperatures",
            fontsize=14,
            fontweight="bold",
        )

        hot_extremes = (max_temp > 35).sum()
        ax4.text(
            0.95,
            0.95,
            f"Years > 35°C: {hot_extremes}",
            transform=ax4.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        self._setup_plot_style(ax4)

        plt.suptitle(
            "Temperature Extremes Analysis - Potsdam Säkularstation (1893-2024)",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        self._add_data_attribution(ax4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"📊 Plot saved as '{save_path}'")

        return fig

    def plot_threshold_exceedance_analysis(
        self, all_data: Dict[int, pd.DataFrame], save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create threshold exceedance analysis plot.

        Args:
            all_data: Dictionary of weather data by year
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        years = []
        heavy_rain_days = []
        extreme_rain_days = []
        hot_days = []
        cold_days = []

        for year, data in all_data.items():
            if year != 2025:  # Exclude incomplete 2025
                years.append(year)
                heavy_rain_days.append((data["prcp"] > 20).sum())
                extreme_rain_days.append((data["prcp"] > 50).sum())
                hot_days.append((data["tmax"] > 30).sum())
                cold_days.append((data["tmin"] < -10).sum())

        years = np.array(years)
        heavy_rain_days = np.array(heavy_rain_days)
        extreme_rain_days = np.array(extreme_rain_days)
        hot_days = np.array(hot_days)
        cold_days = np.array(cold_days)

        trend_years = np.linspace(years.min(), years.max(), 300)

        # Plot 1: Heavy rain days
        ax1.scatter(
            years, heavy_rain_days, color=self.colors["primary"], s=40, alpha=0.7
        )
        z1 = np.polyfit(years, heavy_rain_days, 2)
        p1 = np.poly1d(z1)
        ax1.plot(
            trend_years,
            p1(trend_years),
            color=self.colors["danger"],
            linewidth=2,
            alpha=0.8,
        )
        ax1.set_ylabel("Days per Year", fontsize=12, fontweight="bold")
        ax1.set_title("Heavy Rain Days (> 20mm/day)", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax1)

        # Plot 2: Extreme rain days
        ax2.scatter(
            years, extreme_rain_days, color=self.colors["warning"], s=40, alpha=0.7
        )
        z2 = np.polyfit(years, extreme_rain_days, 2)
        p2 = np.poly1d(z2)
        ax2.plot(
            trend_years,
            p2(trend_years),
            color=self.colors["danger"],
            linewidth=2,
            alpha=0.8,
        )
        ax2.set_ylabel("Days per Year", fontsize=12, fontweight="bold")
        ax2.set_title("Extreme Rain Days (> 50mm/day)", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax2)

        # Plot 3: Hot days
        ax3.scatter(years, hot_days, color=self.colors["danger"], s=40, alpha=0.7)
        z3 = np.polyfit(years, hot_days, 2)
        p3 = np.poly1d(z3)
        ax3.plot(
            trend_years,
            p3(trend_years),
            color=self.colors["danger"],
            linewidth=2,
            alpha=0.8,
        )
        ax3.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Days per Year", fontsize=12, fontweight="bold")
        ax3.set_title("Hot Days (> 30°C)", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax3)

        # Plot 4: Cold days
        ax4.scatter(years, cold_days, color=self.colors["success"], s=40, alpha=0.7)
        z4 = np.polyfit(years, cold_days, 2)
        p4 = np.poly1d(z4)
        ax4.plot(
            trend_years,
            p4(trend_years),
            color=self.colors["danger"],
            linewidth=2,
            alpha=0.8,
        )
        ax4.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Days per Year", fontsize=12, fontweight="bold")
        ax4.set_title("Cold Days (< -10°C)", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax4)

        plt.suptitle(
            "Threshold Exceedance Analysis - Potsdam Säkularstation (1893-2024)",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        self._add_data_attribution(ax4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"📊 Plot saved as '{save_path}'")

        return fig

    def plot_statistics_summary(
        self, extremes_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create extreme statistics summary dashboard.

        Args:
            extremes_df: DataFrame with extreme values
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Box plots of extremes
        data_to_plot = [
            extremes_df["max_precip"],
            extremes_df["max_temp"],
            extremes_df["min_temp"],
            extremes_df["temp_range"],
        ]
        labels = [
            "Max Precip\\n(mm)",
            "Max Temp\\n(°C)",
            "Min Temp\\n(°C)",
            "Temp Range\\n(°C)",
        ]

        box_colors = [
            self.colors["primary"],
            self.colors["danger"],
            self.colors["success"],
            self.colors["warning"],
        ]

        boxes = ax1.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )

        for patch, color in zip(boxes["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_title("Distribution of Annual Extremes", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax1)

        # Plot 2: Correlation matrix
        corr_data = extremes_df[
            ["max_precip", "max_temp", "min_temp", "temp_range"]
        ].corr()
        im = ax2.imshow(corr_data, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

        # Add correlation values
        for i in range(len(corr_data)):
            for j in range(len(corr_data)):
                text = ax2.text(
                    j,
                    i,
                    f"{corr_data.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

        ax2.set_xticks(range(len(corr_data)))
        ax2.set_yticks(range(len(corr_data)))
        ax2.set_xticklabels(
            ["Max Precip", "Max Temp", "Min Temp", "Temp Range"], rotation=45
        )
        ax2.set_yticklabels(["Max Precip", "Max Temp", "Min Temp", "Temp Range"])
        ax2.set_title(
            "Correlation Matrix - Extreme Values", fontsize=14, fontweight="bold"
        )

        # Plot 3: Trend analysis
        from scipy.stats import linregress

        years = extremes_df["year"]

        variables = ["max_precip", "max_temp", "min_temp", "temp_range"]
        trends = []
        p_values = []

        for var in variables:
            slope, _, _, p_value, _ = linregress(years, extremes_df[var])
            trends.append(slope * 100)  # Per century
            p_values.append(p_value)

        colors = [
            self.colors["success"] if trend >= 0 else self.colors["danger"]
            for trend in trends
        ]
        bars = ax3.bar(labels, trends, color=colors, alpha=0.7, edgecolor="black")

        # Add significance indicators
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            significance = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            )
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01 if height > 0 else height - 0.01,
                significance,
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=12,
                fontweight="bold",
            )

        ax3.set_ylabel("Trend (change per century)", fontsize=12, fontweight="bold")
        ax3.set_title(
            "Long-term Trends in Extreme Values", fontsize=14, fontweight="bold"
        )
        self._setup_plot_style(ax3)

        # Plot 4: Data availability by decade
        decades = (years // 10) * 10
        decade_counts = decades.value_counts().sort_index()

        ax4.bar(
            decade_counts.index,
            decade_counts.values,
            width=8,
            alpha=0.7,
            color=self.colors["info"],
            edgecolor="black",
        )
        ax4.set_xlabel("Decade", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Number of Years", fontsize=12, fontweight="bold")
        ax4.set_title("Data Availability by Decade", fontsize=14, fontweight="bold")
        self._setup_plot_style(ax4)

        plt.suptitle(
            "Extreme Value Statistics Summary - Potsdam Säkularstation (1893-2024)",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        self._add_data_attribution(ax4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"📊 Plot saved as '{save_path}'")

        return fig

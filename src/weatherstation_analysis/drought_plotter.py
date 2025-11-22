"""
Drought Visualization Module
============================

Creates publication-quality visualizations for drought analysis.
Designed specifically for Iran drought analysis (2018-2025).
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# Set publication-quality defaults
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


class DroughtPlotter:
    """
    Creates visualizations for drought analysis.

    Generates publication-quality plots suitable for scientific papers.
    """

    # Color scheme for drought severity
    DROUGHT_COLORS = {
        "extreme_drought": "#8B0000",  # Dark red
        "severe_drought": "#FF4500",  # Orange red
        "moderate_drought": "#FFA500",  # Orange
        "normal": "#90EE90",  # Light green
        "wet": "#4169E1",  # Royal blue
    }

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize the drought plotter.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
            sns.set_palette("husl")

    def plot_precipitation_deficit_timeseries(
        self,
        deficit_data: pd.DataFrame,
        station_name: str,
        baseline_mean: float,
        output_file: Optional[str] = None,
    ) -> None:
        """
        Plot precipitation deficit time series.

        Args:
            deficit_data: DataFrame with annual precipitation and deficits
            station_name: Name of the weather station
            baseline_mean: Baseline mean annual precipitation
            output_file: Path to save figure (None = display only)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        years = deficit_data["year"].values
        actual_prcp = deficit_data[
            deficit_data.columns[0]
        ].values  # First column is precipitation

        # Top panel: Actual vs Normal precipitation
        ax1.bar(
            years,
            actual_prcp,
            color="steelblue",
            alpha=0.7,
            label="Actual Precipitation",
        )
        ax1.axhline(
            baseline_mean,
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label=f"Baseline Normal ({baseline_mean:.0f} mm)",
        )

        ax1.set_ylabel("Annual Precipitation (mm)", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"Precipitation Analysis - {station_name}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Add text annotations for extreme years
        for idx, row in deficit_data.iterrows():
            year = int(row["year"])
            prcp = row[deficit_data.columns[0]]
            if row["deficit_percent"] > 30 or row["deficit_percent"] < -30:
                ax1.text(
                    year,
                    prcp,
                    f"{row['percent_of_normal']:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        # Bottom panel: Deficit/Surplus
        colors = [
            "darkred" if x > 0 else "darkgreen" for x in deficit_data["deficit_mm"]
        ]
        ax2.bar(years, deficit_data["deficit_mm"], color=colors, alpha=0.7)
        ax2.axhline(0, color="black", linestyle="-", linewidth=1)

        ax2.set_xlabel("Year", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Deficit/Surplus (mm)", fontsize=11, fontweight="bold")
        ax2.set_title(
            "Precipitation Deficit (positive = deficit, negative = surplus)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)

        # Add cumulative deficit text
        cumulative_deficit = deficit_data["deficit_mm"].sum()
        ax2.text(
            0.02,
            0.98,
            f"Cumulative Deficit: {cumulative_deficit:.0f} mm",
            transform=ax2.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"📊 Saved: {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_spi_timeseries(
        self,
        spi_data: pd.DataFrame,
        station_name: str,
        scale_months: int = 12,
        drought_period: Optional[Tuple[int, int]] = None,
        output_file: Optional[str] = None,
    ) -> None:
        """
        Plot Standardized Precipitation Index (SPI) time series.

        Args:
            spi_data: DataFrame with SPI values
            station_name: Name of the weather station
            scale_months: SPI time scale
            drought_period: Tuple of (start_year, end_year) to highlight
            output_file: Path to save figure (None = display only)
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        spi_col = f"SPI_{scale_months}"
        dates = spi_data.index
        spi_values = spi_data[spi_col].values

        # Color-code by drought severity
        colors = []
        for spi in spi_values:
            if pd.isna(spi):
                colors.append("gray")
            elif spi <= -2.0:
                colors.append(self.DROUGHT_COLORS["extreme_drought"])
            elif spi <= -1.5:
                colors.append(self.DROUGHT_COLORS["severe_drought"])
            elif spi <= -1.0:
                colors.append(self.DROUGHT_COLORS["moderate_drought"])
            elif spi >= 1.0:
                colors.append(self.DROUGHT_COLORS["wet"])
            else:
                colors.append(self.DROUGHT_COLORS["normal"])

        # Plot SPI as bars
        ax.bar(dates, spi_values, color=colors, width=25, alpha=0.7)

        # Add drought category lines
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.axhline(
            -1.0,
            color="orange",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Moderate Drought",
        )
        ax.axhline(
            -1.5,
            color="orangered",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Severe Drought",
        )
        ax.axhline(
            -2.0,
            color="darkred",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Extreme Drought",
        )
        ax.axhline(
            1.0, color="royalblue", linestyle="--", linewidth=1, alpha=0.5, label="Wet"
        )

        # Highlight drought period
        if drought_period:
            start_year, end_year = drought_period
            ax.axvspan(
                pd.Timestamp(f"{start_year}-01-01"),
                pd.Timestamp(f"{end_year}-12-31"),
                alpha=0.2,
                color="red",
                label=f"Drought Period ({start_year}-{end_year})",
            )

        ax.set_xlabel("Year", fontsize=11, fontweight="bold")
        ax.set_ylabel(
            f"SPI-{scale_months} (Standardized)", fontsize=11, fontweight="bold"
        )
        ax.set_title(
            f"Standardized Precipitation Index (SPI-{scale_months}) - {station_name}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.legend(loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

        # Add interpretation text
        interpretation = (
            "SPI Interpretation:\n"
            "  > 2.0: Extremely Wet | 1.5-2.0: Very Wet | 1.0-1.5: Moderately Wet\n"
            "  -1.0 to 1.0: Normal | -1.5 to -1.0: Moderate Drought\n"
            "  -2.0 to -1.5: Severe Drought | < -2.0: Extreme Drought"
        )
        ax.text(
            0.02,
            0.02,
            interpretation,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"📊 Saved: {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_multi_station_comparison(
        self,
        regional_results: Dict[str, Dict],
        metric: str = "mean_annual_deficit_percent",
        output_file: Optional[str] = None,
    ) -> None:
        """
        Compare drought metrics across multiple stations.

        Args:
            regional_results: Dict from MultiStationDroughtAnalyzer
            metric: Metric to compare ('mean_annual_deficit_percent', 'total_deficit_mm', etc.)
            output_file: Path to save figure (None = display only)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        stations = list(regional_results.keys())
        values = [regional_results[s][metric] for s in stations]

        # Left panel: Bar chart of metric
        colors_bar = ["darkred" if v > 0 else "darkgreen" for v in values]
        bars = ax1.barh(stations, values, color=colors_bar, alpha=0.7)
        ax1.axvline(0, color="black", linestyle="-", linewidth=1)

        metric_labels = {
            "mean_annual_deficit_percent": "Mean Annual Deficit (%)",
            "total_deficit_mm": "Total Cumulative Deficit (mm)",
            "worst_deficit_percent": "Worst Year Deficit (%)",
        }
        ax1.set_xlabel(
            metric_labels.get(metric, metric), fontsize=11, fontweight="bold"
        )
        ax1.set_title("Station Comparison", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="x")

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            x_pos = value + (2 if value > 0 else -2)
            ha = "left" if value > 0 else "right"
            ax1.text(
                x_pos,
                i,
                f"{value:.1f}",
                va="center",
                ha=ha,
                fontsize=9,
                fontweight="bold",
            )

        # Right panel: Worst years comparison
        worst_years = [regional_results[s]["worst_year"] for s in stations]
        worst_deficits = [
            regional_results[s]["worst_deficit_percent"] for s in stations
        ]

        scatter = ax2.scatter(
            worst_years,
            worst_deficits,
            s=200,
            alpha=0.6,
            c=worst_deficits,
            cmap="YlOrRd",
            edgecolors="black",
            linewidth=1,
        )

        # Add station labels
        for station, year, deficit in zip(stations, worst_years, worst_deficits):
            ax2.annotate(
                station,
                (year, deficit),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax2.set_xlabel("Worst Year", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Deficit in Worst Year (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Worst Drought Year by Station", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax2, label="Deficit (%)")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"📊 Saved: {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_comprehensive_drought_dashboard(
        self,
        drought_results: Dict,
        station_name: str,
        output_file: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive 4-panel drought analysis dashboard.

        Args:
            drought_results: Results from DroughtAnalyzer.analyze_drought_period()
            station_name: Name of the weather station
            output_file: Path to save figure (None = display only)
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        deficit_data = drought_results["deficit_data"]
        baseline_mean = drought_results["baseline_mean"]
        years = deficit_data["year"].values
        actual_prcp = deficit_data[deficit_data.columns[0]].values

        # Panel 1: Annual precipitation with baseline
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(years, actual_prcp, color="steelblue", alpha=0.7, label="Actual")
        ax1.axhline(
            baseline_mean,
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label=f"Baseline ({baseline_mean:.0f} mm)",
        )
        ax1.fill_between(
            years,
            baseline_mean,
            actual_prcp,
            where=(actual_prcp < baseline_mean),
            color="red",
            alpha=0.2,
            label="Deficit",
        )
        ax1.set_ylabel("Precipitation (mm)", fontweight="bold")
        ax1.set_title("Annual Precipitation vs Baseline", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Percent of normal
        ax2 = fig.add_subplot(gs[0, 1])
        colors = [
            "darkred" if x < 100 else "darkgreen"
            for x in deficit_data["percent_of_normal"]
        ]
        ax2.bar(years, deficit_data["percent_of_normal"], color=colors, alpha=0.7)
        ax2.axhline(100, color="black", linestyle="-", linewidth=2)
        ax2.axhline(75, color="orange", linestyle="--", linewidth=1, alpha=0.5)
        ax2.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_ylabel("Percent of Normal (%)", fontweight="bold")
        ax2.set_title("Precipitation as % of Baseline Normal", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Add annotations for critical years
        for idx, row in deficit_data.iterrows():
            if row["percent_of_normal"] < 60:
                ax2.text(
                    row["year"],
                    row["percent_of_normal"],
                    f"{row['percent_of_normal']:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="red",
                )

        # Panel 3: Cumulative deficit
        ax3 = fig.add_subplot(gs[1, 0])
        cumulative_deficit = deficit_data["deficit_mm"].cumsum()
        ax3.fill_between(
            years,
            0,
            cumulative_deficit,
            where=(cumulative_deficit >= 0),
            color="darkred",
            alpha=0.6,
            label="Cumulative Deficit",
        )
        ax3.plot(years, cumulative_deficit, color="darkred", linewidth=2.5)
        ax3.axhline(0, color="black", linestyle="-", linewidth=1)
        ax3.set_xlabel("Year", fontweight="bold")
        ax3.set_ylabel("Cumulative Deficit (mm)", fontweight="bold")
        ax3.set_title("Cumulative Precipitation Deficit", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # Add final cumulative value
        final_cumulative = cumulative_deficit.iloc[-1]
        ax3.text(
            0.98,
            0.98,
            f"Total: {final_cumulative:.0f} mm",
            transform=ax3.transAxes,
            fontsize=11,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

        # Panel 4: Statistical summary table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        # Prepare summary data
        summary_text = f"""
DROUGHT ANALYSIS SUMMARY
{station_name}

Period: {drought_results['period']}
──────────────────────────────────────

BASELINE STATISTICS (1981-2010)
  Mean Annual Precipitation: {baseline_mean:.1f} mm

DROUGHT PERIOD STATISTICS
  Total Years Analyzed: {drought_results['total_years']}
  Years with Deficit: {drought_results['deficit_years']}

  Total Cumulative Deficit: {drought_results['total_deficit_mm']:.1f} mm
  Mean Annual Deficit: {drought_results['mean_annual_deficit_percent']:.1f}%

  Worst Year: {drought_results['worst_year']}
  Worst Year Deficit: {drought_results['worst_deficit_mm']:.1f} mm
                      ({drought_results['worst_deficit_percent']:.1f}% below normal)

SEVERITY ASSESSMENT
"""
        if drought_results["mean_annual_deficit_percent"] > 40:
            summary_text += "  🔴 EXTREME DROUGHT CONDITIONS\n"
        elif drought_results["mean_annual_deficit_percent"] > 25:
            summary_text += "  🟠 SEVERE DROUGHT CONDITIONS\n"
        elif drought_results["mean_annual_deficit_percent"] > 15:
            summary_text += "  🟡 MODERATE DROUGHT CONDITIONS\n"
        else:
            summary_text += "  🟢 MILD DROUGHT CONDITIONS\n"

        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        # Overall title
        fig.suptitle(
            f"Comprehensive Drought Analysis Dashboard - {station_name}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"📊 Saved: {output_file}")
        else:
            plt.show()

        plt.close()

#!/usr/bin/env python3
"""
Hottest / Coldest Temperature Analysis for the Potsdam Säkularstation.

Fetches the full daily temperature record of the Potsdam Säkularstation from
Meteostat (backed by DWD) and produces Instagram-style annual-extreme plots
that always extend up to *today*.

The current (incomplete) year is handled dynamically via ``date.today()`` and
shown as a separate "so far" marker on the hottest-temperature plot, so the
record never goes stale and never needs a hardcoded cutoff year.

Note: this module targets the Meteostat **2.x** functional API
(``daily`` / ``Point`` / ``Parameter`` / ``stations.nearby``), which replaced
the 1.x ``Stations`` / ``Daily`` classes.
"""

from __future__ import annotations

from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meteostat
from meteostat import Parameter, Point, daily

# The full record spans ~130 years; Meteostat 2.x blocks daily requests longer
# than 30 years unless this guard is lifted.
meteostat.config.block_large_requests = False

# Potsdam Säkularstation (lat, lon, elevation in m). Meteostat station id 10379.
POTSDAM = Point(52.3833, 13.0667, 81)
START_YEAR = 1893  # first full year of the Säkularstation record
MIN_COVERAGE = 80.0  # percent of valid days required to count a complete year
OUTPUT_DIR = "plots"

# Instagram-style palette
GREY = "#888888"
RED = "#D7263D"
BLUE = "#1B6CA8"
ORANGE = "#F18F01"  # current partial year marker


def resolve_station() -> str:
    """Return the Meteostat station id closest to the Säkularstation."""
    near = meteostat.stations.nearby(POTSDAM, radius=50000, limit=1)
    if near.empty:
        raise RuntimeError("No Meteostat station found near Potsdam")
    sid = near.index[0]
    print(
        f"📍 Station: {near.loc[sid, 'name']} ({sid}) @ "
        f"{near.loc[sid, 'latitude']:.4f}°N, {near.loc[sid, 'longitude']:.4f}°E"
    )
    return sid


def fetch_daily_extremes() -> tuple[pd.DataFrame, date]:
    """Fetch the full daily tmax/tmin record up to today in a single request."""
    today = date.today()
    sid = resolve_station()
    print(f"📡 Downloading daily tmax/tmin {START_YEAR}-01-01 … {today} …")
    df = daily(
        sid,
        datetime(START_YEAR, 1, 1),
        datetime(today.year, today.month, today.day),
        parameters=[Parameter.TMAX, Parameter.TMIN],
    ).fetch()
    if df.empty:
        raise RuntimeError("Meteostat returned no data")
    df.index = pd.to_datetime(df.index)
    print(f"✅ {len(df)} daily records, latest = {df.index.max().date()}")
    return df, today


def _expected_days(year: int) -> int:
    """Number of days in *year* (handles leap years)."""
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return 366 if is_leap else 365


def annual_aggregate(
    daily_col: pd.Series, today: date, agg: str
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Aggregate a daily column to one value per year applying the coverage rule.

    Complete years (year < current year) are kept only when they meet
    ``MIN_COVERAGE``. The current, still-incomplete year is returned separately
    and never gated on coverage.

    Args:
        daily_col: Daily ``tmax`` or ``tmin`` series indexed by date.
        today: Reference "today" used to identify the current partial year.
        agg: Either ``"max"`` or ``"min"``.

    Returns:
        ``(years, values, current_year_value)`` where ``current_year_value`` is
        ``None`` when the current year has no data yet.
    """
    series = daily_col.dropna()
    years: list[int] = []
    values: list[float] = []
    current_value: float | None = None

    for year, group in series.groupby(series.index.year):
        year_value = group.max() if agg == "max" else group.min()
        if year == today.year:
            current_value = float(year_value)
            continue
        coverage = len(group) / _expected_days(year) * 100
        if coverage >= MIN_COVERAGE:
            years.append(int(year))
            values.append(float(year_value))

    return np.array(years), np.array(values), current_value


def annual_threshold_count(
    daily_col: pd.Series, today: date, predicate
) -> tuple[np.ndarray, np.ndarray]:
    """Count days per *complete* year where ``predicate(value)`` is True.

    The current partial year is excluded because a partial count is not
    comparable with full-year counts.
    """
    series = daily_col.dropna()
    years: list[int] = []
    counts: list[int] = []
    for year, group in series.groupby(series.index.year):
        if year == today.year:
            continue
        coverage = len(group) / _expected_days(year) * 100
        if coverage >= MIN_COVERAGE:
            years.append(int(year))
            counts.append(int(predicate(group).sum()))
    return np.array(years), np.array(counts)


def _cubic_trend(years: np.ndarray, values: np.ndarray):
    """Return (trend_years, trend_values) for a 3rd-degree fit.

    Years are mean-centred before fitting to keep the Vandermonde matrix
    well-conditioned (avoids numpy's RankWarning on raw four-digit years).
    """
    centre = years.mean()
    coeffs = np.polyfit(years - centre, values, 3)
    poly = np.poly1d(coeffs)
    trend_years = np.linspace(years.min(), years.max(), 300)
    return trend_years, poly(trend_years - centre)


def _style_axes(ax, xlabel: str, ylabel: str, title: str, today: date) -> None:
    """Apply the shared clean, modern, grid-free styling."""
    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold", labelpad=15)
    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold", labelpad=15)
    ax.set_title(title, fontsize=22, fontweight="bold", pad=25)
    ax.tick_params(axis="both", labelsize=14, length=6, width=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.grid(False)
    ax.text(
        0.99,
        0.01,
        f"Data: Meteostat/DWD\nUpdated: {today.strftime('%d.%m.%Y')}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="#444444",
        alpha=0.8,
    )


def create_hottest_temperature_plot(
    years: np.ndarray,
    max_temps: np.ndarray,
    today: date,
    current_year_max: float | None = None,
) -> plt.Figure:
    """Plot the yearly maximum temperature with a cubic trend and today's record."""
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    trend_years, trend_line = _cubic_trend(years, max_temps)
    ax.plot(years, max_temps, "-", color=GREY, linewidth=2, alpha=0.7, zorder=1)
    ax.scatter(years, max_temps, color=RED, s=40, zorder=2)
    ax.plot(
        trend_years, trend_line, "-", color="black", linewidth=3, alpha=0.9, zorder=3
    )

    overall_max = max_temps.max()
    overall_max_year = years[max_temps.argmax()]
    ax.scatter(
        [overall_max_year],
        [overall_max],
        color="black",
        s=100,
        edgecolor="white",
        linewidth=2,
        zorder=4,
    )
    ax.annotate(
        f"{overall_max:.1f}°C",
        xy=(overall_max_year, overall_max),
        xytext=(overall_max_year + 5, overall_max),
        fontsize=16,
        fontweight="bold",
        color="black",
        arrowprops=dict(facecolor="black", arrowstyle="->", lw=2, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8),
    )

    y_top = max_temps.max()
    # Show the current, still-incomplete year as a distinct "so far" marker.
    if current_year_max is not None:
        ax.scatter(
            [today.year],
            [current_year_max],
            marker="*",
            color=ORANGE,
            s=320,
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"{today.year} so far:\n{current_year_max:.1f}°C",
            xy=(today.year, current_year_max),
            xytext=(today.year - 28, current_year_max),
            fontsize=13,
            fontweight="bold",
            color=ORANGE,
            ha="right",
            va="center",
            arrowprops=dict(facecolor=ORANGE, arrowstyle="->", lw=2, alpha=0.8),
        )
        y_top = max(y_top, current_year_max)

    last_year = today.year if current_year_max is not None else years.max()
    ax.set_xlim(years.min() - 2, last_year + 4)
    ax.set_ylim(max_temps.min() - 2, y_top + 2)
    _style_axes(
        ax,
        "Year",
        "Maximum Temperature (°C)",
        f"Hottest Temperature Each Year\nPotsdam Säkularstation, Germany "
        f"({years.min()}-{last_year})",
        today,
    )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/hottest_temperature_plot.png", dpi=300, bbox_inches="tight"
    )
    print(f"📊 Saved {OUTPUT_DIR}/hottest_temperature_plot.png")
    return fig


def create_coldest_temperature_plot(
    years: np.ndarray, min_temps: np.ndarray, today: date
) -> plt.Figure:
    """Plot the yearly minimum temperature with a cubic trend."""
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    trend_years, trend_line = _cubic_trend(years, min_temps)
    ax.plot(years, min_temps, "-", color=GREY, linewidth=2, alpha=0.7, zorder=1)
    ax.scatter(years, min_temps, color=BLUE, s=40, zorder=2)
    ax.plot(
        trend_years, trend_line, "-", color="black", linewidth=3, alpha=0.9, zorder=3
    )

    overall_min = min_temps.min()
    overall_min_year = years[min_temps.argmin()]
    ax.scatter(
        [overall_min_year],
        [overall_min],
        color="black",
        s=100,
        edgecolor="white",
        linewidth=2,
        zorder=4,
    )
    ax.annotate(
        f"{overall_min:.1f}°C",
        xy=(overall_min_year, overall_min),
        xytext=(overall_min_year + 5, overall_min),
        fontsize=16,
        fontweight="bold",
        color="black",
        arrowprops=dict(facecolor="black", arrowstyle="->", lw=2, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8),
    )

    ax.set_ylim(min_temps.min() - 2, min_temps.max() + 2)
    _style_axes(
        ax,
        "Year",
        "Minimum Temperature (°C)",
        f"Coldest Temperature Each Year\nPotsdam Säkularstation, Germany "
        f"({years.min()}-{years.max()})",
        today,
    )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/coldest_temperature_plot.png", dpi=300, bbox_inches="tight"
    )
    print(f"📊 Saved {OUTPUT_DIR}/coldest_temperature_plot.png")
    return fig


def plot_threshold_count(
    years: np.ndarray,
    counts: np.ndarray,
    today: date,
    title: str,
    color: str,
    filename: str,
) -> plt.Figure:
    """Generic 'number of days per year over/under a threshold' plot."""
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    trend_years, trend_line = _cubic_trend(years, counts)
    ax.plot(years, counts, "-", color=GREY, linewidth=2, alpha=0.7, zorder=1)
    ax.scatter(years, counts, color=color, s=40, zorder=2)
    ax.plot(
        trend_years, trend_line, "-", color="black", linewidth=3, alpha=0.9, zorder=3
    )

    ax.set_ylim(max(0, counts.min() - 5), counts.max() + 5)
    _style_axes(ax, "Year", "Number of Days", title, today)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches="tight")
    print(f"📊 Saved {OUTPUT_DIR}/{filename}")
    return fig


def main() -> None:
    print("🌡️ POTSDAM SÄKULARSTATION TEMPERATURE ANALYSIS")
    print("=" * 60)

    df, today = fetch_daily_extremes()

    # Hottest temperature each year (+ current year so far)
    years, max_temps, current_max = annual_aggregate(df["tmax"], today, "max")
    if current_max is not None:
        print(f"🔥 {today.year} hottest so far: {current_max:.1f}°C")
    create_hottest_temperature_plot(years, max_temps, today, current_max)

    # Days per year above 30 °C
    hot_years, hot_counts = annual_threshold_count(df["tmax"], today, lambda g: g > 30)
    plot_threshold_count(
        hot_years,
        hot_counts,
        today,
        f"Days per Year with Maximum Temperature > 30°C\n"
        f"Potsdam Säkularstation, Germany ({hot_years.min()}-{hot_years.max()})",
        RED,
        "days_above_30C_plot.png",
    )

    # Coldest temperature each year
    cold_years, min_temps, _ = annual_aggregate(df["tmin"], today, "min")
    create_coldest_temperature_plot(cold_years, min_temps, today)

    # Days per year below 0 °C
    frost_years, frost_counts = annual_threshold_count(
        df["tmin"], today, lambda g: g < 0
    )
    plot_threshold_count(
        frost_years,
        frost_counts,
        today,
        f"Days per Year with Minimum Temperature < 0°C\n"
        f"Potsdam Säkularstation, Germany ({frost_years.min()}-{frost_years.max()})",
        BLUE,
        "days_below_0C_plot.png",
    )

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

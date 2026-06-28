#!/usr/bin/env python3
"""
Annual temperature-cycle chart (English labels).

For a given station and year, plots every day's maximum temperature as a
coloured bar against the smoothed climatological daily maximum of a reference
period (default 1991-2020):

* warm bar / up-triangle   -> day warmer than the climatological daily maximum
* cool bar / down-triangle -> day cooler than the climatological daily maximum

Behind the bars, nested grey bands show the 5/10/25/75/90/95 percentile
distribution of the reference period; a table underneath lists the monthly mean
daily maximum and its anomaly, and the warmest day so far is annotated.

Targets the Meteostat 2.x functional API. See CLAUDE.md for the API caveat.
"""

from __future__ import annotations

from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import meteostat
from meteostat import Parameter, Point, daily

meteostat.config.block_large_requests = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATIONS = {
    "potsdam": {
        "id": "10379",
        "label": "Potsdam Secular Station",
        "point": Point(52.3833, 13.0667, 81),
        "note": (
            "Obwohl die Station auf dem Telegrafenberg in einem kleinen "
            "Wäldchen liegt, ist es hier statistisch wärmer als üblich!"
        ),
    },
    "berlin-dahlem": {
        "id": "10381",
        "label": "Berlin-Dahlem",
        "point": Point(52.4667, 13.3000, 51),
    },
}

REF_START, REF_END = 1991, 2020  # climatological reference period
SMOOTH_WINDOW = 15  # days, odd circular moving-average window for climatology
DAYS_IN_YEAR = 365  # fixed (leap-day-folded) day-of-year grid length
PCT_LEVELS = [5, 10, 25, 75, 90, 95]  # percentile bands of the distribution
OUTPUT_DIR = "plots"

# Axes box in figure coordinates (shared by the plot and the bottom table)
PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, PLOT_BOTTOM = 0.135, 0.93, 0.90, 0.21
JAHR_X = 0.965  # figure-x of the annual ('Jahr') table column
Y_HEADER = 0.185  # figure-y of the month / 'Jahr' header row
Y_MEAN, Y_ANOM = 0.125, 0.075  # figure-y of the two table value rows

# Divergent warm/cool palette (its own look, not the classic grey-on-crimson)
RED = "#E4572E"  # warmer than normal
BLUE = "#3D7EA6"  # cooler than normal
LINE_COLOR = "#1F2D3D"  # climatological mean line
HILITE = "#1F2D3D"  # warmest-day annotation
BAND_COLORS = {  # (low, high) percentile -> grey level, darkest = innermost
    (25, 75): "#bdbdbd",
    (10, 90): "#d6d6d6",
    (5, 95): "#ececec",
}

# Non-leap calendar: cumulative first-day-of-month day numbers (1..365)
_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_STARTS = np.concatenate([[1], 1 + np.cumsum(_DAYS_IN_MONTH)])  # len 13, ends 366
MONTH_CENTERS = (MONTH_STARTS[:-1] + MONTH_STARTS[1:]) / 2.0
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_tmax(station: dict, start_year: int, end: date) -> pd.Series:
    """Fetch the daily maximum-temperature series for one station.

    Uses the curated Meteostat station id when present; otherwise falls back to
    the station nearest the configured coordinates.
    """
    sid = station.get("id")
    if not sid:
        near = meteostat.stations.nearby(station["point"], radius=50000, limit=1)
        if near.empty:
            raise RuntimeError("No Meteostat station found")
        sid = near.index[0]
    df = daily(
        sid,
        datetime(start_year, 1, 1),
        datetime(end.year, end.month, end.day),
        parameters=[Parameter.TMAX],
    ).fetch()
    if df.empty or "tmax" not in df.columns:
        raise RuntimeError(f"No tmax data returned for station {sid}")
    s = df["tmax"].copy()
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def common_doy(index: pd.DatetimeIndex) -> np.ndarray:
    """Map dates to a fixed 1..365 day-of-year that ignores Feb 29.

    Feb 29 (leap years) is folded onto day 59 (= Feb 28) and every later day in
    a leap year is shifted back by one, so March 1st is always day 60 and all
    years align regardless of leap status.
    """
    doy = np.asarray(index.dayofyear)
    leap = np.asarray(index.is_leap_year)
    # Single pass driven by the ORIGINAL doy: in leap years fold Feb 29 (doy 60)
    # onto Feb 28 (-> 59) and shift every later day back by one, so Mar 1 (61)
    # -> 60 and Dec 31 (366) -> 365. Non-leap years are unchanged.
    return np.where(leap & (doy >= 60), doy - 1, doy)


def circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred circular moving average (wraps Dec->Jan) over a 1..365 grid."""
    if window <= 1:
        return values
    window = window if window % 2 == 1 else window + 1  # 'same' needs odd kernel
    pad = window
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(window) / window
    smoothed = np.convolve(ext, kernel, mode="same")
    return smoothed[pad:-pad]


def _ref_slice(series: pd.Series, ref_start: int, ref_end: int) -> pd.Series:
    """Return the reference-period slice (inclusive year range) of a series."""
    return series[(series.index.year >= ref_start) & (series.index.year <= ref_end)]


def build_climatology(series: pd.Series, ref_start: int, ref_end: int) -> dict:
    """Compute the smoothed climatological mean and percentile bands (1..365)."""
    ref = _ref_slice(series, ref_start, ref_end)
    if ref.empty:
        raise RuntimeError("No data in the reference period")
    frame = pd.DataFrame({"tmax": ref.to_numpy(), "doy": common_doy(ref.index)})
    grid = np.arange(1, DAYS_IN_YEAR + 1)

    grouped = frame.groupby("doy")["tmax"]
    mean = grouped.mean().reindex(grid).interpolate(limit_direction="both").to_numpy()

    quantiles = grouped.quantile([p / 100 for p in PCT_LEVELS]).unstack()
    quantiles = quantiles.reindex(grid).interpolate(limit_direction="both")

    out = {"mean": circular_smooth(mean, SMOOTH_WINDOW), "grid": grid}
    for p in PCT_LEVELS:
        out[p] = circular_smooth(quantiles[p / 100].to_numpy(), SMOOTH_WINDOW)
    return out


def monthly_table(series: pd.Series, year: int, ref_start: int, ref_end: int) -> dict:
    """Monthly mean daily maximum for ``year`` and its anomaly vs the reference.

    Monthly figures are like-for-like: for an in-progress month the reference is
    restricted to the same day-of-month window, so a partial month is not
    compared against a full reference month. The annual ('Year') anomaly instead
    compares the year-to-date mean against the *full-year* reference climatology,
    so it reflects the seasonal sampling of the elapsed part of the year rather
    than a same-window anomaly.
    """
    cur = series[series.index.year == year]
    ref = _ref_slice(series, ref_start, ref_end)

    means, anomalies = {}, {}
    for m in range(1, 13):
        cur_m = cur[cur.index.month == m].dropna()
        if cur_m.empty:
            means[m], anomalies[m] = None, None
            continue
        # Match the reference to the same day-of-month span as the current month.
        last_day = int(cur_m.index.day.max())
        ref_m = ref[(ref.index.month == m) & (ref.index.day <= last_day)]
        means[m] = float(cur_m.mean())
        anomalies[m] = float(cur_m.mean() - ref_m.mean()) if not ref_m.empty else None

    # 'Year' column: year-to-date mean vs the full-year reference climatology
    # (intentionally not a same-period anomaly).
    if not cur.empty and not ref.empty:
        year_mean = float(cur.mean())
        year_anom = float(cur.mean() - ref.mean())
    else:
        year_mean, year_anom = None, None

    return {"means": means, "anomalies": anomalies,
            "year_mean": year_mean, "year_anom": year_anom}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _fig_x(data_x: float) -> float:
    """Map an x value in data coords (1..365) to a figure-x coordinate."""
    return PLOT_LEFT + (data_x - 1) / 364.0 * (PLOT_RIGHT - PLOT_LEFT)


def make_plot(key: str, year: int | None = None) -> str:
    """Render the yearly-cycle chart for one station and save it as a PNG.

    Args:
        key: A key into ``STATIONS``.
        year: Year to plot; defaults to the current year (a partial current
            year is drawn up to today).

    Returns:
        The path of the written PNG file.
    """
    station = STATIONS[key]
    today = date.today()
    year = year or today.year

    series = fetch_tmax(station, REF_START, today)
    clim = build_climatology(series, REF_START, REF_END)
    table = monthly_table(series, year, REF_START, REF_END)

    cur = series[series.index.year == year]
    if cur.empty:
        raise RuntimeError(f"No data for {year} at {station['label']}")
    cur_doy = common_doy(cur.index)
    cur_val = cur.to_numpy()
    clim_at_cur = clim["mean"][cur_doy - 1]
    above = cur_val >= clim_at_cur

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(left=PLOT_LEFT, right=PLOT_RIGHT, top=PLOT_TOP,
                        bottom=PLOT_BOTTOM)
    grid = clim["grid"]

    # Percentile bands (outermost first so inner ones paint on top)
    for (lo, hi), colour in sorted(BAND_COLORS.items(), key=lambda kv: kv[0][0]):
        ax.fill_between(grid, clim[lo], clim[hi], color=colour, linewidth=0, zorder=1)

    # Daily bars + triangle markers for the current year
    ax.vlines(cur_doy[above], clim_at_cur[above], cur_val[above],
              color=RED, linewidth=0.7, zorder=3)
    ax.vlines(cur_doy[~above], cur_val[~above], clim_at_cur[~above],
              color=BLUE, linewidth=0.7, zorder=3)
    ax.scatter(cur_doy[above], cur_val[above], marker="^", s=7,
               color=RED, zorder=4)
    ax.scatter(cur_doy[~above], cur_val[~above], marker="v", s=7,
               color=BLUE, zorder=4)

    # Climatological mean line on top
    ax.plot(grid, clim["mean"], color=LINE_COLOR, linewidth=1.1, zorder=5)

    # Highlight the warmest day of the year so far
    hot_i = int(np.argmax(cur_val))
    hot_x, hot_y, hot_d = cur_doy[hot_i], cur_val[hot_i], cur.index[hot_i]
    ax.scatter([hot_x], [hot_y], marker="o", s=80, facecolors="none",
               edgecolors=HILITE, linewidths=1.4, zorder=6)
    ax.annotate(
        f"Warmest so far: {hot_y:.1f} °C "
        f"({hot_d.day} {MONTH_NAMES[hot_d.month - 1]})",
        xy=(hot_x, hot_y), xytext=(hot_x - 14, hot_y),
        fontsize=9, fontweight="bold", color=HILITE, ha="right", va="center",
        arrowprops=dict(arrowstyle="->", lw=1.2, color=HILITE),
    )

    # Percentile labels just inside both margins
    for p in PCT_LEVELS:
        ax.text(4, clim[p][0], str(p), color="#8f8f8f", fontsize=8,
                ha="left", va="center", zorder=6)
        ax.text(362, clim[p][-1], str(p), color="#8f8f8f", fontsize=8,
                ha="right", va="center", zorder=6)

    # Axes / grid
    ax.set_xlim(1, DAYS_IN_YEAR)
    data_lo = min(cur_val.min(), clim[5].min())
    data_hi = max(cur_val.max(), clim[95].max())
    y_lo = min(-12, float(np.floor(data_lo - 2)))
    y_hi = max(42, float(np.ceil(data_hi + 2.5)))
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(MONTH_CENTERS)
    ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(np.arange(-10, int(np.floor(y_hi)) + 1, 10))
    ax.set_ylabel("Temperature [°C]", fontsize=12)
    ax.yaxis.grid(True, color="#e9e9e9", linewidth=0.8, zorder=0)
    # Faint vertical month separators (a distinct touch)
    for x0 in MONTH_STARTS[1:-1]:
        ax.axvline(x0, color="#eeeeee", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Titles / credits (left-aligned title + grey subtitle)
    fig.text(PLOT_LEFT, 0.955,
             f"{station['label']} — Daily Maximum Temperature",
             ha="left", fontsize=14, fontweight="bold")
    fig.text(PLOT_LEFT, 0.925,
             f"{year} compared with the {REF_START}–{REF_END} climatology",
             ha="left", fontsize=11, color="#555555")
    gen = f"Generated {today.day} {MONTH_NAMES[today.month - 1]} {today.year}"
    fig.text(0.985, 0.955, gen, ha="right", fontsize=9, color="#555555")
    fig.text(0.985, 0.925, "Data: Meteostat / DWD", ha="right", fontsize=9,
             color="#555555")

    # Legend
    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=RED,
               markersize=9, label="Warmer than normal"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=BLUE,
               markersize=9, label="Cooler than normal"),
        Line2D([0], [0], color=LINE_COLOR, linewidth=1.3,
               label=f"Climatological daily max ({REF_START}–{REF_END})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#cccccc",
               markeredgecolor="none", markersize=11,
               label="Daily-max distribution (5–95th pct)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False,
              handletextpad=0.6, borderpad=0.4)

    _draw_table(fig, table)

    # Optional station-specific footnote (e.g. the Potsdam siting caveat)
    if station.get("note"):
        fig.text(PLOT_LEFT, 0.022, station["note"], ha="left", va="center",
                 fontsize=8, style="italic", color="#666666")

    out = f"{OUTPUT_DIR}/yearly_cycle_{key}_{year}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"📊 Saved {out}")
    return out


def _draw_table(fig, table: dict) -> None:
    """Render the monthly-mean / anomaly rows beneath the axes (figure coords)."""
    # Thin separator rule underlining the month / 'Year' header
    sep = Line2D([PLOT_LEFT - 0.02, 0.99], [Y_HEADER - 0.03, Y_HEADER - 0.03],
                 color="#cccccc", linewidth=0.8, transform=fig.transFigure)
    fig.add_artist(sep)

    # Row labels at the far left, right-aligned so they end before the plot box
    fig.text(PLOT_LEFT - 0.008, Y_MEAN, "Monthly mean [°C]", ha="right",
             va="center", fontsize=9)
    fig.text(PLOT_LEFT - 0.008, Y_ANOM, "Anomaly [°C]", ha="right",
             va="center", fontsize=9)

    def _cell(x, mean, anom):
        fig.text(x, Y_MEAN, _fmt(mean), ha="center", va="center", fontsize=9)
        colour = "#555555" if anom is None else (RED if anom >= 0 else BLUE)
        atxt = "-" if anom is None else f"{anom:+.1f}"
        fig.text(x, Y_ANOM, atxt, ha="center", va="center", fontsize=9,
                 color=colour)

    for m in range(1, 13):
        _cell(_fig_x(MONTH_CENTERS[m - 1]), table["means"][m],
              table["anomalies"][m])

    # 'Year' column header (aligned with the month names) + its values
    fig.text(JAHR_X, Y_HEADER, "Year", ha="center", va="center", fontsize=10,
             fontweight="bold")
    _cell(JAHR_X, table["year_mean"], table["year_anom"])


def main() -> None:
    """Render the yearly-cycle chart for all configured stations."""
    for key in ("potsdam", "berlin-dahlem"):
        print(f"=== {STATIONS[key]['label']} ===")
        make_plot(key)


if __name__ == "__main__":
    main()

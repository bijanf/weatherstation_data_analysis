#!/usr/bin/env python3
"""
Annual temperature-cycle chart (English labels).

For a given station and year, plots every day's maximum temperature as a
coloured bar against the smoothed climatological daily maximum of a reference
period (default 1991-2020):

* warm bar / up-triangle   -> day warmer than the climatological daily maximum
* cool bar / down-triangle -> day cooler than the climatological daily maximum

Behind the bars, nested grey bands show the reference-period distribution: the
min-max envelope plus the 10/25/75/90 percentile bands. A table underneath lists
the monthly mean daily maximum and its anomaly, and the warmest day so far is
annotated.

Each station is rendered in three layouts (see ``LAYOUTS``): a landscape chart,
plus Instagram-ready vertical exports (4:5 feed post and 9:16 story/reel).

Targets the Meteostat 2.x functional API. See CLAUDE.md for the API caveat.
"""

from __future__ import annotations

import textwrap
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
PCT_LEVELS = [10, 25, 75, 90]  # inner percentile bands of the distribution
OUTPUT_DIR = "plots"

# Divergent warm/cool palette (its own look, not the classic grey-on-crimson)
RED = "#E4572E"  # warmer than normal
BLUE = "#3D7EA6"  # cooler than normal
LINE_COLOR = "#1F2D3D"  # climatological mean line
HILITE = "#1F2D3D"  # warmest-day annotation
# Bands painted widest (outermost) first so inner bands sit on top. Edge keys
# index into the climatology dict ("min"/"max" or a percentile int).
BAND_COLORS = {
    ("min", "max"): "#ececec",
    (10, 90): "#d6d6d6",
    (25, 75): "#bdbdbd",
}
# Band-edge labels drawn just inside the margins (top -> bottom)
MARGIN_LABELS = [("max", "max"), (90, "90"), (75, "75"),
                 (25, "25"), (10, "10"), ("min", "min")]

# Non-leap calendar: cumulative first-day-of-month day numbers (1..365)
_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_STARTS = np.concatenate([[1], 1 + np.cumsum(_DAYS_IN_MONTH)])  # len 13
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

    def _smoothed(s: pd.Series) -> np.ndarray:
        filled = s.reindex(grid).interpolate(limit_direction="both")
        return circular_smooth(filled.to_numpy(), SMOOTH_WINDOW)

    out = {
        "grid": grid,
        "mean": _smoothed(grouped.mean()),
        "min": _smoothed(grouped.min()),
        "max": _smoothed(grouped.max()),
    }
    quantiles = grouped.quantile([p / 100 for p in PCT_LEVELS]).unstack()
    for p in PCT_LEVELS:
        out[p] = _smoothed(quantiles[p / 100])
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


# Output layouts: chart axes box (l, b, w, h in figure fractions) plus text
# anchors and font sizes. "wide" is the landscape chart; "portrait"/"story" are
# vertical, phone-sized exports for Instagram.
LAYOUTS = {
    "wide": {
        "suffix": "", "figsize": (13, 8), "dpi": 200, "vertical": False,
        "axes": (0.135, 0.21, 0.795, 0.69),
        "title_size": 14, "sub_size": 11, "caption_size": 8,
        "fs": {"tick": 10, "ylabel": 12, "legend": 9, "margin": 8, "annot": 9},
        "table": True, "table_y": (0.185, 0.125, 0.075), "jahr_x": 0.965,
        "caption_y": 0.022,
    },
    "portrait": {  # Instagram feed post, 1080x1350 (4:5)
        "suffix": "_portrait", "figsize": (9, 11.25), "dpi": 120,
        "vertical": True, "axes": (0.13, 0.42, 0.84, 0.40),
        "title_size": 21, "sub_size": 13, "caption_size": 11,
        "fs": {"tick": 12, "ylabel": 14, "legend": 11, "margin": 9, "annot": 12},
        "table": False, "legend_y": 0.375, "stat_y": (0.31, 0.27),
        "caption_y": 0.15, "brand_y": 0.045,
    },
    "story": {  # Instagram story / reel, 1080x1920 (9:16)
        "suffix": "_story", "figsize": (9, 16), "dpi": 120,
        "vertical": True, "axes": (0.13, 0.48, 0.84, 0.34),
        "title_size": 24, "sub_size": 15, "caption_size": 12,
        "fs": {"tick": 13, "ylabel": 16, "legend": 12, "margin": 10, "annot": 13},
        "table": False, "legend_y": 0.45, "stat_y": (0.39, 0.35),
        "caption_y": 0.27, "brand_y": 0.07,
    },
}


def _draw_chart(ax, clim: dict, cur: pd.Series, fs: dict) -> dict:
    """Draw bands, daily bars, climatology line and labels onto ``ax``.

    Returns summary stats (warm-day fraction, day count, warmest day).
    """
    grid = clim["grid"]
    cur_doy = common_doy(cur.index)
    cur_val = cur.to_numpy()
    clim_at_cur = clim["mean"][cur_doy - 1]
    above = cur_val >= clim_at_cur

    for (lo, hi), colour in BAND_COLORS.items():
        ax.fill_between(grid, clim[lo], clim[hi], color=colour, linewidth=0,
                        zorder=1)

    ax.vlines(cur_doy[above], clim_at_cur[above], cur_val[above],
              color=RED, linewidth=0.7, zorder=3)
    ax.vlines(cur_doy[~above], cur_val[~above], clim_at_cur[~above],
              color=BLUE, linewidth=0.7, zorder=3)
    ax.scatter(cur_doy[above], cur_val[above], marker="^", s=7, color=RED,
               zorder=4)
    ax.scatter(cur_doy[~above], cur_val[~above], marker="v", s=7, color=BLUE,
               zorder=4)

    ax.plot(grid, clim["mean"], color=LINE_COLOR, linewidth=1.1, zorder=5)

    # Warmest day of the year so far
    hot_i = int(np.argmax(cur_val))
    hot_x, hot_y, hot_d = cur_doy[hot_i], cur_val[hot_i], cur.index[hot_i]
    ax.scatter([hot_x], [hot_y], marker="o", s=80, facecolors="none",
               edgecolors=HILITE, linewidths=1.4, zorder=6)
    ax.annotate(
        f"Warmest so far: {hot_y:.1f} °C "
        f"({hot_d.day} {MONTH_NAMES[hot_d.month - 1]})",
        xy=(hot_x, hot_y), xytext=(hot_x - 14, hot_y),
        fontsize=fs["annot"], fontweight="bold", color=HILITE, ha="right",
        va="center", arrowprops=dict(arrowstyle="->", lw=1.2, color=HILITE),
    )

    for edge, lab in MARGIN_LABELS:
        ax.text(4, clim[edge][0], lab, color="#8f8f8f", fontsize=fs["margin"],
                ha="left", va="center", zorder=6)
        ax.text(362, clim[edge][-1], lab, color="#8f8f8f",
                fontsize=fs["margin"], ha="right", va="center", zorder=6)

    ax.set_xlim(1, DAYS_IN_YEAR)
    data_lo = min(cur_val.min(), clim["min"].min())
    data_hi = max(cur_val.max(), clim["max"].max())
    ax.set_ylim(min(-12, float(np.floor(data_lo - 2))),
                max(42, float(np.ceil(data_hi + 2.5))))
    y_hi = ax.get_ylim()[1]
    ax.set_xticks(MONTH_CENTERS)
    ax.set_xticklabels(MONTH_NAMES, fontsize=fs["tick"])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.set_yticks(np.arange(-10, int(np.floor(y_hi)) + 1, 10))
    ax.set_ylabel("Temperature [°C]", fontsize=fs["ylabel"])
    ax.yaxis.grid(True, color="#e9e9e9", linewidth=0.8, zorder=0)
    for x0 in MONTH_STARTS[1:-1]:
        ax.axvline(x0, color="#eeeeee", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return {"warm_frac": float(above.mean()), "n_days": int(above.size),
            "hot_y": float(hot_y), "hot_d": hot_d}


def _legend_handles(short: bool) -> list:
    """Legend handles; ``short`` uses compact labels for the vertical layouts."""
    clim_lab = ("Climatology 1991–2020" if short
                else f"Climatological daily max ({REF_START}–{REF_END})")
    dist_lab = ("Distribution (min–max)" if short
                else "Daily-max distribution (min–max & 10–90 pct)")
    return [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=RED,
               markersize=9, label="Warmer than normal"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=BLUE,
               markersize=9, label="Cooler than normal"),
        Line2D([0], [0], color=LINE_COLOR, linewidth=1.3, label=clim_lab),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#cccccc",
               markeredgecolor="none", markersize=11, label=dist_lab),
    ]


def make_plot(key: str, year: int | None = None,
              formats: tuple = ("wide", "portrait", "story")) -> list:
    """Render the yearly-cycle chart for one station in several formats.

    Args:
        key: A key into ``STATIONS``.
        year: Year to plot; defaults to the current year.
        formats: Layout names from ``LAYOUTS`` to render.

    Returns:
        The list of written PNG paths.
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
    n_ref = int(_ref_slice(series, REF_START, REF_END).index.year.nunique())

    return [_render(LAYOUTS[f], key, station, year, today, clim, cur, table,
                    n_ref) for f in formats]


def _render(lay: dict, key, station, year, today, clim, cur, table,
            n_ref: int) -> str:
    """Compose one figure for the given layout and save it as a PNG."""
    fig = plt.figure(figsize=lay["figsize"])
    ax = fig.add_axes(lay["axes"])
    info = _draw_chart(ax, clim, cur, lay["fs"])

    if lay["vertical"]:
        fig.legend(handles=_legend_handles(True), loc="upper center",
                   bbox_to_anchor=(0.5, lay["legend_y"]), ncol=2,
                   fontsize=lay["fs"]["legend"], frameon=False,
                   handletextpad=0.5, columnspacing=1.6)
    else:
        ax.legend(handles=_legend_handles(False), loc="upper left",
                  fontsize=lay["fs"]["legend"], frameon=False,
                  handletextpad=0.6, borderpad=0.4)

    ax_l, ax_b, ax_w, ax_h = lay["axes"]
    ax_top = ax_b + ax_h
    gen = f"Generated {today.day} {MONTH_NAMES[today.month - 1]} {today.year}"

    if lay["vertical"]:
        fig.text(0.5, ax_top + 0.10, station["label"], ha="center",
                 fontsize=lay["title_size"], fontweight="bold")
        fig.text(0.5, ax_top + 0.065, f"Daily Maximum Temperature · {year}",
                 ha="center", fontsize=lay["sub_size"], color="#333333")
        fig.text(0.5, ax_top + 0.038,
                 f"vs {REF_START}–{REF_END} climatology · {n_ref} years of data",
                 ha="center", fontsize=lay["sub_size"] - 1, color="#888888")
        _draw_keystats(fig, table, info, lay)
        fig.text(0.5, lay["brand_y"], f"{gen}  ·  Data: Meteostat / DWD",
                 ha="center", fontsize=9, color="#999999")
    else:
        fig.text(ax_l, 0.955,
                 f"{station['label']} — Daily Maximum Temperature",
                 ha="left", fontsize=lay["title_size"], fontweight="bold")
        fig.text(ax_l, 0.925,
                 f"{year} compared with the {REF_START}–{REF_END} climatology "
                 f"({n_ref} years of data)",
                 ha="left", fontsize=lay["sub_size"], color="#555555")
        fig.text(0.985, 0.955, gen, ha="right", fontsize=9, color="#555555")
        fig.text(0.985, 0.925, "Data: Meteostat / DWD", ha="right",
                 fontsize=9, color="#555555")
        _draw_table(fig, table, lay)

    note = station.get("note")
    if note and lay["vertical"]:
        fig.text(0.5, lay["caption_y"], textwrap.fill(note, width=46),
                 ha="center", va="center", fontsize=lay["caption_size"],
                 style="italic", color="#666666")
    elif note:
        fig.text(ax_l, lay["caption_y"], note, ha="left", va="center",
                 fontsize=lay["caption_size"], style="italic", color="#666666")

    out = f"{OUTPUT_DIR}/yearly_cycle_{key}_{year}{lay['suffix']}.png"
    fig.savefig(out, dpi=lay["dpi"])
    plt.close(fig)
    print(f"📊 Saved {out}")
    return out


def _draw_keystats(fig, table: dict, info: dict, lay: dict) -> None:
    """Big mobile-friendly summary lines below the chart (vertical layouts)."""
    y1, y2 = lay["stat_y"]
    pct = round(info["warm_frac"] * 100)
    fig.text(0.5, y1,
             f"Warmer than normal on {pct}% of {info['n_days']} days so far",
             ha="center", fontsize=lay["sub_size"] + 2, fontweight="bold",
             color="#222222")
    ym, ya = table["year_mean"], table["year_anom"]
    if ym is not None and ya is not None:
        colour = RED if ya >= 0 else BLUE
        fig.text(0.5, y2,
                 f"Year-to-date mean daily max {ym:.1f} °C  "
                 f"({ya:+.1f} °C vs {REF_START}–{REF_END})",
                 ha="center", fontsize=lay["sub_size"], color=colour)


def _draw_table(fig, table: dict, lay: dict) -> None:
    """Render the monthly-mean / anomaly table beneath the axes (wide layout)."""
    ax_l = lay["axes"][0]
    ax_r = lay["axes"][0] + lay["axes"][2]
    y_header, y_mean, y_anom = lay["table_y"]
    jahr_x = lay["jahr_x"]

    def fx(data_x):
        return ax_l + (data_x - 1) / (DAYS_IN_YEAR - 1) * (ax_r - ax_l)

    sep = Line2D([ax_l - 0.02, 0.99], [y_header - 0.03, y_header - 0.03],
                 color="#cccccc", linewidth=0.8, transform=fig.transFigure)
    fig.add_artist(sep)
    fig.text(ax_l - 0.008, y_mean, "Monthly mean [°C]", ha="right",
             va="center", fontsize=9)
    fig.text(ax_l - 0.008, y_anom, "Anomaly [°C]", ha="right", va="center",
             fontsize=9)

    def _cell(x, mean, anom):
        fig.text(x, y_mean, _fmt(mean), ha="center", va="center", fontsize=9)
        colour = "#555555" if anom is None else (RED if anom >= 0 else BLUE)
        atxt = "-" if anom is None else f"{anom:+.1f}"
        fig.text(x, y_anom, atxt, ha="center", va="center", fontsize=9,
                 color=colour)

    for m in range(1, 13):
        _cell(fx(MONTH_CENTERS[m - 1]), table["means"][m], table["anomalies"][m])

    fig.text(jahr_x, y_header, "Year", ha="center", va="center", fontsize=10,
             fontweight="bold")
    _cell(jahr_x, table["year_mean"], table["year_anom"])


def main() -> None:
    """Render every configured station in all output formats."""
    for key in ("potsdam", "berlin-dahlem"):
        print(f"=== {STATIONS[key]['label']} ===")
        make_plot(key)


if __name__ == "__main__":
    main()

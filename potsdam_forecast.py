#!/usr/bin/env python3
"""ECMWF open-data 15-day ENS outlook for a point station (Phase-1 overlay).

Fetches the most recent *long* ECMWF ensemble run (the 00/12 UTC ``enfo`` runs
reach 360 h; the 06/18 UTC runs stop at 144 h), reads the 50 perturbed members'
daily maximum 2 m temperature at a station by bilinear interpolation, applies a
simple persisted rolling bias offset, and returns a percentile *plume* suitable
for overlaying on the annual climatology chart (see ``potsdam_yearly_cycle.py``).

Design choices and honesty constraints (free, real-time, CC-BY-4.0 data):

* Open data exposes only perturbed members (``type=pf``, numbers 1-50) for
  ``enfo``; there is no control member, so the plume uses the 50 members.
* The max-temperature field is ``mx2t3`` (3 h windows) out to 144 h and
  ``mx2t6`` (6 h windows) afterwards. The daily maximum is the max over the
  fetched afternoon windows (12-18 UTC), which contain the European daytime peak.
* Useful *daily* Tmax skill fades by ~day 10 (``SKILL_HORIZON_DAYS``); the plume
  is still drawn to day 15 for context, but the caller marks the skill horizon.
* Bias correction is a single additive offset learned from a persisted archive of
  past short-lead member-mean forecasts vs. observations. Until enough verifiable
  pairs accrue it is 0 and ``calibrated`` is False (the model is near-unbiased at
  Potsdam, so this is acceptable for a first overlay).

Requires ``ecmwf-opendata`` and ``cfgrib``/``eccodes`` (optional dependencies);
if they are missing or the fetch fails, callers should treat the forecast as
unavailable and render the chart without it.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_MEMBERS = 50  # perturbed ENS members exposed by open data (1..50)
FORECAST_DAYS = 15  # nominal horizon of the long enfo run
SKILL_HORIZON_DAYS = 10  # daily-Tmax skill is meaningful only to roughly here
SWITCH_STEP = 144  # <=144 h: mx2t3 (3-hourly); >144 h: mx2t6 (6-hourly)
SOURCE_LABEL = "ECMWF ENS (open data)"

# Bias-correction archive
CACHE_DIR = "results/potsdam_forecast"
BIAS_MIN_PAIRS = 10  # need at least this many verified pairs before correcting
BIAS_WINDOW_DAYS = 45  # rolling window of recent verifications used for the offset
BIAS_LEADS = (1, 2)  # short leads whose error best estimates the systematic bias

_INDEX_URL = (
    "https://data.ecmwf.int/forecasts/{ymd}/{hh:02d}z/ifs/0p25/enfo/"
    "{ymd}{hh:02d}0000-{step}h-enfo-ef.index"
)


@dataclass
class PointForecast:
    """A station point forecast as an ensemble plume of daily maximum temp."""

    run: datetime  # model run (init) time, UTC
    valid_dates: list  # list[date], one per forecast day
    members_c: np.ndarray  # [n_members, n_days] bias-corrected daily Tmax (degC)
    pct: dict  # {10,25,50,75,90: np.ndarray[n_days]} after bias correction
    bias_c: float  # additive offset applied (degC)
    calibrated: bool  # True once the offset is learned from enough pairs
    n_members: int
    skill_horizon_days: int = SKILL_HORIZON_DAYS
    source: str = SOURCE_LABEL


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------
def _index_exists(run: datetime, step: int) -> bool:
    """Return True if the open-data index for ``run`` at ``step`` is published."""
    url = _INDEX_URL.format(ymd=run.strftime("%Y%m%d"), hh=run.hour, step=step)
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_latest_long_run(now: datetime | None = None) -> datetime:
    """Find the most recent 00/12 UTC enfo run whose full 360 h is published.

    The 06/18 UTC runs only reach 144 h, so they are skipped; we require the
    360 h index to exist to be sure the long run has finished publishing.
    """
    now = now or datetime.now(timezone.utc)
    base = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    candidates = []
    for back in range(0, 3):
        day = base - timedelta(days=back)
        for hh in (12, 0):
            run = day.replace(hour=hh)
            if run <= now:
                candidates.append(run)
    for run in sorted(candidates, reverse=True):
        if _index_exists(run, 24 * (FORECAST_DAYS - 1) + 18):
            return run
    raise RuntimeError("No fully-published long ENS run found in the last 3 days")


def _candidate_steps(run_hour: int) -> dict:
    """Map each max-temp param to the afternoon forecast steps to fetch.

    Returns ``{"mx2t3": [...], "mx2t6": [...]}`` selecting only windows whose
    valid time ends in the 12-18 UTC afternoon (15 or 18 UTC for 3 h windows,
    18 UTC for 6 h windows), which bracket the European daily maximum.
    """
    steps3, steps6 = [], []
    for step in range(3, SWITCH_STEP + 1, 3):  # 3-hourly portion
        if (run_hour + step) % 24 in (15, 18):
            steps3.append(step)
    last = 24 * FORECAST_DAYS + run_hour  # cover up to ~15 days of valid time
    for step in range(SWITCH_STEP + 6, last + 1, 6):  # 6-hourly portion
        if (run_hour + step) % 24 == 18:
            steps6.append(step)
    return {"mx2t3": steps3, "mx2t6": steps6}


# ---------------------------------------------------------------------------
# Fetch + interpolate
# ---------------------------------------------------------------------------
def _retrieve(run: datetime, param: str, steps: list, n_members: int,
              target: str) -> str:
    """Download one max-temp param (all members, given steps) to ``target``."""
    from ecmwf.opendata import Client

    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    client = Client(source="ecmwf")
    client.retrieve(
        date=run.strftime("%Y%m%d"),
        time=run.hour,
        type="pf",
        stream="enfo",
        param=param,
        step=steps,
        number=list(range(1, n_members + 1)),
        target=target,
    )
    return target


def _interp_members(grib_path: str, lat: float, lon: float) -> pd.DataFrame:
    """Bilinearly interpolate a multi-member/-step GRIB to a point.

    Returns a DataFrame indexed by ensemble ``number`` with one column per valid
    time (values in Kelvin). Empty DataFrame if the file holds no data.
    """
    import xarray as xr

    ds = xr.open_dataset(grib_path, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    var = [v for v in ds.data_vars][0]
    lon_q = lon % 360 if float(ds.longitude.max()) > 180 else lon
    da = ds[var].interp(latitude=lat, longitude=lon_q, method="linear")
    if "step" not in da.dims:  # single step -> add the dim back
        da = da.expand_dims("step")
    da = da.transpose("number", "step")
    valid = pd.to_datetime(np.atleast_1d(da["valid_time"].values))
    return pd.DataFrame(da.values, index=np.asarray(da["number"].values),
                        columns=valid)


def _daily_max_matrix(frames: list) -> tuple:
    """Combine member frames into a [member, day] daily-max matrix (deg C).

    Takes, per member and per calendar day, the maximum over all fetched
    afternoon windows of that day, then converts Kelvin to Celsius.
    """
    combined = pd.concat([f for f in frames if not f.empty], axis=1)
    combined = combined.sort_index(axis=1)
    col_dates = np.array([pd.Timestamp(c).date() for c in combined.columns])
    valid_dates = sorted(set(col_dates))
    members = combined.index.to_numpy()
    out = np.full((len(members), len(valid_dates)), np.nan)
    for j, d in enumerate(valid_dates):
        block = combined.iloc[:, col_dates == d].to_numpy()
        out[:, j] = np.nanmax(block, axis=1)
    return members, valid_dates, out - 273.15


# ---------------------------------------------------------------------------
# Bias correction (persisted rolling offset)
# ---------------------------------------------------------------------------
def _archive_path() -> str:
    return os.path.join(CACHE_DIR, "fc_archive.csv")


def _update_archive(run: datetime, valid_dates: list,
                    member_mean_c: np.ndarray) -> pd.DataFrame:
    """Append this run's member-mean point forecast to the on-disk archive.

    One row per (init_date, valid_date); re-running the same init replaces it.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = pd.DataFrame({
        "init_date": [run.date().isoformat()] * len(valid_dates),
        "valid_date": [d.isoformat() for d in valid_dates],
        "lead_days": [(d - run.date()).days for d in valid_dates],
        "fc_membmean_c": np.round(member_mean_c, 3),
    })
    path = _archive_path()
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["init_date"] != run.date().isoformat()]
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(path, index=False)
    return rows


def _compute_bias(archive: pd.DataFrame, obs_series: pd.Series | None,
                  run: datetime) -> tuple:
    """Learn an additive offset from recent short-lead forecast errors.

    Matches archived short-lead (``BIAS_LEADS``) member-mean forecasts against
    observed daily maxima over the last ``BIAS_WINDOW_DAYS`` and returns the
    median ``obs - forecast`` error. Falls back to ``(0.0, False)`` until at
    least ``BIAS_MIN_PAIRS`` verifiable pairs exist.
    """
    if obs_series is None or archive.empty:
        return 0.0, False
    obs_by_date = {ts.date(): float(v)
                   for ts, v in obs_series.dropna().items()}
    cutoff = run.date() - timedelta(days=BIAS_WINDOW_DAYS)
    errors = []
    for _, r in archive.iterrows():
        if r["lead_days"] not in BIAS_LEADS:
            continue
        vd = date.fromisoformat(r["valid_date"])
        if vd < cutoff or vd >= run.date():  # only verifiable past days
            continue
        if vd in obs_by_date:
            errors.append(obs_by_date[vd] - float(r["fc_membmean_c"]))
    if len(errors) < BIAS_MIN_PAIRS:
        return 0.0, False
    return float(np.median(errors)), True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_point_forecast(lat: float, lon: float,
                       obs_series: pd.Series | None = None,
                       n_members: int = N_MEMBERS,
                       run: datetime | None = None,
                       cache: bool = True) -> PointForecast:
    """Fetch and post-process the ENS daily-Tmax plume for a station point.

    Args:
        lat: Station latitude (deg N).
        lon: Station longitude (deg E).
        obs_series: Observed daily-max series (date-indexed) used to learn the
            bias offset; if None, no correction is applied.
        n_members: Number of perturbed members to use (<= 50).
        run: Force a specific model run (UTC); defaults to the latest long run.
        cache: Reuse cached GRIB downloads / processed arrays for the run.

    Returns:
        A :class:`PointForecast`.
    """
    run = run or find_latest_long_run()
    runstamp = run.strftime("%Y%m%d%H")
    grib_dir = os.path.join(CACHE_DIR, "grib")
    os.makedirs(grib_dir, exist_ok=True)
    proc = os.path.join(grib_dir, f"{runstamp}_{n_members}m.npz")

    if cache and os.path.exists(proc):
        z = np.load(proc, allow_pickle=True)
        valid_dates = [d for d in z["valid_dates"].tolist()]
        members_c = z["members_c"]
    else:
        plan = _candidate_steps(run.hour)
        frames = []
        for param, steps in plan.items():
            if not steps:
                continue
            target = os.path.join(grib_dir, f"{runstamp}_{param}_{n_members}m.grib2")
            _retrieve(run, param, steps, n_members, target if cache else target)
            frames.append(_interp_members(target, lat, lon))
        _, valid_dates, members_c = _daily_max_matrix(frames)
        if cache:
            np.savez(proc, valid_dates=np.array(valid_dates, dtype=object),
                     members_c=members_c)

    member_mean = np.nanmean(members_c, axis=0)
    archive = _update_archive(run, valid_dates, member_mean)
    bias, calibrated = _compute_bias(archive, obs_series, run)

    corrected = members_c + bias
    levels = [10, 25, 50, 75, 90]
    pct = {p: np.nanpercentile(corrected, p, axis=0) for p in levels}

    return PointForecast(
        run=run, valid_dates=valid_dates, members_c=corrected, pct=pct,
        bias_c=bias, calibrated=calibrated, n_members=members_c.shape[0],
    )


def main() -> None:
    """Print a quick summary of the latest Potsdam point forecast."""
    fc = get_point_forecast(52.3833, 13.0667)
    print(f"Run: {fc.run:%Y-%m-%d %HZ}  members={fc.n_members}  "
          f"bias={fc.bias_c:+.2f} degC calibrated={fc.calibrated}")
    for i, d in enumerate(fc.valid_dates):
        print(f"  {d}  lead {(d - fc.run.date()).days:>2}d  "
              f"median {fc.pct[50][i]:5.1f}  "
              f"10-90 [{fc.pct[10][i]:5.1f}, {fc.pct[90][i]:5.1f}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple Iran Precipitation Plot
==============================

Creates a clear visualization of yearly precipitation for all available
Iranian weather stations from the full historical record (1950-2025).

This script fetches data from NOAA GHCN-Daily using verified station IDs.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Verified Iranian GHCN Station IDs
IRAN_STATIONS = {
    "Tehran": "IR000407540",
    "Mashhad": "IR000040745",
    "Isfahan": "IRM00040800",
    "Tabriz": "IR000040706",
    "Shiraz": "IR000040848",
    "Ahvaz": "IRM00040811",
    "Kerman": "IR000040841",
    "Zahedan": "IR000408560",
    "Bandar Abbas": "IRM00040875",
    "Kermanshah": "IR000407660",
    "Yazd": "IRM00040821",
    "Bushehr": "IRM00040858",
}

GHCN_BASE_URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/"


def fetch_station_data(station_id: str, station_name: str) -> pd.DataFrame:
    """Fetch precipitation data for a station."""
    url = f"{GHCN_BASE_URL}{station_id}.csv"
    print(f"  Fetching {station_name} ({station_id})...", end=" ")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        df['DATE'] = pd.to_datetime(df['DATE'])

        # Extract precipitation (PRCP in tenths of mm)
        if 'PRCP' in df.columns:
            df['precipitation_mm'] = df['PRCP'] / 10.0
            df.loc[df['precipitation_mm'] < 0, 'precipitation_mm'] = np.nan

            # Calculate annual totals
            df['year'] = df['DATE'].dt.year
            annual = df.groupby('year')['precipitation_mm'].sum().reset_index()
            annual.columns = ['year', 'precipitation_mm']
            annual['station'] = station_name

            print(f"OK ({len(annual)} years, {annual['year'].min()}-{annual['year'].max()})")
            return annual
        else:
            print("No PRCP column")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def main():
    print("=" * 70)
    print("IRAN PRECIPITATION DATA - ALL AVAILABLE STATIONS")
    print("=" * 70)
    print(f"\nFetching data from {len(IRAN_STATIONS)} verified GHCN stations...\n")

    # Fetch all station data
    all_data = []
    for name, station_id in IRAN_STATIONS.items():
        df = fetch_station_data(station_id, name)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("\nNo data fetched. Check network connection.")
        return

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    stations_with_data = combined['station'].unique()

    print(f"\n{'=' * 70}")
    print(f"Successfully fetched data from {len(stations_with_data)} stations")
    print(f"{'=' * 70}")

    # Create output directory
    output_dir = Path("results/iran_precipitation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PLOT 1: Simple multi-line yearly precipitation
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 10))

    colors = plt.cm.tab20(np.linspace(0, 1, len(stations_with_data)))

    for i, station in enumerate(sorted(stations_with_data)):
        station_data = combined[combined['station'] == station]
        ax.plot(station_data['year'], station_data['precipitation_mm'],
                label=station, linewidth=1.5, alpha=0.8, color=colors[i])

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Annual Precipitation - All Iranian Weather Stations (NOAA GHCN-Daily)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1950, 2025)

    plt.tight_layout()
    plt.savefig(output_dir / "iran_all_stations_precipitation.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'iran_all_stations_precipitation.png'}")
    plt.close()

    # =========================================================================
    # PLOT 2: Faceted plot - one subplot per station
    # =========================================================================
    n_stations = len(stations_with_data)
    n_cols = 3
    n_rows = (n_stations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, station in enumerate(sorted(stations_with_data)):
        ax = axes[i]
        station_data = combined[combined['station'] == station].sort_values('year')

        # Calculate baseline mean (1981-2010)
        baseline = station_data[(station_data['year'] >= 1981) & (station_data['year'] <= 2010)]
        baseline_mean = baseline['precipitation_mm'].mean() if len(baseline) > 0 else station_data['precipitation_mm'].mean()

        # Color bars by deficit/surplus
        colors_bar = ['#d62728' if p < baseline_mean else '#2ca02c'
                      for p in station_data['precipitation_mm']]

        ax.bar(station_data['year'], station_data['precipitation_mm'],
               color=colors_bar, alpha=0.7, width=0.8)
        ax.axhline(baseline_mean, color='black', linestyle='--', linewidth=1.5,
                  label=f'Baseline: {baseline_mean:.0f} mm')

        ax.set_title(station, fontsize=11, fontweight='bold')
        ax.set_ylabel('mm')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)

    # Hide empty subplots
    for i in range(n_stations, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Annual Precipitation by Station (Red = Below Baseline, Green = Above)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "iran_stations_faceted.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'iran_stations_faceted.png'}")
    plt.close()

    # =========================================================================
    # PLOT 3: Mean across all stations with trend
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate mean precipitation across all stations per year
    yearly_mean = combined.groupby('year')['precipitation_mm'].agg(['mean', 'std', 'count']).reset_index()
    yearly_mean.columns = ['year', 'mean_prcp', 'std_prcp', 'n_stations']

    # Only use years with at least 3 stations
    yearly_mean = yearly_mean[yearly_mean['n_stations'] >= 3]

    # Plot with error bars
    ax.fill_between(yearly_mean['year'],
                    yearly_mean['mean_prcp'] - yearly_mean['std_prcp'],
                    yearly_mean['mean_prcp'] + yearly_mean['std_prcp'],
                    alpha=0.3, color='steelblue', label='±1 Std Dev')
    ax.plot(yearly_mean['year'], yearly_mean['mean_prcp'],
            color='steelblue', linewidth=2, label='Mean across stations')

    # Add trend line
    z = np.polyfit(yearly_mean['year'], yearly_mean['mean_prcp'], 1)
    p = np.poly1d(z)
    ax.plot(yearly_mean['year'], p(yearly_mean['year']),
            'r--', linewidth=2, label=f'Trend: {z[0]:.2f} mm/year')

    # Highlight 2018-2025
    ax.axvspan(2018, 2025, alpha=0.2, color='red', label='2018-2025 Period')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Iran Mean Annual Precipitation (Average Across All Stations)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "iran_mean_precipitation_trend.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'iran_mean_precipitation_trend.png'}")
    plt.close()

    # =========================================================================
    # Export data to CSV
    # =========================================================================
    combined.to_csv(output_dir / "iran_all_stations_annual_precipitation.csv", index=False)
    print(f"Saved: {output_dir / 'iran_all_stations_annual_precipitation.csv'}")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 70}")

    for station in sorted(stations_with_data):
        station_data = combined[combined['station'] == station]
        recent = station_data[station_data['year'] >= 2018]['precipitation_mm'].mean()
        baseline = station_data[(station_data['year'] >= 1981) & (station_data['year'] <= 2010)]['precipitation_mm'].mean()
        if baseline > 0:
            change = ((recent - baseline) / baseline) * 100
            status = "DEFICIT" if change < 0 else "SURPLUS"
            print(f"{station:15s}: {station_data['year'].min()}-{station_data['year'].max()} | "
                  f"Baseline: {baseline:6.1f} mm | 2018-2025: {recent:6.1f} mm | {change:+5.1f}% ({status})")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Outputs saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

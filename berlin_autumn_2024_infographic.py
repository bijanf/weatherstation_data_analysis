#!/usr/bin/env python3
"""
Berlin's Disappearing Frost - October/November 2024 Infographic
================================================================

Creates a dramatic infographic showing record-breaking warm autumn 2024
across the Berlin region using multiple weather stations.

Optimized for social media sharing on Bluesky.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
from meteostat import Stations, Daily
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================

STATIONS_CONFIG = {
    "Berlin-Tempelhof": {"lat": 52.4675, "lon": 13.4021},
    "Berlin-Tegel": {"lat": 52.5597, "lon": 13.2877},
    "Potsdam": {"lat": 52.3833, "lon": 13.0667},
    "Schönefeld": {"lat": 52.3906, "lon": 13.5226}
}

ANALYSIS_MONTHS = [10, 11]  # October and November
HISTORICAL_START_YEAR = 1893
HISTORICAL_END_YEAR = 2026
TARGET_YEAR = 2024


# ==================== DATA FETCHING ====================

def fetch_station_data(station_name, lat, lon, year, months):
    """
    Fetch temperature data for specific months and year from a station.

    Args:
        station_name: Name of the station
        lat: Latitude
        lon: Longitude
        year: Year to fetch
        months: List of month numbers (e.g., [10, 11] for Oct/Nov)

    Returns:
        DataFrame with temperature data or None
    """
    print(f"📡 Fetching {year} data for {station_name}...")

    try:
        # Find station
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)

        if station.empty:
            print(f"❌ No station found near {station_name}")
            return None

        station_id = station.index[0]

        # Fetch data for each month and combine
        all_month_data = []
        for month in months:
            # Determine last day of month
            if month in [1, 3, 5, 7, 8, 10, 12]:
                last_day = 31
            elif month in [4, 6, 9, 11]:
                last_day = 30
            else:  # February
                last_day = 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28

            start = datetime(year, month, 1)
            end = datetime(year, month, last_day)

            data = Daily(station_id, start, end)
            data = data.fetch()

            if not data.empty and 'tmin' in data.columns:
                all_month_data.append(data[['tmin']])

        if all_month_data:
            combined_data = pd.concat(all_month_data)
            print(f"✅ {station_name}: {len(combined_data)} days retrieved")
            return combined_data
        else:
            print(f"❌ {station_name}: No temperature data")
            return None

    except Exception as e:
        print(f"❌ {station_name}: Error - {e}")
        return None


def fetch_multi_station_data(year, months):
    """
    Fetch data from all configured stations for given year and months.

    Returns:
        Dictionary mapping station names to DataFrames
    """
    print(f"\n{'='*60}")
    print(f"FETCHING DATA FOR {year}")
    print(f"Months: {', '.join([datetime(2000, m, 1).strftime('%B') for m in months])}")
    print(f"{'='*60}\n")

    station_data = {}
    for station_name, coords in STATIONS_CONFIG.items():
        data = fetch_station_data(
            station_name,
            coords['lat'],
            coords['lon'],
            year,
            months
        )
        if data is not None:
            station_data[station_name] = data

    print(f"\n✅ Successfully fetched data from {len(station_data)}/{len(STATIONS_CONFIG)} stations")
    return station_data


def fetch_historical_frost_days(station_name, lat, lon, months, start_year=1893, end_year=2026):
    """
    Fetch historical frost day counts for October/November.

    Returns:
        Dictionary mapping year to frost day count
    """
    print(f"\n📊 Fetching historical data for {station_name}...")

    try:
        # Find station
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)

        if station.empty:
            print(f"❌ No station found")
            return {}

        station_id = station.index[0]
        frost_days_by_year = {}

        for year in range(start_year, end_year):
            if year % 10 == 0:
                print(f"  Processing {year}...")

            all_month_data = []
            for month in months:
                if month in [1, 3, 5, 7, 8, 10, 12]:
                    last_day = 31
                elif month in [4, 6, 9, 11]:
                    last_day = 30
                else:
                    last_day = 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28

                try:
                    start = datetime(year, month, 1)
                    end = datetime(year, month, last_day)

                    data = Daily(station_id, start, end)
                    data = data.fetch()

                    if not data.empty and 'tmin' in data.columns:
                        all_month_data.append(data[['tmin']])
                except:
                    continue

            if all_month_data:
                combined_data = pd.concat(all_month_data)
                # Count frost days (days with tmin < 0)
                frost_days = (combined_data['tmin'] < 0.0).sum()
                total_days = len(combined_data['tmin'].dropna())

                # Only include if we have at least 50 days of data (out of ~61)
                if total_days >= 50 or year == 2024:
                    frost_days_by_year[year] = frost_days

        print(f"✅ Retrieved {len(frost_days_by_year)} years of historical data")
        return frost_days_by_year

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


# ==================== STATISTICAL ANALYSIS ====================

def calculate_percentile_ranking(values_dict, target_year):
    """
    Calculate percentile ranking for target year.

    Args:
        values_dict: Dictionary mapping year to value
        target_year: Year to rank

    Returns:
        Dictionary with ranking statistics
    """
    if target_year not in values_dict:
        return None

    target_value = values_dict[target_year]
    all_values = list(values_dict.values())

    # Count how many values are greater than or equal to target
    # (for frost days, lower is warmer/more unusual)
    values_below = sum(1 for v in all_values if v < target_value)
    values_equal = sum(1 for v in all_values if v == target_value)

    # Percentile (lower frost days = lower percentile = warmer)
    percentile = ((values_below + values_equal / 2) / len(all_values)) * 100

    # Rank (1 = lowest frost days = warmest)
    rank = sum(1 for v in all_values if v < target_value) + 1

    return {
        'value': target_value,
        'percentile': percentile,
        'rank': rank,
        'total_years': len(all_values),
        'warmest_year': min(values_dict.items(), key=lambda x: x[1])[0],
        'warmest_value': min(values_dict.values()),
        'coldest_year': max(values_dict.items(), key=lambda x: x[1])[0],
        'coldest_value': max(values_dict.values())
    }


def find_last_occurrence(values_dict, threshold, comparison='below'):
    """
    Find the last year a threshold was crossed.

    Args:
        values_dict: Dictionary mapping year to value
        threshold: Threshold value
        comparison: 'below' or 'above'

    Returns:
        Last year meeting condition, or None
    """
    matching_years = []
    for year, value in values_dict.items():
        if comparison == 'below' and value < threshold:
            matching_years.append(year)
        elif comparison == 'above' and value > threshold:
            matching_years.append(year)

    if matching_years:
        return max(matching_years)
    return None


def detect_records(current_value, historical_dict):
    """
    Detect if current value is a record.

    Args:
        current_value: Current year's value
        historical_dict: Dictionary of historical values

    Returns:
        String describing record status
    """
    min_val = min(historical_dict.values())
    max_val = max(historical_dict.values())

    if current_value < min_val:
        return "ALL-TIME RECORD LOW"
    elif current_value == min_val:
        min_years = [year for year, val in historical_dict.items() if val == min_val]
        return f"TIES RECORD ({', '.join(map(str, sorted(min_years)))})"
    elif current_value > max_val:
        return "ALL-TIME RECORD HIGH"
    else:
        return "Not a record"


def calculate_era_comparison(values_dict):
    """
    Compare different historical eras.

    Returns:
        Dictionary with era statistics
    """
    era_1 = [v for y, v in values_dict.items() if 1893 <= y < 1950]
    era_2 = [v for y, v in values_dict.items() if 1950 <= y < 2000]
    era_3 = [v for y, v in values_dict.items() if 2000 <= y <= 2024]

    return {
        '1893-1949': {'mean': np.mean(era_1), 'std': np.std(era_1), 'n': len(era_1)},
        '1950-1999': {'mean': np.mean(era_2), 'std': np.std(era_2), 'n': len(era_2)},
        '2000-2024': {'mean': np.mean(era_3), 'std': np.std(era_3), 'n': len(era_3)}
    }


# ==================== VISUALIZATION ====================

def create_infographic(station_data_2024, historical_data, output_path='plots/berlin_autumn_2024_bluesky.png'):
    """
    Create the mega-infographic for Bluesky.

    Args:
        station_data_2024: Dictionary with 2024 data for each station
        historical_data: Dictionary with historical frost day counts
        output_path: Where to save the infographic
    """
    # Calculate statistics
    print("\n📊 Calculating statistics...")

    # Count 2024 frost days per station
    frost_days_2024 = {}
    for station_name, data in station_data_2024.items():
        frost_days = (data['tmin'] < 0.0).sum()
        frost_days_2024[station_name] = frost_days
        print(f"  {station_name}: {frost_days} frost days")

    # Historical statistics
    ranking = calculate_percentile_ranking(historical_data, 2024)
    era_stats = calculate_era_comparison(historical_data)
    record_status = detect_records(historical_data.get(2024, 0),
                                   {k: v for k, v in historical_data.items() if k != 2024})

    # Find last time we had this few or fewer frost days
    if 2024 in historical_data:
        last_time = find_last_occurrence(
            {k: v for k, v in historical_data.items() if k < 2024},
            historical_data[2024],
            'below'
        )
    else:
        last_time = None

    # Historical mean
    historical_mean = np.mean([v for y, v in historical_data.items() if y != 2024])

    # Create the figure
    print("\n🎨 Creating infographic...")
    fig = plt.figure(figsize=(12, 16), facecolor='white')
    gs = GridSpec(5, 1, height_ratios=[1, 1.5, 2, 1.5, 0.3], hspace=0.35)

    # ============ PANEL 1: HEADLINE ============
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')

    # Main title
    ax_title.text(0.5, 0.7, "BERLIN'S DISAPPEARING FROST",
                 ha='center', va='center', fontsize=32, fontweight='black',
                 color='#1a1a1a')

    # Subtitle
    ax_title.text(0.5, 0.35, "October/November 2024 — Record-Breaking Warmth Across Region",
                 ha='center', va='center', fontsize=16, fontweight='normal',
                 color='#4a4a4a', style='italic')

    # Year span
    ax_title.text(0.5, 0.05, f"Historical Analysis: {HISTORICAL_START_YEAR}-2024 ({2024 - HISTORICAL_START_YEAR + 1} years)",
                 ha='center', va='center', fontsize=12, fontweight='normal',
                 color='#666666')

    # ============ PANEL 2: MULTI-STATION COMPARISON ============
    ax_bars = fig.add_subplot(gs[1])

    stations = list(frost_days_2024.keys())
    frost_2024_values = [frost_days_2024[s] for s in stations]
    historical_avg = [historical_mean] * len(stations)

    x = np.arange(len(stations))
    width = 0.35

    bars1 = ax_bars.bar(x - width/2, frost_2024_values, width,
                       label='Oct/Nov 2024', color='#d62728', alpha=0.9, edgecolor='darkred', linewidth=2)
    bars2 = ax_bars.bar(x + width/2, historical_avg, width,
                       label=f'Historical Avg ({HISTORICAL_START_YEAR}-2023)', color='#4a90e2', alpha=0.7)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        ax_bars.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                    f'{int(height1)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=14, color='#d62728')

        ax_bars.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                    f'{height2:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11, color='#4a90e2')

    ax_bars.set_ylabel('Days with Temperature < 0°C', fontsize=13, fontweight='bold')
    ax_bars.set_title('Frost Days Comparison — 4 Berlin Region Stations',
                     fontsize=14, fontweight='bold', pad=15)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([s.replace('Berlin-', 'B-') for s in stations],
                           rotation=15, ha='right', fontsize=11)
    ax_bars.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True)
    ax_bars.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bars.set_ylim(0, max(historical_avg) * 1.2)

    # ============ PANEL 3: HISTORICAL TIMELINE ============
    ax_timeline = fig.add_subplot(gs[2])

    years = sorted([y for y in historical_data.keys() if y != 2024])
    values = [historical_data[y] for y in years]

    # Plot historical data
    ax_timeline.plot(years, values, color='#cccccc', linewidth=1.5, alpha=0.6, zorder=1)
    ax_timeline.scatter(years, values, color='#4a90e2', s=30, alpha=0.5, zorder=2)

    # Highlight 2024
    if 2024 in historical_data:
        ax_timeline.scatter([2024], [historical_data[2024]], color='#d62728', s=400,
                          marker='*', edgecolors='darkred', linewidths=2, zorder=5,
                          label='2024')

    # Add trend line
    if len(years) > 1:
        z = np.polyfit(years, values, 1)
        p = np.poly1d(z)
        ax_timeline.plot(years, p(years), "k--", alpha=0.5, linewidth=2,
                        label=f'Trend: {z[0]:.3f} days/year', zorder=3)

    # Shade eras
    ax_timeline.axvspan(1893, 1949, alpha=0.1, color='blue', label='1893-1949')
    ax_timeline.axvspan(1950, 1999, alpha=0.1, color='green')
    ax_timeline.axvspan(2000, 2024, alpha=0.1, color='red', label='2000-2024')

    # Add historical mean line
    ax_timeline.axhline(y=historical_mean, color='#4a90e2', linestyle=':',
                       linewidth=2, alpha=0.7, label=f'Historical Mean: {historical_mean:.1f}')

    ax_timeline.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax_timeline.set_ylabel('Frost Days (Oct + Nov)', fontsize=13, fontweight='bold')
    ax_timeline.set_title(f'132 Years of October/November Frost Days — Potsdam Station',
                         fontsize=14, fontweight='bold', pad=15)
    ax_timeline.grid(True, alpha=0.3, linewidth=0.5)
    ax_timeline.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, ncol=2)
    ax_timeline.set_xlim(1890, 2027)
    ax_timeline.set_ylim(-2, max(values) * 1.1)

    # ============ PANEL 4: RECORD-BREAKING CALLOUTS ============
    ax_stats = fig.add_subplot(gs[3])
    ax_stats.axis('off')

    # Create 4 callout boxes
    box_props = dict(boxstyle='round,pad=0.8', facecolor='#fff3cd', edgecolor='#856404', linewidth=2.5, alpha=0.9)

    callouts = []

    # Callout 1: Ranking
    if ranking:
        if ranking['rank'] <= 5:
            callout1 = f"🏆 TOP {ranking['rank']} WARMEST\nOut of {ranking['total_years']} years\n({ranking['percentile']:.1f}th percentile)"
        else:
            callout1 = f"📊 RANKED #{ranking['rank']}\nOut of {ranking['total_years']} years\n({ranking['percentile']:.1f}th percentile)"
        callouts.append(callout1)

    # Callout 2: First time since
    if last_time and 2024 in historical_data:
        years_ago = 2024 - last_time
        callout2 = f"🕐 LAST THIS WARM:\n{last_time}\n({years_ago} years ago)"
        callouts.append(callout2)
    elif not last_time and 2024 in historical_data:
        callout2 = f"⭐ UNPRECEDENTED\nFewest frost days\nSINCE RECORDS BEGAN"
        callouts.append(callout2)

    # Callout 3: Era comparison
    if era_stats:
        early = era_stats['1893-1949']['mean']
        recent = era_stats['2000-2024']['mean']
        change = ((recent - early) / early) * 100
        callout3 = f"📉 ERA COMPARISON\n2000s: {recent:.1f} days\n1900s: {early:.1f} days ({change:+.0f}%)"
        callouts.append(callout3)

    # Callout 4: 2024 specific
    avg_2024 = np.mean(list(frost_days_2024.values()))
    if avg_2024 == 0:
        callout4 = f"❄️ 2024 AVERAGE:\n0 FROST DAYS\nAcross all 4 stations"
    else:
        callout4 = f"❄️ 2024 AVERAGE:\n{avg_2024:.1f} frost days\nAcross 4 stations"
    callouts.append(callout4)

    # Position callouts
    positions = [(0.125, 0.5), (0.375, 0.5), (0.625, 0.5), (0.875, 0.5)]

    for (x, y), text in zip(positions, callouts):
        ax_stats.text(x, y, text,
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     bbox=box_props, family='monospace', color='#1a1a1a')

    # ============ PANEL 5: ATTRIBUTION ============
    ax_attr = fig.add_subplot(gs[4])
    ax_attr.axis('off')

    ax_attr.text(0.5, 0.5,
                f"Data: Meteostat/DWD • Stations: {', '.join([s.replace('Berlin-', 'B-') for s in STATIONS_CONFIG.keys()])} • Updated: {datetime.now().strftime('%d.%m.%Y')}",
                ha='center', va='center', fontsize=10, style='italic', color='#666666')

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Infographic saved: {output_path}")

    return fig, {
        'frost_days_2024': frost_days_2024,
        'ranking': ranking,
        'era_stats': era_stats,
        'record_status': record_status,
        'last_similar': last_time,
        'historical_mean': historical_mean
    }


def generate_bluesky_caption(stats):
    """
    Generate a punchy caption for Bluesky post.

    Args:
        stats: Statistics dictionary from create_infographic

    Returns:
        String caption
    """
    caption = "🔥 BERLIN'S VANISHING FROST 🔥\n\n"

    frost_avg = np.mean(list(stats['frost_days_2024'].values()))
    caption += f"Oct/Nov 2024: {frost_avg:.0f} frost days across Berlin region\n"
    caption += f"Historical avg: {stats['historical_mean']:.1f} days\n\n"

    if stats['ranking']:
        rank = stats['ranking']['rank']
        total = stats['ranking']['total_years']
        if rank <= 3:
            caption += f"🏆 #{rank} WARMEST autumn in {total} years of records!\n\n"
        else:
            caption += f"Ranks #{rank} out of {total} years (top {(rank/total*100):.0f}%)\n\n"

    if stats['last_similar']:
        years_ago = 2024 - stats['last_similar']
        caption += f"Last this warm: {stats['last_similar']} ({years_ago} years ago)\n\n"

    caption += "Climate change is here. The data speaks.\n\n"
    caption += "#ClimateChange #Berlin #Weather #DataViz"

    return caption


# ==================== MAIN ====================

def main():
    """
    Main function to create the Berlin autumn 2024 infographic.
    """
    print("🔥 BERLIN'S DISAPPEARING FROST — INFOGRAPHIC GENERATOR")
    print("="*60)

    # Fetch 2024 data from all stations
    station_data_2024 = fetch_multi_station_data(TARGET_YEAR, ANALYSIS_MONTHS)

    if not station_data_2024:
        print("\n❌ ERROR: Could not fetch 2024 data")
        return

    # Fetch historical data (use Potsdam for longest record)
    potsdam_coords = STATIONS_CONFIG['Potsdam']
    historical_data = fetch_historical_frost_days(
        'Potsdam',
        potsdam_coords['lat'],
        potsdam_coords['lon'],
        ANALYSIS_MONTHS,
        HISTORICAL_START_YEAR,
        HISTORICAL_END_YEAR
    )

    if not historical_data:
        print("\n❌ ERROR: Could not fetch historical data")
        return

    # Create infographic
    fig, stats = create_infographic(station_data_2024, historical_data)

    # Generate Bluesky caption
    caption = generate_bluesky_caption(stats)

    print("\n" + "="*60)
    print("📱 SUGGESTED BLUESKY CAPTION:")
    print("="*60)
    print(caption)
    print("="*60)

    # Save caption to file
    with open('plots/berlin_autumn_2024_caption.txt', 'w') as f:
        f.write(caption)
    print("\n✅ Caption saved to: plots/berlin_autumn_2024_caption.txt")

    print("\n🎉 INFOGRAPHIC COMPLETE!")
    print(f"📊 Image: plots/berlin_autumn_2024_bluesky.png")
    print(f"📝 Caption: plots/berlin_autumn_2024_caption.txt")
    print("\nReady to post on Bluesky! 🚀")


if __name__ == "__main__":
    main()

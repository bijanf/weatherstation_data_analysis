#!/usr/bin/env python3
"""
Berlin's Disappearing Frost - FROST SEASON ANALYSIS (Oct-Mar)
==============================================================

Shows systematic decline in frost season (Oct-Mar) frost days across decades.
Analyzes the full cold season from October through March.
More robust than single-year analysis, demonstrates clear climate trend.

Optimized for social media sharing on Bluesky.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
from meteostat import Stations, Daily
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================

STATIONS_CONFIG = {
    "Berlin-Tempelhof": {"lat": 52.4675, "lon": 13.4021},
    "Berlin-Tegel": {"lat": 52.5597, "lon": 13.2877},
    "Potsdam": {"lat": 52.3833, "lon": 13.0667},
    "Schönefeld": {"lat": 52.3906, "lon": 13.5226}
}

ANALYSIS_MONTHS = [10, 11, 12, 1, 2, 3]  # Oct, Nov, Dec, Jan, Feb, Mar (frost season)
HISTORICAL_START_YEAR = 1893
HISTORICAL_END_YEAR = 2025  # For Oct-Mar, we need complete seasons


# ==================== DATA FETCHING ====================

def fetch_station_data(station_name, lat, lon, year, months):
    """
    Fetch temperature data for specific months and year from a station.
    """
    try:
        # Find station
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)

        if station.empty:
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
            return combined_data
        else:
            return None

    except Exception as e:
        return None


def fetch_historical_frost_days(station_name, lat, lon, months, start_year=1893, end_year=2025):
    """
    Fetch historical frost day counts for frost season (Oct-Mar).

    For each season year, fetches:
    - Oct, Nov, Dec of year Y
    - Jan, Feb, Mar of year Y+1

    Returns:
        Dictionary mapping season year to frost day count
    """
    print(f"\n📊 Fetching historical frost season data for {station_name}...")

    try:
        # Find station
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)

        if station.empty:
            print(f"❌ No station found")
            return {}

        station_id = station.index[0]
        print(f"✅ Found station: {station.loc[station_id, 'name']}")

        frost_days_by_year = {}

        for year in range(start_year, end_year):
            if year % 10 == 0:
                print(f"  Processing {year}/{year+1} seasons...")

            all_month_data = []
            for month in months:
                # Determine which calendar year to use
                if month >= 10:  # Oct, Nov, Dec - use current year
                    calendar_year = year
                else:  # Jan, Feb, Mar - use next year
                    calendar_year = year + 1

                # Determine last day of month
                if month in [1, 3, 5, 7, 8, 10, 12]:
                    last_day = 31
                elif month in [4, 6, 9, 11]:
                    last_day = 30
                else:  # February
                    last_day = 29 if (calendar_year % 4 == 0 and calendar_year % 100 != 0) or (calendar_year % 400 == 0) else 28

                try:
                    start = datetime(calendar_year, month, 1)
                    end = datetime(calendar_year, month, last_day)

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

                # Only include if we have at least 150 days of data (out of ~181 for 6 months)
                if total_days >= 150:
                    frost_days_by_year[year] = frost_days

        print(f"✅ Retrieved {len(frost_days_by_year)} frost seasons")
        return frost_days_by_year

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


# ==================== DECADAL ANALYSIS ====================

def calculate_decadal_statistics(frost_days_dict):
    """
    Calculate decadal averages and statistics.

    Returns:
        Dictionary with decadal statistics
    """
    print("\n📊 Calculating decadal statistics...")

    # Group by decade
    decades = {}
    for year, frost_days in frost_days_dict.items():
        decade = (year // 10) * 10
        if decade not in decades:
            decades[decade] = []
        decades[decade].append(frost_days)

    # Calculate statistics for each decade
    decadal_stats = {}
    for decade, values in sorted(decades.items()):
        decadal_stats[decade] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'n_years': len(values),
            'values': values
        }
        print(f"  {decade}s: {decadal_stats[decade]['mean']:.1f} ± {decadal_stats[decade]['std']:.1f} days (n={decadal_stats[decade]['n_years']})")

    return decadal_stats


def calculate_trend_statistics(decadal_stats):
    """
    Calculate linear trend and statistical significance.

    Returns:
        Dictionary with trend statistics
    """
    decades = sorted(decadal_stats.keys())
    means = [decadal_stats[d]['mean'] for d in decades]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(decades, means)

    # Calculate total change
    first_decade_mean = decadal_stats[decades[0]]['mean']
    last_decade_mean = decadal_stats[decades[-1]]['mean']
    absolute_change = last_decade_mean - first_decade_mean
    percent_change = (absolute_change / first_decade_mean) * 100

    print(f"\n📈 TREND ANALYSIS:")
    print(f"  Slope: {slope:.3f} days/decade")
    print(f"  R² = {r_value**2:.3f}")
    print(f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '(not significant)'}")
    print(f"  Total change: {absolute_change:.1f} days ({percent_change:+.0f}%)")

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'absolute_change': absolute_change,
        'percent_change': percent_change,
        'first_decade': decades[0],
        'last_decade': decades[-1],
        'first_mean': first_decade_mean,
        'last_mean': last_decade_mean
    }


def compare_eras(decadal_stats):
    """
    Compare early vs recent eras.
    """
    # Define eras
    early_decades = [d for d in decadal_stats.keys() if d < 1950]
    recent_decades = [d for d in decadal_stats.keys() if d >= 2000]

    early_values = []
    for d in early_decades:
        early_values.extend(decadal_stats[d]['values'])

    recent_values = []
    for d in recent_decades:
        recent_values.extend(decadal_stats[d]['values'])

    early_mean = np.mean(early_values)
    recent_mean = np.mean(recent_values)

    # T-test
    t_stat, p_value = stats.ttest_ind(early_values, recent_values)

    change = recent_mean - early_mean
    percent_change = (change / early_mean) * 100

    print(f"\n🔄 ERA COMPARISON:")
    print(f"  Pre-1950: {early_mean:.1f} ± {np.std(early_values):.1f} days (n={len(early_values)})")
    print(f"  Post-2000: {recent_mean:.1f} ± {np.std(recent_values):.1f} days (n={len(recent_values)})")
    print(f"  Change: {change:.1f} days ({percent_change:+.0f}%)")
    print(f"  t-test p-value: {p_value:.4e} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

    return {
        'early_mean': early_mean,
        'early_std': np.std(early_values),
        'recent_mean': recent_mean,
        'recent_std': np.std(recent_values),
        'change': change,
        'percent_change': percent_change,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


# ==================== VISUALIZATION ====================

def create_decadal_infographic(decadal_stats, trend_stats, era_comparison,
                               historical_data, output_path='plots/berlin_autumn_decadal_bluesky.png'):
    """
    Create decadal analysis infographic for Bluesky.
    """
    print("\n🎨 Creating decadal infographic...")

    # Create the figure
    fig = plt.figure(figsize=(12, 16), facecolor='white')
    gs = GridSpec(4, 1, height_ratios=[1, 2, 2, 0.4], hspace=0.45)

    # ============ PANEL 1: HEADLINE ============
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')

    ax_title.text(0.5, 0.7, "BERLIN'S DISAPPEARING FROST",
                 ha='center', va='center', fontsize=32, fontweight='black',
                 color='#1a1a1a')

    ax_title.text(0.5, 0.35, "13 Decades of Frost Season Data (Oct-Mar) Reveal Dramatic Decline",
                 ha='center', va='center', fontsize=15, fontweight='normal',
                 color='#4a4a4a', style='italic')

    ax_title.text(0.5, 0.05, f"Systematic Analysis: 1890s-2020s • {sum([s['n_years'] for s in decadal_stats.values()])} frost seasons",
                 ha='center', va='center', fontsize=12, fontweight='normal',
                 color='#666666')

    # ============ PANEL 2: DECADAL BAR CHART ============
    ax_bars = fig.add_subplot(gs[1])

    decades = sorted(decadal_stats.keys())
    means = [decadal_stats[d]['mean'] for d in decades]
    stds = [decadal_stats[d]['std'] for d in decades]
    n_years = [decadal_stats[d]['n_years'] for d in decades]

    # Color gradient from blue (cold/many frost days) to red (warm/few frost days)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(decades)))

    x = np.arange(len(decades))
    bars = ax_bars.bar(x, means, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add error bars (standard deviation)
    ax_bars.errorbar(x, means, yerr=stds, fmt='none', ecolor='black',
                    capsize=5, capthick=2, alpha=0.6)

    # Add value labels on bars
    for i, (bar, mean, n) in enumerate(zip(bars, means, n_years)):
        height = bar.get_height()
        label = f'{mean:.1f}'
        ax_bars.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.5,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Add sample size below bar
        ax_bars.text(bar.get_x() + bar.get_width()/2., -1.8,
                    f'n={n}', ha='center', va='top', fontsize=8, style='italic', alpha=0.7)

    # Add trend line
    trend_line = [trend_stats['intercept'] + trend_stats['slope'] * d for d in decades]
    ax_bars.plot(x, trend_line, 'k--', linewidth=3, alpha=0.7)

    ax_bars.set_ylabel('Frost Days (Oct-Mar)', fontsize=13, fontweight='bold')
    ax_bars.set_title(f'Frost Season Averages by Decade: {trend_stats["absolute_change"]:.1f} Day Decline ({trend_stats["percent_change"]:+.0f}%)',
                     fontsize=15, fontweight='bold', pad=20)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([f"{d}s" for d in decades], rotation=45, ha='right', fontsize=10)
    ax_bars.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bars.set_ylim(-3, 90)  # Extended range to accommodate 6-month frost season

    # ============ PANEL 3: INDIVIDUAL YEARS SCATTER ============
    ax_scatter = fig.add_subplot(gs[2])

    years = sorted(historical_data.keys())
    values = [historical_data[y] for y in years]

    # Color by decade
    decade_colors = {}
    for d in decades:
        decade_colors[d] = plt.cm.RdYlBu_r((d - min(decades)) / (max(decades) - min(decades)))

    point_colors = [decade_colors[(y // 10) * 10] for y in years]

    ax_scatter.scatter(years, values, c=point_colors, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add decadal means as larger points
    for decade in decades:
        decade_years = [y for y in years if (y // 10) * 10 == decade]
        if decade_years:
            mid_year = np.mean(decade_years)
            ax_scatter.scatter([mid_year], [decadal_stats[decade]['mean']],
                             color=decade_colors[decade], s=300, marker='D',
                             edgecolors='black', linewidth=2, zorder=5)

    # Add trend line
    trend_line_full = [trend_stats['intercept'] + trend_stats['slope'] * y for y in years]
    ax_scatter.plot(years, trend_line_full, 'k--', linewidth=2.5, alpha=0.7)

    ax_scatter.set_xlabel('Frost Season Year', fontsize=13, fontweight='bold')
    ax_scatter.set_ylabel('Frost Days (Oct-Mar)', fontsize=13, fontweight='bold')
    ax_scatter.set_title('Individual Seasons (circles) & Decadal Means (diamonds)',
                        fontsize=14, fontweight='bold', pad=15)
    ax_scatter.grid(True, alpha=0.3, linewidth=0.5)
    ax_scatter.set_xlim(1890, 2027)
    ax_scatter.set_ylim(-2, max(values) * 1.1)

    # ============ PANEL 4: ATTRIBUTION ============
    ax_attr = fig.add_subplot(gs[3])
    ax_attr.axis('off')

    # Line 1: Data source
    ax_attr.text(0.5, 0.75,
                f"Data: Meteostat/DWD • Station: Potsdam Säkularstation • Updated: {datetime.now().strftime('%d.%m.%Y')}",
                ha='center', va='center', fontsize=10, style='italic', color='#666666')

    # Line 2: Author and repository
    ax_attr.text(0.5, 0.25,
                "Analysis: Bijan Fallah • github.com/bijanf/weatherstation_data_analysis",
                ha='center', va='center', fontsize=10, style='italic', color='#666666')

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Infographic saved: {output_path}")

    return fig


def generate_decadal_caption(decadal_stats, trend_stats, era_comparison):
    """
    Generate a punchy caption for Bluesky post emphasizing systematic change.
    """
    caption = "🔥 BERLIN'S VANISHING FROST — FROST SEASON ANALYSIS 🔥\n\n"

    caption += f"Frost season (Oct-Mar) over 13 decades reveals dramatic decline:\n\n"

    first_decade = trend_stats['first_decade']
    last_decade = trend_stats['last_decade']
    caption += f"❄️ {first_decade}s: {trend_stats['first_mean']:.1f} frost days per season\n"
    caption += f"🌡️ {last_decade}s: {trend_stats['last_mean']:.1f} frost days per season\n\n"

    caption += f"📉 {trend_stats['percent_change']:+.0f}% decline ({trend_stats['absolute_change']:.1f} days)\n"
    caption += f"📈 Trend: {trend_stats['slope']:.2f} days/decade\n\n"

    if trend_stats['p_value'] < 0.001:
        caption += "🔬 Statistically significant (p < 0.001)\n\n"

    caption += f"Pre-1950 avg: {era_comparison['early_mean']:.1f} days/season\n"
    caption += f"Post-2000 avg: {era_comparison['recent_mean']:.1f} days/season\n"
    caption += f"Difference: {era_comparison['percent_change']:+.0f}%\n\n"

    caption += "Winter is warming. The full frost season tells the story.\n\n"
    caption += f"Data: {sum([s['n_years'] for s in decadal_stats.values()])} frost seasons (Oct-Mar) • Potsdam station\n\n"
    caption += "#ClimateChange #Berlin #Weather #DataViz #Science #WinterWarming"

    return caption


# ==================== MAIN ====================

def main():
    """
    Main function to create the decadal analysis infographic.
    """
    print("🔥 BERLIN'S DISAPPEARING FROST — DECADAL ANALYSIS")
    print("="*60)

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

    # Calculate decadal statistics
    decadal_stats = calculate_decadal_statistics(historical_data)

    # Calculate trends
    trend_stats = calculate_trend_statistics(decadal_stats)

    # Compare eras
    era_comparison = compare_eras(decadal_stats)

    # Create infographic
    fig = create_decadal_infographic(decadal_stats, trend_stats, era_comparison, historical_data)

    # Generate Bluesky caption
    caption = generate_decadal_caption(decadal_stats, trend_stats, era_comparison)

    print("\n" + "="*60)
    print("📱 SUGGESTED BLUESKY CAPTION:")
    print("="*60)
    print(caption)
    print("="*60)

    # Save caption to file
    with open('plots/berlin_autumn_decadal_caption.txt', 'w') as f:
        f.write(caption)
    print("\n✅ Caption saved to: plots/berlin_autumn_decadal_caption.txt")

    print("\n🎉 DECADAL INFOGRAPHIC COMPLETE!")
    print(f"📊 Image: plots/berlin_autumn_decadal_bluesky.png")
    print(f"📝 Caption: plots/berlin_autumn_decadal_caption.txt")
    print("\n🔬 Scientifically robust • Ready for Bluesky! 🚀")


if __name__ == "__main__":
    main()

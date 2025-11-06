#!/usr/bin/env python3
"""
Quick script to update Potsdam precipitation plot using wetterdienst
Based on the working germany_centennial_climate_dwd.py approach
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import warnings
warnings.filterwarnings('ignore')

print("🌧️ Fetching Potsdam precipitation data from DWD...")

# Potsdam station ID
station_id = "02925"

try:
    # Request historical + recent precipitation data
    print("📡 Requesting data from DWD...")
    request = DwdObservationRequest(
        parameters="daily/climate_summary/precipitation_height",
        start_date=datetime(1890, 1, 1),
        end_date=datetime(2025, 11, 5)
    ).filter_by_station_id(station_id=station_id)

    # Fetch data
    print("📥 Downloading data...")
    values = request.values.all()
    df = values.df

    if df.empty:
        print("❌ No data available")
        exit(1)

    print(f"✅ Downloaded {len(df)} records")

    # Process data by year
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df[['date', 'year', 'value']].copy()
    df = df.dropna(subset=['value'])

    # Group by year
    all_data = {}
    for year in range(1890, 2026):
        year_data = df[df['year'] == year].copy()
        if len(year_data) > 0:
            year_data = year_data.set_index('date')['value']

            # Calculate coverage
            if year == 2025:
                expected_days = 309  # Jan 1 to Nov 5
            else:
                expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365

            coverage = (len(year_data) / expected_days) * 100

            # Only include if coverage >= 80% or year 2025
            if year == 2025 or coverage >= 80.0:
                # Fill missing dates with 0
                date_range = pd.date_range(f'{year}-01-01',
                                         f'{year}-11-05' if year == 2025 else f'{year}-12-31',
                                         freq='D')
                year_data = year_data.reindex(date_range, fill_value=0.0)
                all_data[year] = year_data
                status = "(partial)" if year == 2025 else ""
                print(f"  {year}: {year_data.sum():.1f}mm ({len(year_data)} days, {coverage:.0f}%) {status}")

    print(f"\n📊 Creating plot for {len(all_data)} years...")

    # Create plot
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    max_precip = 0
    years_list = sorted(all_data.keys())
    gray_plotted = False

    # Plot all years except 2018 and 2025 in gray
    for year in years_list:
        if year != 2025:
            daily_data = all_data[year]
            cumulative = daily_data.cumsum()
            day_of_year = [d.timetuple().tm_yday for d in daily_data.index]

            if year == 2018:
                ax.plot(day_of_year, cumulative,
                        color='#FF0000', linewidth=3, alpha=1.0, label='2018 (346mm)')
            else:
                label = f'{min(years_list)}-{max([y for y in years_list if y != 2025])} (all other years)' if not gray_plotted else None
                gray_plotted = True
                ax.plot(day_of_year, cumulative,
                        color='#808080', linewidth=1.0, alpha=0.4, label=label)

            max_precip = max(max_precip, cumulative.max())

    # Plot 2025 in blue
    if 2025 in all_data:
        daily_data = all_data[2025]
        cumulative = daily_data.cumsum()
        day_of_year = [d.timetuple().tm_yday for d in daily_data.index]

        ax.plot(day_of_year, cumulative,
                color='#000080', linewidth=3, alpha=1.0, label='2025')

        max_precip = max(max_precip, cumulative.max())

        ax.text(0.98, 0.08, f'Data: DWD (German Weather Service)',
               transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='darkblue', linewidth=2, alpha=0.95))

    # Customize plot
    ax.set_xlabel('Month', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=18, fontweight='bold')
    ax.set_title('Cumulative Daily Precipitation — Potsdam, Germany\n130+ Years of Climate History (1893-2025)',
                fontsize=18, fontweight='bold', pad=25)

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=16, fontweight='bold')

    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_ylim(0, max_precip * 1.1)
    ax.grid(True, alpha=0.4, linewidth=0.8)

    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=16)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    today = date.today()
    ax.text(0.98, 0.98, f'Updated: {today.strftime("%d.%m.%Y")}',
           transform=ax.transAxes, ha='right', va='top', fontsize=10,
           color='gray', alpha=0.7)

    plt.tight_layout()
    plt.savefig('plots/cumulative_precipitation_plot.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved to plots/cumulative_precipitation_plot.png")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

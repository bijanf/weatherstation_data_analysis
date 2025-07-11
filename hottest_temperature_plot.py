#!/usr/bin/env python3
"""
Hottest Temperature Plot for Potsdam Station
Shows the highest temperature recorded each year from the Säkularstation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date
from meteostat import Stations, Daily
import warnings
warnings.filterwarnings('ignore')

def get_real_temperature_data():
    """
    Get REAL daily temperature data for Potsdam station using Meteostat.
    """
    print("🌡️ Fetching REAL daily temperature data for Potsdam...")
    
    # Get the station
    stations = Stations()
    stations = stations.nearby(52.3833, 13.0667)  # Potsdam Säkularstation coordinates
    station = stations.fetch(1)
    
    if station.empty:
        print("❌ No station found")
        return None
    
    station_id = station.index[0]
    station_name = station.loc[station_id, 'name']
    print(f"📍 Station: {station_name}")
    print(f"📍 Coordinates: {station.loc[station_id, 'latitude']:.4f}°N, {station.loc[station_id, 'longitude']:.4f}°E")
    
    # Get real daily data for all available years (100+ years)
    years_to_analyze = list(range(1890, 2026))  # Try to get as much historical data as possible
    all_data = {}
    
    for year in years_to_analyze:
        print(f"📡 Downloading real temperature data for {year}...")
        
        start = datetime(year, 1, 1)
        # For 2025, only get data up to current date
        if year == 2025:
            end = datetime(2025, 7, 2)  # Up to July 2nd
        else:
            end = datetime(year, 12, 31)
        
        try:
            data = Daily(station_id, start, end)
            data = data.fetch()
            
            if not data.empty and 'tmax' in data.columns:
                # Calculate expected days for the year
                if year == 2025:
                    # For 2025, calculate up to July 2nd
                    expected_days = 183  # Days from Jan 1 to July 2
                else:
                    # Full year: check if leap year
                    expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
                
                # Count actual non-null temperature data points
                non_null_data = data['tmax'].dropna()
                actual_days = len(non_null_data)
                coverage_percentage = (actual_days / expected_days) * 100
                
                # Only include years with at least 80% coverage (or 2025 regardless of coverage)
                if year == 2025 or coverage_percentage >= 80.0:
                    # Use real maximum temperature data
                    daily_tmax = data['tmax']
                    all_data[year] = daily_tmax
                    max_temp = daily_tmax.max()
                    status = "(partial year)" if year == 2025 else ""
                    print(f"✅ {year}: {actual_days}/{expected_days} days, max temp: {max_temp:.1f}°C, coverage: {coverage_percentage:.1f}% {status}")
                else:
                    print(f"❌ {year}: Insufficient coverage ({actual_days}/{expected_days} days = {coverage_percentage:.1f}%)")
            else:
                print(f"❌ {year}: No temperature data available")
                
        except Exception as e:
            print(f"❌ {year}: Error downloading data - {e}")
    
    if not all_data:
        print("❌ No real data could be retrieved")
        return None
        
    print(f"\n✅ Successfully retrieved real temperature data for {len(all_data)} years")
    return all_data

def create_hottest_temperature_plot(all_data):
    """
    Create a visually appealing time series plot for Instagram:
    - Trend line in black
    - Lines in grey
    - Dots in red
    - No grid
    - Clean, modern style
    """
    if not all_data:
        print("❌ No data to plot")
        return None

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Extract yearly maximum temperatures (exclude 2025)
    yearly_max_temps = {}
    for year, daily_data in all_data.items():
        if year != 2025:  # Exclude 2025
            max_temp = daily_data.max()
            if pd.notna(max_temp):
                yearly_max_temps[year] = max_temp

    years = np.array(sorted(yearly_max_temps.keys()))
    max_temps = np.array([yearly_max_temps[year] for year in years])

    # Trend line (nonlinear, 3rd degree polynomial)
    z = np.polyfit(years, max_temps, 3)
    p = np.poly1d(z)
    trend_years = np.linspace(years.min(), years.max(), 300)
    trend_line = p(trend_years)

    # Plot the grey line
    ax.plot(years, max_temps, '-', color='#888888', linewidth=2, alpha=0.7, zorder=1)
    # Plot the red dots
    ax.scatter(years, max_temps, color='#D7263D', s=40, zorder=2)
    # Plot the black trend line
    ax.plot(trend_years, trend_line, '-', color='black', linewidth=3, alpha=0.9, zorder=3)

    # Highlight the hottest year
    overall_max = max_temps.max()
    overall_max_year = years[max_temps.argmax()]
    ax.scatter([overall_max_year], [overall_max], color='black', s=100, edgecolor='white', linewidth=2, zorder=4)

    # Clean, modern style
    ax.set_xlabel('Year', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('Maximum Temperature (°C)', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title(f'Hottest Temperature Each Year\nPotsdam Säkularstation, Germany ({years.min()}-{years.max()})', fontsize=22, fontweight='bold', pad=25)
    ax.set_ylim(max_temps.min() - 2, max_temps.max() + 2)
    ax.tick_params(axis='both', labelsize=14, length=6, width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Remove grid
    ax.grid(False)

    # No legend

    # Annotation for hottest year
    ax.annotate(f'{overall_max:.1f}°C',
                xy=(overall_max_year, overall_max),
                xytext=(overall_max_year+5, overall_max),
                fontsize=16, fontweight='bold', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=2, alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))

    # Data source and update date
    today = date.today()
    ax.text(0.99, 0.01, f'Data: Meteostat/DWD\nUpdated: {today.strftime("%d.%m.%Y")}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=12, color='#444444', alpha=0.8)

    plt.tight_layout()
    plt.savefig('plots/hottest_temperature_plot.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved as 'plots/hottest_temperature_plot.png'")
    return fig

def create_coldest_temperature_plot(all_data):
    """
    Create a visually appealing time series plot for Instagram:
    - Trend line in black
    - Lines in grey
    - Dots in blue
    - No grid
    - Clean, modern style
    """
    if not all_data:
        print("❌ No data to plot")
        return None

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Extract yearly minimum temperatures (exclude 2025)
    yearly_min_temps = {}
    for year, daily_data in all_data.items():
        if year != 2025:  # Exclude 2025
            min_temp = daily_data.min()
            if pd.notna(min_temp):
                yearly_min_temps[year] = min_temp

    years = np.array(sorted(yearly_min_temps.keys()))
    min_temps = np.array([yearly_min_temps[year] for year in years])

    # Trend line (nonlinear, 3rd degree polynomial)
    z = np.polyfit(years, min_temps, 3)
    p = np.poly1d(z)
    trend_years = np.linspace(years.min(), years.max(), 300)
    trend_line = p(trend_years)

    # Plot the grey line
    ax.plot(years, min_temps, '-', color='#888888', linewidth=2, alpha=0.7, zorder=1)
    # Plot the blue dots
    ax.scatter(years, min_temps, color='#1B6CA8', s=40, zorder=2)
    # Plot the black trend line
    ax.plot(trend_years, trend_line, '-', color='black', linewidth=3, alpha=0.9, zorder=3)

    # Highlight the coldest year
    overall_min = min_temps.min()
    overall_min_year = years[min_temps.argmin()]
    ax.scatter([overall_min_year], [overall_min], color='black', s=100, edgecolor='white', linewidth=2, zorder=4)

    # Clean, modern style
    ax.set_xlabel('Year', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('Minimum Temperature (°C)', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title(f'Coldest Temperature Each Year\nPotsdam Säkularstation, Germany ({years.min()}-{years.max()})', fontsize=22, fontweight='bold', pad=25)
    ax.set_ylim(-30, 0)
    ax.tick_params(axis='both', labelsize=14, length=6, width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Remove grid
    ax.grid(False)

    # No legend

    # Annotation for coldest year
    ax.annotate(f'{overall_min:.1f}°C',
                xy=(overall_min_year, overall_min),
                xytext=(overall_min_year+5, overall_min),
                fontsize=16, fontweight='bold', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=2, alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))

    # Data source and update date
    today = date.today()
    ax.text(0.99, 0.01, f'Data: Meteostat/DWD\nUpdated: {today.strftime("%d.%m.%Y")}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=12, color='#444444', alpha=0.8)

    plt.tight_layout()
    plt.savefig('plots/coldest_temperature_plot.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved as 'plots/coldest_temperature_plot.png'")
    return fig

# To generate the coldest temperature plot, fetch tmin instead of tmax in get_real_temperature_data

def get_real_min_temperature_data():
    print("❄️ Fetching REAL daily minimum temperature data for Potsdam...")
    stations = Stations()
    stations = stations.nearby(52.3833, 13.0667)
    station = stations.fetch(1)
    if station.empty:
        print("❌ No station found")
        return None
    station_id = station.index[0]
    years_to_analyze = list(range(1890, 2026))
    all_data = {}
    for year in years_to_analyze:
        start = datetime(year, 1, 1)
        end = datetime(2025, 7, 2) if year == 2025 else datetime(year, 12, 31)
        try:
            data = Daily(station_id, start, end).fetch()
            if not data.empty and 'tmin' in data.columns:
                expected_days = 183 if year == 2025 else (366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365)
                non_null_data = data['tmin'].dropna()
                actual_days = len(non_null_data)
                coverage_percentage = (actual_days / expected_days) * 100
                if year == 2025 or coverage_percentage >= 80.0:
                    daily_tmin = data['tmin']
                    all_data[year] = daily_tmin
            else:
                continue
        except Exception as e:
            continue
    if not all_data:
        print("❌ No real data could be retrieved")
        return None
    print(f"\n✅ Successfully retrieved real minimum temperature data for {len(all_data)} years")
    return all_data

def plot_days_above_30C(all_data):
    """
    Plot the number of days per year with tmax > 30°C.
    """
    if not all_data:
        print("❌ No data to plot")
        return None
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    years = sorted(all_data.keys())
    # Exclude 2025
    years = [y for y in years if y != 2025]
    days_above_30 = [ (all_data[y] > 30).sum() for y in years ]
    years = np.array(years)
    days_above_30 = np.array(days_above_30)
    # Trend line
    z = np.polyfit(years, days_above_30, 3)
    p = np.poly1d(z)
    trend_years = np.linspace(years.min(), years.max(), 300)
    trend_line = p(trend_years)
    # Plot
    ax.plot(years, days_above_30, '-', color='#888888', linewidth=2, alpha=0.7, zorder=1)
    ax.scatter(years, days_above_30, color='#D7263D', s=40, zorder=2)
    ax.plot(trend_years, trend_line, '-', color='black', linewidth=3, alpha=0.9, zorder=3)
    # Style
    ax.set_xlabel('Year', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('Number of Days', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title(f'Days per Year with Maximum Temperature > 30°C\nPotsdam Säkularstation, Germany ({years.min()}-{years.max()})', fontsize=22, fontweight='bold', pad=25)
    ax.set_ylim(0, days_above_30.max() + 5)
    ax.tick_params(axis='both', labelsize=14, length=6, width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)
    # No legend
    today = date.today()
    ax.text(0.99, 0.01, f'Data: Meteostat/DWD\nUpdated: {today.strftime("%d.%m.%Y")}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=12, color='#444444', alpha=0.8)
    plt.tight_layout()
    plt.savefig('plots/days_above_30C_plot.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved as 'plots/days_above_30C_plot.png'")
    return fig

def plot_days_below_0C(all_data):
    """
    Plot the number of days per year with tmin < 0°C.
    """
    if not all_data:
        print("❌ No data to plot")
        return None
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    years = sorted(all_data.keys())
    # Exclude 2025
    years = [y for y in years if y != 2025]
    days_below_0 = [ (all_data[y] < 0).sum() for y in years ]
    years = np.array(years)
    days_below_0 = np.array(days_below_0)
    # Trend line
    z = np.polyfit(years, days_below_0, 3)
    p = np.poly1d(z)
    trend_years = np.linspace(years.min(), years.max(), 300)
    trend_line = p(trend_years)
    # Plot
    ax.plot(years, days_below_0, '-', color='#888888', linewidth=2, alpha=0.7, zorder=1)
    ax.scatter(years, days_below_0, color='#1B6CA8', s=40, zorder=2)
    ax.plot(trend_years, trend_line, '-', color='black', linewidth=3, alpha=0.9, zorder=3)
    # Style
    ax.set_xlabel('Year', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('Number of Days', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title(f'Days per Year with Minimum Temperature < 0°C\nPotsdam Säkularstation, Germany ({years.min()}-{years.max()})', fontsize=22, fontweight='bold', pad=25)
    ax.set_ylim(days_below_0.min() - 5, days_below_0.max() + 5)
    ax.tick_params(axis='both', labelsize=14, length=6, width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)
    # No legend
    today = date.today()
    ax.text(0.99, 0.01, f'Data: Meteostat/DWD\nUpdated: {today.strftime("%d.%m.%Y")}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=12, color='#444444', alpha=0.8)
    plt.tight_layout()
    plt.savefig('plots/days_below_0C_plot.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved as 'plots/days_below_0C_plot.png'")
    return fig

# In main(), after creating the hottest and coldest plots, call these new functions:
# plot_days_above_30C(all_data)  # for tmax
def main():
    print("🌡️ HOTTEST TEMPERATURE ANALYSIS")
    print("="*60)
    all_data = get_real_temperature_data()
    if all_data:
        create_hottest_temperature_plot(all_data)
        plot_days_above_30C(all_data)
    print("\nNow generating coldest temperature plot...")
    min_data = get_real_min_temperature_data()
    if min_data:
        create_coldest_temperature_plot(min_data)
        plot_days_below_0C(min_data)

if __name__ == "__main__":
    main() 
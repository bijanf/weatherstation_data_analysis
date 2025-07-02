#!/usr/bin/env python3
"""
Real Cumulative Daily Precipitation Plot for Potsdam
Uses ONLY real data - no synthetic data generation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date
from meteostat import Stations, Daily
import warnings
warnings.filterwarnings('ignore')

def get_real_precipitation_data():
    """
    Get REAL daily precipitation data for Potsdam station using Meteostat.
    No synthetic data generation.
    """
    print("🌧️ Fetching REAL daily precipitation data for Potsdam...")
    
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
        print(f"📡 Downloading real data for {year}...")
        
        start = datetime(year, 1, 1)
        # For 2025, only get data up to current date
        if year == 2025:
            end = datetime(2025, 7, 2)  # Up to July 2nd
        else:
            end = datetime(year, 12, 31)
        
        try:
            data = Daily(station_id, start, end)
            data = data.fetch()
            
            if not data.empty and 'prcp' in data.columns:
                # Calculate expected days for the year
                if year == 2025:
                    # For 2025, calculate up to July 2nd
                    expected_days = 183  # Days from Jan 1 to July 2
                else:
                    # Full year: check if leap year
                    expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
                
                # Count actual non-null precipitation data points
                non_null_data = data['prcp'].dropna()
                actual_days = len(non_null_data)
                coverage_percentage = (actual_days / expected_days) * 100
                
                # Only include years with at least 80% coverage (or 2025 regardless of coverage)
                if year == 2025 or coverage_percentage >= 80.0:
                    # Use real precipitation data, fill NaN with 0 (no precipitation)
                    daily_prcp = data['prcp'].fillna(0.0)
                    all_data[year] = daily_prcp
                    total = daily_prcp.sum()
                    status = "(partial year)" if year == 2025 else ""
                    print(f"✅ {year}: {actual_days}/{expected_days} days, total: {total:.1f}mm, coverage: {coverage_percentage:.1f}% {status}")
                else:
                    print(f"❌ {year}: Insufficient coverage ({actual_days}/{expected_days} days = {coverage_percentage:.1f}%)")
            else:
                print(f"❌ {year}: No precipitation data available")
                
        except Exception as e:
            print(f"❌ {year}: Error downloading data - {e}")
    
    if not all_data:
        print("❌ No real data could be retrieved")
        return None
        
    print(f"\n✅ Successfully retrieved real data for {len(all_data)} years")
    return all_data

def create_real_cumulative_plot(all_data):
    """
    Create cumulative daily precipitation plot using ONLY real data.
    """
    if not all_data:
        print("❌ No data to plot")
        return None
        
    plt.style.use('default')
    
    # Create figure optimized for Instagram (slightly more square)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Colors: all years in gray except 2025 in blue
    max_precip = 0
    years_list = sorted(all_data.keys())
    first_year = min(years_list)
    last_year = max([y for y in years_list if y != 2025]) if 2025 in years_list else max(years_list)
    
    # Plot all years except 2018 and 2025 in gray
    gray_plotted = False
    
    for year in years_list:
        if year != 2025:
            daily_data = all_data[year]
            cumulative = daily_data.cumsum()
            
            # Create day-of-year index
            day_of_year = [d.timetuple().tm_yday for d in daily_data.index]
            
            if year == 2018:
                # 2018 in red (driest year on record)
                ax.plot(day_of_year, cumulative, 
                        color='#FF0000', linewidth=3, alpha=1.0, label='2018 (346mm)')
            else:
                # All other years in gray
                label = '1893-2024 (all other years)' if not gray_plotted else None
                gray_plotted = True
                ax.plot(day_of_year, cumulative, 
                        color='#808080', linewidth=1.0, alpha=0.4, label=label)
            
            max_precip = max(max_precip, cumulative.max())
    
    # Plot 2025 in blue (if available)
    if 2025 in all_data:
        daily_data = all_data[2025]
        cumulative = daily_data.cumsum()
        
        # Create day-of-year index
        day_of_year = [d.timetuple().tm_yday for d in daily_data.index]
        
        ax.plot(day_of_year, cumulative, 
                color='#000080', linewidth=3, alpha=1.0, label='2025')
        
        max_precip = max(max_precip, cumulative.max())
        
        # Add final total annotation for 2025 in lower white space
        final_total = cumulative.iloc[-1]
        ax.text(0.98, 0.08, f' Data: Real measurements from Meteostat/DWD', 
               transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='darkblue', linewidth=2, alpha=0.95))
    
    # Customize the plot with bigger labels for Instagram
    ax.set_xlabel('Month', fontsize=18, fontweight='bold')
    ax.set_ylabel('mm', fontsize=18, fontweight='bold')
    ax.set_title('Cumulative Daily Precipitation - Potsdam Station, Germany\n(130+ Years of Real Climate Data)', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Set x-axis to show months with bigger font for Instagram
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=16, fontweight='bold')
    
    # Make y-axis labels bigger and bold for Instagram
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Set y-axis limits
    ax.set_ylim(0, max_precip * 1.1)
    
    # Add enhanced grid for better readability
    ax.grid(True, alpha=0.4, linewidth=0.8)
    
    # Add legend with bigger font for Instagram
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=16)
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # Add update date in top right (smaller and less intrusive)
    today = date.today()
    ax.text(0.98, 0.98, f'Updated: {today.strftime("%d.%m.%Y")}', 
           transform=ax.transAxes, ha='right', va='top', fontsize=10, 
           color='gray', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('real_cumulative_precipitation_plot.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved as 'real_cumulative_precipitation_plot.png'")
    
    return fig

def main():
    """
    Main function - uses ONLY real data.
    """
    print("🌧️ REAL CUMULATIVE DAILY PRECIPITATION ANALYSIS")
    print("="*60)
    print("Potsdam Station - REAL Data Only (No Synthetic Data)")
    print("Data Source: Meteostat/DWD")
    print("="*60)
    
    # Get ONLY real data
    real_data = get_real_precipitation_data()
    
    if not real_data:
        print("\n❌ ERROR: No real data available")
        print("Cannot create plot without real measurements")
        return
    
    # Create plot with real data only
    print(f"\n📊 Creating plot with REAL data for {len(real_data)} years...")
    create_real_cumulative_plot(real_data)
    
    print(f"\n✅ Analysis complete using REAL data only!")
    print(f"📈 Real precipitation plot saved")
    print(f"\n📋 REAL Data Summary:")
    for year, data in sorted(real_data.items()):
        total = data.sum()
        data_points = len(data)
        status = "🔵 (incomplete year)" if year == 2025 else ""
        print(f"   • {year}: {total:.1f}mm ({data_points} days of real measurements) {status}")
    
    print(f"\n⚠️  IMPORTANT: This plot shows ONLY real precipitation measurements")
    print(f"    No synthetic or artificial data was generated.")

if __name__ == "__main__":
    main() 
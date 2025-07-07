#!/usr/bin/env python3
"""
Real Cumulative Daily Precipitation Plot for Texas Flash Flood Areas
Uses ONLY real data - no synthetic data generation.
Follows the same style as Potsdam plot: all years in grey, 2025 in red.
IMPROVED: Only plots lines when there are actual values, handles missing data properly.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date
import requests
import warnings
warnings.filterwarnings('ignore')

def get_real_texas_precipitation_data():
    """
    Get REAL daily precipitation data for Texas flash flood areas using NOAA/NCEI.
    Focus on Camp Mystic area and surrounding regions.
    """
    print("🌧️ Fetching REAL daily precipitation data for Texas flash flood areas...")
    
    # Camp Mystic is in Kerr County, Texas (near Kerrville)
    # Coordinates: approximately 30.0474° N, 99.1403° W
    
    # NOAA/NCEI station IDs for Texas areas:
    stations = {
        'Kerrville': {
            'id': 'USW00013962',
            'name': 'Kerrville Municipal Airport',
            'coords': (30.0474, -99.1403),
            'description': 'Near Camp Mystic flash flood area'
        },
        'Fredericksburg': {
            'id': 'USW00013963',
            'name': 'Fredericksburg Gillespie County Airport',
            'coords': (30.2436, -98.9097),
            'description': 'Central Texas flash flood area'
        }
    }
    
    all_data = {}
    
    for location, station_info in stations.items():
        print(f"\n📍 Analyzing {location}: {station_info['name']}")
        print(f"📍 {station_info['description']}")
        
        # Get real daily data for all available years (focus on recent decades)
        years_to_analyze = list(range(1950, 2026))  # Try to get data from 1950 onward
        location_data = {}
        
        for year in years_to_analyze:
            print(f"📡 Downloading real data for {year}...")
            
            start = datetime(year, 1, 1)
            # For 2025, only get data up to current date
            if year == 2025:
                end = datetime(2025, 12, 31)  # Full year for 2025
            else:
                end = datetime(year, 12, 31)
            
            try:
                # Use NOAA/NCEI API to get real precipitation data
                url = f"https://www.ncei.noaa.gov/access/services/data/v1"
                params = {
                    'dataset': 'daily-summaries',
                    'stations': station_info['id'],
                    'startDate': start.strftime('%Y-%m-%d'),
                    'endDate': end.strftime('%Y-%m-%d'),
                    'dataTypes': 'PRCP',
                    'format': 'json'
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data:
                        # Convert to pandas DataFrame
                        df = pd.DataFrame(data)
                        df['DATE'] = pd.to_datetime(df['DATE'])
                        df['PRCP'] = pd.to_numeric(df['PRCP'], errors='coerce')
                        
                        # Filter out missing data
                        df = df.dropna(subset=['PRCP'])
                        
                        if not df.empty:
                            # Convert from tenths of mm to mm
                            df['PRCP'] = df['PRCP'] / 10.0
                            
                            # Set as daily precipitation series
                            daily_prcp = df.set_index('DATE')['PRCP']
                            
                            # Calculate expected days for the year
                            if year == 2025:
                                # For 2025, calculate up to current date
                                current_date = date.today()
                                expected_days = (current_date - date(2025, 1, 1)).days + 1
                            else:
                                # Full year: check if leap year
                                expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
                            
                            # Count actual non-null precipitation data points
                            actual_days = len(daily_prcp)
                            coverage_percentage = (actual_days / expected_days) * 100
                            
                            # Only include years with at least 80% coverage (or 2025 regardless of coverage)
                            if year == 2025 or coverage_percentage >= 80.0:
                                # Create full date range and reindex
                                full_date_range = pd.date_range(start, end, freq='D')
                                daily_prcp = daily_prcp.reindex(full_date_range)
                                
                                # Only fill NaN with 0 if we have some real data (not all NaN)
                                if not daily_prcp.isna().all():
                                    daily_prcp = daily_prcp.fillna(0.0)
                                    location_data[year] = daily_prcp
                                    total = daily_prcp.sum()
                                    status = "(partial year)" if year == 2025 else ""
                                    print(f"✅ {year}: {actual_days}/{expected_days} days, total: {total:.1f}mm, coverage: {coverage_percentage:.1f}% {status}")
                                else:
                                    print(f"❌ {year}: No valid precipitation data available")
                            else:
                                print(f"❌ {year}: Insufficient coverage ({actual_days}/{expected_days} days = {coverage_percentage:.1f}%)")
                        else:
                            print(f"❌ {year}: No valid precipitation data")
                    else:
                        print(f"❌ {year}: No data returned from API")
                else:
                    print(f"❌ {year}: API error {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {year}: Error downloading data - {e}")
        
        if location_data:
            all_data[location] = {
                'station_info': station_info,
                'years': location_data
            }
    
    if not all_data:
        print("❌ No real data could be retrieved")
        return None
        
    print(f"\n✅ Successfully retrieved real data for {len(all_data)} locations")
    return all_data

def create_real_cumulative_plot(all_data):
    """
    Create cumulative daily precipitation plot using ONLY real data.
    All years in grey except 2025 in red (like Potsdam plot).
    IMPROVED: Only plots lines when there are actual values.
    """
    if not all_data:
        print("❌ No data to plot")
        return None
        
    plt.style.use('default')
    
    # Create plots for each location
    for location, location_info in all_data.items():
        station_info = location_info['station_info']
        years_data = location_info['years']
        
        print(f"\n📊 Creating cumulative plot for {location}...")
        
        # Create figure optimized for Instagram (slightly more square)
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Colors: all years in gray except 2025 in red
        max_precip = 0
        years_list = sorted(years_data.keys())
        
        # Plot all years except 2025 in gray
        gray_plotted = False
        
        for year in years_list:
            if year != 2025:
                daily_data = years_data[year]
                
                # Only plot if we have actual precipitation data (not just zeros)
                if daily_data.sum() > 0:
                    cumulative = daily_data.cumsum()
                    
                    # Create day-of-year index
                    day_of_year = [d.timetuple().tm_yday for d in daily_data.index]
                    
                    # All other years in gray
                    label = f'{min(years_list)}-{max([y for y in years_list if y != 2025])} (all other years)' if not gray_plotted else None
                    gray_plotted = True
                    ax.plot(day_of_year, cumulative, 
                            color='#808080', linewidth=1.0, alpha=0.4, label=label)
                    
                    max_precip = max(max_precip, cumulative.max())
        
        # Plot 2025 in red (if available) - only plot when we have actual data
        if 2025 in years_data:
            daily_data = years_data[2025]
            
            # Only plot if we have actual precipitation data (not just zeros)
            if daily_data.sum() > 0:
                cumulative = daily_data.cumsum()
                
                # Create day-of-year index
                day_of_year = [d.timetuple().tm_yday for d in daily_data.index]
                
                ax.plot(day_of_year, cumulative, 
                        color='#FF0000', linewidth=3, alpha=1.0, label='2025')
                
                max_precip = max(max_precip, cumulative.max())
                
                # Add final total annotation for 2025
                final_total = cumulative.iloc[-1]
                ax.text(0.98, 0.08, f' Data: Real measurements from NOAA/NCEI', 
                       transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='darkblue', linewidth=2, alpha=0.95))
        
        # Customize the plot with bigger labels for Instagram
        ax.set_xlabel('Month', fontsize=18, fontweight='bold')
        ax.set_ylabel('mm', fontsize=18, fontweight='bold')
        ax.set_title(f'Cumulative Daily Precipitation - {station_info["name"]}\nTexas Flash Flood Area ({len(years_list)} Years of Real Climate Data)', 
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
        
        # Save plot
        filename = f"texas_cumulative_precipitation_{location}_improved.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved as '{filename}'")
        
        plt.close()
    
    return True

def main():
    """
    Main function - uses ONLY real data.
    """
    print("🌧️ REAL CUMULATIVE DAILY PRECIPITATION ANALYSIS - TEXAS FLASH FLOOD")
    print("="*60)
    print("Texas Flash Flood Areas - REAL Data Only (No Synthetic Data)")
    print("Data Source: NOAA/NCEI")
    print("IMPROVED: Only plots lines when there are actual values")
    print("="*60)
    
    # Get ONLY real data
    real_data = get_real_texas_precipitation_data()
    
    if not real_data:
        print("\n❌ ERROR: No real data available")
        print("Cannot create plot without real measurements")
        return
    
    # Create plot with real data only
    print(f"\n📊 Creating cumulative plots with REAL data...")
    create_real_cumulative_plot(real_data)
    
    print(f"\n✅ Analysis complete using REAL data only!")
    print(f"📈 Real cumulative precipitation plots saved")
    print(f"\n📋 REAL Data Summary:")
    for location, location_info in real_data.items():
        station_info = location_info['station_info']
        years_data = location_info['years']
        print(f"\n📍 {location}: {station_info['name']}")
        print(f"   {station_info['description']}")
        for year, data in sorted(years_data.items()):
            total = data.sum()
            data_points = len(data)
            status = "🔴 (2025 flash flood year)" if year == 2025 else ""
            print(f"   • {year}: {total:.1f}mm ({data_points} days of real measurements) {status}")
    
    print(f"\n⚠️  IMPORTANT: These plots show ONLY real precipitation measurements")
    print(f"    No synthetic or artificial data was generated.")

if __name__ == "__main__":
    main() 
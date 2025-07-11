#!/usr/bin/env python3
"""
Extreme Value Statistics for German Cities
Comprehensive analysis of climate extremes from weather station data.

This script supports any German city with fuzzy matching and suggestions.
Usage: python potsdam_extreme_values.py [city_name]
Example: python potsdam_extreme_values.py Berlin
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from weatherstation_analysis import ExtremeValueAnalyzer, WeatherPlotter
from weatherstation_analysis.weather_fetcher import WeatherDataFetcher
from weatherstation_analysis.city_manager import CityManager
import warnings
warnings.filterwarnings('ignore')

def get_weather_station_data(city_name: str = "Potsdam"):
    """
    Get comprehensive daily weather data for the specified city.
    
    Args:
        city_name: Name of the German city
        
    Returns:
        Weather data dictionary and fetcher instance
    """
    try:
        fetcher = WeatherDataFetcher(city_name)
        data = fetcher.fetch_comprehensive_data()
        return data, fetcher
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None, None

def analyze_annual_extremes(all_data):
    """
    Extract annual extreme values for analysis using new architecture.
    """
    analyzer = ExtremeValueAnalyzer(all_data)
    return analyzer.analyze_annual_extremes()

def plot_annual_precipitation_extremes(extremes_df, city_name: str = "Potsdam"):
    """
    Plot annual maximum daily precipitation with return period analysis.
    """
    plotter = WeatherPlotter()
    filename = f'plots/{city_name.lower()}_extreme_precipitation_analysis.png'
    return plotter.plot_annual_precipitation_extremes(extremes_df, filename)

def plot_temperature_extremes_analysis(extremes_df, city_name: str = "Potsdam"):
    """
    Plot comprehensive temperature extremes analysis.
    """
    plotter = WeatherPlotter()
    filename = f'plots/{city_name.lower()}_temperature_extremes_analysis.png'
    return plotter.plot_temperature_extremes_analysis(extremes_df, filename)

def plot_threshold_exceedance_analysis(all_data, city_name: str = "Potsdam"):
    """
    Analyze frequency of threshold exceedances over time.
    """
    plotter = WeatherPlotter()
    filename = f'plots/{city_name.lower()}_threshold_exceedance_analysis.png'
    return plotter.plot_threshold_exceedance_analysis(all_data, filename)

def plot_extreme_statistics_summary(extremes_df, city_name: str = "Potsdam"):
    """
    Create a summary dashboard of extreme value statistics.
    """
    plotter = WeatherPlotter()
    filename = f'plots/{city_name.lower()}_extreme_statistics_summary.png'
    return plotter.plot_statistics_summary(extremes_df, filename)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extreme Value Statistics for German Cities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python potsdam_extreme_values.py                  # Default: Potsdam
  python potsdam_extreme_values.py Berlin           # Analyze Berlin
  python potsdam_extreme_values.py munich           # Case insensitive
  python potsdam_extreme_values.py --list-cities    # Show available cities
        """
    )
    
    parser.add_argument(
        "city", 
        nargs="?", 
        default="Potsdam",
        help="German city name (default: Potsdam)"
    )
    
    parser.add_argument(
        "--list-cities", 
        action="store_true",
        help="List all available German cities"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run all extreme value analyses.
    """
    args = parse_arguments()
    
    # Handle city listing
    if args.list_cities:
        city_manager = CityManager()
        cities = city_manager.list_available_cities()
        print("🏙️ Available German Cities:")
        print("=" * 40)
        for i, city in enumerate(cities, 1):
            print(f"{i:2d}. {city}")
        print(f"\nTotal: {len(cities)} cities")
        return
    
    city_name = args.city
    
    print("🌪️ EXTREME VALUE STATISTICS ANALYSIS")
    print("=" * 70)
    print(f"German Weather Station Analysis - {city_name.title()}")
    print("Data Source: Meteostat/DWD")
    print("=" * 70)
    
    # Get comprehensive weather data
    all_data, fetcher = get_weather_station_data(city_name)
    
    if not all_data or not fetcher:
        print("\n❌ ERROR: No data available")
        return
    
    # Get station info
    station_info = fetcher.get_station_info()
    actual_city = station_info["city"]
    
    # Analyze annual extremes
    print("\n📊 Analyzing annual extreme values...")
    extremes_df = analyze_annual_extremes(all_data)
    
    # Create all plots
    print("\n🎨 Creating extreme value visualizations...")
    
    plot_annual_precipitation_extremes(extremes_df, actual_city)
    plot_temperature_extremes_analysis(extremes_df, actual_city)
    plot_threshold_exceedance_analysis(all_data, actual_city)
    plot_extreme_statistics_summary(extremes_df, actual_city)
    
    # Print summary statistics
    print("\n📈 EXTREME VALUE SUMMARY:")
    print(f"   • City: {actual_city}")
    print(f"   • Station: {station_info['station_name']}")
    print(f"   • Distance: {station_info['distance_km']:.1f} km from city center")
    print(f"   • Analysis period: {extremes_df['year'].min()}-{extremes_df['year'].max()}")
    print(f"   • Maximum daily precipitation: {extremes_df['max_precip'].max():.1f}mm ({extremes_df.loc[extremes_df['max_precip'].idxmax(), 'year']})")
    print(f"   • Highest temperature: {extremes_df['max_temp'].max():.1f}°C ({extremes_df.loc[extremes_df['max_temp'].idxmax(), 'year']})")
    print(f"   • Lowest temperature: {extremes_df['min_temp'].min():.1f}°C ({extremes_df.loc[extremes_df['min_temp'].idxmin(), 'year']})")
    print(f"   • Largest temperature range: {extremes_df['temp_range'].max():.1f}°C ({extremes_df.loc[extremes_df['temp_range'].idxmax(), 'year']})")
    
    city_lower = actual_city.lower()
    print(f"\n✅ Analysis complete! 4 extreme value plots created.")
    print(f"📊 Files saved:")
    print(f"   • plots/{city_lower}_extreme_precipitation_analysis.png")
    print(f"   • plots/{city_lower}_temperature_extremes_analysis.png")
    print(f"   • plots/{city_lower}_threshold_exceedance_analysis.png")
    print(f"   • plots/{city_lower}_extreme_statistics_summary.png")

if __name__ == "__main__":
    main()
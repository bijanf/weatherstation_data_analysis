#!/usr/bin/env python3
"""
Extreme Value Statistics for Potsdam Säkularstation
Comprehensive analysis of climate extremes from 133 years of real data.

This script uses the new modular architecture for better maintainability.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from weatherstation_analysis import PotsdamDataFetcher, ExtremeValueAnalyzer, WeatherPlotter
import warnings
warnings.filterwarnings('ignore')

def get_potsdam_station_data():
    """
    Get comprehensive daily weather data for Potsdam station using new architecture.
    """
    fetcher = PotsdamDataFetcher()
    return fetcher.fetch_comprehensive_data()

def analyze_annual_extremes(all_data):
    """
    Extract annual extreme values for analysis using new architecture.
    """
    analyzer = ExtremeValueAnalyzer(all_data)
    return analyzer.analyze_annual_extremes()

def plot_annual_precipitation_extremes(extremes_df):
    """
    Plot annual maximum daily precipitation with return period analysis.
    """
    plotter = WeatherPlotter()
    return plotter.plot_annual_precipitation_extremes(
        extremes_df, 'plots/extreme_precipitation_analysis.png')

def plot_temperature_extremes_analysis(extremes_df):
    """
    Plot comprehensive temperature extremes analysis.
    """
    plotter = WeatherPlotter()
    return plotter.plot_temperature_extremes_analysis(
        extremes_df, 'plots/temperature_extremes_analysis.png')

def plot_threshold_exceedance_analysis(all_data):
    """
    Analyze frequency of threshold exceedances over time.
    """
    plotter = WeatherPlotter()
    return plotter.plot_threshold_exceedance_analysis(
        all_data, 'plots/threshold_exceedance_analysis.png')

def plot_extreme_statistics_summary(extremes_df):
    """
    Create a summary dashboard of extreme value statistics.
    """
    plotter = WeatherPlotter()
    return plotter.plot_statistics_summary(
        extremes_df, 'plots/extreme_statistics_summary.png')

def main():
    """
    Main function to run all extreme value analyses.
    """
    print("🌪️ EXTREME VALUE STATISTICS ANALYSIS")
    print("="*70)
    print("Potsdam Säkularstation - 133 Years of Climate Extremes")
    print("Data Source: Meteostat/DWD")
    print("="*70)
    
    # Get comprehensive weather data
    all_data = get_potsdam_station_data()
    
    if not all_data:
        print("\n❌ ERROR: No data available")
        return
    
    # Analyze annual extremes
    print("\n📊 Analyzing annual extreme values...")
    extremes_df = analyze_annual_extremes(all_data)
    
    # Create all plots
    print("\n🎨 Creating extreme value visualizations...")
    
    plot_annual_precipitation_extremes(extremes_df)
    plot_temperature_extremes_analysis(extremes_df)
    plot_threshold_exceedance_analysis(all_data)
    plot_extreme_statistics_summary(extremes_df)
    
    # Print summary statistics
    print("\n📈 EXTREME VALUE SUMMARY:")
    print(f"   • Analysis period: {extremes_df['year'].min()}-{extremes_df['year'].max()}")
    print(f"   • Maximum daily precipitation: {extremes_df['max_precip'].max():.1f}mm ({extremes_df.loc[extremes_df['max_precip'].idxmax(), 'year']})")
    print(f"   • Highest temperature: {extremes_df['max_temp'].max():.1f}°C ({extremes_df.loc[extremes_df['max_temp'].idxmax(), 'year']})")
    print(f"   • Lowest temperature: {extremes_df['min_temp'].min():.1f}°C ({extremes_df.loc[extremes_df['min_temp'].idxmin(), 'year']})")
    print(f"   • Largest temperature range: {extremes_df['temp_range'].max():.1f}°C ({extremes_df.loc[extremes_df['temp_range'].idxmax(), 'year']})")
    
    print(f"\n✅ Analysis complete! 4 extreme value plots created.")
    print(f"📊 Files saved:")
    print(f"   • plots/extreme_precipitation_analysis.png")
    print(f"   • plots/temperature_extremes_analysis.png")
    print(f"   • plots/threshold_exceedance_analysis.png")
    print(f"   • plots/extreme_statistics_summary.png")

if __name__ == "__main__":
    main()
"""
Germany's Centennial Weather Stations Analysis using DWD Data
==============================================================

Analyzes 100+ years of weather data from Germany's longest-running weather stations
using DWD (Deutscher Wetterdienst) Climate Data Center.

Features:
- Identifies German stations with 100+ years of continuous records
- Annual temperature anomalies showing warming trend
- Cumulative annual precipitation patterns
- Color-coded extreme years vs normal years
- Recent years highlighted to show climate change impact
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# Try importing wetterdienst
try:
    from wetterdienst import Parameter, Period, Resolution, Settings
    from wetterdienst.provider.dwd.observation import DwdObservationRequest
    WETTERDIENST_AVAILABLE = True
except ImportError as e:
    WETTERDIENST_AVAILABLE = False
    print(f"Warning: wetterdienst not available: {e}")
    print("Install with: pip install wetterdienst")
except Exception as e:
    WETTERDIENST_AVAILABLE = False
    print(f"Error importing wetterdienst: {e}")


class GermanCentennialClimate:
    """
    Analyzes century-long weather data from German weather stations using DWD data.

    Attributes:
        min_years: Minimum years of data required (default: 100)
        stations_data: Dict mapping station IDs to weather data
    """

    def __init__(self, min_years: int = 100):
        """
        Initialize the analyzer.

        Args:
            min_years: Minimum number of years of data required
        """
        self.min_years = min_years
        self.stations_data: Dict[str, pd.DataFrame] = {}

    def find_longest_stations(self) -> pd.DataFrame:
        """
        Find DWD weather stations with the longest temperature records.

        Returns:
            DataFrame with station information sorted by data availability
        """
        print(f"🔍 Searching for German DWD stations with {self.min_years}+ years of data...")

        # Request historical stations with temperature data
        request = DwdObservationRequest(
            parameter=[Parameter.TEMPERATURE_AIR_MEAN_2M, Parameter.PRECIPITATION_HEIGHT],
            resolution=Resolution.DAILY,
            period=Period.HISTORICAL,
        ).all()

        # Get stations metadata
        stations_df = request.df

        if stations_df.empty:
            print("❌ No stations found")
            return pd.DataFrame()

        print(f"📊 Found {len(stations_df)} DWD stations total")

        # Calculate data range in years
        stations_df['data_range'] = (
            pd.to_datetime(stations_df['end_date']) -
            pd.to_datetime(stations_df['start_date'])
        ).dt.days / 365.25

        # Filter for long-term stations
        long_stations = stations_df[stations_df['data_range'] >= self.min_years].copy()
        long_stations = long_stations.sort_values('data_range', ascending=False)

        print(f"\n✅ Found {len(long_stations)} stations with {self.min_years}+ years of data:\n")
        print("-" * 90)

        for idx, row in long_stations.head(15).iterrows():
            print(f"📍 {row['name']:40s} | {row['data_range']:.1f} years | "
                  f"{row['start_date']} to {row['end_date']}")

        return long_stations

    def fetch_station_data(
        self,
        station_id: str,
        station_name: str = "Unknown"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch complete weather data for a specific station from DWD.

        Args:
            station_id: DWD station identifier
            station_name: Name of the station (for display)

        Returns:
            DataFrame with complete weather data
        """
        print(f"\n📡 Fetching historical climate data for {station_name} (ID: {station_id})...")

        try:
            # Request climate data (temperature and precipitation)
            request = DwdObservationRequest(
                parameter=[
                    Parameter.TEMPERATURE_AIR_MEAN_2M,
                    Parameter.TEMPERATURE_AIR_MAX_2M,
                    Parameter.TEMPERATURE_AIR_MIN_2M,
                    Parameter.PRECIPITATION_HEIGHT
                ],
                resolution=Resolution.DAILY,
                period=Period.HISTORICAL,
                start_date=datetime(1800, 1, 1),
                end_date=datetime(2025, 12, 31)
            ).filter_by_station_id(station_id=station_id)

            # Fetch data
            print("  📥 Downloading data from DWD CDC...")
            values = request.values.all()
            df = values.df

            if df.empty:
                print("  ❌ No data available")
                return None

            # Pivot data to wide format
            df_pivot = df.pivot_table(
                index='date',
                columns='parameter',
                values='value'
            )

            print(f"  ✅ Retrieved {len(df_pivot)} days of data")
            print(f"  📊 Available parameters: {df_pivot.columns.tolist()}")

            self.stations_data[station_id] = df_pivot
            return df_pivot

        except Exception as e:
            print(f"  ❌ Error fetching data: {e}")
            return None

    def calculate_annual_statistics(
        self,
        station_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate annual temperature and precipitation statistics.

        Args:
            station_data: DataFrame with daily weather data

        Returns:
            DataFrame with annual statistics
        """
        print("\n📊 Calculating annual statistics...")

        # Extract year from index
        station_data = station_data.copy()
        station_data['year'] = station_data.index.year

        # Group by year and calculate statistics
        annual_stats_list = []

        for year, year_data in station_data.groupby('year'):
            # Temperature statistics
            temp_col = None
            for col in ['temperature_air_mean_2m', 'temperature_air_2m', 'tmean', 'temperature_air_mean_200']:
                if col in year_data.columns:
                    temp_col = col
                    break

            tmax_col = None
            for col in ['temperature_air_max_2m', 'temperature_air_max_200', 'tmax']:
                if col in year_data.columns:
                    tmax_col = col
                    break

            tmin_col = None
            for col in ['temperature_air_min_2m', 'temperature_air_min_200', 'tmin']:
                if col in year_data.columns:
                    tmin_col = col
                    break

            # Precipitation statistics
            prcp_col = None
            for col in ['precipitation_height', 'prcp', 'precipitation_height_rocker']:
                if col in year_data.columns:
                    prcp_col = col
                    break

            # Calculate statistics only if we have temperature data
            if temp_col and temp_col in year_data.columns:
                mean_temp = year_data[temp_col].mean()

                # Check data coverage
                expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
                actual_days = year_data[temp_col].notna().sum()
                coverage = (actual_days / expected_days) * 100

                # Only include years with good coverage (>80%)
                if coverage >= 80:
                    stats_dict = {
                        'year': int(year),
                        'mean_temp': mean_temp,
                        'n_days': actual_days,
                        'coverage': coverage
                    }

                    if tmax_col and tmax_col in year_data.columns:
                        stats_dict['max_temp'] = year_data[tmax_col].max()

                    if tmin_col and tmin_col in year_data.columns:
                        stats_dict['min_temp'] = year_data[tmin_col].min()

                    if prcp_col and prcp_col in year_data.columns:
                        stats_dict['total_prcp'] = year_data[prcp_col].sum()

                    annual_stats_list.append(stats_dict)

        df = pd.DataFrame(annual_stats_list)
        df = df.sort_values('year')

        if len(df) == 0:
            print("  ❌ No valid annual statistics could be calculated")
            return df

        # Calculate temperature anomaly (deviation from 1961-1990 baseline)
        baseline_period = df[(df['year'] >= 1961) & (df['year'] <= 1990)]
        if len(baseline_period) >= 20:
            baseline_mean = baseline_period['mean_temp'].mean()
            df['temp_anomaly'] = df['mean_temp'] - baseline_mean
            print(f"  ✅ Using 1961-1990 baseline: {baseline_mean:.2f}°C")
        else:
            # Fallback to overall mean
            df['temp_anomaly'] = df['mean_temp'] - df['mean_temp'].mean()
            print(f"  ⚠️  Using overall mean baseline: {df['mean_temp'].mean():.2f}°C")

        print(f"  ✅ Calculated statistics for {len(df)} years ({df['year'].min()}-{df['year'].max()})")

        return df

    def identify_extreme_years(
        self,
        annual_stats: pd.DataFrame,
        temp_threshold: float = 1.5,
        prcp_threshold: float = 1.5
    ) -> Dict[int, str]:
        """
        Identify extreme years based on temperature and precipitation anomalies.

        Args:
            annual_stats: DataFrame with annual statistics
            temp_threshold: Standard deviations for temperature extremes
            prcp_threshold: Standard deviations for precipitation extremes

        Returns:
            Dict mapping years to extreme type ('hot', 'cold', 'wet', 'dry')
        """
        extreme_years = {}

        # Temperature extremes
        temp_std = annual_stats['temp_anomaly'].std()
        temp_mean = annual_stats['temp_anomaly'].mean()

        # Precipitation extremes (if available)
        has_prcp = 'total_prcp' in annual_stats.columns and annual_stats['total_prcp'].notna().sum() > 50
        if has_prcp:
            prcp_std = annual_stats['total_prcp'].std()
            prcp_mean = annual_stats['total_prcp'].mean()

        for _, row in annual_stats.iterrows():
            year = int(row['year'])
            temp_z = (row['temp_anomaly'] - temp_mean) / temp_std if temp_std > 0 else 0

            # Classify temperature extremes
            if temp_z > temp_threshold:
                extreme_years[year] = 'hot'
            elif temp_z < -temp_threshold:
                extreme_years[year] = 'cold'
            elif has_prcp and not pd.isna(row['total_prcp']):
                prcp_z = (row['total_prcp'] - prcp_mean) / prcp_std if prcp_std > 0 else 0
                if prcp_z > prcp_threshold:
                    extreme_years[year] = 'wet'
                elif prcp_z < -prcp_threshold:
                    extreme_years[year] = 'dry'

        return extreme_years

    def plot_centennial_climate_trends(
        self,
        annual_stats: pd.DataFrame,
        station_name: str,
        save_path: str = "plots/germany_centennial_climate_dwd.png"
    ):
        """
        Create comprehensive visualization of centennial climate trends.

        Args:
            annual_stats: DataFrame with annual statistics
            station_name: Name of the weather station
            save_path: Path to save the plot
        """
        print(f"\n🎨 Creating climate trends visualization...")

        # Identify extreme years
        extreme_years = self.identify_extreme_years(annual_stats)

        # Define color scheme
        colors = {
            'hot': '#d62728',      # Red
            'cold': '#1f77b4',     # Blue
            'wet': '#2ca02c',      # Green
            'dry': '#ff7f0e',      # Orange
            'normal': '#7f7f7f',   # Grey
            'recent': '#9467bd'    # Purple for recent years
        }

        # Determine if we have precipitation data
        has_prcp = 'total_prcp' in annual_stats.columns and annual_stats['total_prcp'].notna().sum() > 50

        # Create figure
        if has_prcp:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))

        fig.suptitle(
            f'Century-Long Climate Analysis: {station_name}\n'
            f'{int(annual_stats["year"].min())}–{int(annual_stats["year"].max())} '
            f'({len(annual_stats)} years of data)',
            fontsize=16, fontweight='bold', y=0.98
        )

        # Recent year threshold (highlight last 30 years)
        recent_year_threshold = annual_stats['year'].max() - 30

        # --- Panel 1: Temperature Anomaly (Warming Trend) ---
        print("  📈 Plotting temperature anomalies...")

        for _, row in annual_stats.iterrows():
            year = int(row['year'])
            anomaly = row['temp_anomaly']

            # Determine color and style
            if year in extreme_years:
                color = colors[extreme_years[year]]
                alpha = 0.8
                linewidth = 2
            elif year > recent_year_threshold:
                color = colors['recent']
                alpha = 0.7
                linewidth = 1.5
            else:
                color = colors['normal']
                alpha = 0.4
                linewidth = 1

            ax1.plot([year, year], [0, anomaly],
                    color=color, alpha=alpha, linewidth=linewidth)

        # Add trend line
        z = np.polyfit(annual_stats['year'], annual_stats['temp_anomaly'], 1)
        p = np.poly1d(z)
        trend_line = p(annual_stats['year'])
        ax1.plot(annual_stats['year'], trend_line,
                'k--', linewidth=2.5, alpha=0.8,
                label=f'Linear trend: +{z[0]*100:.2f}°C per century')

        # Calculate warming statistics
        early_period = annual_stats[annual_stats['year'] < 1950]
        late_period = annual_stats[annual_stats['year'] >= 1990]
        if len(early_period) > 0 and len(late_period) > 0:
            warming = late_period['mean_temp'].mean() - early_period['mean_temp'].mean()
        else:
            warming = 0

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temperature Anomaly (°C)\nrelative to 1961-1990',
                      fontsize=12, fontweight='bold')
        ax1.set_title(
            'Annual Mean Temperature Anomaly - Warming Trend Visible in Recent Decades',
            fontsize=13, pad=10
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)

        # Add warming statistics text box
        if warming > 0:
            textstr = f'Warming: +{warming:.2f}°C\n(pre-1950 to post-1990)'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

        # --- Panel 2: Annual Precipitation (if available) ---
        if has_prcp:
            print("  🌧️  Plotting precipitation totals...")

            for _, row in annual_stats.iterrows():
                if pd.isna(row['total_prcp']):
                    continue

                year = int(row['year'])
                prcp = row['total_prcp']

                # Determine color and style
                if year in extreme_years:
                    color = colors[extreme_years[year]]
                    alpha = 0.8
                    linewidth = 2
                elif year > recent_year_threshold:
                    color = colors['recent']
                    alpha = 0.7
                    linewidth = 1.5
                else:
                    color = colors['normal']
                    alpha = 0.4
                    linewidth = 1

                ax2.plot([year, year], [0, prcp],
                        color=color, alpha=alpha, linewidth=linewidth)

            # Add mean line
            mean_prcp = annual_stats['total_prcp'].mean()
            ax2.axhline(y=mean_prcp, color='black', linestyle='--',
                       linewidth=2, alpha=0.7, label=f'Mean: {mean_prcp:.0f} mm')

            ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Annual Precipitation (mm)', fontsize=12, fontweight='bold')
            ax2.set_title(
                'Annual Total Precipitation - Extreme Dry and Wet Years Highlighted',
                fontsize=13, pad=10
            )
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)

            # Add custom legend for color coding
            legend_elements = [
                Line2D([0], [0], color=colors['hot'], linewidth=3, label='Extreme Hot Year'),
                Line2D([0], [0], color=colors['cold'], linewidth=3, label='Extreme Cold Year'),
                Line2D([0], [0], color=colors['wet'], linewidth=3, label='Extreme Wet Year'),
                Line2D([0], [0], color=colors['dry'], linewidth=3, label='Extreme Dry Year'),
                Line2D([0], [0], color=colors['recent'], linewidth=3, label='Recent Years (1995+)'),
                Line2D([0], [0], color=colors['normal'], linewidth=2, label='Normal Years'),
            ]

            ax2.legend(handles=legend_elements, loc='upper right',
                      fontsize=10, framealpha=0.9, ncol=2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {save_path}")

        return fig

    def generate_summary_report(
        self,
        annual_stats: pd.DataFrame,
        station_name: str
    ):
        """
        Generate a text summary of the climate analysis.

        Args:
            annual_stats: DataFrame with annual statistics
            station_name: Name of the weather station
        """
        print(f"\n" + "="*90)
        print(f"CLIMATE SUMMARY: {station_name}")
        print("="*90)

        # Overall statistics
        print(f"\n📊 Data Coverage:")
        print(f"  Period: {int(annual_stats['year'].min())}–{int(annual_stats['year'].max())}")
        print(f"  Total years: {len(annual_stats)}")

        # Temperature statistics
        print(f"\n🌡️  Temperature Analysis:")
        print(f"  Mean annual temperature: {annual_stats['mean_temp'].mean():.2f}°C")
        print(f"  Warmest year: {int(annual_stats.loc[annual_stats['mean_temp'].idxmax(), 'year'])} "
              f"({annual_stats['mean_temp'].max():.2f}°C)")
        print(f"  Coldest year: {int(annual_stats.loc[annual_stats['mean_temp'].idxmin(), 'year'])} "
              f"({annual_stats['mean_temp'].min():.2f}°C)")

        if 'max_temp' in annual_stats.columns:
            print(f"  All-time record high: {annual_stats['max_temp'].max():.1f}°C "
                  f"(year {int(annual_stats.loc[annual_stats['max_temp'].idxmax(), 'year'])})")

        if 'min_temp' in annual_stats.columns:
            print(f"  All-time record low: {annual_stats['min_temp'].min():.1f}°C "
                  f"(year {int(annual_stats.loc[annual_stats['min_temp'].idxmin(), 'year'])})")

        # Warming trend
        early_period = annual_stats[annual_stats['year'] < 1950]
        late_period = annual_stats[annual_stats['year'] >= 1990]
        if len(early_period) > 0 and len(late_period) > 0:
            warming = late_period['mean_temp'].mean() - early_period['mean_temp'].mean()
            print(f"\n🔥 Warming Trend:")
            print(f"  Pre-1950 mean: {early_period['mean_temp'].mean():.2f}°C")
            print(f"  Post-1990 mean: {late_period['mean_temp'].mean():.2f}°C")
            print(f"  Total warming: +{warming:.2f}°C")
            print(f"  Warming rate: +{(warming / len(range(1950, 1990))) * 100:.2f}°C per century")

        # Precipitation statistics
        if 'total_prcp' in annual_stats.columns and annual_stats['total_prcp'].notna().sum() > 50:
            print(f"\n🌧️  Precipitation Analysis:")
            print(f"  Mean annual precipitation: {annual_stats['total_prcp'].mean():.0f} mm")
            print(f"  Wettest year: {int(annual_stats.loc[annual_stats['total_prcp'].idxmax(), 'year'])} "
                  f"({annual_stats['total_prcp'].max():.0f} mm)")
            print(f"  Driest year: {int(annual_stats.loc[annual_stats['total_prcp'].idxmin(), 'year'])} "
                  f"({annual_stats['total_prcp'].min():.0f} mm)")

        # Extreme years
        extreme_years = self.identify_extreme_years(annual_stats)
        print(f"\n⚠️  Extreme Years ({len(extreme_years)} total):")
        for extreme_type in ['hot', 'cold', 'wet', 'dry']:
            years = [y for y, t in extreme_years.items() if t == extreme_type]
            if years:
                years_sorted = sorted(years)
                if len(years_sorted) <= 10:
                    print(f"  {extreme_type.capitalize()}: {', '.join(map(str, years_sorted))}")
                else:
                    print(f"  {extreme_type.capitalize()}: {', '.join(map(str, years_sorted[:10]))} "
                          f"... ({len(years_sorted)} total)")

        print("="*90 + "\n")


def main():
    """
    Main execution function.
    """
    if not WETTERDIENST_AVAILABLE:
        print("❌ Error: wetterdienst library is not installed")
        print("   Install with: pip install wetterdienst")
        return

    print("="*90)
    print("GERMANY'S CENTENNIAL WEATHER STATIONS ANALYSIS (DWD Data)")
    print("Analyzing 100+ years of temperature and precipitation data")
    print("="*90 + "\n")

    # Initialize analyzer
    analyzer = GermanCentennialClimate(min_years=100)

    # Find longest-running stations
    stations = analyzer.find_longest_stations()

    if stations.empty:
        print("\n❌ No stations found with 100+ years of data")
        return

    # Analyze the top station
    top_station = stations.iloc[0]
    station_id = str(top_station['station_id'])
    station_name = top_station['name']

    print(f"\n🎯 Analyzing primary station: {station_name} (ID: {station_id})")
    print(f"   Data range: {top_station['data_range']:.1f} years "
          f"({top_station['start_date']} to {top_station['end_date']})")

    # Fetch data
    data = analyzer.fetch_station_data(station_id, station_name)

    if data is None or data.empty:
        print("\n❌ Failed to retrieve station data")
        return

    # Calculate annual statistics
    annual_stats = analyzer.calculate_annual_statistics(data)

    if annual_stats.empty or len(annual_stats) < 50:
        print(f"\n❌ Insufficient data for analysis (only {len(annual_stats)} years)")
        return

    # Generate visualization
    analyzer.plot_centennial_climate_trends(
        annual_stats,
        station_name,
        save_path="plots/germany_centennial_climate_dwd.png"
    )

    # Generate summary report
    analyzer.generate_summary_report(annual_stats, station_name)

    print("✅ Analysis complete!")
    print(f"📊 Analyzed {len(annual_stats)} years of data from {station_name}")
    print(f"📈 Visualization saved to: plots/germany_centennial_climate_dwd.png")


if __name__ == "__main__":
    main()

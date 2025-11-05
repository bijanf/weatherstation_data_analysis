"""
Germany's Centennial Weather Stations Analysis
===============================================

Analyzes 100+ years of weather data from Germany's longest-running weather stations.
Visualizes warming trends and cumulative rainfall patterns with extreme years highlighted.

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
from matplotlib.patches import Rectangle
from meteostat import Daily, Stations
from scipy import stats

warnings.filterwarnings("ignore")


class GermanCentennialStations:
    """
    Analyzes weather data from Germany's longest-running weather stations.

    Attributes:
        min_years: Minimum years of data required (default: 100)
        stations_info: DataFrame with station metadata
        stations_data: Dict mapping station IDs to weather data
    """

    def __init__(self, min_years: int = 100):
        """
        Initialize the analyzer.

        Args:
            min_years: Minimum number of years of data required
        """
        self.min_years = min_years
        self.stations_info: Optional[pd.DataFrame] = None
        self.stations_data: Dict[str, Dict[int, pd.DataFrame]] = {}

    def find_longest_stations(self, country_code: str = "DE") -> pd.DataFrame:
        """
        Find weather stations with longest records in Germany.

        Args:
            country_code: ISO country code (default: "DE" for Germany)

        Returns:
            DataFrame with station information sorted by data availability
        """
        print(f"🔍 Searching for German stations with {self.min_years}+ years of data...")

        # Fetch all stations in Germany
        stations = Stations()
        stations = stations.region(country_code)
        all_stations = stations.fetch()

        if all_stations.empty:
            print("❌ No stations found in Germany")
            return pd.DataFrame()

        print(f"📊 Found {len(all_stations)} stations in Germany")

        # Filter stations with sufficient data range
        all_stations = all_stations.copy()
        all_stations['data_range'] = (
            pd.to_datetime(all_stations['daily_end']) -
            pd.to_datetime(all_stations['daily_start'])
        ).dt.days / 365.25

        # Filter for stations with 100+ years
        long_stations = all_stations[all_stations['data_range'] >= self.min_years].copy()
        long_stations = long_stations.sort_values('data_range', ascending=False)

        print(f"\n✅ Found {len(long_stations)} stations with {self.min_years}+ years of data:")
        print("-" * 80)

        for idx, row in long_stations.head(10).iterrows():
            print(f"📍 {row['name']:35s} | {row['data_range']:.1f} years | "
                  f"{row['daily_start']} to {row['daily_end']}")

        self.stations_info = long_stations
        return long_stations

    def fetch_station_data(
        self,
        station_id: str,
        start_year: int = 1880,
        end_year: int = 2025
    ) -> Optional[Dict[int, pd.DataFrame]]:
        """
        Fetch complete weather data for a specific station.

        Args:
            station_id: Meteostat station identifier
            start_year: Start year for data retrieval
            end_year: End year for data retrieval

        Returns:
            Dict mapping years to DataFrames with weather data
        """
        station_name = "Unknown"
        if self.stations_info is not None and station_id in self.stations_info.index:
            station_name = self.stations_info.loc[station_id, 'name']

        print(f"\n📡 Fetching data for {station_name} ({station_id})...")

        years_data = {}

        for year in range(start_year, end_year + 1):
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)

            try:
                data = Daily(station_id, start_date, end_date).fetch()

                if not data.empty and 'tavg' in data.columns and 'prcp' in data.columns:
                    # Check data coverage
                    expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
                    actual_temp_days = data['tavg'].notna().sum()
                    actual_prcp_days = data['prcp'].notna().sum()

                    temp_coverage = (actual_temp_days / expected_days) * 100
                    prcp_coverage = (actual_prcp_days / expected_days) * 100

                    # Require at least 80% coverage
                    if temp_coverage >= 80 and prcp_coverage >= 80:
                        years_data[year] = data[['tavg', 'tmin', 'tmax', 'prcp']].copy()
                        if year % 10 == 0:  # Print every 10th year
                            print(f"  ✓ {year}: {actual_temp_days}/{expected_days} days "
                                  f"({temp_coverage:.0f}% temp, {prcp_coverage:.0f}% prcp)")

            except Exception as e:
                if year % 10 == 0:
                    print(f"  ✗ {year}: {e}")

        print(f"✅ Retrieved {len(years_data)} years of data for {station_name}")

        self.stations_data[station_id] = years_data
        return years_data if years_data else None

    def calculate_annual_statistics(
        self,
        station_data: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate annual temperature and precipitation statistics.

        Args:
            station_data: Dict mapping years to weather DataFrames

        Returns:
            DataFrame with annual statistics
        """
        annual_stats = []

        for year, data in sorted(station_data.items()):
            stats_dict = {
                'year': year,
                'mean_temp': data['tavg'].mean(),
                'max_temp': data['tmax'].max(),
                'min_temp': data['tmin'].min(),
                'total_prcp': data['prcp'].sum(),
                'n_days': len(data)
            }
            annual_stats.append(stats_dict)

        df = pd.DataFrame(annual_stats)

        # Calculate temperature anomaly (deviation from long-term mean)
        baseline_period = df[(df['year'] >= 1961) & (df['year'] <= 1990)]
        if len(baseline_period) > 0:
            baseline_mean = baseline_period['mean_temp'].mean()
            df['temp_anomaly'] = df['mean_temp'] - baseline_mean
        else:
            # Fallback to overall mean
            df['temp_anomaly'] = df['mean_temp'] - df['mean_temp'].mean()

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

        # Precipitation extremes
        prcp_std = annual_stats['total_prcp'].std()
        prcp_mean = annual_stats['total_prcp'].mean()

        for _, row in annual_stats.iterrows():
            year = int(row['year'])
            temp_z = (row['temp_anomaly'] - temp_mean) / temp_std
            prcp_z = (row['total_prcp'] - prcp_mean) / prcp_std

            # Classify extremes
            if temp_z > temp_threshold:
                extreme_years[year] = 'hot'
            elif temp_z < -temp_threshold:
                extreme_years[year] = 'cold'
            elif prcp_z > prcp_threshold:
                extreme_years[year] = 'wet'
            elif prcp_z < -prcp_threshold:
                extreme_years[year] = 'dry'

        return extreme_years

    def plot_centennial_climate_trends(
        self,
        annual_stats: pd.DataFrame,
        station_name: str,
        save_path: str = "plots/germany_centennial_climate_trends.png"
    ):
        """
        Create comprehensive visualization of centennial climate trends.

        Args:
            annual_stats: DataFrame with annual statistics
            station_name: Name of the weather station
            save_path: Path to save the plot
        """
        print(f"\n🎨 Creating climate trends visualization for {station_name}...")

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

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle(
            f'Century-Long Climate Analysis: {station_name}\n'
            f'{int(annual_stats["year"].min())}–{int(annual_stats["year"].max())} '
            f'({len(annual_stats)} years)',
            fontsize=16, fontweight='bold', y=0.995
        )

        # Recent year threshold (highlight last 30 years)
        recent_year_threshold = annual_stats['year'].max() - 30

        # --- Panel 1: Temperature Anomaly (Warming Trend) ---
        print("  📊 Plotting temperature anomalies...")

        for _, row in annual_stats.iterrows():
            year = int(row['year'])
            anomaly = row['temp_anomaly']

            # Determine color
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
                label=f'Trend: +{z[0]*100:.2f}°C per century')

        # Calculate and display warming rate
        early_period = annual_stats[annual_stats['year'] < 1950]['temp_anomaly'].mean()
        late_period = annual_stats[annual_stats['year'] >= 1990]['temp_anomaly'].mean()
        warming = late_period - early_period

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

        # Add text box with warming statistics
        textstr = f'Warming: {warming:.2f}°C\n(pre-1950 to post-1990)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

        # --- Panel 2: Annual Cumulative Precipitation ---
        print("  📊 Plotting precipitation totals...")

        for _, row in annual_stats.iterrows():
            year = int(row['year'])
            prcp = row['total_prcp']

            # Determine color
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
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors['hot'], linewidth=3, label='Extreme Hot'),
            Line2D([0], [0], color=colors['cold'], linewidth=3, label='Extreme Cold'),
            Line2D([0], [0], color=colors['wet'], linewidth=3, label='Extreme Wet'),
            Line2D([0], [0], color=colors['dry'], linewidth=3, label='Extreme Dry'),
            Line2D([0], [0], color=colors['recent'], linewidth=3, label='Recent Years (1995+)'),
            Line2D([0], [0], color=colors['normal'], linewidth=2, label='Normal Years'),
        ]

        ax2.legend(handles=legend_elements, loc='upper right',
                  fontsize=10, framealpha=0.9, ncol=2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")

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
        print(f"\n" + "="*80)
        print(f"CLIMATE SUMMARY: {station_name}")
        print("="*80)

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
        print(f"  All-time record high: {annual_stats['max_temp'].max():.1f}°C "
              f"(year {int(annual_stats.loc[annual_stats['max_temp'].idxmax(), 'year'])})")
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
            print(f"  Warming: +{warming:.2f}°C")

        # Precipitation statistics
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
                print(f"  {extreme_type.capitalize()}: {', '.join(map(str, sorted(years)[:10]))}"
                      + (f" ... ({len(years)} total)" if len(years) > 10 else ""))

        print("="*80 + "\n")


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("GERMANY'S CENTENNIAL WEATHER STATIONS ANALYSIS")
    print("Analyzing 100+ years of temperature and precipitation data")
    print("="*80 + "\n")

    # Initialize analyzer
    analyzer = GermanCentennialStations(min_years=100)

    # Find longest-running stations
    stations = analyzer.find_longest_stations()

    if stations.empty:
        print("❌ No stations found with 100+ years of data")
        return

    # Analyze the top station with most data
    top_station_id = stations.index[0]
    station_name = stations.loc[top_station_id, 'name']

    print(f"\n🎯 Analyzing primary station: {station_name}")

    # Fetch data
    data = analyzer.fetch_station_data(
        top_station_id,
        start_year=1880,
        end_year=2025
    )

    if not data:
        print("❌ Failed to retrieve station data")
        return

    # Calculate annual statistics
    annual_stats = analyzer.calculate_annual_statistics(data)

    # Generate visualization
    analyzer.plot_centennial_climate_trends(
        annual_stats,
        station_name,
        save_path="plots/germany_centennial_climate_trends.png"
    )

    # Generate summary report
    analyzer.generate_summary_report(annual_stats, station_name)

    print("✅ Analysis complete!")
    print(f"📊 Analyzed {len(annual_stats)} years of data")
    print(f"📈 Plot saved to: plots/germany_centennial_climate_trends.png")


if __name__ == "__main__":
    main()

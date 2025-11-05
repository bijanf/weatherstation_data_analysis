"""
Germany's Centennial Weather Stations - Simplified Version
===========================================================

Uses known German weather stations with 100+ years of records.
This version bypasses API complexities by using well-known station IDs.
"""

import warnings
from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# Known German weather stations with 100+ years of records
# Data from DWD Climate Data Center
GERMAN_CENTENNIAL_STATIONS = {
    '00433': {'name': 'Hohenpeißenberg', 'start': 1781, 'lat': 47.8011, 'lon': 11.0089},
    '01048': {'name': 'Dresden-Klotzsche', 'start': 1886, 'lat': 51.1283, 'lon': 13.7536},
    '01214': {'name': 'Jena', 'start': 1896, 'lat': 50.9283, 'lon': 11.5892},
    '01766': {'name': 'Kassel', 'start': 1891, 'lat': 51.2997, 'lon': 9.4433},
    '02014': {'name': 'Leck', 'start': 1887, 'lat': 54.7667, 'lon': 8.9667},
    '02159': {'name': 'List auf Sylt', 'start': 1898, 'lat': 55.0167, 'lon': 8.4167},
    '02290': {'name': 'Hamburg-Fuhlsbüttel', 'start': 1891, 'lat': 53.6342, 'lon': 10.0000},
    '02559': {'name': 'München-Nymphenburg', 'start': 1879, 'lat': 48.1619, 'lon': 11.5036},
    '02834': {'name': 'Nürnberg', 'start': 1879, 'lat': 49.5025, 'lon': 11.0542},
    '02925': {'name': 'Potsdam', 'start': 1893, 'lat': 52.3833, 'lon': 13.0667},
    '03379': {'name': 'Rostock-Warnemünde', 'start': 1947, 'lat': 54.1775, 'lon': 12.0819},
    '04270': {'name': 'Trier-Petrisberg', 'start': 1951, 'lat': 49.7489, 'lon': 6.6578},
    '04887': {'name': 'Würzburg', 'start': 1879, 'lat': 49.7686, 'lon': 9.9572},
}


try:
    from wetterdienst import Parameter
    from wetterdienst.provider.dwd.observation import DwdObservationRequest
    WETTERDIENST_AVAILABLE = True
except:
    WETTERDIENST_AVAILABLE = False


class GermanCentennialAnalysis:
    """Analyze century-long weather data from German stations"""

    def __init__(self):
        self.station_data = {}

    def fetch_station_data_wetterdienst(self, station_id: str, station_info: dict) -> Optional[pd.DataFrame]:
        """Fetch data using wetterdienst library"""
        if not WETTERDIENST_AVAILABLE:
            return None

        print(f"  Fetching data for {station_info['name']}...")
        try:
            request = DwdObservationRequest(
                parameters=[Parameter.TEMPERATURE_AIR_MEAN_2M, Parameter.PRECIPITATION_HEIGHT],
                start_date=datetime(1800, 1, 1),
                end_date=datetime(2025, 12, 31)
            ).filter_by_station_id(station_id=station_id)

            df = request.values.all().df
            if not df.empty:
                df_pivot = df.pivot_table(index='date', columns='parameter', values='value')
                print(f"    ✓ Retrieved {len(df_pivot)} days")
                return df_pivot
        except Exception as e:
            print(f"    ✗ Error: {e}")

        return None

    def fetch_all_stations(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all centennial stations"""
        print("\n📡 Fetching data for German centennial stations...")

        for station_id, station_info in list(GERMAN_CENTENNIAL_STATIONS.items())[:3]:  # Limit to first 3 for faster testing
            data = self.fetch_station_data_wetterdienst(station_id, station_info)
            if data is not None:
                self.station_data[station_id] = {
                    'data': data,
                    'info': station_info
                }
                break  # Use first successful station

        return self.station_data

    def calculate_annual_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate annual statistics from daily data"""
        data = data.copy()
        data['year'] = data.index.year

        annual_stats = []
        for year, year_data in data.groupby('year'):
            temp_col = 'temperature_air_mean_2m'
            prcp_col = 'precipitation_height'

            if temp_col in year_data.columns:
                expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
                actual_days = year_data[temp_col].notna().sum()
                coverage = (actual_days / expected_days) * 100

                if coverage >= 80:
                    stats = {
                        'year': int(year),
                        'mean_temp': year_data[temp_col].mean(),
                        'n_days': actual_days
                    }

                    if prcp_col in year_data.columns:
                        stats['total_prcp'] = year_data[prcp_col].sum()

                    annual_stats.append(stats)

        df = pd.DataFrame(annual_stats)

        # Calculate temperature anomaly
        baseline = df[(df['year'] >= 1961) & (df['year'] <= 1990)]
        if len(baseline) >= 20:
            baseline_mean = baseline['mean_temp'].mean()
        else:
            baseline_mean = df['mean_temp'].mean()

        df['temp_anomaly'] = df['mean_temp'] - baseline_mean

        return df

    def identify_extreme_years(self, annual_stats: pd.DataFrame) -> Dict[int, str]:
        """Identify extreme years"""
        extreme_years = {}
        threshold = 1.5

        temp_std = annual_stats['temp_anomaly'].std()
        temp_mean = annual_stats['temp_anomaly'].mean()

        if 'total_prcp' in annual_stats.columns:
            prcp_std = annual_stats['total_prcp'].std()
            prcp_mean = annual_stats['total_prcp'].mean()
            has_prcp = True
        else:
            has_prcp = False

        for _, row in annual_stats.iterrows():
            year = int(row['year'])
            temp_z = (row['temp_anomaly'] - temp_mean) / temp_std

            if temp_z > threshold:
                extreme_years[year] = 'hot'
            elif temp_z < -threshold:
                extreme_years[year] = 'cold'
            elif has_prcp and not pd.isna(row.get('total_prcp')):
                prcp_z = (row['total_prcp'] - prcp_mean) / prcp_std
                if prcp_z > threshold:
                    extreme_years[year] = 'wet'
                elif prcp_z < -threshold:
                    extreme_years[year] = 'dry'

        return extreme_years

    def plot_climate_trends(self, annual_stats: pd.DataFrame, station_name: str, save_path: str):
        """Create visualization"""
        print(f"\n🎨 Creating visualization...")

        extreme_years = self.identify_extreme_years(annual_stats)

        colors = {
            'hot': '#d62728', 'cold': '#1f77b4',
            'wet': '#2ca02c', 'dry': '#ff7f0e',
            'normal': '#7f7f7f', 'recent': '#9467bd'
        }

        has_prcp = 'total_prcp' in annual_stats.columns and annual_stats['total_prcp'].notna().sum() > 50

        if has_prcp:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))

        fig.suptitle(
            f'Century-Long Climate Analysis: {station_name}\n'
            f'{int(annual_stats["year"].min())}–{int(annual_stats["year"].max())} '
            f'({len(annual_stats)} years)',
            fontsize=16, fontweight='bold', y=0.98
        )

        recent_threshold = annual_stats['year'].max() - 30

        # Temperature anomaly plot
        for _, row in annual_stats.iterrows():
            year = int(row['year'])
            anomaly = row['temp_anomaly']

            if year in extreme_years:
                color, alpha, lw = colors[extreme_years[year]], 0.8, 2
            elif year > recent_threshold:
                color, alpha, lw = colors['recent'], 0.7, 1.5
            else:
                color, alpha, lw = colors['normal'], 0.4, 1

            ax1.plot([year, year], [0, anomaly], color=color, alpha=alpha, linewidth=lw)

        # Trend line
        z = np.polyfit(annual_stats['year'], annual_stats['temp_anomaly'], 1)
        p = np.poly1d(z)
        ax1.plot(annual_stats['year'], p(annual_stats['year']),
                'k--', linewidth=2.5, alpha=0.8,
                label=f'Linear trend: +{z[0]*100:.2f}°C per century')

        early = annual_stats[annual_stats['year'] < 1950]
        late = annual_stats[annual_stats['year'] >= 1990]
        if len(early) > 0 and len(late) > 0:
            warming = late['mean_temp'].mean() - early['mean_temp'].mean()
            textstr = f'Warming: +{warming:.2f}°C\n(pre-1950 to post-1990)'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temperature Anomaly (°C)\nrelative to 1961-1990', fontsize=12, fontweight='bold')
        ax1.set_title('Annual Mean Temperature Anomaly - Warming Trend Visible in Recent Decades',
                     fontsize=13, pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)

        # Precipitation plot (if available)
        if has_prcp:
            for _, row in annual_stats.iterrows():
                if pd.isna(row['total_prcp']):
                    continue

                year = int(row['year'])
                prcp = row['total_prcp']

                if year in extreme_years:
                    color, alpha, lw = colors[extreme_years[year]], 0.8, 2
                elif year > recent_threshold:
                    color, alpha, lw = colors['recent'], 0.7, 1.5
                else:
                    color, alpha, lw = colors['normal'], 0.4, 1

                ax2.plot([year, year], [0, prcp], color=color, alpha=alpha, linewidth=lw)

            mean_prcp = annual_stats['total_prcp'].mean()
            ax2.axhline(y=mean_prcp, color='black', linestyle='--',
                       linewidth=2, alpha=0.7, label=f'Mean: {mean_prcp:.0f} mm')

            ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Annual Precipitation (mm)', fontsize=12, fontweight='bold')
            ax2.set_title('Annual Total Precipitation - Extreme Dry and Wet Years Highlighted',
                         fontsize=13, pad=10)
            ax2.grid(True, alpha=0.3, linestyle='--')

            legend_elements = [
                Line2D([0], [0], color=colors['hot'], linewidth=3, label='Extreme Hot'),
                Line2D([0], [0], color=colors['cold'], linewidth=3, label='Extreme Cold'),
                Line2D([0], [0], color=colors['wet'], linewidth=3, label='Extreme Wet'),
                Line2D([0], [0], color=colors['dry'], linewidth=3, label='Extreme Dry'),
                Line2D([0], [0], color=colors['recent'], linewidth=3, label='Recent Years (1995+)'),
                Line2D([0], [0], color=colors['normal'], linewidth=2, label='Normal Years'),
            ]
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9, ncol=2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")

    def print_summary(self, annual_stats: pd.DataFrame, station_name: str):
        """Print summary statistics"""
        print(f"\n{'='*90}")
        print(f"CLIMATE SUMMARY: {station_name}")
        print(f"{'='*90}")

        print(f"\n📊 Data Coverage: {int(annual_stats['year'].min())}–{int(annual_stats['year'].max())} ({len(annual_stats)} years)")

        print(f"\n🌡️  Temperature:")
        print(f"  Mean: {annual_stats['mean_temp'].mean():.2f}°C")
        print(f"  Warmest year: {int(annual_stats.loc[annual_stats['mean_temp'].idxmax(), 'year'])} "
              f"({annual_stats['mean_temp'].max():.2f}°C)")
        print(f"  Coldest year: {int(annual_stats.loc[annual_stats['mean_temp'].idxmin(), 'year'])} "
              f"({annual_stats['mean_temp'].min():.2f}°C)")

        early = annual_stats[annual_stats['year'] < 1950]
        late = annual_stats[annual_stats['year'] >= 1990]
        if len(early) > 0 and len(late) > 0:
            warming = late['mean_temp'].mean() - early['mean_temp'].mean()
            print(f"\n🔥 Warming: +{warming:.2f}°C (pre-1950 to post-1990)")

        if 'total_prcp' in annual_stats.columns:
            print(f"\n🌧️  Precipitation:")
            print(f"  Mean annual: {annual_stats['total_prcp'].mean():.0f} mm")
            print(f"  Wettest year: {int(annual_stats.loc[annual_stats['total_prcp'].idxmax(), 'year'])} "
                  f"({annual_stats['total_prcp'].max():.0f} mm)")
            print(f"  Driest year: {int(annual_stats.loc[annual_stats['total_prcp'].idxmin(), 'year'])} "
                  f"({annual_stats['total_prcp'].min():.0f} mm)")

        print(f"{'='*90}\n")


def main():
    print("="*90)
    print("GERMANY'S CENTENNIAL WEATHER STATIONS ANALYSIS")
    print("="*90 + "\n")

    if not WETTERDIENST_AVAILABLE:
        print("❌ wetterdienst library required. Install with: pip install wetterdienst")
        return

    analyzer = GermanCentennialAnalysis()
    stations_data = analyzer.fetch_all_stations()

    if not stations_data:
        print("❌ No data could be retrieved")
        return

    # Use the first successful station
    station_id = list(stations_data.keys())[0]
    station_info = stations_data[station_id]

    print(f"\n🎯 Analyzing: {station_info['info']['name']}")

    # Calculate statistics
    annual_stats = analyzer.calculate_annual_statistics(station_info['data'])

    if len(annual_stats) < 50:
        print(f"❌ Insufficient data ({len(annual_stats)} years)")
        return

    # Create visualization
    analyzer.plot_climate_trends(
        annual_stats,
        station_info['info']['name'],
        "plots/germany_centennial_climate.png"
    )

    # Print summary
    analyzer.print_summary(annual_stats, station_info['info']['name'])

    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()

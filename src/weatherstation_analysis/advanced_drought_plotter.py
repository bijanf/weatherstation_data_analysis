"""
Advanced Drought Visualization Module
=====================================

Publication-quality visualizations for advanced drought analysis:
- Return period plots with Gumbel/GEV fits
- Compound drought-heat event visualizations
- DSA (Duration-Severity-Area) curves
- Multi-station spatial analysis
- Regime shift visualizations
- Historical context plots

Designed for scientific publication on Iran's 2018-2025 megadrought.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


class AdvancedDroughtPlotter:
    """
    Creates advanced visualizations for drought analysis.

    Generates publication-quality plots suitable for high-impact journals.
    """

    # Color schemes
    COLORS = {
        'drought_severe': '#8B0000',
        'drought_moderate': '#FF6347',
        'drought_mild': '#FFA500',
        'normal': '#228B22',
        'wet': '#4169E1',
        'compound': '#800080',
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'highlight': '#C73E1D'
    }

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """Initialize plotter with style."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")

    def plot_return_period_analysis(
        self,
        ranked_droughts: pd.DataFrame,
        return_levels: pd.DataFrame,
        gumbel_params: Dict[str, float],
        drought_highlight_years: Optional[List[int]] = None,
        station_name: str = "Station",
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot return period analysis with Gumbel fit.

        Args:
            ranked_droughts: DataFrame from get_drought_ranking()
            return_levels: DataFrame from calculate_return_levels()
            gumbel_params: Gumbel distribution parameters
            drought_highlight_years: Years to highlight (e.g., [2018, 2019, ...])
            station_name: Name of station
            output_file: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Return period plot (Gumbel probability paper)
        ax1 = axes[0]

        # Calculate empirical return periods
        n = len(ranked_droughts)
        ranked_droughts = ranked_droughts.sort_values('deficit_percent', ascending=False).reset_index(drop=True)
        ranked_droughts['empirical_rp'] = (n + 1) / (ranked_droughts.index + 1)

        # Plot empirical points
        highlight_mask = ranked_droughts['year'].isin(drought_highlight_years or [])

        ax1.scatter(
            ranked_droughts.loc[~highlight_mask, 'empirical_rp'],
            ranked_droughts.loc[~highlight_mask, 'deficit_percent'],
            c=self.COLORS['primary'], s=60, alpha=0.6, label='Historical droughts',
            edgecolors='white', linewidth=0.5
        )

        if highlight_mask.any():
            ax1.scatter(
                ranked_droughts.loc[highlight_mask, 'empirical_rp'],
                ranked_droughts.loc[highlight_mask, 'deficit_percent'],
                c=self.COLORS['highlight'], s=120, alpha=0.9,
                label='2018-2025 drought years', marker='*',
                edgecolors='black', linewidth=0.5
            )

            # Annotate highlighted years
            for _, row in ranked_droughts[highlight_mask].iterrows():
                ax1.annotate(
                    str(int(row['year'])),
                    (row['empirical_rp'], row['deficit_percent']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color=self.COLORS['highlight']
                )

        # Plot theoretical Gumbel curve
        rp_range = np.logspace(0, 3, 100)
        deficit_theoretical = stats.gumbel_r.ppf(
            1 - 1/rp_range,
            gumbel_params['location'],
            gumbel_params['scale']
        )
        ax1.plot(rp_range, deficit_theoretical, 'k-', linewidth=2,
                label='Gumbel fit', zorder=1)

        # Add return level markers
        for _, row in return_levels.iterrows():
            ax1.axhline(row['deficit_percent'], color='gray', linestyle='--',
                       alpha=0.3, linewidth=1)
            ax1.text(1.1, row['deficit_percent'],
                    f"{int(row['return_period_years'])}-yr: {row['deficit_percent']:.0f}%",
                    fontsize=8, va='center')

        ax1.set_xscale('log')
        ax1.set_xlabel('Return Period (years)', fontweight='bold')
        ax1.set_ylabel('Precipitation Deficit (%)', fontweight='bold')
        ax1.set_title('Drought Return Period Analysis', fontweight='bold', fontsize=13)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 200)

        # Panel 2: Historical ranking bar chart
        ax2 = axes[1]

        # Show top 20 droughts
        top_droughts = ranked_droughts.head(20).copy()
        top_droughts = top_droughts.sort_values('deficit_percent', ascending=True)

        colors = [
            self.COLORS['highlight'] if y in (drought_highlight_years or [])
            else self.COLORS['primary']
            for y in top_droughts['year']
        ]

        bars = ax2.barh(
            range(len(top_droughts)),
            top_droughts['deficit_percent'],
            color=colors, alpha=0.8, edgecolor='white'
        )

        ax2.set_yticks(range(len(top_droughts)))
        ax2.set_yticklabels([str(int(y)) for y in top_droughts['year']])
        ax2.set_xlabel('Precipitation Deficit (%)', fontweight='bold')
        ax2.set_ylabel('Year', fontweight='bold')
        ax2.set_title('Top 20 Drought Years Ranked', fontweight='bold', fontsize=13)

        # Add return period labels
        for i, (idx, row) in enumerate(top_droughts.iterrows()):
            rp = row.get('return_period_gev', row.get('return_period_gumbel', np.nan))
            if not np.isnan(rp) and rp < 1000:
                ax2.text(row['deficit_percent'] + 1, i, f'{rp:.0f}-yr',
                        va='center', fontsize=8, fontweight='bold')

        ax2.axvline(0, color='black', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_file}")

        plt.close()

    def plot_compound_event_analysis(
        self,
        annual_anomalies: pd.DataFrame,
        probability_stats: Dict[str, float],
        drought_period: Tuple[int, int],
        station_name: str = "Station",
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot compound drought-heat event analysis.

        Args:
            annual_anomalies: DataFrame with annual precipitation and temperature anomalies
            probability_stats: Dict with joint/conditional probabilities
            drought_period: Tuple of (start_year, end_year)
            station_name: Name of station
            output_file: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Panel 1: Scatter plot of anomalies
        ax1 = fig.add_subplot(gs[0, 0:2])

        # Color by event type
        colors = []
        for _, row in annual_anomalies.iterrows():
            if row['is_compound']:
                colors.append(self.COLORS['compound'])
            elif row['is_drought']:
                colors.append(self.COLORS['drought_moderate'])
            elif row['is_heat']:
                colors.append(self.COLORS['accent'])
            else:
                colors.append(self.COLORS['normal'])

        scatter = ax1.scatter(
            annual_anomalies['prcp_anomaly_std'],
            annual_anomalies['temp_anomaly_std'],
            c=colors, s=80, alpha=0.7, edgecolors='white', linewidth=0.5
        )

        # Highlight drought period
        drought_mask = (
            (annual_anomalies['year'] >= drought_period[0]) &
            (annual_anomalies['year'] <= drought_period[1])
        )
        drought_data = annual_anomalies[drought_mask]

        ax1.scatter(
            drought_data['prcp_anomaly_std'],
            drought_data['temp_anomaly_std'],
            s=150, facecolors='none', edgecolors='black', linewidth=2,
            label=f'{drought_period[0]}-{drought_period[1]}'
        )

        # Add year labels for drought period
        for _, row in drought_data.iterrows():
            ax1.annotate(
                str(int(row['year'])),
                (row['prcp_anomaly_std'], row['temp_anomaly_std']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold'
            )

        # Add quadrant lines and shading
        ax1.axhline(1, color='orange', linestyle='--', alpha=0.5)
        ax1.axhline(-1, color='blue', linestyle='--', alpha=0.5)
        ax1.axvline(1, color='blue', linestyle='--', alpha=0.5)
        ax1.axvline(-1, color='orange', linestyle='--', alpha=0.5)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(0, color='black', linestyle='-', alpha=0.3)

        # Shade compound region
        ax1.fill_between([-4, -1], [1, 1], [4, 4], alpha=0.1, color=self.COLORS['compound'])
        ax1.text(-2.5, 2.5, 'COMPOUND\nDROUGHT-HEAT', ha='center', va='center',
                fontsize=10, fontweight='bold', color=self.COLORS['compound'])

        ax1.set_xlabel('Precipitation Anomaly (standardized)', fontweight='bold')
        ax1.set_ylabel('Temperature Anomaly (standardized)', fontweight='bold')
        ax1.set_title('Compound Drought-Heat Event Analysis', fontweight='bold', fontsize=13)
        ax1.set_xlim(-3.5, 3.5)
        ax1.set_ylim(-3.5, 3.5)
        ax1.grid(True, alpha=0.3)

        # Custom legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['compound'],
                   markersize=10, label='Compound event'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['drought_moderate'],
                   markersize=10, label='Drought only'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['accent'],
                   markersize=10, label='Heat only'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['normal'],
                   markersize=10, label='Normal'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Panel 2: Probability statistics
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')

        prob_text = f"""
COMPOUND EVENT STATISTICS
{station_name}
{'─' * 35}

MARGINAL PROBABILITIES
  P(Drought):     {probability_stats['p_drought']*100:.1f}%
  P(Heat):        {probability_stats['p_heat']*100:.1f}%

JOINT PROBABILITIES
  P(Compound) observed:    {probability_stats['p_compound_observed']*100:.1f}%
  P(Compound) if independent: {probability_stats['p_compound_independent']*100:.1f}%

DEPENDENCE
  Ratio: {probability_stats['dependence_ratio']:.2f}x
  {"Events MORE likely together" if probability_stats['dependence_ratio'] > 1 else "Events LESS likely together"}

CONDITIONAL PROBABILITIES
  P(Heat | Drought):  {probability_stats['p_heat_given_drought']*100:.1f}%
  P(Drought | Heat):  {probability_stats['p_drought_given_heat']*100:.1f}%

COUNTS ({probability_stats['n_years_total']} years)
  Drought years:    {probability_stats['n_drought_years']}
  Heat years:       {probability_stats['n_heat_years']}
  Compound years:   {probability_stats['n_compound_years']}
"""
        ax2.text(0.1, 0.95, prob_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Panel 3: Time series of anomalies
        ax3 = fig.add_subplot(gs[1, :])

        years = annual_anomalies['year'].values
        width = 0.4

        # Precipitation bars (inverted so deficit is up)
        prcp_bars = ax3.bar(
            years - width/2,
            -annual_anomalies['prcp_anomaly_std'],  # Invert so deficit is positive
            width, label='Precip. deficit (inverted)',
            color=self.COLORS['primary'], alpha=0.7
        )

        # Temperature bars
        temp_bars = ax3.bar(
            years + width/2,
            annual_anomalies['temp_anomaly_std'],
            width, label='Temp. anomaly',
            color=self.COLORS['accent'], alpha=0.7
        )

        # Highlight drought period
        ax3.axvspan(drought_period[0] - 0.5, drought_period[1] + 0.5,
                   alpha=0.2, color='red', label=f'Drought period')

        # Mark compound years
        compound_years = annual_anomalies[annual_anomalies['is_compound']]['year'].values
        for cy in compound_years:
            ax3.axvline(cy, color=self.COLORS['compound'], linestyle='--',
                       alpha=0.5, linewidth=1.5)

        ax3.axhline(0, color='black', linewidth=1)
        ax3.axhline(1, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(-1, color='gray', linestyle='--', alpha=0.5)

        ax3.set_xlabel('Year', fontweight='bold')
        ax3.set_ylabel('Standardized Anomaly', fontweight='bold')
        ax3.set_title('Precipitation Deficit and Temperature Anomaly Time Series',
                     fontweight='bold', fontsize=13)
        ax3.legend(loc='upper left', ncol=3)
        ax3.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Compound Drought-Heat Event Analysis - {station_name}',
                    fontsize=15, fontweight='bold', y=1.02)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_file}")

        plt.close()

    def plot_multi_station_drought_map(
        self,
        station_results: Dict[str, Dict],
        station_coords: Dict[str, Tuple[float, float]],
        metric: str = 'mean_annual_deficit_percent',
        drought_period: Tuple[int, int] = (2018, 2025),
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot multi-station drought analysis as a spatial map-like visualization.

        Args:
            station_results: Dict mapping station names to analysis results
            station_coords: Dict mapping station names to (lat, lon) tuples
            metric: Metric to visualize
            drought_period: Drought period being analyzed
            output_file: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        stations = list(station_results.keys())
        values = [station_results[s].get(metric, 0) for s in stations]
        lats = [station_coords[s][0] for s in stations]
        lons = [station_coords[s][1] for s in stations]

        # Panel 1: Geographic scatter
        ax1 = axes[0]

        # Normalize values for color mapping
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
        cmap = plt.cm.YlOrRd

        scatter = ax1.scatter(
            lons, lats, c=values, s=500, cmap=cmap, norm=norm,
            edgecolors='black', linewidth=2, alpha=0.8
        )

        # Add station labels with values
        for station, lon, lat, val in zip(stations, lons, lats, values):
            short_name = station.split('(')[0].strip()[:10]
            ax1.annotate(
                f'{short_name}\n{val:.0f}%',
                (lon, lat), fontsize=9, fontweight='bold',
                ha='center', va='center', color='white',
                path_effects=[patheffects.withStroke(linewidth=2, foreground='black')]
            )

        ax1.set_xlabel('Longitude (°E)', fontweight='bold')
        ax1.set_ylabel('Latitude (°N)', fontweight='bold')
        ax1.set_title(f'Spatial Distribution of Drought Severity\n{drought_period[0]}-{drought_period[1]}',
                     fontweight='bold', fontsize=13)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
        cbar.set_label('Precipitation Deficit (%)', fontweight='bold')

        # Add Iran outline approximation
        ax1.set_xlim(44, 64)
        ax1.set_ylim(25, 40)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Bar chart comparison
        ax2 = axes[1]

        # Sort by deficit
        sorted_idx = np.argsort(values)[::-1]
        sorted_stations = [stations[i] for i in sorted_idx]
        sorted_values = [values[i] for i in sorted_idx]

        colors = [cmap(norm(v)) for v in sorted_values]

        bars = ax2.barh(range(len(sorted_stations)), sorted_values,
                       color=colors, edgecolor='black', alpha=0.8)

        ax2.set_yticks(range(len(sorted_stations)))
        ax2.set_yticklabels([s.split('(')[0].strip() for s in sorted_stations])
        ax2.set_xlabel('Precipitation Deficit (%)', fontweight='bold')
        ax2.set_title('Station Ranking by Drought Severity', fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(sorted_values):
            ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)

        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.1f}%')
        ax2.legend(loc='lower right')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_file}")

        plt.close()

    def plot_dsa_analysis(
        self,
        dsa_timeseries: pd.DataFrame,
        drought_period: Tuple[int, int],
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot Duration-Severity-Area analysis.

        Args:
            dsa_timeseries: DataFrame with monthly DSA metrics
            drought_period: Drought period being analyzed
            output_file: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        dates = pd.to_datetime(dsa_timeseries['date'])

        # Panel 1: Area coverage
        ax1 = axes[0]
        ax1.fill_between(dates, 0, dsa_timeseries['area_fraction'] * 100,
                        color=self.COLORS['drought_moderate'], alpha=0.7)
        ax1.set_ylabel('Area Coverage (%)', fontweight='bold')
        ax1.set_title('Drought Area Coverage (% of stations in drought)', fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Severity
        ax2 = axes[1]
        ax2.fill_between(dates, 0, dsa_timeseries['mean_severity'],
                        color=self.COLORS['drought_severe'], alpha=0.7)
        ax2.set_ylabel('Mean Severity (|SPI|)', fontweight='bold')
        ax2.set_title('Drought Severity (mean absolute SPI of affected stations)', fontweight='bold')
        ax2.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Severe threshold')
        ax2.axhline(2.0, color='darkred', linestyle='--', alpha=0.5, label='Extreme threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Combined DSA index
        ax3 = axes[2]
        dsa_index = dsa_timeseries['area_fraction'] * dsa_timeseries['mean_severity']
        ax3.fill_between(dates, 0, dsa_index, color=self.COLORS['compound'], alpha=0.7)
        ax3.set_ylabel('DSA Index', fontweight='bold')
        ax3.set_xlabel('Year', fontweight='bold')
        ax3.set_title('Combined DSA Index (Area × Severity)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Highlight drought period on all panels
        for ax in axes:
            ax.axvspan(
                pd.Timestamp(f'{drought_period[0]}-01-01'),
                pd.Timestamp(f'{drought_period[1]}-12-31'),
                alpha=0.2, color='red'
            )

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_file}")

        plt.close()

    def plot_regime_shift_analysis(
        self,
        change_point_results: Dict[str, Any],
        decadal_trends: pd.DataFrame,
        moving_stats: pd.DataFrame,
        station_name: str = "Station",
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot regime shift and trend analysis.

        Args:
            change_point_results: Results from detect_change_points_cusum()
            decadal_trends: DataFrame with decadal statistics
            moving_stats: DataFrame with moving window statistics
            station_name: Name of station
            output_file: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Panel 1: CUSUM change point detection
        ax1 = fig.add_subplot(gs[0, 0])

        years = moving_stats['year'].values
        cusum = change_point_results['cusum_series']

        ax1.plot(years, cusum, 'b-', linewidth=2)
        ax1.fill_between(years, 0, cusum, where=(cusum >= 0), color='blue', alpha=0.3)
        ax1.fill_between(years, 0, cusum, where=(cusum < 0), color='red', alpha=0.3)

        change_year = change_point_results['change_point_year']
        ax1.axvline(change_year, color='red', linestyle='--', linewidth=2,
                   label=f'Change point: {change_year}')

        ax1.set_xlabel('Year', fontweight='bold')
        ax1.set_ylabel('CUSUM', fontweight='bold')
        ax1.set_title('CUSUM Change Point Analysis', fontweight='bold', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linewidth=1)

        # Panel 2: Before vs After comparison
        ax2 = fig.add_subplot(gs[0, 1])

        categories = ['Before\n' + str(change_year), 'After\n' + str(change_year)]
        means = [change_point_results['mean_before'], change_point_results['mean_after']]
        colors = [self.COLORS['normal'], self.COLORS['drought_moderate']]

        bars = ax2.bar(categories, means, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_ylabel('Mean Annual Precipitation (mm)', fontweight='bold')
        ax2.set_title('Precipitation Before vs After Change Point', fontweight='bold', fontsize=13)

        # Add change magnitude
        change_pct = change_point_results['change_percent']
        ax2.text(0.5, max(means) * 0.5,
                f'Change: {change_pct:+.1f}%\np-value: {change_point_results["p_value"]:.4f}',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{mean:.0f} mm', ha='center', fontweight='bold')

        ax2.grid(True, alpha=0.3, axis='y')

        # Panel 3: Moving average trend
        ax3 = fig.add_subplot(gs[1, 0])

        ax3.scatter(moving_stats['year'], moving_stats['precipitation'],
                   c=self.COLORS['primary'], s=30, alpha=0.5, label='Annual')
        ax3.plot(moving_stats['year'], moving_stats['moving_mean'],
                'r-', linewidth=2.5, label='10-year moving mean')
        ax3.fill_between(
            moving_stats['year'],
            moving_stats['moving_mean'] - moving_stats['moving_std'],
            moving_stats['moving_mean'] + moving_stats['moving_std'],
            alpha=0.2, color='red', label='±1 std'
        )

        ax3.set_xlabel('Year', fontweight='bold')
        ax3.set_ylabel('Precipitation (mm)', fontweight='bold')
        ax3.set_title('Long-term Precipitation Trend', fontweight='bold', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Decadal comparison
        ax4 = fig.add_subplot(gs[1, 1])

        decades = decadal_trends.index.astype(str) + 's'
        means = decadal_trends['mean_mm'].values
        drought_freq = decadal_trends['drought_frequency'].values * 100

        x = np.arange(len(decades))
        width = 0.35

        bars1 = ax4.bar(x - width/2, means, width, label='Mean precip. (mm)',
                       color=self.COLORS['primary'], alpha=0.8)
        ax4.set_ylabel('Mean Precipitation (mm)', fontweight='bold', color=self.COLORS['primary'])
        ax4.tick_params(axis='y', labelcolor=self.COLORS['primary'])

        ax4b = ax4.twinx()
        bars2 = ax4b.bar(x + width/2, drought_freq, width, label='Drought freq. (%)',
                        color=self.COLORS['drought_moderate'], alpha=0.8)
        ax4b.set_ylabel('Drought Frequency (%)', fontweight='bold',
                       color=self.COLORS['drought_moderate'])
        ax4b.tick_params(axis='y', labelcolor=self.COLORS['drought_moderate'])

        ax4.set_xticks(x)
        ax4.set_xticklabels(decades, rotation=45)
        ax4.set_title('Decadal Precipitation and Drought Frequency', fontweight='bold', fontsize=13)

        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.suptitle(f'Precipitation Regime Analysis - {station_name}',
                    fontsize=15, fontweight='bold', y=1.02)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_file}")

        plt.close()

    def plot_historical_context(
        self,
        historical_context: Dict[str, Any],
        annual_prcp: pd.Series,
        baseline_mean: float,
        drought_period: Tuple[int, int],
        station_name: str = "Station",
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot historical context analysis.

        Args:
            historical_context: Dict with historical context statistics
            annual_prcp: Series of annual precipitation
            baseline_mean: Baseline mean precipitation
            drought_period: Drought period being analyzed
            station_name: Name of station
            output_file: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        years = annual_prcp.index
        values = annual_prcp.values

        # Panel 1: Full time series with context
        ax1 = axes[0, 0]

        ax1.bar(years, values, color=self.COLORS['primary'], alpha=0.6)
        ax1.axhline(baseline_mean, color='green', linestyle='--', linewidth=2,
                   label=f'Baseline mean: {baseline_mean:.0f} mm')

        # Highlight drought period
        drought_mask = (years >= drought_period[0]) & (years <= drought_period[1])
        ax1.bar(years[drought_mask], values[drought_mask],
               color=self.COLORS['highlight'], alpha=0.9,
               label=f'{drought_period[0]}-{drought_period[1]} drought')

        ax1.set_xlabel('Year', fontweight='bold')
        ax1.set_ylabel('Annual Precipitation (mm)', fontweight='bold')
        ax1.set_title(f'Full Historical Record ({historical_context["record_start_year"]}-{historical_context["record_end_year"]})',
                     fontweight='bold', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Distribution with drought period highlighted
        ax2 = axes[0, 1]

        ax2.hist(values, bins=20, color=self.COLORS['primary'], alpha=0.6,
                edgecolor='white', label='All years')

        # Highlight drought years
        drought_values = values[drought_mask]
        for dv in drought_values:
            ax2.axvline(dv, color=self.COLORS['highlight'], linewidth=2, alpha=0.7)

        mean_drought = np.mean(drought_values)
        ax2.axvline(mean_drought, color=self.COLORS['highlight'], linewidth=3,
                   linestyle='--', label=f'Drought mean: {mean_drought:.0f} mm')
        ax2.axvline(baseline_mean, color='green', linewidth=3, linestyle='--',
                   label=f'Baseline mean: {baseline_mean:.0f} mm')

        ax2.set_xlabel('Annual Precipitation (mm)', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Precipitation Distribution', fontweight='bold', fontsize=13)
        ax2.legend()

        # Panel 3: Percentile ranking
        ax3 = axes[1, 0]

        percentile_rank = historical_context['percentile_rank']
        ax3.barh(['Current Drought'], [percentile_rank],
                color=self.COLORS['highlight'], edgecolor='black')
        ax3.set_xlim(0, 100)
        ax3.set_xlabel('Percentile (higher = more severe)', fontweight='bold')
        ax3.set_title('Drought Severity Percentile Ranking', fontweight='bold', fontsize=13)

        ax3.text(percentile_rank + 2, 0, f'{percentile_rank:.1f}th percentile',
                va='center', fontweight='bold', fontsize=12)

        # Add context text
        context_text = f"""
Only {historical_context['years_with_similar_or_worse']} years
out of {historical_context['total_years_in_record']} years
had similar or worse conditions
"""
        ax3.text(50, -0.3, context_text, ha='center', fontsize=10,
                style='italic')

        # Panel 4: Summary statistics box
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""
HISTORICAL CONTEXT SUMMARY
{station_name}
{'═' * 45}

CURRENT DROUGHT ({historical_context['drought_period']})
  Mean precipitation:  {historical_context['mean_precipitation_mm']:.1f} mm/year
  Deficit vs baseline: {historical_context['deficit_percent']:.1f}%
  Consecutive years:   {historical_context['consecutive_deficit_years']}

HISTORICAL COMPARISON
  Record period:       {historical_context['record_start_year']}-{historical_context['record_end_year']}
  Total years:         {historical_context['total_years_in_record']}

  Similar/worse years: {historical_context['years_with_similar_or_worse']}
  Percentile rank:     {historical_context['percentile_rank']:.1f}th percentile

  Max consecutive
  deficit (historical): {historical_context['max_consecutive_historical']} years

{'═' * 45}

INTERPRETATION:
"""
        if historical_context['percentile_rank'] >= 95:
            summary_text += "  ⚠️ UNPRECEDENTED - Top 5% most severe"
        elif historical_context['percentile_rank'] >= 90:
            summary_text += "  🔴 EXTREME - Top 10% most severe"
        elif historical_context['percentile_rank'] >= 75:
            summary_text += "  🟠 SEVERE - Top 25% most severe"
        else:
            summary_text += "  🟡 NOTABLE - Above median severity"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_file}")

        plt.close()

    def plot_all_stations_spi_heatmap(
        self,
        multi_station_spi: Dict[str, pd.DataFrame],
        drought_period: Tuple[int, int],
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot SPI heatmap for all stations over time.

        Args:
            multi_station_spi: Dict mapping station names to SPI DataFrames
            drought_period: Drought period to highlight
            output_file: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # Prepare data matrix
        stations = list(multi_station_spi.keys())

        # Find common date range
        all_dates = set()
        for df in multi_station_spi.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        # Create matrix
        spi_matrix = np.full((len(stations), len(all_dates)), np.nan)

        for i, station in enumerate(stations):
            df = multi_station_spi[station]
            spi_col = [c for c in df.columns if 'SPI' in c][0]
            for j, date in enumerate(all_dates):
                if date in df.index:
                    spi_matrix[i, j] = df.loc[date, spi_col]

        # Create heatmap
        cmap = plt.cm.RdYlBu
        im = ax.imshow(spi_matrix, aspect='auto', cmap=cmap, vmin=-3, vmax=3)

        # Set axis labels
        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels([s.split('(')[0].strip() for s in stations])

        # Set x-axis to show years
        year_indices = [i for i, d in enumerate(all_dates) if d.month == 1]
        year_labels = [all_dates[i].year for i in year_indices]
        ax.set_xticks(year_indices[::5])  # Show every 5 years
        ax.set_xticklabels(year_labels[::5])

        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Station', fontweight='bold')
        ax.set_title('Multi-Station SPI-12 Heatmap (Blue = Wet, Red = Drought)',
                    fontweight='bold', fontsize=13)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('SPI-12', fontweight='bold')
        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cbar.set_ticklabels(['<-3\nExtreme\nDrought', '-2', '-1', '0\nNormal', '1', '2', '>3\nExtreme\nWet'])

        # Highlight drought period
        start_idx = None
        end_idx = None
        for i, d in enumerate(all_dates):
            if d.year == drought_period[0] and d.month == 1:
                start_idx = i
            if d.year == drought_period[1] and d.month == 12:
                end_idx = i

        if start_idx and end_idx:
            ax.axvline(start_idx, color='black', linewidth=2, linestyle='--')
            ax.axvline(end_idx, color='black', linewidth=2, linestyle='--')
            ax.text(start_idx, -0.7, f'{drought_period[0]}', ha='center', fontweight='bold')
            ax.text(end_idx, -0.7, f'{drought_period[1]}', ha='center', fontweight='bold')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_file}")

        plt.close()

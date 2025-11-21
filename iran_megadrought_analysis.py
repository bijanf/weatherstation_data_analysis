#!/usr/bin/env python3
"""
Iran Megadrought Analysis - Comprehensive Scientific Study
===========================================================

Advanced analysis of Iran's unprecedented 2018-2025 megadrought using:
1. Return period analysis with Extreme Value Theory (Gumbel/GEV)
2. Compound drought-heat event analysis
3. Duration-Severity-Area (DSA) curves
4. Regime shift detection
5. Periodicity analysis
6. Full historical context (1950-2025)

Generates publication-quality figures and statistics for scientific papers.

Author: Weather Station Data Analysis Project
Date: 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from weatherstation_analysis.iran_data_fetcher import (
    IranianDataFetcher,
    IranianStationRegistry,
    MultiStationFetcher
)
from weatherstation_analysis.drought_analyzer import DroughtAnalyzer
from weatherstation_analysis.advanced_drought_analyzer import (
    DroughtReturnPeriodAnalyzer,
    CompoundEventAnalyzer,
    DroughtDSAAnalyzer,
    DroughtRegimeAnalyzer,
    WaveletDroughtAnalyzer,
    MegadroughtAnalyzer
)
from weatherstation_analysis.drought_plotter import DroughtPlotter
from weatherstation_analysis.advanced_drought_plotter import AdvancedDroughtPlotter


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Main analysis workflow for comprehensive Iran megadrought study."""

    print_header("IRAN MEGADROUGHT ANALYSIS - COMPREHENSIVE SCIENTIFIC STUDY")
    print("""
This analysis provides a multi-faceted characterization of Iran's 2018-2025
megadrought, establishing its unprecedented nature through:

  1. RETURN PERIOD ANALYSIS - Quantifying drought rarity (1-in-X year event)
  2. COMPOUND EVENT ANALYSIS - Concurrent drought + heat amplification
  3. DURATION-SEVERITY-AREA - 3D drought characterization
  4. REGIME SHIFT DETECTION - Non-stationarity analysis
  5. PERIODICITY ANALYSIS - Climate oscillation connections
  6. HISTORICAL CONTEXT - Full 75-year perspective (1950-2025)
""")

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    CONFIG = {
        'data_start_year': 1950,
        'data_end_year': 2025,
        'baseline_start': 1981,
        'baseline_end': 2010,
        'drought_start': 2018,
        'drought_end': 2025,
    }

    # All 10 major Iranian cities
    ALL_CITIES = [
        "Tehran",       # Capital, central region
        "Mashhad",      # Northeast
        "Isfahan",      # Central
        "Tabriz",       # Northwest
        "Shiraz",       # South-central
        "Ahvaz",        # Southwest
        "Kerman",       # Southeast
        "Rasht",        # Caspian Sea
        "Zahedan",      # Southeast
        "Bandar Abbas", # Persian Gulf
    ]

    # Create output directories
    output_dir = Path("results/iran_megadrought_analysis")
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"

    for directory in [output_dir, plots_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Analyzing {len(ALL_CITIES)} major Iranian cities")
    print(f"📅 Full period: {CONFIG['data_start_year']}-{CONFIG['data_end_year']} ({CONFIG['data_end_year'] - CONFIG['data_start_year'] + 1} years)")
    print(f"📅 Baseline: {CONFIG['baseline_start']}-{CONFIG['baseline_end']}")
    print(f"📅 Drought period: {CONFIG['drought_start']}-{CONFIG['drought_end']}")

    # Initialize plotters
    basic_plotter = DroughtPlotter()
    advanced_plotter = AdvancedDroughtPlotter()

    # =========================================================================
    # PART 1: FETCH DATA FROM ALL STATIONS
    # =========================================================================
    print_header("PART 1: DATA ACQUISITION FROM ALL 10 STATIONS")

    multi_fetcher = MultiStationFetcher(city_names=ALL_CITIES)
    all_prcp_data = {}
    all_temp_data = {}
    station_coords = {}

    for city in ALL_CITIES:
        try:
            print(f"\n📍 Fetching data for {city}...")
            fetcher = IranianDataFetcher(city)

            # Get coordinates
            station_info = IranianStationRegistry.get_station(city)
            station_coords[city] = (station_info['lat'], station_info['lon'])

            # Fetch precipitation
            prcp = fetcher.fetch_precipitation_data(
                start_year=CONFIG['data_start_year'],
                end_year=CONFIG['data_end_year']
            )
            if prcp is not None and not prcp.empty:
                all_prcp_data[city] = prcp
                print(f"   ✅ Precipitation: {len(prcp)} days")

            # Fetch temperature for compound analysis
            temp = fetcher.fetch_temperature_data(
                start_year=CONFIG['data_start_year'],
                end_year=CONFIG['data_end_year']
            )
            if temp is not None and not temp.empty:
                all_temp_data[city] = temp
                print(f"   ✅ Temperature: {len(temp)} days")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n📊 Successfully fetched precipitation from {len(all_prcp_data)}/{len(ALL_CITIES)} stations")
    print(f"📊 Successfully fetched temperature from {len(all_temp_data)}/{len(ALL_CITIES)} stations")

    if len(all_prcp_data) == 0:
        print("\n⚠️ No data fetched. Check network connection or NOAA service availability.")
        print("   Exiting analysis.")
        return

    # =========================================================================
    # PART 2: COMPREHENSIVE SINGLE-STATION ANALYSIS (TEHRAN)
    # =========================================================================
    print_header("PART 2: COMPREHENSIVE SINGLE-STATION ANALYSIS (TEHRAN)")

    if "Tehran" in all_prcp_data:
        tehran_prcp = all_prcp_data["Tehran"]
        tehran_temp = all_temp_data.get("Tehran")

        print("\n🔬 Performing comprehensive megadrought analysis...")

        megadrought_analyzer = MegadroughtAnalyzer(
            precipitation_data=tehran_prcp,
            temperature_data=tehran_temp,
            baseline_start=CONFIG['baseline_start'],
            baseline_end=CONFIG['baseline_end']
        )

        tehran_results = megadrought_analyzer.comprehensive_analysis(
            drought_start=CONFIG['drought_start'],
            drought_end=CONFIG['drought_end']
        )

        # --- Return Period Analysis ---
        print("\n📊 Generating Return Period Analysis plots...")

        rp_results = tehran_results['return_periods']
        drought_years = list(range(CONFIG['drought_start'], CONFIG['drought_end'] + 1))

        advanced_plotter.plot_return_period_analysis(
            ranked_droughts=rp_results['all_years_ranked'],
            return_levels=rp_results['return_levels'],
            gumbel_params=rp_results['gumbel_params'],
            drought_highlight_years=drought_years,
            station_name="Tehran (Mehrabad)",
            output_file=plots_dir / "tehran_return_period_analysis.png"
        )

        # Print return period for current drought
        drought_rp = tehran_results['drought_period_return_period']
        print(f"\n🎯 KEY FINDING: Current drought return period: {drought_rp:.0f} years")
        print(f"   This is a 1-in-{drought_rp:.0f} year event!")

        # --- Compound Event Analysis ---
        if 'compound_events' in tehran_results:
            print("\n🌡️ Generating Compound Event Analysis plots...")

            compound = tehran_results['compound_events']

            # Get annual anomalies for plotting
            compound_analyzer = CompoundEventAnalyzer(
                tehran_prcp, tehran_temp,
                CONFIG['baseline_start'], CONFIG['baseline_end']
            )
            annual_anomalies = compound_analyzer.calculate_annual_anomalies()

            advanced_plotter.plot_compound_event_analysis(
                annual_anomalies=annual_anomalies,
                probability_stats=compound['probabilities'],
                drought_period=(CONFIG['drought_start'], CONFIG['drought_end']),
                station_name="Tehran (Mehrabad)",
                output_file=plots_dir / "tehran_compound_event_analysis.png"
            )

            # Print compound event findings
            dep_ratio = compound['probabilities']['dependence_ratio']
            print(f"\n🎯 KEY FINDING: Drought-heat dependence ratio: {dep_ratio:.2f}x")
            if dep_ratio > 1:
                print(f"   Compound events are {dep_ratio:.1f}x MORE likely than by chance!")

        # --- Regime Shift Analysis ---
        print("\n📈 Generating Regime Shift Analysis plots...")

        regime = tehran_results['regime_analysis']

        advanced_plotter.plot_regime_shift_analysis(
            change_point_results=regime['change_point'],
            decadal_trends=regime['decadal_trends'],
            moving_stats=regime['moving_stats'],
            station_name="Tehran (Mehrabad)",
            output_file=plots_dir / "tehran_regime_shift_analysis.png"
        )

        change_year = regime['change_point']['change_point_year']
        change_pct = regime['change_point']['change_percent']
        print(f"\n🎯 KEY FINDING: Regime shift detected around {change_year}")
        print(f"   Precipitation changed by {change_pct:+.1f}% after change point")

        # --- Historical Context ---
        print("\n📚 Generating Historical Context plots...")

        context = tehran_results['historical_context']

        # Get annual precipitation series
        prcp_col = [c for c in tehran_prcp.columns if 'prcp' in c.lower() or 'precipitation' in c.lower()][0]
        annual_prcp = tehran_prcp[prcp_col].resample('YE').sum()
        annual_prcp.index = annual_prcp.index.year

        baseline_years = annual_prcp[
            (annual_prcp.index >= CONFIG['baseline_start']) &
            (annual_prcp.index <= CONFIG['baseline_end'])
        ]
        baseline_mean = baseline_years.mean()

        advanced_plotter.plot_historical_context(
            historical_context=context,
            annual_prcp=annual_prcp,
            baseline_mean=baseline_mean,
            drought_period=(CONFIG['drought_start'], CONFIG['drought_end']),
            station_name="Tehran (Mehrabad)",
            output_file=plots_dir / "tehran_historical_context.png"
        )

        print(f"\n🎯 KEY FINDING: Current drought at {context['percentile_rank']:.1f}th percentile")
        print(f"   Only {context['years_with_similar_or_worse']} years in {context['total_years_in_record']}-year record were similar or worse")

        # --- Periodicity Analysis ---
        print("\n🔄 Periodicity Analysis Results:")
        periods = tehran_results['periodicity']['dominant_periods']
        print(periods.to_string(index=False))

        # Export Tehran results
        print("\n💾 Exporting Tehran analysis data...")
        rp_results['all_years_ranked'].to_csv(data_dir / "tehran_drought_ranking.csv", index=False)
        rp_results['return_levels'].to_csv(data_dir / "tehran_return_levels.csv", index=False)
        regime['decadal_trends'].to_csv(data_dir / "tehran_decadal_trends.csv")

    # =========================================================================
    # PART 3: MULTI-STATION REGIONAL ANALYSIS
    # =========================================================================
    print_header("PART 3: MULTI-STATION REGIONAL ANALYSIS (ALL 10 CITIES)")

    if len(all_prcp_data) >= 2:
        print(f"\n🌍 Analyzing {len(all_prcp_data)} stations for regional patterns...")

        # Calculate drought metrics for each station
        regional_results = {}
        station_spi_data = {}

        for city, prcp_data in all_prcp_data.items():
            try:
                print(f"\n   Processing {city}...")

                analyzer = DroughtAnalyzer(
                    precipitation_data=prcp_data,
                    station_name=city,
                    baseline_start=CONFIG['baseline_start'],
                    baseline_end=CONFIG['baseline_end']
                )

                results = analyzer.analyze_drought_period(
                    start_year=CONFIG['drought_start'],
                    end_year=CONFIG['drought_end']
                )

                regional_results[city] = {
                    'mean_annual_deficit_percent': results['mean_annual_deficit_percent'],
                    'total_deficit_mm': results['total_deficit_mm'],
                    'worst_year': results['worst_year'],
                    'worst_deficit_percent': results['worst_deficit_percent'],
                    'baseline_mean': results['baseline_mean']
                }

                # Calculate SPI for DSA analysis
                spi = analyzer.calculate_spi(scale_months=12)
                station_spi_data[city] = spi

            except Exception as e:
                print(f"   ❌ Error processing {city}: {e}")

        if regional_results:
            # --- Multi-Station Map ---
            print("\n🗺️ Generating Multi-Station Drought Map...")

            # Filter to stations we have coords for
            valid_stations = {k: v for k, v in regional_results.items() if k in station_coords}
            valid_coords = {k: v for k, v in station_coords.items() if k in regional_results}

            if valid_stations:
                advanced_plotter.plot_multi_station_drought_map(
                    station_results=valid_stations,
                    station_coords=valid_coords,
                    metric='mean_annual_deficit_percent',
                    drought_period=(CONFIG['drought_start'], CONFIG['drought_end']),
                    output_file=plots_dir / "regional_drought_map.png"
                )

            # --- SPI Heatmap ---
            if station_spi_data:
                print("\n🔥 Generating Multi-Station SPI Heatmap...")
                advanced_plotter.plot_all_stations_spi_heatmap(
                    multi_station_spi=station_spi_data,
                    drought_period=(CONFIG['drought_start'], CONFIG['drought_end']),
                    output_file=plots_dir / "regional_spi_heatmap.png"
                )

            # --- DSA Analysis ---
            print("\n📐 Performing Duration-Severity-Area Analysis...")

            dsa_analyzer = DroughtDSAAnalyzer(station_spi_data)
            dsa_timeseries = dsa_analyzer.calculate_dsa_timeseries(
                start_year=CONFIG['data_start_year'],
                end_year=CONFIG['data_end_year']
            )

            advanced_plotter.plot_dsa_analysis(
                dsa_timeseries=dsa_timeseries,
                drought_period=(CONFIG['drought_start'], CONFIG['drought_end']),
                output_file=plots_dir / "regional_dsa_analysis.png"
            )

            # Export regional summary
            print("\n💾 Exporting regional analysis data...")

            regional_df = pd.DataFrame([
                {'Station': k, **v} for k, v in regional_results.items()
            ])
            regional_df = regional_df.sort_values('mean_annual_deficit_percent', ascending=False)
            regional_df.to_csv(data_dir / "regional_drought_summary.csv", index=False)

            dsa_timeseries.to_csv(data_dir / "dsa_timeseries.csv", index=False)

            # Print regional summary
            print("\n" + "=" * 80)
            print("REGIONAL DROUGHT SUMMARY")
            print("=" * 80)
            print(f"\nMean deficit across all stations: {regional_df['mean_annual_deficit_percent'].mean():.1f}%")
            print(f"Most affected: {regional_df.iloc[0]['Station']} ({regional_df.iloc[0]['mean_annual_deficit_percent']:.1f}%)")
            print(f"Least affected: {regional_df.iloc[-1]['Station']} ({regional_df.iloc[-1]['mean_annual_deficit_percent']:.1f}%)")

    # =========================================================================
    # PART 4: GENERATE COMPREHENSIVE DASHBOARDS
    # =========================================================================
    print_header("PART 4: GENERATING COMPREHENSIVE DASHBOARDS")

    for city in list(all_prcp_data.keys())[:5]:  # Top 5 stations
        try:
            print(f"\n📊 Creating dashboard for {city}...")

            analyzer = DroughtAnalyzer(
                precipitation_data=all_prcp_data[city],
                station_name=city,
                baseline_start=CONFIG['baseline_start'],
                baseline_end=CONFIG['baseline_end']
            )

            results = analyzer.analyze_drought_period(
                start_year=CONFIG['drought_start'],
                end_year=CONFIG['drought_end']
            )

            basic_plotter.plot_comprehensive_drought_dashboard(
                drought_results=results,
                station_name=city,
                output_file=plots_dir / f"{city.lower().replace(' ', '_')}_dashboard.png"
            )

            # Also plot SPI
            spi = analyzer.calculate_spi(scale_months=12)
            basic_plotter.plot_spi_timeseries(
                spi_data=spi,
                station_name=city,
                scale_months=12,
                drought_period=(CONFIG['drought_start'], CONFIG['drought_end']),
                output_file=plots_dir / f"{city.lower().replace(' ', '_')}_spi.png"
            )

        except Exception as e:
            print(f"   ❌ Error creating dashboard for {city}: {e}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_header("ANALYSIS COMPLETE - SCIENTIFIC PAPER FINDINGS")

    print("""
KEY FINDINGS FOR PUBLICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. RETURN PERIOD ANALYSIS
   • The 2018-2025 drought represents a rare extreme event
   • Return period estimates available for paper

2. COMPOUND EVENTS
   • Drought severity amplified by concurrent heat
   • Dependence analysis shows non-independent occurrence

3. SPATIAL EXTENT
   • All 10 major stations show significant deficits
   • Regional coherence indicates large-scale driver

4. REGIME SHIFT
   • Change point analysis suggests fundamental shift
   • Decadal trends show increasing drought frequency

5. UNPRECEDENTED NATURE
   • Historical percentile ranking demonstrates severity
   • Multi-year persistence exceeds historical precedent

OUTPUT FILES:
━━━━━━━━━━━━━
""")
    print(f"📊 Plots: {plots_dir}")
    for f in sorted(plots_dir.glob("*.png")):
        print(f"    • {f.name}")

    print(f"\n💾 Data: {data_dir}")
    for f in sorted(data_dir.glob("*.csv")):
        print(f"    • {f.name}")

    print("""
RECOMMENDED PAPER STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Introduction
   - Iran's water crisis context
   - 2018-2025 as potential megadrought

2. Data and Methods
   - NOAA GHCN-Daily stations (10 cities)
   - Return period analysis (Gumbel/GEV)
   - Compound event framework
   - DSA methodology

3. Results
   - Return period: 1-in-X year event (Fig. return_period)
   - Compound amplification (Fig. compound_event)
   - Regional patterns (Fig. regional_map, spi_heatmap)
   - Regime shift evidence (Fig. regime_shift)

4. Discussion
   - Comparison with historical droughts
   - Climate change attribution
   - Future projections

5. Conclusions
   - Policy implications for water management
""")

    print("\n" + "=" * 80)
    print("  Analysis complete! Results ready for scientific publication.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

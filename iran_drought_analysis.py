#!/usr/bin/env python3
"""
Iran Drought Analysis - Comprehensive Study
============================================

Analyzes Iran's severe drought conditions (2018-2025) using NOAA GHCN-Daily data.
Suitable for scientific paper preparation.

This script:
1. Fetches precipitation data from multiple Iranian weather stations
2. Calculates precipitation deficits compared to baseline (1981-2010)
3. Computes drought indices (SPI)
4. Generates publication-quality visualizations
5. Exports statistical summaries

Author: Weather Station Data Analysis Project
Date: 2025
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from weatherstation_analysis.iran_data_fetcher import (
    IranianDataFetcher,
    IranianStationRegistry,
    MultiStationFetcher
)
from weatherstation_analysis.drought_analyzer import (
    DroughtAnalyzer,
    MultiStationDroughtAnalyzer
)
from weatherstation_analysis.drought_plotter import DroughtPlotter


def main():
    """
    Main analysis workflow for Iran drought study.
    """
    print("=" * 80)
    print("🇮🇷 IRAN DROUGHT ANALYSIS - COMPREHENSIVE STUDY (2018-2025)")
    print("=" * 80)
    print("\nThis analysis examines Iran's severe drought conditions over the past 6-7 years")
    print("using precipitation data from NOAA's Global Historical Climatology Network.\n")

    # Configuration
    DROUGHT_PERIOD_START = 2018
    DROUGHT_PERIOD_END = 2025
    BASELINE_START = 1981
    BASELINE_END = 2010
    DATA_START_YEAR = 1950  # Fetch data from 1950 onwards

    # Create output directories
    output_dir = Path("results/iran_drought_analysis")
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"

    for directory in [output_dir, plots_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Plots will be saved to: {plots_dir}")
    print(f"💾 Data will be saved to: {data_dir}\n")

    # =========================================================================
    # PART 1: SINGLE STATION ANALYSIS (Tehran - Capital City)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: SINGLE STATION DETAILED ANALYSIS - TEHRAN")
    print("=" * 80)

    try:
        # Fetch Tehran data
        print("\n📡 Fetching precipitation data for Tehran...")
        tehran_fetcher = IranianDataFetcher("Tehran")
        tehran_prcp = tehran_fetcher.fetch_precipitation_data(
            start_year=DATA_START_YEAR,
            end_year=DROUGHT_PERIOD_END
        )

        if tehran_prcp is None or tehran_prcp.empty:
            print("\n⚠️ WARNING: Could not fetch Tehran data from NOAA GHCN.")
            print("   This may be due to:")
            print("   1. Network connectivity issues")
            print("   2. NOAA API temporary unavailability")
            print("   3. Station ID changes")
            print("\n💡 SOLUTION: You can use alternative data sources:")
            print("   - Download CSV data manually from: https://www.ncei.noaa.gov/cdo-web/")
            print("   - Use Iran Meteorological Organization data")
            print("   - Try again later when NOAA services are restored")
            print("\n   For demonstration purposes, the script structure is complete.")
            print("   Simply replace the data fetching with your local data files.")

        else:
            # Analyze drought for Tehran
            print("\n📊 Analyzing drought conditions for Tehran...")
            tehran_analyzer = DroughtAnalyzer(
                precipitation_data=tehran_prcp,
                station_name="Tehran (Mehrabad)",
                baseline_start=BASELINE_START,
                baseline_end=BASELINE_END
            )

            # Perform comprehensive analysis
            tehran_results = tehran_analyzer.analyze_drought_period(
                start_year=DROUGHT_PERIOD_START,
                end_year=DROUGHT_PERIOD_END
            )

            # Calculate SPI
            print("\n📈 Calculating Standardized Precipitation Index (SPI)...")
            tehran_spi = tehran_analyzer.calculate_spi(scale_months=12)

            # Generate visualizations
            print("\n🎨 Generating visualizations...")
            plotter = DroughtPlotter()

            # Plot 1: Comprehensive dashboard
            plotter.plot_comprehensive_drought_dashboard(
                drought_results=tehran_results,
                station_name="Tehran (Mehrabad)",
                output_file=plots_dir / "tehran_drought_dashboard.png"
            )

            # Plot 2: Deficit time series
            plotter.plot_precipitation_deficit_timeseries(
                deficit_data=tehran_results['deficit_data'],
                station_name="Tehran (Mehrabad)",
                baseline_mean=tehran_results['baseline_mean'],
                output_file=plots_dir / "tehran_deficit_timeseries.png"
            )

            # Plot 3: SPI time series
            plotter.plot_spi_timeseries(
                spi_data=tehran_spi,
                station_name="Tehran (Mehrabad)",
                scale_months=12,
                drought_period=(DROUGHT_PERIOD_START, DROUGHT_PERIOD_END),
                output_file=plots_dir / "tehran_spi_timeseries.png"
            )

            # Export data for scientific paper
            print("\n💾 Exporting analysis results...")
            tehran_results['deficit_data'].to_csv(
                data_dir / "tehran_annual_deficit.csv",
                index=True
            )
            tehran_spi.to_csv(
                data_dir / "tehran_spi12.csv",
                index=True
            )

            print("\n✅ Tehran analysis complete!")

    except Exception as e:
        print(f"\n❌ Error in Tehran analysis: {e}")
        print("   See warning message above for solutions.")

    # =========================================================================
    # PART 2: MULTI-STATION REGIONAL ANALYSIS
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("PART 2: MULTI-STATION REGIONAL ANALYSIS - ALL MAJOR IRANIAN CITIES")
    print("=" * 80)

    # Select key stations representing different regions
    selected_cities = [
        "Tehran",      # Capital, central region
        "Mashhad",     # Northeast
        "Isfahan",     # Central
        "Tabriz",      # Northwest
        "Shiraz",      # South-central
        "Ahvaz",       # Southwest
    ]

    print(f"\n📍 Analyzing {len(selected_cities)} major stations:")
    for city in selected_cities:
        print(f"   - {city}")

    try:
        # Fetch data from multiple stations
        print("\n📡 Fetching data from multiple stations...")
        multi_fetcher = MultiStationFetcher(city_names=selected_cities)
        all_station_data = multi_fetcher.fetch_all_precipitation(
            start_year=DATA_START_YEAR,
            end_year=DROUGHT_PERIOD_END
        )

        if not all_station_data:
            print("\n⚠️ WARNING: Could not fetch multi-station data.")
            print("   Refer to the solutions mentioned in Part 1.")

        else:
            # Analyze regional drought
            print("\n🌍 Performing regional drought analysis...")
            regional_analyzer = MultiStationDroughtAnalyzer(all_station_data)
            regional_results = regional_analyzer.analyze_regional_drought(
                start_year=DROUGHT_PERIOD_START,
                end_year=DROUGHT_PERIOD_END
            )

            # Generate regional comparison plots
            print("\n🎨 Generating regional comparison visualizations...")
            plotter = DroughtPlotter()

            plotter.plot_multi_station_comparison(
                regional_results=regional_results,
                metric='mean_annual_deficit_percent',
                output_file=plots_dir / "regional_drought_comparison.png"
            )

            # Export regional summary
            print("\n💾 Exporting regional analysis results...")

            # Create summary DataFrame
            summary_data = []
            for station, results in regional_results.items():
                summary_data.append({
                    'Station': station,
                    'Baseline_Mean_mm': results['baseline_mean'],
                    'Total_Deficit_mm': results['total_deficit_mm'],
                    'Mean_Annual_Deficit_Percent': results['mean_annual_deficit_percent'],
                    'Worst_Year': results['worst_year'],
                    'Worst_Year_Deficit_mm': results['worst_deficit_mm'],
                    'Worst_Year_Deficit_Percent': results['worst_deficit_percent'],
                    'Years_with_Deficit': results['deficit_years'],
                    'Total_Years_Analyzed': results['total_years']
                })

            import pandas as pd
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                data_dir / "regional_drought_summary.csv",
                index=False
            )

            print("\n✅ Regional analysis complete!")

            # Print key findings
            print("\n" + "=" * 80)
            print("KEY FINDINGS")
            print("=" * 80)

            print(f"\nBaseline Period: {BASELINE_START}-{BASELINE_END}")
            print(f"Drought Period Analyzed: {DROUGHT_PERIOD_START}-{DROUGHT_PERIOD_END}\n")

            print("Regional Drought Severity:")
            for _, row in summary_df.iterrows():
                severity = "🔴 SEVERE" if row['Mean_Annual_Deficit_Percent'] > 25 else \
                          "🟠 MODERATE" if row['Mean_Annual_Deficit_Percent'] > 15 else \
                          "🟡 MILD"
                print(f"  {row['Station']:20s}: {row['Mean_Annual_Deficit_Percent']:>6.1f}% deficit   {severity}")

            print(f"\nMost affected city: {summary_df.loc[summary_df['Mean_Annual_Deficit_Percent'].idxmax(), 'Station']}")
            print(f"Least affected city: {summary_df.loc[summary_df['Mean_Annual_Deficit_Percent'].idxmin(), 'Station']}")

            most_common_worst_year = summary_df['Worst_Year'].mode()[0]
            print(f"\nMost common worst year: {most_common_worst_year}")

    except Exception as e:
        print(f"\n❌ Error in multi-station analysis: {e}")
        print("   See warning message above for solutions.")

    # =========================================================================
    # FINAL SUMMARY AND RECOMMENDATIONS
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)

    print(f"\n📊 Generated outputs:")
    print(f"   Plots: {plots_dir}")
    print(f"   Data: {data_dir}")

    print("\n📄 For Scientific Paper:")
    print("   1. Deficit time series plots show precipitation trends")
    print("   2. SPI analysis quantifies drought severity")
    print("   3. Regional comparison identifies spatial patterns")
    print("   4. Statistical summaries provide quantitative metrics")

    print("\n📚 Recommended Sections for Paper:")
    print("   • Introduction: Context of Iran's water crisis")
    print("   • Methodology: GHCN-Daily data, SPI calculation, baseline comparison")
    print("   • Results: Regional patterns, deficit quantification, SPI trends")
    print("   • Discussion: Potential causes (climate change, anthropogenic factors)")
    print("   • Conclusion: Implications for water resource management")

    print("\n🔍 Key Metrics to Report:")
    print("   • Mean annual precipitation deficit (%)")
    print("   • Total cumulative deficit (mm)")
    print("   • Standardized Precipitation Index (SPI-12)")
    print("   • Worst drought year and severity")
    print("   • Regional variation patterns")

    print("\n💡 Next Steps:")
    print("   1. If data fetching failed, try alternative sources (see warnings above)")
    print("   2. Consider adding temperature analysis (heat stress + drought)")
    print("   3. Include groundwater level data if available")
    print("   4. Compare with satellite-based drought indices (GRACE, NDVI)")
    print("   5. Analyze correlation with climate oscillations (ENSO, NAO)")

    print("\n" + "=" * 80)
    print("Thank you for using the Iran Drought Analysis toolkit!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

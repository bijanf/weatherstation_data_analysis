"""
Drought Analysis Module
========================

Analyzes drought conditions using precipitation data.
Calculates precipitation deficits, anomalies, and drought indices.

Designed specifically for analyzing Iran's severe drought (2018-2025).
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class DroughtAnalyzer:
    """
    Analyzes drought conditions from precipitation data.

    Implements multiple drought assessment methods:
    - Precipitation deficits (absolute and percentage)
    - Precipitation anomalies (standardized)
    - Standardized Precipitation Index (SPI)
    - Consecutive dry days analysis
    - Multi-year drought period analysis
    """

    def __init__(
        self,
        precipitation_data: pd.DataFrame,
        station_name: str = "Unknown Station",
        baseline_start: int = 1981,
        baseline_end: int = 2010,
    ):
        """
        Initialize the drought analyzer.

        Args:
            precipitation_data: DataFrame with precipitation data (mm)
                                Index should be datetime, column: 'precipitation_mm'
            station_name: Name of the weather station
            baseline_start: Start year for baseline (climatological normal) period
            baseline_end: End year for baseline period
                          (typically 1981-2010 or 1991-2020)
        """
        self.precipitation_data = precipitation_data.copy()
        self.station_name = station_name
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end

        # Ensure index is datetime
        if not isinstance(self.precipitation_data.index, pd.DatetimeIndex):
            raise ValueError("Precipitation data index must be DatetimeIndex")

        # Determine precipitation column name
        prcp_cols = [
            c
            for c in self.precipitation_data.columns
            if "prcp" in c.lower() or "precipitation" in c.lower()
        ]
        if not prcp_cols:
            raise ValueError("No precipitation column found in data")
        self.prcp_col = prcp_cols[0]

        print(f"📊 Initializing DroughtAnalyzer for {station_name}")
        print(
            f"   Data period: {self.precipitation_data.index.min().year} - "
            f"{self.precipitation_data.index.max().year}"
        )
        print(f"   Baseline period: {baseline_start} - {baseline_end}")

    def calculate_annual_totals(self) -> pd.DataFrame:
        """
        Calculate annual precipitation totals.

        Returns:
            DataFrame with annual precipitation totals
        """
        annual_totals = self.precipitation_data.resample("YE").sum()
        annual_totals["year"] = annual_totals.index.year

        return annual_totals

    def calculate_baseline_statistics(self) -> Dict[str, float]:
        """
        Calculate baseline (normal) precipitation statistics.

        Returns:
            Dict with baseline mean, std, median, percentiles
        """
        # Filter to baseline period
        baseline_data = self.precipitation_data[
            (self.precipitation_data.index.year >= self.baseline_start)
            & (self.precipitation_data.index.year <= self.baseline_end)
        ]

        if baseline_data.empty:
            raise ValueError(
                f"No data available for baseline period "
                f"{self.baseline_start}-{self.baseline_end}"
            )

        # Calculate annual totals for baseline period
        baseline_annual = baseline_data.resample("YE").sum()[self.prcp_col]

        baseline_stats = {
            "mean_annual": baseline_annual.mean(),
            "std_annual": baseline_annual.std(),
            "median_annual": baseline_annual.median(),
            "min_annual": baseline_annual.min(),
            "max_annual": baseline_annual.max(),
            "p25_annual": baseline_annual.quantile(0.25),
            "p75_annual": baseline_annual.quantile(0.75),
            "n_years": len(baseline_annual),
        }

        print(f"\n📈 Baseline Statistics ({self.baseline_start}-{self.baseline_end}):")
        print(f"   Mean annual precipitation: {baseline_stats['mean_annual']:.1f} mm")
        print(f"   Std deviation: {baseline_stats['std_annual']:.1f} mm")
        print(
            f"   Range: {baseline_stats['min_annual']:.1f} - "
            f"{baseline_stats['max_annual']:.1f} mm"
        )

        return baseline_stats

    def calculate_precipitation_deficits(
        self, analysis_start_year: int, analysis_end_year: int
    ) -> pd.DataFrame:
        """
        Calculate precipitation deficits for analysis period.

        Args:
            analysis_start_year: Start year of analysis (e.g., 2018)
            analysis_end_year: End year of analysis (e.g., 2025)

        Returns:
            DataFrame with annual precipitation, deficit, and percentage deficit
        """
        baseline_stats = self.calculate_baseline_statistics()
        baseline_mean = baseline_stats["mean_annual"]

        # Get annual totals
        annual_totals = self.calculate_annual_totals()

        # Filter to analysis period
        analysis_data = annual_totals[
            (annual_totals["year"] >= analysis_start_year)
            & (annual_totals["year"] <= analysis_end_year)
        ].copy()

        # Calculate deficits
        analysis_data["baseline_normal"] = baseline_mean
        analysis_data["deficit_mm"] = baseline_mean - analysis_data[self.prcp_col]
        analysis_data["deficit_percent"] = (
            (baseline_mean - analysis_data[self.prcp_col]) / baseline_mean * 100
        )
        analysis_data["anomaly_mm"] = analysis_data[self.prcp_col] - baseline_mean
        analysis_data["percent_of_normal"] = (
            analysis_data[self.prcp_col] / baseline_mean * 100
        )

        print(
            f"\n🔍 Precipitation Deficits ({analysis_start_year}-{analysis_end_year}):"
        )
        print(
            "Analysis focuses on the severe drought period "
            "starting in water year 2017-2018."
        )
        print(
            "Baseline for comparison: 1981-2010 "
            "(WMO standard for climate change assessment)."
        )
        for _, row in analysis_data.iterrows():
            status = "🔴 DEFICIT" if row["deficit_mm"] > 0 else "🟢 SURPLUS"
            print(
                f"({row['percent_of_normal']:.1f}% of normal) {status}"
            )

        return analysis_data

    def calculate_spi(
        self, scale_months: int = 12, distribution: str = "gamma"
    ) -> pd.DataFrame:
        """
        Calculate Standardized Precipitation Index (SPI).

        SPI is a widely used drought index that represents precipitation
        as a standardized departure from a probability distribution.

        Args:
            scale_months: Time scale in months (3, 6, 12, 24, etc.)
            distribution: Distribution to fit ('gamma' or 'normal')

        Returns:
            DataFrame with SPI values
        """
        # Resample to monthly data
        monthly_prcp = self.precipitation_data.resample("ME").sum()

        # Calculate rolling sum for the specified scale
        rolling_prcp = (
            monthly_prcp[self.prcp_col]
            .rolling(window=scale_months, min_periods=scale_months)
            .sum()
        )

        # Fit distribution to baseline period
        baseline_monthly = rolling_prcp[
            (rolling_prcp.index.year >= self.baseline_start)
            & (rolling_prcp.index.year <= self.baseline_end)
        ].dropna()

        spi_values = None
        use_gamma = distribution == "gamma"

        if use_gamma:
            # Fit gamma distribution - filter out zeros for fitting
            baseline_nonzero = baseline_monthly[baseline_monthly > 0]

            if len(baseline_nonzero) >= 10:
                try:
                    shape, loc, scale = stats.gamma.fit(baseline_nonzero, floc=0)

                    # Calculate SPI using gamma distribution
                    spi_values = []
                    for value in rolling_prcp:
                        if pd.isna(value) or value <= 0:
                            spi_values.append(np.nan)
                        else:
                            # Calculate cumulative probability
                            cdf = stats.gamma.cdf(value, shape, loc, scale)
                            # Clip to avoid inf values at extremes
                            cdf = np.clip(cdf, 0.001, 0.999)
                            # Transform to standard normal
                            spi = stats.norm.ppf(cdf)
                            spi_values.append(spi)
                except Exception:
                    spi_values = None

        # Fall back to normal distribution
        if spi_values is None:
            mean = baseline_monthly.mean()
            std = baseline_monthly.std()
            if std > 0:
                spi_values = (rolling_prcp - mean) / std
            else:
                spi_values = rolling_prcp * 0  # All zeros if no variance

        spi_df = pd.DataFrame(
            {"precipitation": rolling_prcp, f"SPI_{scale_months}": spi_values},
            index=rolling_prcp.index,
        )

        return spi_df

    def classify_drought_severity(self, spi_value: float) -> Tuple[str, str]:
        """
        Classify drought severity based on SPI value.

        Args:
            spi_value: Standardized Precipitation Index value

        Returns:
            Tuple of (category, description)
        """
        if spi_value >= 2.0:
            return ("Extremely Wet", "🟦 Extremely Wet")
        elif spi_value >= 1.5:
            return ("Very Wet", "🔵 Very Wet")
        elif spi_value >= 1.0:
            return ("Moderately Wet", "🟢 Moderately Wet")
        elif spi_value >= -1.0:
            return ("Normal", "⚪ Normal")
        elif spi_value >= -1.5:
            return ("Moderate Drought", "🟡 Moderate Drought")
        elif spi_value >= -2.0:
            return ("Severe Drought", "🟠 Severe Drought")
        else:
            return ("Extreme Drought", "🔴 Extreme Drought")

    def analyze_drought_period(self, start_year: int, end_year: int) -> Dict[str, any]:
        """
        Comprehensive drought analysis for a specific period.

        Args:
            start_year: Start year of drought period
            end_year: End year of drought period

        Returns:
            Dictionary with comprehensive drought statistics
        """
        print(f"\n🔬 Comprehensive Drought Analysis: {start_year}-{end_year}")
        print("=" * 80)

        # Get baseline statistics
        baseline_stats = self.calculate_baseline_statistics()

        # Calculate deficits
        deficit_data = self.calculate_precipitation_deficits(start_year, end_year)

        # Calculate SPI
        try:
            spi_df = self.calculate_spi(scale_months=12)
            # Get SPI for drought period
            drought_spi = spi_df[
                (spi_df.index.year >= start_year) & (spi_df.index.year <= end_year)
            ]
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate SPI - {e}")
            drought_spi = None

        # Calculate cumulative deficit
        total_deficit = deficit_data["deficit_mm"].sum()
        mean_deficit_percent = deficit_data["deficit_percent"].mean()
        deficit_years = len(deficit_data[deficit_data["deficit_mm"] > 0])

        # Identify worst year
        worst_year_idx = deficit_data["deficit_mm"].idxmax()
        worst_year = int(deficit_data.loc[worst_year_idx, "year"])
        worst_deficit = deficit_data.loc[worst_year_idx, "deficit_mm"]
        worst_percent = deficit_data.loc[worst_year_idx, "deficit_percent"]

        results = {
            "period": f"{start_year}-{end_year}",
            "baseline_mean": baseline_stats["mean_annual"],
            "total_deficit_mm": total_deficit,
            "mean_annual_deficit_percent": mean_deficit_percent,
            "deficit_years": deficit_years,
            "total_years": len(deficit_data),
            "worst_year": worst_year,
            "worst_deficit_mm": worst_deficit,
            "worst_deficit_percent": worst_percent,
            "deficit_data": deficit_data,
            "baseline_stats": baseline_stats,
            "spi_data": drought_spi,
        }

        # Print summary
        print("\n📊 Summary:")
        print(f"   Baseline annual mean: {baseline_stats['mean_annual']:.1f} mm")
        print(f"   Total cumulative deficit: {total_deficit:.1f} mm")
        print(f"   Mean annual deficit: {mean_deficit_percent:.1f}%")
        print(f"   Years with deficit: {deficit_years}/{len(deficit_data)}")
        print(
            f"   Worst year: {worst_year} ({worst_deficit:.1f} mm deficit, "
            f"{worst_percent:.1f}% below normal)"
        )

        if drought_spi is not None and not drought_spi.empty:
            mean_spi = drought_spi["SPI_12"].mean()
            min_spi = drought_spi["SPI_12"].min()
            _, spi_category = self.classify_drought_severity(mean_spi)
            print(f"   Mean SPI-12: {mean_spi:.2f} {spi_category}")
            print(f"   Minimum SPI-12: {min_spi:.2f}")

        return results

    def calculate_consecutive_dry_days(
        self,
        threshold_mm: float = 1.0,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate consecutive dry days (CDD).

        Args:
            threshold_mm: Threshold for defining a dry day (mm)
            start_year: Start year for analysis (None = all data)
            end_year: End year for analysis (None = all data)

        Returns:
            DataFrame with yearly maximum consecutive dry days
        """
        # Filter data by years if specified
        data = self.precipitation_data.copy()
        if start_year:
            data = data[data.index.year >= start_year]
        if end_year:
            data = data[data.index.year <= end_year]

        # Identify dry days
        is_dry = data[self.prcp_col] < threshold_mm

        # Calculate consecutive dry days by year
        yearly_cdd = {}

        for year in data.index.year.unique():
            year_data = is_dry[is_dry.index.year == year]

            # Find consecutive dry periods
            consecutive = 0
            max_consecutive = 0

            for dry in year_data:
                if dry:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0

            yearly_cdd[year] = max_consecutive

        cdd_df = pd.DataFrame.from_dict(
            yearly_cdd, orient="index", columns=["max_consecutive_dry_days"]
        )
        cdd_df.index.name = "year"

        print(f"\n🌵 Consecutive Dry Days Analysis (threshold: {threshold_mm} mm):")
        for year, row in cdd_df.iterrows():
            print(f"   {year}: {row['max_consecutive_dry_days']} days")

        return cdd_df

    def export_analysis_summary(
        self, output_file: str, drought_period: Tuple[int, int]
    ) -> None:
        """
        Export comprehensive drought analysis summary to CSV.

        Args:
            output_file: Path to output CSV file
            drought_period: Tuple of (start_year, end_year)
        """
        start_year, end_year = drought_period

        results = self.analyze_drought_period(start_year, end_year)

        # Prepare export data
        deficit_data = results["deficit_data"].copy()
        deficit_data["station"] = self.station_name

        # Export
        deficit_data.to_csv(output_file, index=True)
        print(f"\n💾 Analysis exported to: {output_file}")


class MultiStationDroughtAnalyzer:
    """
    Analyzes drought across multiple stations for regional assessment.
    """

    def __init__(self, station_data: Dict[str, pd.DataFrame]):
        """
        Initialize multi-station analyzer.

        Args:
            station_data: Dict mapping station names to precipitation DataFrames
        """
        self.station_data = station_data
        self.analyzers = {}

        for station_name, data in station_data.items():
            try:
                self.analyzers[station_name] = DroughtAnalyzer(
                    precipitation_data=data, station_name=station_name
                )
            except Exception as e:
                print(
                    f"⚠️ Warning: Could not initialize analyzer for {station_name}: {e}"
                )

    def analyze_regional_drought(
        self, start_year: int, end_year: int
    ) -> Dict[str, Dict]:
        """
        Analyze drought across all stations.

        Args:
            start_year: Start year of drought period
            end_year: End year of drought period

        Returns:
            Dict mapping station names to drought analysis results
        """
        print("\n=== Iran Drought Analysis Report (2018-2025) ===")
        print("=" * 80)

        regional_results = {}

        for station_name, analyzer in self.analyzers.items():
            print(f"\n📍 Analyzing {station_name}...")
            try:
                results = analyzer.analyze_drought_period(start_year, end_year)
                regional_results[station_name] = results
            except Exception as e:
                print(f"❌ Error analyzing {station_name}: {e}")

        # Calculate regional summary
        print("\n🌍 Regional Summary:")
        print("=" * 80)

        deficits = [r["mean_annual_deficit_percent"] for r in regional_results.values()]
        worst_years = [r["worst_year"] for r in regional_results.values()]

        print(f"   Stations analyzed: {len(regional_results)}")
        print(f"   Mean regional deficit: {np.mean(deficits):.1f}%")
        print(f"   Range: {np.min(deficits):.1f}% to {np.max(deficits):.1f}%")
        print(
            f"   Most common worst year: {max(set(worst_years), key=worst_years.count)}"
        )

        return regional_results

"""
Advanced Drought Analysis Module
================================

Novel scientific analyses for drought characterization:
- Return period analysis using extreme value theory (Gumbel/GEV)
- Compound drought-heat event analysis
- Duration-Severity-Area (DSA) curves
- Change point detection for regime shifts
- Wavelet analysis for periodicity detection
- Multi-decadal historical context

Designed for publication-quality research on Iran's 2018-2025 megadrought.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import warnings

warnings.filterwarnings("ignore")


class DroughtReturnPeriodAnalyzer:
    """
    Calculates drought return periods using Extreme Value Theory.

    Applies Gumbel and GEV distributions to annual drought severity
    to quantify the rarity of extreme drought events.
    """

    def __init__(self, annual_precipitation: pd.Series, baseline_mean: float):
        """
        Initialize return period analyzer.

        Args:
            annual_precipitation: Series of annual precipitation totals (index=year)
            baseline_mean: Baseline (normal) annual precipitation in mm
        """
        self.annual_prcp = annual_precipitation.dropna()
        self.baseline_mean = baseline_mean
        self.years = self.annual_prcp.index

        # Calculate deficit series (positive = drought)
        self.deficit_mm = baseline_mean - self.annual_prcp
        self.deficit_percent = (self.deficit_mm / baseline_mean) * 100

    def fit_gumbel_to_deficits(self) -> Dict[str, float]:
        """
        Fit Gumbel distribution to annual precipitation deficits.

        Returns:
            Dict with location (mu) and scale (beta) parameters
        """
        # Use maximum deficit per year (for drought analysis, we want the minima inverted)
        deficits = self.deficit_percent.values

        # Fit Gumbel distribution (for maxima - deficits are "maxima" of water shortage)
        loc, scale = stats.gumbel_r.fit(deficits)

        return {
            'location': loc,
            'scale': scale,
            'distribution': 'gumbel_r'
        }

    def fit_gev_to_deficits(self) -> Dict[str, float]:
        """
        Fit Generalized Extreme Value distribution to deficits.

        GEV is more flexible than Gumbel, with shape parameter.

        Returns:
            Dict with shape (c), location (loc), and scale parameters
        """
        deficits = self.deficit_percent.values

        # Fit GEV distribution
        shape, loc, scale = stats.genextreme.fit(deficits)

        return {
            'shape': shape,
            'location': loc,
            'scale': scale,
            'distribution': 'gev'
        }

    def calculate_return_period(
        self,
        deficit_value: float,
        distribution: str = 'gumbel'
    ) -> float:
        """
        Calculate return period for a given deficit value.

        Args:
            deficit_value: Precipitation deficit in percent
            distribution: 'gumbel' or 'gev'

        Returns:
            Return period in years
        """
        if distribution == 'gumbel':
            params = self.fit_gumbel_to_deficits()
            exceedance_prob = 1 - stats.gumbel_r.cdf(
                deficit_value, params['location'], params['scale']
            )
        else:
            params = self.fit_gev_to_deficits()
            exceedance_prob = 1 - stats.genextreme.cdf(
                deficit_value, params['shape'], params['location'], params['scale']
            )

        # Return period = 1 / exceedance probability
        if exceedance_prob > 0:
            return 1 / exceedance_prob
        else:
            return float('inf')

    def calculate_return_levels(
        self,
        return_periods: List[int] = [2, 5, 10, 25, 50, 100],
        distribution: str = 'gumbel'
    ) -> pd.DataFrame:
        """
        Calculate deficit levels for various return periods.

        Args:
            return_periods: List of return periods in years
            distribution: 'gumbel' or 'gev'

        Returns:
            DataFrame with return periods and corresponding deficit levels
        """
        if distribution == 'gumbel':
            params = self.fit_gumbel_to_deficits()
            levels = [
                stats.gumbel_r.ppf(1 - 1/T, params['location'], params['scale'])
                for T in return_periods
            ]
        else:
            params = self.fit_gev_to_deficits()
            levels = [
                stats.genextreme.ppf(1 - 1/T, params['shape'], params['location'], params['scale'])
                for T in return_periods
            ]

        return pd.DataFrame({
            'return_period_years': return_periods,
            'deficit_percent': levels,
            'distribution': distribution
        })

    def analyze_historical_extremes(self) -> pd.DataFrame:
        """
        Analyze all years and calculate their return periods.

        Returns:
            DataFrame with year, deficit, and return period for each year
        """
        results = []

        for year in self.years:
            deficit = self.deficit_percent.loc[year]

            # Calculate return period using both distributions
            rp_gumbel = self.calculate_return_period(deficit, 'gumbel')
            rp_gev = self.calculate_return_period(deficit, 'gev')

            results.append({
                'year': year,
                'precipitation_mm': self.annual_prcp.loc[year],
                'deficit_mm': self.deficit_mm.loc[year],
                'deficit_percent': deficit,
                'return_period_gumbel': rp_gumbel,
                'return_period_gev': rp_gev,
                'is_drought': deficit > 0
            })

        df = pd.DataFrame(results)
        df = df.sort_values('deficit_percent', ascending=False)

        return df

    def get_drought_ranking(self) -> pd.DataFrame:
        """
        Rank all drought years by severity with return periods.

        Returns:
            DataFrame ranking drought years (deficit > 0)
        """
        all_years = self.analyze_historical_extremes()
        droughts = all_years[all_years['is_drought']].copy()
        droughts['rank'] = range(1, len(droughts) + 1)

        return droughts


class CompoundEventAnalyzer:
    """
    Analyzes compound drought-heat events.

    Compound events (concurrent drought + heat) have amplified impacts.
    This is a key research focus in climate science.
    """

    def __init__(
        self,
        precipitation_data: pd.DataFrame,
        temperature_data: pd.DataFrame,
        baseline_start: int = 1981,
        baseline_end: int = 2010
    ):
        """
        Initialize compound event analyzer.

        Args:
            precipitation_data: Daily precipitation data (index=date)
            temperature_data: Daily temperature data (index=date, columns: tmax_celsius, tmin_celsius)
            baseline_start: Start year for baseline period
            baseline_end: End year for baseline period
        """
        self.prcp_data = precipitation_data
        self.temp_data = temperature_data
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end

        # Merge datasets
        self.combined = self._merge_data()

    def _merge_data(self) -> pd.DataFrame:
        """Merge precipitation and temperature data."""
        combined = self.prcp_data.join(self.temp_data, how='outer')
        return combined

    def calculate_annual_anomalies(self) -> pd.DataFrame:
        """
        Calculate annual precipitation and temperature anomalies.

        Returns:
            DataFrame with annual anomalies relative to baseline
        """
        # Determine column names
        prcp_col = [c for c in self.combined.columns if 'prcp' in c.lower() or 'precipitation' in c.lower()]
        temp_col = [c for c in self.combined.columns if 'tmax' in c.lower() or 'tmean' in c.lower()]

        if not prcp_col or not temp_col:
            raise ValueError("Missing precipitation or temperature columns")

        prcp_col = prcp_col[0]
        temp_col = temp_col[0]

        # Calculate annual values
        annual = self.combined.resample('YE').agg({
            prcp_col: 'sum',
            temp_col: 'mean'
        })
        annual['year'] = annual.index.year

        # Calculate baseline means
        baseline = annual[
            (annual['year'] >= self.baseline_start) &
            (annual['year'] <= self.baseline_end)
        ]

        prcp_baseline_mean = baseline[prcp_col].mean()
        prcp_baseline_std = baseline[prcp_col].std()
        temp_baseline_mean = baseline[temp_col].mean()
        temp_baseline_std = baseline[temp_col].std()

        # Calculate standardized anomalies
        annual['prcp_anomaly_std'] = (annual[prcp_col] - prcp_baseline_mean) / prcp_baseline_std
        annual['temp_anomaly_std'] = (annual[temp_col] - temp_baseline_mean) / temp_baseline_std
        annual['prcp_anomaly_percent'] = ((annual[prcp_col] - prcp_baseline_mean) / prcp_baseline_mean) * 100
        annual['temp_anomaly_celsius'] = annual[temp_col] - temp_baseline_mean

        # Identify compound events (drought + heat)
        # Drought: prcp anomaly < -1 std, Heat: temp anomaly > +1 std
        annual['is_drought'] = annual['prcp_anomaly_std'] < -1.0
        annual['is_heat'] = annual['temp_anomaly_std'] > 1.0
        annual['is_compound'] = annual['is_drought'] & annual['is_heat']

        return annual

    def calculate_joint_probability(self) -> Dict[str, float]:
        """
        Calculate joint and conditional probabilities for compound events.

        Returns:
            Dict with probability statistics
        """
        annual = self.calculate_annual_anomalies()

        n_total = len(annual)
        n_drought = annual['is_drought'].sum()
        n_heat = annual['is_heat'].sum()
        n_compound = annual['is_compound'].sum()

        # Marginal probabilities
        p_drought = n_drought / n_total
        p_heat = n_heat / n_total

        # Joint probability (observed)
        p_compound_obs = n_compound / n_total

        # Expected under independence
        p_compound_indep = p_drought * p_heat

        # Dependence ratio
        if p_compound_indep > 0:
            dependence_ratio = p_compound_obs / p_compound_indep
        else:
            dependence_ratio = float('nan')

        # Conditional probabilities
        p_heat_given_drought = n_compound / n_drought if n_drought > 0 else 0
        p_drought_given_heat = n_compound / n_heat if n_heat > 0 else 0

        return {
            'n_years_total': n_total,
            'n_drought_years': n_drought,
            'n_heat_years': n_heat,
            'n_compound_years': n_compound,
            'p_drought': p_drought,
            'p_heat': p_heat,
            'p_compound_observed': p_compound_obs,
            'p_compound_independent': p_compound_indep,
            'dependence_ratio': dependence_ratio,
            'p_heat_given_drought': p_heat_given_drought,
            'p_drought_given_heat': p_drought_given_heat
        }

    def identify_compound_events(self) -> pd.DataFrame:
        """
        Identify and characterize all compound drought-heat events.

        Returns:
            DataFrame with compound event details
        """
        annual = self.calculate_annual_anomalies()
        compound_years = annual[annual['is_compound']].copy()

        if len(compound_years) == 0:
            print("No compound events found in the record")
            return pd.DataFrame()

        # Calculate severity index (combined drought + heat)
        compound_years['severity_index'] = (
            abs(compound_years['prcp_anomaly_std']) +
            compound_years['temp_anomaly_std']
        )

        compound_years = compound_years.sort_values('severity_index', ascending=False)

        return compound_years

    def analyze_drought_period_temperature(
        self,
        start_year: int,
        end_year: int
    ) -> Dict[str, Any]:
        """
        Analyze temperature conditions during a specific drought period.

        Args:
            start_year: Start year of drought period
            end_year: End year of drought period

        Returns:
            Dict with temperature analysis during drought
        """
        annual = self.calculate_annual_anomalies()

        drought_period = annual[
            (annual['year'] >= start_year) &
            (annual['year'] <= end_year)
        ]

        return {
            'period': f"{start_year}-{end_year}",
            'mean_temp_anomaly': drought_period['temp_anomaly_celsius'].mean(),
            'max_temp_anomaly': drought_period['temp_anomaly_celsius'].max(),
            'compound_years': drought_period['is_compound'].sum(),
            'total_years': len(drought_period),
            'compound_fraction': drought_period['is_compound'].mean(),
            'mean_prcp_anomaly_percent': drought_period['prcp_anomaly_percent'].mean(),
            'years_detail': drought_period[['year', 'prcp_anomaly_percent', 'temp_anomaly_celsius', 'is_compound']].to_dict('records')
        }


class DroughtDSAAnalyzer:
    """
    Duration-Severity-Area (DSA) analysis for drought characterization.

    Provides 3D characterization of drought events:
    - Duration: How long?
    - Severity: How intense?
    - Area: How widespread? (multi-station)
    """

    def __init__(self, multi_station_spi: Dict[str, pd.DataFrame]):
        """
        Initialize DSA analyzer.

        Args:
            multi_station_spi: Dict mapping station names to SPI DataFrames
        """
        self.station_spi = multi_station_spi
        self.stations = list(multi_station_spi.keys())

    def identify_drought_events(
        self,
        spi_threshold: float = -1.0,
        min_duration_months: int = 3
    ) -> pd.DataFrame:
        """
        Identify distinct drought events across all stations.

        Args:
            spi_threshold: SPI threshold for drought (default -1.0)
            min_duration_months: Minimum duration to qualify as drought event

        Returns:
            DataFrame with drought events and DSA characteristics
        """
        events = []

        for station, spi_df in self.station_spi.items():
            # Find SPI column
            spi_col = [c for c in spi_df.columns if 'SPI' in c][0]
            spi_series = spi_df[spi_col].dropna()

            # Identify drought periods (SPI below threshold)
            in_drought = spi_series < spi_threshold

            # Find continuous drought periods
            drought_starts = in_drought & ~in_drought.shift(1).fillna(False)
            drought_ends = in_drought & ~in_drought.shift(-1).fillna(False)

            start_dates = spi_series.index[drought_starts]
            end_dates = spi_series.index[drought_ends]

            for start, end in zip(start_dates, end_dates):
                duration = (end - start).days // 30 + 1  # Approximate months

                if duration >= min_duration_months:
                    drought_spi = spi_series[(spi_series.index >= start) & (spi_series.index <= end)]

                    events.append({
                        'station': station,
                        'start_date': start,
                        'end_date': end,
                        'duration_months': duration,
                        'mean_spi': drought_spi.mean(),
                        'min_spi': drought_spi.min(),
                        'severity': abs(drought_spi.mean()) * duration,  # Severity index
                        'peak_month': drought_spi.idxmin()
                    })

        return pd.DataFrame(events)

    def calculate_area_coverage(
        self,
        date: pd.Timestamp,
        spi_threshold: float = -1.0
    ) -> Dict[str, Any]:
        """
        Calculate drought area coverage for a specific date.

        Args:
            date: Date to analyze
            spi_threshold: SPI threshold for drought

        Returns:
            Dict with area coverage statistics
        """
        stations_in_drought = []
        total_stations = 0

        for station, spi_df in self.station_spi.items():
            spi_col = [c for c in spi_df.columns if 'SPI' in c][0]

            # Find closest date
            if date in spi_df.index:
                spi_value = spi_df.loc[date, spi_col]
            else:
                # Find nearest date
                idx = spi_df.index.get_indexer([date], method='nearest')[0]
                spi_value = spi_df.iloc[idx][spi_col]

            total_stations += 1
            if not pd.isna(spi_value) and spi_value < spi_threshold:
                stations_in_drought.append(station)

        return {
            'date': date,
            'stations_in_drought': stations_in_drought,
            'n_stations_drought': len(stations_in_drought),
            'n_stations_total': total_stations,
            'area_fraction': len(stations_in_drought) / total_stations if total_stations > 0 else 0
        }

    def calculate_dsa_timeseries(
        self,
        start_year: int,
        end_year: int,
        spi_threshold: float = -1.0
    ) -> pd.DataFrame:
        """
        Calculate monthly DSA metrics over time.

        Args:
            start_year: Start year
            end_year: End year
            spi_threshold: SPI threshold for drought

        Returns:
            DataFrame with monthly DSA metrics
        """
        # Generate monthly date range
        dates = pd.date_range(
            start=f'{start_year}-01-01',
            end=f'{end_year}-12-31',
            freq='ME'
        )

        results = []

        for date in dates:
            coverage = self.calculate_area_coverage(date, spi_threshold)

            # Calculate mean severity across stations
            severities = []
            for station, spi_df in self.station_spi.items():
                spi_col = [c for c in spi_df.columns if 'SPI' in c][0]
                if date in spi_df.index:
                    spi_val = spi_df.loc[date, spi_col]
                    if not pd.isna(spi_val) and spi_val < spi_threshold:
                        severities.append(abs(spi_val))

            results.append({
                'date': date,
                'year': date.year,
                'month': date.month,
                'area_fraction': coverage['area_fraction'],
                'n_stations_drought': coverage['n_stations_drought'],
                'mean_severity': np.mean(severities) if severities else 0
            })

        return pd.DataFrame(results)


class DroughtRegimeAnalyzer:
    """
    Analyzes drought regime shifts and non-stationarity.

    Uses change point detection to identify when drought
    characteristics fundamentally changed.
    """

    def __init__(self, annual_precipitation: pd.Series):
        """
        Initialize regime analyzer.

        Args:
            annual_precipitation: Series of annual precipitation totals
        """
        self.annual_prcp = annual_precipitation.dropna().sort_index()
        self.years = self.annual_prcp.index

    def detect_change_points_cusum(self) -> Dict[str, Any]:
        """
        Detect change points using CUSUM method.

        Returns:
            Dict with change point information
        """
        values = self.annual_prcp.values
        mean_val = np.mean(values)

        # Calculate cumulative sum of deviations
        cusum = np.cumsum(values - mean_val)

        # Find maximum deviation point
        max_idx = np.argmax(np.abs(cusum))
        change_year = self.years[max_idx]

        # Split data and compare
        before = values[:max_idx]
        after = values[max_idx:]

        # Statistical test for difference
        if len(before) > 1 and len(after) > 1:
            t_stat, p_value = stats.ttest_ind(before, after)
        else:
            t_stat, p_value = np.nan, np.nan

        return {
            'change_point_year': int(change_year),
            'change_point_index': max_idx,
            'mean_before': np.mean(before),
            'mean_after': np.mean(after),
            'change_magnitude': np.mean(after) - np.mean(before),
            'change_percent': ((np.mean(after) - np.mean(before)) / np.mean(before)) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'cusum_series': cusum
        }

    def analyze_decadal_trends(self) -> pd.DataFrame:
        """
        Analyze precipitation trends by decade.

        Returns:
            DataFrame with decadal statistics
        """
        df = pd.DataFrame({
            'year': self.years,
            'precipitation': self.annual_prcp.values
        })

        # Assign decades
        df['decade'] = (df['year'] // 10) * 10

        # Calculate decadal statistics
        decadal = df.groupby('decade').agg({
            'precipitation': ['mean', 'std', 'min', 'max', 'count']
        }).round(1)

        decadal.columns = ['mean_mm', 'std_mm', 'min_mm', 'max_mm', 'n_years']

        # Calculate drought frequency per decade (years below mean)
        overall_mean = df['precipitation'].mean()
        df['is_deficit'] = df['precipitation'] < overall_mean
        drought_freq = df.groupby('decade')['is_deficit'].mean()
        decadal['drought_frequency'] = drought_freq

        return decadal

    def calculate_moving_statistics(
        self,
        window: int = 10
    ) -> pd.DataFrame:
        """
        Calculate moving window statistics to visualize trends.

        Args:
            window: Window size in years

        Returns:
            DataFrame with moving statistics
        """
        df = pd.DataFrame({
            'year': self.years,
            'precipitation': self.annual_prcp.values
        })

        df['moving_mean'] = df['precipitation'].rolling(window=window, center=True).mean()
        df['moving_std'] = df['precipitation'].rolling(window=window, center=True).std()
        df['moving_min'] = df['precipitation'].rolling(window=window, center=True).min()
        df['moving_max'] = df['precipitation'].rolling(window=window, center=True).max()

        # Calculate coefficient of variation
        df['moving_cv'] = df['moving_std'] / df['moving_mean'] * 100

        return df


class WaveletDroughtAnalyzer:
    """
    Wavelet analysis for drought periodicity detection.

    Identifies cyclic patterns in precipitation that may relate
    to climate oscillations (ENSO, PDO, NAO, etc.).
    """

    def __init__(self, annual_precipitation: pd.Series):
        """
        Initialize wavelet analyzer.

        Args:
            annual_precipitation: Series of annual precipitation totals
        """
        self.annual_prcp = annual_precipitation.dropna().sort_index()
        self.years = self.annual_prcp.index

    def calculate_spectral_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate spectral density to identify dominant periodicities.

        Returns:
            Tuple of (frequencies, power spectral density)
        """
        # Detrend and normalize
        values = self.annual_prcp.values
        detrended = values - np.mean(values)
        normalized = detrended / np.std(detrended)

        # Calculate FFT
        n = len(normalized)
        fft = np.fft.fft(normalized)
        psd = np.abs(fft) ** 2

        # Calculate frequencies (in cycles per year)
        freqs = np.fft.fftfreq(n, d=1)  # d=1 year

        # Keep only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        psd = psd[pos_mask]

        # Convert to periods
        periods = 1 / freqs

        return periods, psd

    def identify_dominant_periods(
        self,
        n_peaks: int = 5
    ) -> pd.DataFrame:
        """
        Identify dominant periodicities in the precipitation record.

        Args:
            n_peaks: Number of peaks to identify

        Returns:
            DataFrame with dominant periods and their power
        """
        periods, psd = self.calculate_spectral_density()

        # Find peaks
        peak_indices, properties = find_peaks(psd, height=np.percentile(psd, 75))

        # Sort by power
        sorted_idx = np.argsort(psd[peak_indices])[::-1][:n_peaks]
        top_peaks = peak_indices[sorted_idx]

        results = []
        for idx in top_peaks:
            period = periods[idx]
            power = psd[idx]

            # Interpret period
            if 2 <= period <= 7:
                interpretation = "ENSO-related"
            elif 8 <= period <= 12:
                interpretation = "Decadal oscillation"
            elif 20 <= period <= 30:
                interpretation = "Multi-decadal (PDO/AMO)"
            else:
                interpretation = "Other"

            results.append({
                'period_years': round(period, 1),
                'spectral_power': power,
                'interpretation': interpretation
            })

        return pd.DataFrame(results)

    def calculate_running_correlation_enso(
        self,
        enso_index: Optional[pd.Series] = None,
        window: int = 21
    ) -> pd.DataFrame:
        """
        Calculate running correlation with ENSO index.

        Args:
            enso_index: Optional ENSO index time series
            window: Running window size in years

        Returns:
            DataFrame with running correlations
        """
        if enso_index is None:
            # Generate synthetic ENSO proxy (for demonstration)
            # In real application, use actual ENSO data
            print("Note: Using synthetic ENSO proxy. For publication, use actual ENSO index.")
            np.random.seed(42)
            enso_index = pd.Series(
                np.sin(2 * np.pi * np.arange(len(self.years)) / 4) + np.random.normal(0, 0.3, len(self.years)),
                index=self.years
            )

        # Ensure same index
        common_idx = self.annual_prcp.index.intersection(enso_index.index)
        prcp = self.annual_prcp.loc[common_idx]
        enso = enso_index.loc[common_idx]

        # Calculate running correlation
        correlations = []
        center_years = []

        half_window = window // 2

        for i in range(half_window, len(common_idx) - half_window):
            window_prcp = prcp.iloc[i-half_window:i+half_window+1]
            window_enso = enso.iloc[i-half_window:i+half_window+1]

            corr, p_val = stats.pearsonr(window_prcp, window_enso)

            correlations.append(corr)
            center_years.append(common_idx[i])

        return pd.DataFrame({
            'year': center_years,
            'correlation': correlations
        })


class MegadroughtAnalyzer:
    """
    Comprehensive megadrought analysis combining all methods.

    Provides unified analysis framework for characterizing
    the 2018-2025 Iran megadrought in historical context.
    """

    def __init__(
        self,
        precipitation_data: pd.DataFrame,
        temperature_data: Optional[pd.DataFrame] = None,
        baseline_start: int = 1981,
        baseline_end: int = 2010
    ):
        """
        Initialize megadrought analyzer.

        Args:
            precipitation_data: Daily precipitation data
            temperature_data: Optional daily temperature data
            baseline_start: Baseline period start
            baseline_end: Baseline period end
        """
        self.prcp_data = precipitation_data
        self.temp_data = temperature_data
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end

        # Calculate annual totals
        prcp_col = [c for c in self.prcp_data.columns if 'prcp' in c.lower() or 'precipitation' in c.lower()][0]
        self.annual_prcp = self.prcp_data[prcp_col].resample('YE').sum()
        self.annual_prcp.index = self.annual_prcp.index.year

        # Calculate baseline
        baseline_years = self.annual_prcp[
            (self.annual_prcp.index >= baseline_start) &
            (self.annual_prcp.index <= baseline_end)
        ]
        self.baseline_mean = baseline_years.mean()
        self.baseline_std = baseline_years.std()

    def comprehensive_analysis(
        self,
        drought_start: int,
        drought_end: int
    ) -> Dict[str, Any]:
        """
        Perform comprehensive megadrought analysis.

        Args:
            drought_start: Start year of drought period
            drought_end: End year of drought period

        Returns:
            Dict with all analysis results
        """
        results = {}

        # 1. Return Period Analysis
        print("📊 Analyzing return periods...")
        rp_analyzer = DroughtReturnPeriodAnalyzer(self.annual_prcp, self.baseline_mean)

        results['return_periods'] = {
            'all_years_ranked': rp_analyzer.get_drought_ranking(),
            'return_levels': rp_analyzer.calculate_return_levels(),
            'gumbel_params': rp_analyzer.fit_gumbel_to_deficits(),
            'gev_params': rp_analyzer.fit_gev_to_deficits()
        }

        # Calculate return period for drought period average
        drought_years = self.annual_prcp[
            (self.annual_prcp.index >= drought_start) &
            (self.annual_prcp.index <= drought_end)
        ]
        mean_drought_prcp = drought_years.mean()
        mean_deficit_percent = ((self.baseline_mean - mean_drought_prcp) / self.baseline_mean) * 100

        results['drought_period_return_period'] = rp_analyzer.calculate_return_period(
            mean_deficit_percent, 'gev'
        )

        # 2. Compound Event Analysis (if temperature data available)
        if self.temp_data is not None:
            print("🌡️ Analyzing compound drought-heat events...")
            compound_analyzer = CompoundEventAnalyzer(
                self.prcp_data, self.temp_data,
                self.baseline_start, self.baseline_end
            )

            results['compound_events'] = {
                'probabilities': compound_analyzer.calculate_joint_probability(),
                'identified_events': compound_analyzer.identify_compound_events(),
                'drought_period_temp': compound_analyzer.analyze_drought_period_temperature(
                    drought_start, drought_end
                )
            }

        # 3. Regime Analysis
        print("📈 Analyzing drought regime shifts...")
        regime_analyzer = DroughtRegimeAnalyzer(self.annual_prcp)

        results['regime_analysis'] = {
            'change_point': regime_analyzer.detect_change_points_cusum(),
            'decadal_trends': regime_analyzer.analyze_decadal_trends(),
            'moving_stats': regime_analyzer.calculate_moving_statistics()
        }

        # 4. Periodicity Analysis
        print("🔄 Analyzing drought periodicity...")
        wavelet_analyzer = WaveletDroughtAnalyzer(self.annual_prcp)

        results['periodicity'] = {
            'dominant_periods': wavelet_analyzer.identify_dominant_periods()
        }

        # 5. Historical Context
        print("📚 Establishing historical context...")
        results['historical_context'] = self._calculate_historical_context(
            drought_start, drought_end
        )

        return results

    def _calculate_historical_context(
        self,
        drought_start: int,
        drought_end: int
    ) -> Dict[str, Any]:
        """Calculate where current drought stands in historical record."""

        drought_years = self.annual_prcp[
            (self.annual_prcp.index >= drought_start) &
            (self.annual_prcp.index <= drought_end)
        ]

        # Mean deficit
        mean_prcp = drought_years.mean()
        deficit_percent = ((self.baseline_mean - mean_prcp) / self.baseline_mean) * 100

        # Count how many years in record had similar or worse conditions
        annual_deficits = ((self.baseline_mean - self.annual_prcp) / self.baseline_mean) * 100
        worse_years = (annual_deficits >= deficit_percent).sum()

        # Consecutive year analysis
        consecutive_deficit_years = 0
        max_consecutive_historical = 0
        current_consecutive = 0

        for year in sorted(self.annual_prcp.index):
            if annual_deficits.loc[year] > 0:  # Deficit year
                current_consecutive += 1
                if drought_start <= year <= drought_end:
                    consecutive_deficit_years += 1
            else:
                if current_consecutive > max_consecutive_historical:
                    max_consecutive_historical = current_consecutive
                current_consecutive = 0

        return {
            'drought_period': f"{drought_start}-{drought_end}",
            'mean_precipitation_mm': mean_prcp,
            'baseline_mean_mm': self.baseline_mean,
            'deficit_percent': deficit_percent,
            'years_with_similar_or_worse': worse_years,
            'total_years_in_record': len(self.annual_prcp),
            'percentile_rank': (1 - worse_years / len(self.annual_prcp)) * 100,
            'consecutive_deficit_years': consecutive_deficit_years,
            'max_consecutive_historical': max_consecutive_historical,
            'record_start_year': int(self.annual_prcp.index.min()),
            'record_end_year': int(self.annual_prcp.index.max())
        }

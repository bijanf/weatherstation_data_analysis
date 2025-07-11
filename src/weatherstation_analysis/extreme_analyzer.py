"""
Extreme Value Analysis Module
============================

Performs statistical analysis of extreme weather events and climate data.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gumbel_r
import warnings

warnings.filterwarnings("ignore")


class ExtremeValueAnalyzer:
    """
    Analyzes extreme values in weather data.

    This class provides comprehensive statistical analysis of extreme weather
    events including annual maxima, return periods, and threshold exceedances.

    Attributes:
        data (Dict[int, pd.DataFrame]): Weather data by year
        extremes_df (pd.DataFrame): Analyzed extreme values
    """

    def __init__(self, data: Dict[int, pd.DataFrame]):
        """
        Initialize the extreme value analyzer.

        Args:
            data: Dictionary mapping years to weather DataFrames
        """
        self.data = data
        self.extremes_df: Optional[pd.DataFrame] = None

    def analyze_annual_extremes(
        self, exclude_current_year: bool = True
    ) -> pd.DataFrame:
        """
        Extract annual extreme values from the data.

        Args:
            exclude_current_year: Whether to exclude 2025 from analysis

        Returns:
            DataFrame with annual extreme statistics
        """
        annual_extremes = {
            "year": [],
            "max_precip": [],  # Maximum daily precipitation
            "max_temp": [],  # Maximum temperature
            "min_temp": [],  # Minimum temperature
            "precip_95th": [],  # 95th percentile precipitation
            "temp_range": [],  # Temperature range (max - min)
        }

        for year, data in self.data.items():
            if exclude_current_year and year == 2025:
                continue

            annual_extremes["year"].append(year)

            # Precipitation extremes
            if "prcp" in data.columns:
                annual_extremes["max_precip"].append(data["prcp"].max())
                annual_extremes["precip_95th"].append(data["prcp"].quantile(0.95))
            else:
                annual_extremes["max_precip"].append(np.nan)
                annual_extremes["precip_95th"].append(np.nan)

            # Temperature extremes
            if "tmax" in data.columns and "tmin" in data.columns:
                max_temp = data["tmax"].max()
                min_temp = data["tmin"].min()
                annual_extremes["max_temp"].append(max_temp)
                annual_extremes["min_temp"].append(min_temp)
                annual_extremes["temp_range"].append(max_temp - min_temp)
            else:
                annual_extremes["max_temp"].append(np.nan)
                annual_extremes["min_temp"].append(np.nan)
                annual_extremes["temp_range"].append(np.nan)

        self.extremes_df = pd.DataFrame(annual_extremes)
        return self.extremes_df

    def calculate_return_periods(
        self, values: np.ndarray, distribution: str = "gumbel"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate return periods for extreme values.

        Args:
            values: Array of extreme values
            distribution: Statistical distribution to use
                ('gumbel', 'weibull', 'genextreme')

        Returns:
            Tuple of (return_periods, theoretical_values)
        """
        # Empirical return periods
        sorted_values = np.sort(values)[::-1]  # Sort descending
        n = len(sorted_values)
        empirical_rp = (n + 1) / np.arange(1, n + 1)

        # Theoretical return periods
        theoretical_rp = np.logspace(0, 2, 100)

        if distribution == "gumbel":
            params = gumbel_r.fit(values)
            theoretical_values = gumbel_r.ppf(1 - 1 / theoretical_rp, *params)
        elif distribution == "weibull":
            from scipy.stats import weibull_min

            params = weibull_min.fit(values)
            theoretical_values = weibull_min.ppf(1 - 1 / theoretical_rp, *params)
        elif distribution == "genextreme":
            from scipy.stats import genextreme

            params = genextreme.fit(values)
            theoretical_values = genextreme.ppf(1 - 1 / theoretical_rp, *params)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return empirical_rp, theoretical_values, theoretical_rp

    def analyze_threshold_exceedances(
        self, thresholds: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Analyze frequency of threshold exceedances.

        Args:
            thresholds: Dictionary of variable names to threshold values

        Returns:
            DataFrame with exceedance counts by year
        """
        exceedance_data = {"year": []}

        # Initialize columns for each threshold
        for var_name, threshold in thresholds.items():
            exceedance_data[f"{var_name}_exceedances"] = []

        for year, data in self.data.items():
            if year == 2025:  # Skip incomplete year
                continue

            exceedance_data["year"].append(year)

            for var_name, threshold in thresholds.items():
                if var_name in data.columns:
                    if "above" in var_name.lower():
                        count = (data[var_name.split("_")[0]] > threshold).sum()
                    else:
                        count = (data[var_name.split("_")[0]] < threshold).sum()
                    exceedance_data[f"{var_name}_exceedances"].append(count)
                else:
                    exceedance_data[f"{var_name}_exceedances"].append(np.nan)

        return pd.DataFrame(exceedance_data)

    def calculate_trends(self, variable: str) -> Dict[str, float]:
        """
        Calculate linear trends in extreme values.

        Args:
            variable: Variable name to analyze trends for

        Returns:
            Dictionary with trend statistics
        """
        if self.extremes_df is None:
            raise ValueError("Must run analyze_annual_extremes first")

        if variable not in self.extremes_df.columns:
            raise ValueError(f"Variable {variable} not found in extremes data")

        years = self.extremes_df["year"].values
        values = self.extremes_df[variable].values

        # Remove NaN values
        mask = ~np.isnan(values)
        years_clean = years[mask]
        values_clean = values[mask]

        if len(values_clean) < 2:
            return {
                "slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "p_value": np.nan,
                "std_err": np.nan,
            }

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            years_clean, values_clean
        )

        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
            "trend_per_century": slope * 100,
            "significance": (
                "***"
                if p_value < 0.001
                else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            ),
        }

    def get_extreme_events(self, variable: str, n_events: int = 10) -> pd.DataFrame:
        """
        Get the most extreme events for a given variable.

        Args:
            variable: Variable name to find extremes for
            n_events: Number of extreme events to return

        Returns:
            DataFrame with extreme events sorted by magnitude
        """
        if self.extremes_df is None:
            raise ValueError("Must run analyze_annual_extremes first")

        if variable not in self.extremes_df.columns:
            raise ValueError(f"Variable {variable} not found in extremes data")

        # For temperature minimums, we want the lowest values
        if "min_temp" in variable:
            extreme_events = self.extremes_df.nsmallest(n_events, variable)
        else:
            extreme_events = self.extremes_df.nlargest(n_events, variable)

        return extreme_events[["year", variable]].reset_index(drop=True)

    def calculate_statistics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive statistics for all extreme variables.

        Returns:
            Dictionary with statistics for each variable
        """
        if self.extremes_df is None:
            raise ValueError("Must run analyze_annual_extremes first")

        summary = {}

        for column in self.extremes_df.columns:
            if column == "year":
                continue

            values = self.extremes_df[column].dropna()

            if len(values) == 0:
                continue

            summary[column] = {
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
                "median": values.median(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "skewness": values.skew(),
                "kurtosis": values.kurtosis(),
            }

            # Add year of extreme events
            max_idx = values.idxmax()
            min_idx = values.idxmin()

            summary[column]["max_year"] = self.extremes_df.loc[max_idx, "year"]
            summary[column]["min_year"] = self.extremes_df.loc[min_idx, "year"]

        return summary

    def detect_changepoints(
        self, variable: str, method: str = "pettitt"
    ) -> Dict[str, any]:
        """
        Detect changepoints in extreme value time series.

        Args:
            variable: Variable name to analyze
            method: Method to use ('pettitt', 'cusum')

        Returns:
            Dictionary with changepoint analysis results
        """
        if self.extremes_df is None:
            raise ValueError("Must run analyze_annual_extremes first")

        if variable not in self.extremes_df.columns:
            raise ValueError(f"Variable {variable} not found in extremes data")

        values = self.extremes_df[variable].dropna().values
        years = self.extremes_df.loc[self.extremes_df[variable].notna(), "year"].values

        if method == "pettitt":
            # Simplified Pettitt test implementation
            n = len(values)
            K = np.zeros(n)

            for i in range(n):
                K[i] = np.sum(np.sign(values[i] - values[:i])) + np.sum(
                    np.sign(values[i] - values[i + 1 :])
                )

            max_k_idx = np.argmax(np.abs(K))
            changepoint_year = years[max_k_idx]

            return {
                "method": method,
                "changepoint_year": changepoint_year,
                "changepoint_index": max_k_idx,
                "test_statistic": K[max_k_idx],
                "years": years,
                "values": values,
            }

        else:
            raise ValueError(f"Unknown changepoint method: {method}")

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for extreme variables.

        Returns:
            Correlation matrix DataFrame
        """
        if self.extremes_df is None:
            raise ValueError("Must run analyze_annual_extremes first")

        # Select only numeric columns (excluding year)
        numeric_cols = self.extremes_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "year"]

        return self.extremes_df[numeric_cols].corr()

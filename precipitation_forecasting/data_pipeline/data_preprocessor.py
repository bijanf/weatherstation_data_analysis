"""
Data Preprocessing Module

This module handles feature engineering, data cleaning, and preparation for precipitation forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Handles data preprocessing for precipitation forecasting.
    
    This class provides comprehensive data cleaning, feature engineering,
    and preparation functionality for time series forecasting models.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the data preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.scalers = {}
        self.imputers = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the preprocessor."""
        logger = logging.getLogger('DataPreprocessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_processed_data(self, data_path: str) -> pd.DataFrame:
        """
        Load processed ERA5 data.
        
        Args:
            data_path: Path to processed data file
            
        Returns:
            DataFrame with loaded data
        """
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        self.logger.info(f"Loaded data with shape: {df.shape}")
        self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                          target_col: str = 'precipitation',
                          lag_days: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create lagged features for time series forecasting.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            lag_days: List of lag days to create
            
        Returns:
            DataFrame with lag features
        """
        if lag_days is None:
            lag_days = self.config['training']['features']['lag_days']
        
        df_with_lags = df.copy()
        
        for lag in lag_days:
            lag_col = f"{target_col}_lag_{lag}"
            df_with_lags[lag_col] = df_with_lags[target_col].shift(lag)
            
            # Also create lags for other important variables
            for col in ['temperature', 'pressure', 'wind_speed']:
                if col in df_with_lags.columns:
                    df_with_lags[f"{col}_lag_{lag}"] = df_with_lags[col].shift(lag)
        
        self.logger.info(f"Created lag features for lags: {lag_days}")
        return df_with_lags
    
    def create_rolling_features(self, df: pd.DataFrame,
                              target_col: str = 'precipitation',
                              windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = self.config['training']['features']['rolling_windows']
        
        df_with_rolling = df.copy()
        
        for window in windows:
            # Rolling statistics for precipitation
            df_with_rolling[f"{target_col}_rolling_mean_{window}"] = df_with_rolling[target_col].rolling(window=window).mean()
            df_with_rolling[f"{target_col}_rolling_std_{window}"] = df_with_rolling[target_col].rolling(window=window).std()
            df_with_rolling[f"{target_col}_rolling_max_{window}"] = df_with_rolling[target_col].rolling(window=window).max()
            df_with_rolling[f"{target_col}_rolling_min_{window}"] = df_with_rolling[target_col].rolling(window=window).min()
            
            # Rolling statistics for other variables
            for col in ['temperature', 'pressure', 'wind_speed']:
                if col in df_with_rolling.columns:
                    df_with_rolling[f"{col}_rolling_mean_{window}"] = df_with_rolling[col].rolling(window=window).mean()
                    df_with_rolling[f"{col}_rolling_std_{window}"] = df_with_rolling[col].rolling(window=window).std()
        
        self.logger.info(f"Created rolling features for windows: {windows}")
        return df_with_rolling
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal and cyclical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with seasonal features
        """
        if not self.config['training']['features']['seasonal_features']:
            return df
        
        df_with_seasonal = df.copy()
        
        # Basic time features
        df_with_seasonal['year'] = df_with_seasonal.index.year
        df_with_seasonal['month'] = df_with_seasonal.index.month
        df_with_seasonal['day'] = df_with_seasonal.index.day
        df_with_seasonal['day_of_year'] = df_with_seasonal.index.dayofyear
        df_with_seasonal['week_of_year'] = df_with_seasonal.index.isocalendar().week
        df_with_seasonal['day_of_week'] = df_with_seasonal.index.dayofweek
        
        # Seasonal indicators
        df_with_seasonal['is_winter'] = df_with_seasonal['month'].isin([12, 1, 2]).astype(int)
        df_with_seasonal['is_spring'] = df_with_seasonal['month'].isin([3, 4, 5]).astype(int)
        df_with_seasonal['is_summer'] = df_with_seasonal['month'].isin([6, 7, 8]).astype(int)
        df_with_seasonal['is_autumn'] = df_with_seasonal['month'].isin([9, 10, 11]).astype(int)
        
        # Cyclical encoding for periodic features
        if self.config['training']['features']['cyclical_features']:
            # Day of year (365/366 days cycle)
            df_with_seasonal['day_of_year_sin'] = np.sin(2 * np.pi * df_with_seasonal['day_of_year'] / 365.25)
            df_with_seasonal['day_of_year_cos'] = np.cos(2 * np.pi * df_with_seasonal['day_of_year'] / 365.25)
            
            # Month (12 months cycle)
            df_with_seasonal['month_sin'] = np.sin(2 * np.pi * df_with_seasonal['month'] / 12)
            df_with_seasonal['month_cos'] = np.cos(2 * np.pi * df_with_seasonal['month'] / 12)
            
            # Day of week (7 days cycle)
            df_with_seasonal['day_of_week_sin'] = np.sin(2 * np.pi * df_with_seasonal['day_of_week'] / 7)
            df_with_seasonal['day_of_week_cos'] = np.cos(2 * np.pi * df_with_seasonal['day_of_week'] / 7)
        
        self.logger.info("Created seasonal and cyclical features")
        return df_with_seasonal
    
    def create_weather_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-based indices and derived features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with weather indices
        """
        if not self.config['training']['features']['weather_indices']:
            return df
        
        df_with_indices = df.copy()
        
        # Temperature-based indices
        if 'temperature' in df_with_indices.columns:
            # Temperature anomaly (deviation from long-term mean)
            temp_mean = df_with_indices['temperature'].mean()
            df_with_indices['temperature_anomaly'] = df_with_indices['temperature'] - temp_mean
            
            # Heating/cooling degree days
            df_with_indices['heating_degree_days'] = np.maximum(18 - df_with_indices['temperature'], 0)
            df_with_indices['cooling_degree_days'] = np.maximum(df_with_indices['temperature'] - 18, 0)
        
        # Pressure-based indices
        if 'pressure' in df_with_indices.columns:
            # Pressure tendency (change from previous day)
            df_with_indices['pressure_tendency'] = df_with_indices['pressure'].diff()
            
            # Pressure anomaly
            pressure_climatology = df_with_indices.groupby([df_with_indices.index.month, df_with_indices.index.day])['pressure'].mean()
            df_with_indices['pressure_anomaly'] = df_with_indices['pressure'] - df_with_indices.index.map(
                lambda x: pressure_climatology.get((x.month, x.day), df_with_indices['pressure'].mean())
            )
        
        # Humidity-based indices
        if 'temperature' in df_with_indices.columns and 'dewpoint' in df_with_indices.columns:
            # Relative humidity approximation
            df_with_indices['relative_humidity'] = np.exp(
                17.625 * df_with_indices['dewpoint'] / (243.04 + df_with_indices['dewpoint'])
            ) / np.exp(
                17.625 * df_with_indices['temperature'] / (243.04 + df_with_indices['temperature'])
            ) * 100
        
        # Wind-based indices
        if 'wind_speed' in df_with_indices.columns and 'temperature' in df_with_indices.columns:
            # Wind chill index
            df_with_indices['wind_chill'] = 13.12 + 0.6215 * df_with_indices['temperature'] - \
                                           11.37 * (df_with_indices['wind_speed'] ** 0.16) + \
                                           0.3965 * df_with_indices['temperature'] * (df_with_indices['wind_speed'] ** 0.16)
        
        # Precipitation-based indices
        if 'precipitation' in df_with_indices.columns:
            # Dry/wet day indicators
            df_with_indices['is_dry_day'] = (df_with_indices['precipitation'] < 0.1).astype(int)
            df_with_indices['is_wet_day'] = (df_with_indices['precipitation'] >= 0.1).astype(int)
            df_with_indices['is_heavy_rain'] = (df_with_indices['precipitation'] >= 10.0).astype(int)
            
            # Consecutive dry/wet days
            df_with_indices['consecutive_dry_days'] = self._calculate_consecutive_days(df_with_indices['is_dry_day'])
            df_with_indices['consecutive_wet_days'] = self._calculate_consecutive_days(df_with_indices['is_wet_day'])
        
        self.logger.info("Created weather indices and derived features")
        return df_with_indices
    
    def _calculate_consecutive_days(self, binary_series: pd.Series) -> pd.Series:
        """
        Calculate consecutive occurrences of binary events.
        
        Args:
            binary_series: Series with binary values (0/1)
            
        Returns:
            Series with consecutive day counts
        """
        # Create groups for consecutive sequences
        groups = (binary_series != binary_series.shift()).cumsum()
        
        # Calculate consecutive counts only for sequences of 1s
        consecutive = binary_series.groupby(groups).cumsum()
        consecutive[binary_series == 0] = 0
        
        return consecutive
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Check for missing values
        missing_info = df_clean.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            self.logger.info(f"Found missing values in columns: {missing_cols.to_dict()}")
            
            # Strategy 1: Forward fill for short gaps (< 3 days)
            df_clean = df_clean.fillna(method='ffill', limit=3)
            
            # Strategy 2: Interpolation for remaining gaps
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
            
            # Strategy 3: Median imputation for any remaining missing values
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    if col not in self.imputers:
                        self.imputers[col] = SimpleImputer(strategy='median')
                        df_clean[col] = self.imputers[col].fit_transform(df_clean[[col]]).flatten()
                    else:
                        df_clean[col] = self.imputers[col].transform(df_clean[[col]]).flatten()
            
            self.logger.info("Handled missing values using multiple strategies")
        
        return df_clean
    
    def scale_features(self, df: pd.DataFrame, 
                      target_col: str = 'precipitation',
                      scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Scale features for machine learning models.
        
        Args:
            df: Input DataFrame
            target_col: Target column name (not scaled)
            scaler_type: Type of scaler ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        # Identify numeric columns to scale (excluding target)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col != target_col]
        
        if len(cols_to_scale) == 0:
            return df_scaled
        
        # Initialize scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit and transform
        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
        
        # Store scaler for later use
        self.scalers[f"{scaler_type}_scaler"] = scaler
        
        self.logger.info(f"Scaled {len(cols_to_scale)} features using {scaler_type} scaler")
        return df_scaled
    
    def create_target_sequences(self, df: pd.DataFrame,
                              target_col: str = 'precipitation',
                              forecast_horizon: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create target sequences for different forecast horizons.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            forecast_horizon: Number of days to forecast
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        # Create cumulative precipitation targets
        targets_df = pd.DataFrame(index=df.index)
        
        # Create targets for different horizons
        for horizon in range(1, forecast_horizon + 1):
            # Cumulative precipitation for next 'horizon' days
            targets_df[f'cumulative_precip_{horizon}d'] = df[target_col].rolling(
                window=horizon, min_periods=1
            ).sum().shift(-horizon)
            
            # Single day precipitation
            targets_df[f'precip_{horizon}d'] = df[target_col].shift(-horizon)
        
        # Remove rows with NaN targets
        valid_idx = targets_df.dropna().index
        features_df = df.loc[valid_idx]
        targets_df = targets_df.loc[valid_idx]
        
        self.logger.info(f"Created target sequences for {forecast_horizon} day horizon")
        return features_df, targets_df
    
    def prepare_training_data(self, df: pd.DataFrame,
                            target_col: str = 'precipitation',
                            test_size: float = 0.2,
                            validation_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        Prepare complete training dataset with all features.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Dictionary with train/validation/test splits
        """
        self.logger.info("Preparing complete training dataset...")
        
        # Apply all preprocessing steps
        df_processed = self.handle_missing_values(df)
        df_processed = self.create_lag_features(df_processed, target_col)
        df_processed = self.create_rolling_features(df_processed, target_col)
        df_processed = self.create_seasonal_features(df_processed)
        df_processed = self.create_weather_indices(df_processed)
        
        # Remove rows with NaN values (due to lag features)
        df_processed = df_processed.dropna()
        
        # Create train/validation/test splits (chronological)
        n_samples = len(df_processed)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - validation_size))
        
        train_df = df_processed.iloc[:val_start]
        val_df = df_processed.iloc[val_start:test_start]
        test_df = df_processed.iloc[test_start:]
        
        # Scale features (fit on training data only)
        train_df_scaled = self.scale_features(train_df, target_col)
        
        # Apply same scaling to validation and test sets
        val_df_scaled = val_df.copy()
        test_df_scaled = test_df.copy()
        
        if f"standard_scaler" in self.scalers:
            scaler = self.scalers[f"standard_scaler"]
            numeric_cols = train_df_scaled.select_dtypes(include=[np.number]).columns
            cols_to_scale = [col for col in numeric_cols if col != target_col]
            
            val_df_scaled[cols_to_scale] = scaler.transform(val_df_scaled[cols_to_scale])
            test_df_scaled[cols_to_scale] = scaler.transform(test_df_scaled[cols_to_scale])
        
        self.logger.info(f"Training data shape: {train_df_scaled.shape}")
        self.logger.info(f"Validation data shape: {val_df_scaled.shape}")
        self.logger.info(f"Test data shape: {test_df_scaled.shape}")
        
        return {
            'train': train_df_scaled,
            'validation': val_df_scaled,
            'test': test_df_scaled,
            'train_raw': train_df,
            'validation_raw': val_df,
            'test_raw': test_df
        }
    
    def save_preprocessed_data(self, data_dict: Dict[str, pd.DataFrame], 
                             output_dir: str = "data/processed") -> None:
        """
        Save preprocessed data to files.
        
        Args:
            data_dict: Dictionary with train/validation/test data
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, df in data_dict.items():
            filepath = output_path / f"{split_name}_data.parquet"
            df.to_parquet(filepath)
            self.logger.info(f"Saved {split_name} data to {filepath}")
        
        # Save scalers and other preprocessing objects
        import joblib
        scalers_path = output_path / "scalers.pkl"
        joblib.dump(self.scalers, scalers_path)
        
        imputers_path = output_path / "imputers.pkl"
        joblib.dump(self.imputers, imputers_path)
        
        self.logger.info("Saved preprocessing objects")
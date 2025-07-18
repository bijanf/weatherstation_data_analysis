"""
Baseline Models for Precipitation Forecasting

This module implements simple baseline models that serve as benchmarks
for more sophisticated forecasting approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class BaselineModel:
    """Base class for baseline forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger(f'BaselineModel.{self.name}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, train_data: pd.DataFrame, target_col: str = 'precipitation') -> None:
        """Fit the model to training data."""
        raise NotImplementedError
    
    def predict(self, forecast_dates: pd.DatetimeIndex) -> pd.Series:
        """Generate predictions for given dates."""
        raise NotImplementedError
    
    def forecast_cumulative(self, start_date: datetime, forecast_horizon: int) -> Dict:
        """Generate cumulative precipitation forecast."""
        raise NotImplementedError

class ClimatologyModel(BaselineModel):
    """
    Climatology model that uses historical averages.
    
    This model predicts precipitation based on the long-term average
    for each day of the year.
    """
    
    def __init__(self):
        super().__init__('Climatology')
        self.daily_climatology = None
        self.monthly_climatology = None
        self.overall_mean = None
    
    def fit(self, train_data: pd.DataFrame, target_col: str = 'precipitation') -> None:
        """
        Fit climatology model using historical averages.
        
        Args:
            train_data: Training data with datetime index
            target_col: Target column name
        """
        self.logger.info("Fitting climatology model")
        
        # Calculate daily climatology (day of year)
        self.daily_climatology = train_data.groupby(train_data.index.dayofyear)[target_col].mean()
        
        # Calculate monthly climatology
        self.monthly_climatology = train_data.groupby(train_data.index.month)[target_col].mean()
        
        # Overall mean as fallback
        self.overall_mean = train_data[target_col].mean()
        
        self.is_fitted = True
        self.logger.info(f"Fitted climatology model with {len(train_data)} samples")
    
    def predict(self, forecast_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate climatology predictions for given dates.
        
        Args:
            forecast_dates: Dates to predict for
            
        Returns:
            Series with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for date in forecast_dates:
            day_of_year = date.dayofyear
            
            # Handle leap year edge case (day 366)
            if day_of_year == 366:
                day_of_year = 365
            
            # Get climatology value, use monthly mean if daily not available
            if day_of_year in self.daily_climatology.index:
                pred = self.daily_climatology[day_of_year]
            else:
                # Fallback to monthly climatology
                pred = self.monthly_climatology[date.month]
            
            predictions.append(pred)
        
        return pd.Series(predictions, index=forecast_dates)
    
    def forecast_cumulative(self, start_date: datetime, forecast_horizon: int) -> Dict:
        """
        Generate cumulative precipitation forecast.
        
        Args:
            start_date: Start date for forecast
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Generate forecast dates
        forecast_dates = pd.date_range(start=start_date, periods=forecast_horizon, freq='D')
        
        # Get daily predictions
        daily_predictions = self.predict(forecast_dates)
        
        # Calculate cumulative
        cumulative_predictions = daily_predictions.cumsum()
        
        return {
            'dates': forecast_dates,
            'daily_predictions': daily_predictions,
            'cumulative_predictions': cumulative_predictions,
            'total_predicted': cumulative_predictions.iloc[-1],
            'model_name': self.name
        }

class LinearTrendModel(BaselineModel):
    """
    Linear trend model that extrapolates historical trends.
    
    This model fits a linear regression to historical data and
    extrapolates the trend into the future.
    """
    
    def __init__(self, trend_window: int = 365):
        super().__init__('LinearTrend')
        self.trend_window = trend_window
        self.trend_model = None
        self.seasonal_component = None
        self.train_data = None
    
    def fit(self, train_data: pd.DataFrame, target_col: str = 'precipitation') -> None:
        """
        Fit linear trend model.
        
        Args:
            train_data: Training data with datetime index
            target_col: Target column name
        """
        self.logger.info("Fitting linear trend model")
        
        # Store training data for seasonal component
        self.train_data = train_data.copy()
        
        # Use only recent data for trend estimation
        recent_data = train_data.tail(self.trend_window)
        
        # Create time index (days since start)
        time_index = np.arange(len(recent_data)).reshape(-1, 1)
        
        # Fit linear regression
        self.trend_model = LinearRegression()
        self.trend_model.fit(time_index, recent_data[target_col])
        
        # Calculate seasonal component (residuals from trend)
        trend_predictions = self.trend_model.predict(time_index)
        residuals = recent_data[target_col] - trend_predictions
        
        # Calculate seasonal component by day of year
        seasonal_data = pd.DataFrame({
            'residual': residuals,
            'day_of_year': recent_data.index.dayofyear
        })
        self.seasonal_component = seasonal_data.groupby('day_of_year')['residual'].mean()
        
        self.is_fitted = True
        self.logger.info(f"Fitted linear trend model with {len(recent_data)} samples")
        self.logger.info(f"Trend slope: {self.trend_model.coef_[0]:.6f} mm/day")
    
    def predict(self, forecast_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate trend-based predictions.
        
        Args:
            forecast_dates: Dates to predict for
            
        Returns:
            Series with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Calculate time steps from end of training data
        last_train_date = self.train_data.index[-1]
        days_ahead = [(date - last_train_date).days for date in forecast_dates]
        
        # Get trend predictions
        trend_predictions = self.trend_model.predict(
            np.array(days_ahead).reshape(-1, 1) + self.trend_window
        )
        
        # Add seasonal component
        seasonal_adjustments = []
        for date in forecast_dates:
            day_of_year = date.dayofyear
            if day_of_year == 366:
                day_of_year = 365
            
            if day_of_year in self.seasonal_component.index:
                seasonal_adjustments.append(self.seasonal_component[day_of_year])
            else:
                seasonal_adjustments.append(0.0)
        
        final_predictions = trend_predictions + np.array(seasonal_adjustments)
        
        # Ensure non-negative predictions
        final_predictions = np.maximum(final_predictions, 0)
        
        return pd.Series(final_predictions, index=forecast_dates)
    
    def forecast_cumulative(self, start_date: datetime, forecast_horizon: int) -> Dict:
        """
        Generate cumulative precipitation forecast.
        
        Args:
            start_date: Start date for forecast
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Generate forecast dates
        forecast_dates = pd.date_range(start=start_date, periods=forecast_horizon, freq='D')
        
        # Get daily predictions
        daily_predictions = self.predict(forecast_dates)
        
        # Calculate cumulative
        cumulative_predictions = daily_predictions.cumsum()
        
        return {
            'dates': forecast_dates,
            'daily_predictions': daily_predictions,
            'cumulative_predictions': cumulative_predictions,
            'total_predicted': cumulative_predictions.iloc[-1],
            'model_name': self.name
        }

class SeasonalNaiveModel(BaselineModel):
    """
    Seasonal naive model that uses last year's values.
    
    This model predicts precipitation based on the same date
    in the previous year.
    """
    
    def __init__(self):
        super().__init__('SeasonalNaive')
        self.historical_data = None
    
    def fit(self, train_data: pd.DataFrame, target_col: str = 'precipitation') -> None:
        """
        Fit seasonal naive model.
        
        Args:
            train_data: Training data with datetime index
            target_col: Target column name
        """
        self.logger.info("Fitting seasonal naive model")
        
        # Store historical data
        self.historical_data = train_data[target_col].copy()
        
        self.is_fitted = True
        self.logger.info(f"Fitted seasonal naive model with {len(train_data)} samples")
    
    def predict(self, forecast_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate seasonal naive predictions.
        
        Args:
            forecast_dates: Dates to predict for
            
        Returns:
            Series with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for date in forecast_dates:
            # Look for same date in previous year
            prev_year_date = date - timedelta(days=365)
            
            # If leap year, handle Feb 29
            if date.month == 2 and date.day == 29:
                prev_year_date = date - timedelta(days=366)
            
            # Find closest available date in historical data
            available_dates = self.historical_data.index
            closest_date = min(available_dates, key=lambda x: abs((x - prev_year_date).days))
            
            predictions.append(self.historical_data[closest_date])
        
        return pd.Series(predictions, index=forecast_dates)
    
    def forecast_cumulative(self, start_date: datetime, forecast_horizon: int) -> Dict:
        """
        Generate cumulative precipitation forecast.
        
        Args:
            start_date: Start date for forecast
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Generate forecast dates
        forecast_dates = pd.date_range(start=start_date, periods=forecast_horizon, freq='D')
        
        # Get daily predictions
        daily_predictions = self.predict(forecast_dates)
        
        # Calculate cumulative
        cumulative_predictions = daily_predictions.cumsum()
        
        return {
            'dates': forecast_dates,
            'daily_predictions': daily_predictions,
            'cumulative_predictions': cumulative_predictions,
            'total_predicted': cumulative_predictions.iloc[-1],
            'model_name': self.name
        }

class MovingAverageModel(BaselineModel):
    """
    Moving average model that uses recent historical averages.
    
    This model predicts precipitation based on the moving average
    of recent historical data.
    """
    
    def __init__(self, window_size: int = 30):
        super().__init__('MovingAverage')
        self.window_size = window_size
        self.recent_data = None
    
    def fit(self, train_data: pd.DataFrame, target_col: str = 'precipitation') -> None:
        """
        Fit moving average model.
        
        Args:
            train_data: Training data with datetime index
            target_col: Target column name
        """
        self.logger.info("Fitting moving average model")
        
        # Store recent data for predictions
        self.recent_data = train_data[target_col].tail(self.window_size)
        
        self.is_fitted = True
        self.logger.info(f"Fitted moving average model with window size {self.window_size}")
    
    def predict(self, forecast_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate moving average predictions.
        
        Args:
            forecast_dates: Dates to predict for
            
        Returns:
            Series with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use mean of recent data as prediction
        mean_prediction = self.recent_data.mean()
        
        predictions = pd.Series(
            [mean_prediction] * len(forecast_dates),
            index=forecast_dates
        )
        
        return predictions
    
    def forecast_cumulative(self, start_date: datetime, forecast_horizon: int) -> Dict:
        """
        Generate cumulative precipitation forecast.
        
        Args:
            start_date: Start date for forecast
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Generate forecast dates
        forecast_dates = pd.date_range(start=start_date, periods=forecast_horizon, freq='D')
        
        # Get daily predictions
        daily_predictions = self.predict(forecast_dates)
        
        # Calculate cumulative
        cumulative_predictions = daily_predictions.cumsum()
        
        return {
            'dates': forecast_dates,
            'daily_predictions': daily_predictions,
            'cumulative_predictions': cumulative_predictions,
            'total_predicted': cumulative_predictions.iloc[-1],
            'model_name': self.name
        }

class BaselineModelEnsemble:
    """
    Ensemble of baseline models with different weighting schemes.
    
    This class combines multiple baseline models to create
    a more robust forecast.
    """
    
    def __init__(self, models: Optional[List[BaselineModel]] = None):
        """
        Initialize baseline ensemble.
        
        Args:
            models: List of baseline models to include
        """
        if models is None:
            self.models = [
                ClimatologyModel(),
                LinearTrendModel(),
                SeasonalNaiveModel(),
                MovingAverageModel()
            ]
        else:
            self.models = models
        
        self.weights = None
        self.is_fitted = False
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger('BaselineEnsemble')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, train_data: pd.DataFrame, 
            validation_data: Optional[pd.DataFrame] = None,
            target_col: str = 'precipitation') -> None:
        """
        Fit all baseline models and calculate optimal weights.
        
        Args:
            train_data: Training data
            validation_data: Validation data for weight calculation
            target_col: Target column name
        """
        self.logger.info("Fitting baseline ensemble")
        
        # Fit all models
        for model in self.models:
            model.fit(train_data, target_col)
        
        # Calculate weights based on validation performance
        if validation_data is not None:
            self._calculate_weights(validation_data, target_col)
        else:
            # Equal weights if no validation data
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        self.is_fitted = True
        self.logger.info(f"Fitted ensemble with {len(self.models)} models")
        self.logger.info(f"Model weights: {dict(zip([m.name for m in self.models], self.weights))}")
    
    def _calculate_weights(self, validation_data: pd.DataFrame, target_col: str) -> None:
        """Calculate optimal weights based on validation performance."""
        # Generate predictions for validation period
        val_dates = validation_data.index
        model_predictions = []
        
        for model in self.models:
            try:
                preds = model.predict(val_dates)
                model_predictions.append(preds)
            except Exception as e:
                self.logger.warning(f"Failed to get predictions from {model.name}: {e}")
                model_predictions.append(pd.Series([0] * len(val_dates), index=val_dates))
        
        # Calculate RMSE for each model
        rmse_scores = []
        for preds in model_predictions:
            rmse = np.sqrt(mean_squared_error(validation_data[target_col], preds))
            rmse_scores.append(rmse)
        
        # Convert RMSE to weights (inverse relationship)
        rmse_scores = np.array(rmse_scores)
        weights = 1.0 / (rmse_scores + 1e-8)  # Add small epsilon to avoid division by zero
        self.weights = weights / weights.sum()  # Normalize to sum to 1
    
    def predict(self, forecast_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate ensemble predictions.
        
        Args:
            forecast_dates: Dates to predict for
            
        Returns:
            Series with ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            try:
                preds = model.predict(forecast_dates)
                all_predictions.append(preds)
            except Exception as e:
                self.logger.warning(f"Failed to get predictions from {model.name}: {e}")
                all_predictions.append(pd.Series([0] * len(forecast_dates), index=forecast_dates))
        
        # Calculate weighted average
        ensemble_prediction = pd.Series(0, index=forecast_dates)
        for i, preds in enumerate(all_predictions):
            ensemble_prediction += self.weights[i] * preds
        
        return ensemble_prediction
    
    def forecast_cumulative(self, start_date: datetime, forecast_horizon: int) -> Dict:
        """
        Generate cumulative precipitation forecast.
        
        Args:
            start_date: Start date for forecast
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Generate forecast dates
        forecast_dates = pd.date_range(start=start_date, periods=forecast_horizon, freq='D')
        
        # Get daily predictions
        daily_predictions = self.predict(forecast_dates)
        
        # Calculate cumulative
        cumulative_predictions = daily_predictions.cumsum()
        
        # Also get individual model forecasts for comparison
        individual_forecasts = {}
        for model in self.models:
            try:
                model_forecast = model.forecast_cumulative(start_date, forecast_horizon)
                individual_forecasts[model.name] = model_forecast
            except Exception as e:
                self.logger.warning(f"Failed to get forecast from {model.name}: {e}")
        
        return {
            'dates': forecast_dates,
            'daily_predictions': daily_predictions,
            'cumulative_predictions': cumulative_predictions,
            'total_predicted': cumulative_predictions.iloc[-1],
            'model_name': 'BaselineEnsemble',
            'individual_forecasts': individual_forecasts,
            'model_weights': dict(zip([m.name for m in self.models], self.weights))
        }
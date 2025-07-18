"""
ERA5 Data Downloader

This module handles downloading ERA5 reanalysis data from the Copernicus Climate Data Store.
It provides functionality to download various meteorological variables for precipitation forecasting.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

class ERA5Downloader:
    """
    Downloads ERA5 reanalysis data for precipitation forecasting.
    
    This class handles authentication, data requests, and basic preprocessing
    of ERA5 data from the Copernicus Climate Data Store.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the ERA5 downloader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
        self.logger = self._setup_logger()
        
        # Create data directories
        self.raw_data_dir = Path(self.config['paths']['data_raw'])
        self.processed_data_dir = Path(self.config['paths']['data_processed'])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_client(self) -> cdsapi.Client:
        """Initialize CDS API client."""
        try:
            return cdsapi.Client()
        except Exception as e:
            logging.error(f"Failed to initialize CDS client: {e}")
            logging.error("Please ensure you have a valid .cdsapirc file with your CDS API credentials")
            raise
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the downloader."""
        logger = logging.getLogger('ERA5Downloader')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def download_era5_data(self, 
                          start_date: str,
                          end_date: str,
                          variables: Optional[List[str]] = None) -> str:
        """
        Download ERA5 reanalysis data for specified period.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            variables: List of variables to download (if None, uses config defaults)
            
        Returns:
            Path to downloaded file
        """
        if variables is None:
            variables = self.config['data']['era5']['variables']
        
        # Prepare spatial extent
        area = [
            self.config['data']['era5']['spatial_extent']['north'],
            self.config['data']['era5']['spatial_extent']['west'],
            self.config['data']['era5']['spatial_extent']['south'],
            self.config['data']['era5']['spatial_extent']['east']
        ]
        
        # Generate filename
        filename = f"era5_{start_date}_{end_date}.nc"
        filepath = self.raw_data_dir / filename
        
        # Skip if file already exists
        if filepath.exists():
            self.logger.info(f"File {filename} already exists, skipping download")
            return str(filepath)
        
        # Prepare date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date list
        date_list = []
        current_date = start_dt
        while current_date <= end_dt:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # Prepare CDS request
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'date': date_list,
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
            ],
            'area': area,
            'grid': [self.config['data']['era5']['spatial_resolution'], 
                    self.config['data']['era5']['spatial_resolution']]
        }
        
        self.logger.info(f"Downloading ERA5 data for {start_date} to {end_date}")
        self.logger.info(f"Variables: {variables}")
        
        try:
            self.client.retrieve(
                'reanalysis-era5-single-levels',
                request,
                str(filepath)
            )
            self.logger.info(f"Successfully downloaded {filename}")
            return str(filepath)
        
        except Exception as e:
            self.logger.error(f"Failed to download ERA5 data: {e}")
            raise
    
    def download_historical_data(self, 
                               start_year: Optional[int] = None,
                               end_year: Optional[int] = None) -> List[str]:
        """
        Download historical ERA5 data year by year.
        
        Args:
            start_year: Start year (if None, uses config default)
            end_year: End year (if None, uses config default)
            
        Returns:
            List of downloaded file paths
        """
        if start_year is None:
            start_year = self.config['data']['era5']['start_year']
        if end_year is None:
            end_year = self.config['data']['era5']['end_year']
        
        downloaded_files = []
        
        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            # Handle leap years and current year
            if year == datetime.now().year:
                # For current year, only download up to yesterday
                yesterday = datetime.now() - timedelta(days=1)
                end_date = yesterday.strftime('%Y-%m-%d')
            
            try:
                filepath = self.download_era5_data(start_date, end_date)
                downloaded_files.append(filepath)
                self.logger.info(f"Completed download for year {year}")
                
            except Exception as e:
                self.logger.error(f"Failed to download data for year {year}: {e}")
                continue
        
        return downloaded_files
    
    def preprocess_era5_data(self, filepath: str) -> str:
        """
        Preprocess ERA5 data for forecasting.
        
        Args:
            filepath: Path to raw ERA5 NetCDF file
            
        Returns:
            Path to processed file
        """
        self.logger.info(f"Preprocessing ERA5 data: {filepath}")
        
        # Load data
        ds = xr.open_dataset(filepath)
        
        # Extract location data
        lat = self.config['location']['latitude']
        lon = self.config['location']['longitude']
        
        # Select nearest grid point
        ds_location = ds.sel(latitude=lat, longitude=lon, method='nearest')
        
        # Calculate daily aggregates
        daily_data = {}
        
        # Total precipitation - sum over day
        if 'tp' in ds_location.data_vars:
            daily_data['precipitation'] = ds_location['tp'].resample(time='1D').sum()
        
        # Temperature - daily mean
        if 't2m' in ds_location.data_vars:
            daily_data['temperature'] = ds_location['t2m'].resample(time='1D').mean()
        
        # Pressure - daily mean
        if 'msl' in ds_location.data_vars:
            daily_data['pressure'] = ds_location['msl'].resample(time='1D').mean()
        
        # Wind components - daily mean
        if 'u10' in ds_location.data_vars:
            daily_data['u_wind'] = ds_location['u10'].resample(time='1D').mean()
        if 'v10' in ds_location.data_vars:
            daily_data['v_wind'] = ds_location['v10'].resample(time='1D').mean()
        
        # Dewpoint temperature - daily mean
        if 'd2m' in ds_location.data_vars:
            daily_data['dewpoint'] = ds_location['d2m'].resample(time='1D').mean()
        
        # Combine into single dataset
        processed_ds = xr.Dataset(daily_data)
        
        # Convert to pandas DataFrame for easier handling
        df = processed_ds.to_dataframe()
        
        # Convert units
        if 'precipitation' in df.columns:
            df['precipitation'] = df['precipitation'] * 1000  # Convert m to mm
        if 'temperature' in df.columns:
            df['temperature'] = df['temperature'] - 273.15  # Convert K to °C
        if 'dewpoint' in df.columns:
            df['dewpoint'] = df['dewpoint'] - 273.15  # Convert K to °C
        if 'pressure' in df.columns:
            df['pressure'] = df['pressure'] / 100  # Convert Pa to hPa
        
        # Calculate derived variables
        if 'u_wind' in df.columns and 'v_wind' in df.columns:
            df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
            df['wind_direction'] = np.arctan2(df['v_wind'], df['u_wind']) * 180 / np.pi
        
        # Save processed data
        processed_filename = Path(filepath).stem + "_processed.parquet"
        processed_filepath = self.processed_data_dir / processed_filename
        
        df.to_parquet(processed_filepath)
        self.logger.info(f"Processed data saved to: {processed_filepath}")
        
        return str(processed_filepath)
    
    def get_latest_available_date(self) -> str:
        """
        Get the latest available date for ERA5 data.
        
        Returns:
            Latest available date in 'YYYY-MM-DD' format
        """
        # ERA5 has approximately 5-day delay
        lag_days = self.config['operational']['data_update']['era5_lag_days']
        latest_date = datetime.now() - timedelta(days=lag_days)
        return latest_date.strftime('%Y-%m-%d')
    
    def update_data(self) -> str:
        """
        Update ERA5 data with latest available data.
        
        Returns:
            Path to updated processed file
        """
        latest_date = self.get_latest_available_date()
        
        # Find the most recent file
        existing_files = list(self.raw_data_dir.glob("era5_*.nc"))
        
        if not existing_files:
            # No existing files, download from start year
            start_date = f"{self.config['data']['era5']['start_year']}-01-01"
        else:
            # Find the latest date in existing files
            latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
            # Extract end date from filename
            filename_parts = latest_file.stem.split('_')
            last_date = filename_parts[-1]
            start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Download new data
        filepath = self.download_era5_data(start_date, latest_date)
        
        # Preprocess new data
        processed_filepath = self.preprocess_era5_data(filepath)
        
        return processed_filepath
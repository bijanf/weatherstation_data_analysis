"""
Weather Station Analysis Module
==============================

Core analysis modules for weather station data processing and visualization.
"""

from .data_fetcher import PotsdamDataFetcher
from .weather_fetcher import WeatherDataFetcher
from .extreme_analyzer import ExtremeValueAnalyzer
from .visualization import WeatherPlotter
from .city_manager import CityManager

# Iran drought analysis modules
from .iran_data_fetcher import (
    IranianDataFetcher,
    IranianStationRegistry,
    MultiStationFetcher,
)
from .drought_analyzer import DroughtAnalyzer, MultiStationDroughtAnalyzer
from .drought_plotter import DroughtPlotter

# Advanced drought analysis modules
from .advanced_drought_analyzer import (
    DroughtReturnPeriodAnalyzer,
    CompoundEventAnalyzer,
    DroughtDSAAnalyzer,
    DroughtRegimeAnalyzer,
    WaveletDroughtAnalyzer,
    MegadroughtAnalyzer,
)
from .advanced_drought_plotter import AdvancedDroughtPlotter

# Multi-source data fetchers
from .chirps_fetcher import CHIRPSFetcher
from .era5_fetcher import ERA5Fetcher

__all__ = [
    # Core modules
    "PotsdamDataFetcher",
    "WeatherDataFetcher",
    "ExtremeValueAnalyzer",
    "WeatherPlotter",
    "CityManager",
    # Iran drought modules
    "IranianDataFetcher",
    "IranianStationRegistry",
    "MultiStationFetcher",
    "DroughtAnalyzer",
    "MultiStationDroughtAnalyzer",
    "DroughtPlotter",
    # Advanced drought modules
    "DroughtReturnPeriodAnalyzer",
    "CompoundEventAnalyzer",
    "DroughtDSAAnalyzer",
    "DroughtRegimeAnalyzer",
    "WaveletDroughtAnalyzer",
    "MegadroughtAnalyzer",
    "AdvancedDroughtPlotter",
    # Multi-source data fetchers
    "CHIRPSFetcher",
    "ERA5Fetcher",
]

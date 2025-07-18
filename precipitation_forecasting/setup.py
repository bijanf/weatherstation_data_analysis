"""Setup script for ERA5 Precipitation Forecasting project."""

from setuptools import setup, find_packages

setup(
    name="precipitation-forecasting",
    version="0.1.0",
    description="ERA5-based precipitation forecasting using machine learning",
    author="Bijan Fallah",
    author_email="bijan.fallah@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "cdsapi>=0.5.1",
        "xarray>=0.19.0",
        "netcdf4>=1.5.8",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "deep-learning": ["torch>=1.12.0", "tensorflow>=2.9.0"],
        "time-series": ["statsmodels>=0.13.0", "prophet>=1.1.0"],
        "visualization": ["plotly>=5.0.0", "cartopy>=0.20.0"],
        "development": ["pytest>=6.2.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
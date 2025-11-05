"""Quick test to check what German stations are available"""
from meteostat import Stations
import pandas as pd

# Try different approaches to find German stations
print("Testing German weather station retrieval...")

# Method 1: By country
print("\n1. Searching by country code 'DE':")
stations = Stations()
stations_de = stations.region('DE')
result1 = stations_de.fetch()
print(f"   Found {len(result1)} stations")
if not result1.empty:
    print(result1.head())
    print(f"\nColumns: {result1.columns.tolist()}")

# Method 2: By coordinates (Berlin area)
print("\n2. Searching near Berlin (52.5°N, 13.4°E):")
stations2 = Stations()
stations_berlin = stations2.nearby(52.5, 13.4, radius=50000)
result2 = stations_berlin.fetch()
print(f"   Found {len(result2)} stations")
if not result2.empty:
    print(result2.head())

# Method 3: By coordinates (Potsdam)
print("\n3. Searching near Potsdam (52.38°N, 13.07°E):")
stations3 = Stations()
stations_potsdam = stations3.nearby(52.38, 13.07, radius=50000)
result3 = stations_potsdam.fetch()
print(f"   Found {len(result3)} stations")
if not result3.empty:
    print(result3.head())
    # Check for long-term stations
    if 'daily_start' in result3.columns and 'daily_end' in result3.columns:
        result3['years'] = (
            pd.to_datetime(result3['daily_end']) -
            pd.to_datetime(result3['daily_start'])
        ).dt.days / 365.25
        long_stations = result3[result3['years'] >= 50]
        print(f"\n   Stations with 50+ years: {len(long_stations)}")
        if not long_stations.empty:
            print(long_stations[['name', 'years', 'daily_start', 'daily_end']].sort_values('years', ascending=False))

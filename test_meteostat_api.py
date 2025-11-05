"""Test if Meteostat API is working at all"""
from meteostat import Stations, Daily
from datetime import datetime
import pandas as pd

print("Testing Meteostat API functionality...\n")

# Test 1: Direct station ID (Potsdam - as used in existing code)
print("1. Testing with known Potsdam station ID:")
try:
    # Try to find Potsdam station by coordinates (from existing code)
    stations = Stations()
    stations = stations.nearby(52.3833, 13.0667)
    station = stations.fetch(1)

    if not station.empty:
        station_id = station.index[0]
        print(f"   ✓ Found station: {station.loc[station_id, 'name']} (ID: {station_id})")

        # Try to fetch some recent data
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        data = Daily(station_id, start, end).fetch()
        if not data.empty:
            print(f"   ✓ Successfully fetched {len(data)} days of data")
            print(f"   Available columns: {data.columns.tolist()}")
        else:
            print("   ✗ No data available for this period")
    else:
        print("   ✗ No station found near Potsdam coordinates")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Try some known German weather stations by direct ID
print("\n2. Testing known German station IDs:")
known_stations = {
    '10382': 'Potsdam',
    '10384': 'Berlin-Tempelhof',
    '10637': 'Frankfurt am Main',
    '10763': 'Hohenpeißenberg',
}

for sid, name in known_stations.items():
    try:
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)
        data = Daily(sid, start, end).fetch()
        if not data.empty:
            print(f"   ✓ {name} ({sid}): {len(data)} days")
        else:
            print(f"   ✗ {name} ({sid}): No data")
    except Exception as e:
        print(f"   ✗ {name} ({sid}): Error - {e}")

# Test 3: Check Meteostat version and configuration
print("\n3. Meteostat configuration:")
import meteostat
print(f"   Version: {meteostat.__version__}")
print(f"   Cache directory: {meteostat.core.cache_dir}")

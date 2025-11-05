"""Test wetterdienst API to find the correct usage"""
from datetime import datetime
from wetterdienst.provider.dwd.observation import DwdObservationRequest

print("Testing wetterdienst API...")

# Try fetching data for a known station
try:
    print("\n1. Trying with station ID '02925' (Potsdam)...")

    # Try the correct format: resolution/dataset
    request = DwdObservationRequest(
        parameters="daily/kl",  # daily/climate_summary
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 31)
    ).filter_by_station_id(station_id="02925")

    df = request.values.all().df
    print(f"   Success! Got {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   First few rows:")
    print(df.head())

except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

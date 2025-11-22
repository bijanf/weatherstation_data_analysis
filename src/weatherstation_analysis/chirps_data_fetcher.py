"""
CHIRPS Satellite Precipitation Data Fetcher
===========================================

Handles fetching CHIRPS daily precipitation data using the `climateserv` library.
This is designed for hydrological analysis where ground station data is sparse.
"""

from typing import Optional
import pandas as pd
from datetime import datetime
import climateserv.api


class CHIRPSDataFetcher:
    """
    Fetches CHIRPS daily precipitation data for a specific point location.
    """

    def __init__(self, latitude: float, longitude: float, location_name: str):
        """
        Initialize the fetcher for a specific location.

        Args:
            latitude: Latitude of the point.
            longitude: Longitude of the point.
            location_name: A name for the location (e.g., "Zagros Headwaters").
        """
        self.latitude = latitude
        self.longitude = longitude
        self.location_name = location_name
        self.dataset_type = "CHIRPS"

    def fetch_precipitation_data(
        self,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches daily CHIRPS precipitation data for the specified date range.

        Args:
            start_date: Start date in "YYYY-MM-DD" format.
            end_date: End date in "YYYY-MM-DD" format.

        Returns:
            A pandas DataFrame with a datetime index and 'precipitation_mm' column,
            or None if data fetching fails.
        """
        print(
            f"🛰️  Fetching CHIRPS data for {self.location_name} ({self.latitude}, {self.longitude})..."
        )
        print(f"    Date range: {start_date} to {end_date}")

        try:
            # Create a small bounding box for the geometry
            lon, lat = self.longitude, self.latitude
            geometry_coords = [
                [lon - 0.01, lat + 0.01],
                [lon + 0.01, lat + 0.01],
                [lon + 0.01, lat - 0.01],
                [lon - 0.01, lat - 0.01],
                [lon - 0.01, lat + 0.01],
            ]

            # Reformat dates to MM/DD/YYYY
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            start_date_str = start_date_obj.strftime("%m/%d/%Y")
            end_date_str = end_date_obj.strftime("%m/%d/%Y")

            # Request data from ClimateSERV API
            data = climateserv.api.request_data(
                data_set_type=0,  # CHIRPS is dataset 0
                operation_type="Average",
                earliest_date=start_date_str,
                latest_date=end_date_str,
                geometry_coords=geometry_coords,
                seasonal_ensemble="",
                seasonal_variable="",
                outfile="memory_object",
            )

            # Debugging: Print the raw data object
            print(f"DEBUG: Raw data from ClimateSERV API: {data}")

            # The API returns a JSON object with a 'data' key
            # if data is None, it means there was an error in the API call itself
            if data is None or "data" not in data:
                print(
                    "⚠️ No valid data or 'data' key found in the response from ClimateSERV API."
                )
                return None

            df = pd.DataFrame(data["data"])

            if df.empty:
                print("⚠️ No data returned from CHIRPS API.")
                return None

            # Rename columns and set index
            df = df.rename(columns={"value": "precipitation_mm", "date": "DATE"})
            df["DATE"] = pd.to_datetime(df["DATE"])
            df.set_index("DATE", inplace=True)

            # The API might return other columns, so we select only the one we need
            prcp_data = df[["precipitation_mm"]]

            print(f"✅ Successfully fetched {len(prcp_data)} days of CHIRPS data.")
            return prcp_data

        except Exception as e:
            print(f"❌ Error fetching or processing CHIRPS data: {e}")
            return None

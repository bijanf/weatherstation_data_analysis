"""
City Manager Module
==================

Handles city name resolution, fuzzy matching, and station discovery for German cities.
"""

import difflib
from typing import Dict, List, Optional
from meteostat import Stations
import pandas as pd


class CityManager:
    """Manages city-based weather station lookup and fuzzy matching."""

    def __init__(self):
        """Initialize the city manager with German and Iranian city databases."""
        # Major Iranian cities with approximate coordinates
        self.iranian_cities = {
            "tehran": {"lat": 35.689, "lon": 51.313, "name": "Tehran"},
            "mashhad": {"lat": 36.267, "lon": 59.633, "name": "Mashhad"},
            "isfahan": {"lat": 32.750, "lon": 51.667, "name": "Isfahan"},
            "esfahan": {"lat": 32.750, "lon": 51.667, "name": "Isfahan"},
            "tabriz": {"lat": 38.133, "lon": 46.300, "name": "Tabriz"},
            "shiraz": {"lat": 29.533, "lon": 52.600, "name": "Shiraz"},
            "ahvaz": {"lat": 31.333, "lon": 48.667, "name": "Ahvaz"},
            "ahwaz": {"lat": 31.333, "lon": 48.667, "name": "Ahvaz"},
            "kerman": {"lat": 30.250, "lon": 56.967, "name": "Kerman"},
            "rasht": {"lat": 37.317, "lon": 49.617, "name": "Rasht"},
            "zahedan": {"lat": 29.467, "lon": 60.883, "name": "Zahedan"},
            "bandarabbas": {"lat": 27.217, "lon": 56.367, "name": "Bandar Abbas"},
            "bandar abbas": {"lat": 27.217, "lon": 56.367, "name": "Bandar Abbas"},
        }

        # Major German cities with approximate coordinates
        self.german_cities = {
            "berlin": {"lat": 52.5200, "lon": 13.4050, "name": "Berlin"},
            "hamburg": {"lat": 53.5511, "lon": 9.9937, "name": "Hamburg"},
            "munich": {"lat": 48.1351, "lon": 11.5820, "name": "Munich"},
            "münchen": {"lat": 48.1351, "lon": 11.5820, "name": "Munich"},
            "cologne": {"lat": 50.9375, "lon": 6.9603, "name": "Cologne"},
            "köln": {"lat": 50.9375, "lon": 6.9603, "name": "Cologne"},
            "frankfurt": {"lat": 50.1109, "lon": 8.6821, "name": "Frankfurt"},
            "stuttgart": {"lat": 48.7758, "lon": 9.1829, "name": "Stuttgart"},
            "düsseldorf": {"lat": 51.2277, "lon": 6.7735, "name": "Düsseldorf"},
            "dortmund": {"lat": 51.5136, "lon": 7.4653, "name": "Dortmund"},
            "essen": {"lat": 51.4556, "lon": 7.0116, "name": "Essen"},
            "bremen": {"lat": 53.0793, "lon": 8.8017, "name": "Bremen"},
            "dresden": {"lat": 51.0504, "lon": 13.7373, "name": "Dresden"},
            "leipzig": {"lat": 51.3397, "lon": 12.3731, "name": "Leipzig"},
            "hannover": {"lat": 52.3759, "lon": 9.7320, "name": "Hannover"},
            "nuremberg": {"lat": 49.4521, "lon": 11.0767, "name": "Nuremberg"},
            "nürnberg": {"lat": 49.4521, "lon": 11.0767, "name": "Nuremberg"},
            "potsdam": {"lat": 52.3833, "lon": 13.0667, "name": "Potsdam"},
            "mannheim": {"lat": 49.4875, "lon": 8.4660, "name": "Mannheim"},
            "karlsruhe": {"lat": 49.0069, "lon": 8.4037, "name": "Karlsruhe"},
            "augsburg": {"lat": 48.3705, "lon": 10.8978, "name": "Augsburg"},
            "wiesbaden": {"lat": 50.0782, "lon": 8.2397, "name": "Wiesbaden"},
            "mainz": {"lat": 49.9929, "lon": 8.2473, "name": "Mainz"},
            "freiburg": {"lat": 47.9990, "lon": 7.8421, "name": "Freiburg"},
            "rostock": {"lat": 54.0887, "lon": 12.1425, "name": "Rostock"},
            "kiel": {"lat": 54.3233, "lon": 10.1228, "name": "Kiel"},
            "erfurt": {"lat": 50.9848, "lon": 11.0299, "name": "Erfurt"},
            "magdeburg": {"lat": 52.1205, "lon": 11.6276, "name": "Magdeburg"},
        }

    def find_city_match(self, city_input: str, country: str = "auto") -> Optional[Dict]:
        """
        Find the best matching city using fuzzy matching.

        Args:
            city_input: User input city name
            country: Country to search in ("germany", "iran", or "auto" for both)

        Returns:
            Dictionary with city info or None if no good match
        """
        city_lower = city_input.lower().strip()

        # Determine which city databases to search
        databases = []
        if country == "auto":
            databases = [self.iranian_cities, self.german_cities]
        elif country == "iran":
            databases = [self.iranian_cities]
        elif country == "germany":
            databases = [self.german_cities]

        # Exact match first
        for db in databases:
            if city_lower in db:
                return db[city_lower]

        # Fuzzy matching
        for db in databases:
            city_names = list(db.keys())
            matches = difflib.get_close_matches(city_lower, city_names, n=1, cutoff=0.6)
            if matches:
                return db[matches[0]]

        return None

    def get_suggestions(self, city_input: str, n: int = 3) -> List[str]:
        """
        Get city name suggestions for partial/incorrect input.

        Args:
            city_input: User input city name
            n: Number of suggestions to return

        Returns:
            List of suggested city names
        """
        city_lower = city_input.lower().strip()
        city_names = list(self.german_cities.keys())

        suggestions = difflib.get_close_matches(city_lower, city_names, n=n, cutoff=0.3)

        # Return the display names
        return [self.german_cities[name]["name"] for name in suggestions]

    def find_stations_near_city(
        self, city_name: str, radius_km: int = 50
    ) -> Optional[pd.DataFrame]:
        """
        Find weather stations near a given city.

        Args:
            city_name: Name of the city
            radius_km: Search radius in kilometers

        Returns:
            DataFrame with station information or None
        """
        city_info = self.find_city_match(city_name)
        if not city_info:
            return None

        try:
            # Get stations within radius
            stations = Stations()
            stations = stations.nearby(
                city_info["lat"], city_info["lon"], radius_km * 1000
            )  # Convert to meters
            stations_df = stations.fetch()

            if not stations_df.empty:
                # Add distance information
                stations_df["distance_km"] = (
                    (
                        (stations_df["latitude"] - city_info["lat"]) ** 2
                        + (stations_df["longitude"] - city_info["lon"]) ** 2
                    )
                    ** 0.5
                ) * 111  # Rough conversion to km

                # Sort by distance
                stations_df = stations_df.sort_values("distance_km")

                print(f"📍 Found {len(stations_df)} stations near {city_info['name']}")
                return stations_df
            else:
                print(f"❌ No stations found near {city_info['name']}")
                return None

        except Exception as e:
            print(f"❌ Error finding stations: {e}")
            return None

    def get_city_info(self, city_name: str) -> Optional[Dict]:
        """Get city information including coordinates."""
        return self.find_city_match(city_name)

    def list_available_cities(self, country: str = "all") -> List[str]:
        """
        List all available cities.

        Args:
            country: Country filter ("all", "germany", "iran")

        Returns:
            Sorted list of available city names
        """
        unique_cities = set()

        if country in ["all", "iran"]:
            for city_data in self.iranian_cities.values():
                unique_cities.add(city_data["name"])

        if country in ["all", "germany"]:
            for city_data in self.german_cities.values():
                unique_cities.add(city_data["name"])

        return sorted(list(unique_cities))

    def list_iranian_cities(self) -> List[str]:
        """List all available Iranian cities."""
        return self.list_available_cities(country="iran")

    def list_german_cities(self) -> List[str]:
        """List all available German cities."""
        return self.list_available_cities(country="germany")

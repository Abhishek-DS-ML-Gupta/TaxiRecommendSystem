"""
driver_profiles.py — Realistic Driver Profiles and Cold-Start Personas for MORA-LLM.

In MORA-LLM (Ge et al., 2026), user background information (age, experience, preferred zones,
driving style, interaction history) is converted into embedding vectors via LEPS strategy
to overcome the cold start problem for new drivers and personalize multi-objective routes.
"""

from typing import Dict, List, Any
import numpy as np

# Catalog of realistic driver profiles for the Beijing taxi market
DRIVER_PROFILES: Dict[str, Dict[str, Any]] = {
    "new_driver_cold_start": {
        "id": "driver_newbie",
        "name": "Alex Zhang (New Driver - Cold Start)",
        "experience_years": 0.1,
        "is_cold_start": True,
        "historical_rides_count": 3,
        "description": "Newly onboarded taxi driver with minimal trip history. Unfamiliar with peak passenger clusters. Needs safe, reliable, high-probability pickup corridors along arterial roads rather than high-risk speculative cruising.",
        "preferred_zones": ["Commercial", "Residential"],
        "avoid_zones": ["Industrial", "Rural"],
        "driving_preference": "steady_income",
        "max_cruising_tolerance_min": 8.0,
        "target_hourly_income": 45.0,
        "historical_interactions": [
            {"segment_id": 1204, "zone": "Commercial", "rating": 4.5, "earning": 28.5},
            {"segment_id": 1560, "zone": "Residential", "rating": 4.0, "earning": 22.0},
            {"segment_id": 1820, "zone": "Commercial", "rating": 4.2, "earning": 31.0},
        ]
    },
    "airport_hunter": {
        "id": "driver_airport",
        "name": "Master Chen (Airport & Long-Haul Hunter)",
        "experience_years": 8.5,
        "is_cold_start": False,
        "historical_rides_count": 1420,
        "description": "Veteran taxi driver specializing in high-fare airport corridors (Capital Airport Express, 3rd/4th Ring Road, Beijing South Railway Station). Prefers fewer trips with high per-trip revenue (¥60+), willing to cruise slightly longer for premium fares.",
        "preferred_zones": ["Transport Hub", "Business District", "Hotel Cluster"],
        "avoid_zones": ["Narrow Alleys", "Hutong"],
        "driving_preference": "maximum_profit",
        "max_cruising_tolerance_min": 18.0,
        "target_hourly_income": 85.0,
        "historical_interactions": [
            {"segment_id": 3102, "zone": "Transport Hub", "rating": 5.0, "earning": 78.0},
            {"segment_id": 4015, "zone": "Business District", "rating": 4.8, "earning": 64.5},
            {"segment_id": 2901, "zone": "Transport Hub", "rating": 4.9, "earning": 85.0},
            {"segment_id": 5120, "zone": "Hotel Cluster", "rating": 4.7, "earning": 55.0},
            {"segment_id": 3310, "zone": "Business District", "rating": 4.6, "earning": 62.0},
        ]
    },
    "downtown_cruiser": {
        "id": "driver_downtown",
        "name": "Lei Wang (CBD & Downtown Cruiser)",
        "experience_years": 4.2,
        "is_cold_start": False,
        "historical_rides_count": 890,
        "description": "Active urban cruiser focusing on Chaoyang CBD, Sanlitun, Wangfujing, and Zhongguancun. Aims for rapid passenger turnover, minimal waiting, and quick consecutive trips with dense pickup probability.",
        "preferred_zones": ["Commercial", "Entertainment", "Tech Hub"],
        "avoid_zones": ["Outskirts", "Highway Expressways"],
        "driving_preference": "high_turnover",
        "max_cruising_tolerance_min": 6.0,
        "target_hourly_income": 65.0,
        "historical_interactions": [
            {"segment_id": 2045, "zone": "Commercial", "rating": 4.8, "earning": 38.0},
            {"segment_id": 2110, "zone": "Entertainment", "rating": 4.7, "earning": 42.0},
            {"segment_id": 2340, "zone": "Tech Hub", "rating": 4.6, "earning": 35.5},
            {"segment_id": 2580, "zone": "Commercial", "rating": 4.9, "earning": 40.0},
        ]
    },
    "night_owl": {
        "id": "driver_night",
        "name": "Qiang Sun (Night Shift Veteran)",
        "experience_years": 6.0,
        "is_cold_start": False,
        "historical_rides_count": 1150,
        "description": "Night owl driver working from 20:00 to 04:00. Prioritizes late-night dining hotspots, bar districts, karaoke venues, and midnight train arrivals. Seeks premium night surcharge fares while avoiding dead areas.",
        "preferred_zones": ["Entertainment", "Nightlife", "Transport Hub"],
        "avoid_zones": ["Suburban Residential", "Daytime Industrial"],
        "driving_preference": "night_surge",
        "max_cruising_tolerance_min": 12.0,
        "target_hourly_income": 75.0,
        "historical_interactions": [
            {"segment_id": 1980, "zone": "Entertainment", "rating": 4.9, "earning": 52.0},
            {"segment_id": 2215, "zone": "Nightlife", "rating": 4.8, "earning": 58.0},
            {"segment_id": 3105, "zone": "Transport Hub", "rating": 4.7, "earning": 66.0},
        ]
    },
    "eco_explorer": {
        "id": "driver_eco",
        "name": "Hao Zhao (Eco & Underserved Explorer)",
        "experience_years": 3.0,
        "is_cold_start": False,
        "historical_rides_count": 520,
        "description": "Electric vehicle taxi driver mindful of battery range and energy efficiency. Strongly avoids heavy traffic bottlenecks. Actively searches for underserved passenger pockets (novelty) with low taxi competition.",
        "preferred_zones": ["Transit Oriented", "Residential Pocket", "Suburban Hub"],
        "avoid_zones": ["Severe Traffic Gridlocks", "Construction Zones"],
        "driving_preference": "fuel_efficient_exploration",
        "max_cruising_tolerance_min": 10.0,
        "target_hourly_income": 55.0,
        "historical_interactions": [
            {"segment_id": 1430, "zone": "Transit Oriented", "rating": 4.6, "earning": 34.0},
            {"segment_id": 1675, "zone": "Residential Pocket", "rating": 4.5, "earning": 32.0},
            {"segment_id": 1890, "zone": "Suburban Hub", "rating": 4.8, "earning": 48.0},
        ]
    }
}

# Pre-defined Beijing Urban Zone Categories for Topic Diversity Objective
ZONE_TOPICS = [
    "Commercial & Retail",
    "Transport Hub (Airport / Station)",
    "Business District / CBD",
    "Residential Community",
    "Entertainment & Nightlife",
    "Tech Park & University",
    "Government & Cultural",
    "Hospital & Medical Center"
]

def get_zone_for_segment(segment_id: int, lat: float = 39.9, lon: float = 116.4) -> str:
    """Assigns deterministic urban functional zone topics to road segments based on coordinates and ID."""
    # Deterministic spatial hash mapping to zone topics
    h = (int(segment_id) * 31 + int(lat * 1000) * 17 + int(lon * 1000) * 13) % len(ZONE_TOPICS)
    return ZONE_TOPICS[h]

def get_driver_profile(profile_id: str) -> Dict[str, Any]:
    """Retrieves a driver profile by ID or returns the default cold-start profile."""
    return DRIVER_PROFILES.get(profile_id, DRIVER_PROFILES["new_driver_cold_start"])

def list_all_driver_profiles() -> List[Dict[str, Any]]:
    """Lists all available driver personas for selection in UI and APIs."""
    result = []
    for pid, p in DRIVER_PROFILES.items():
        result.append({
            "id": pid,
            "name": p["name"],
            "experience_years": p["experience_years"],
            "is_cold_start": p["is_cold_start"],
            "preferred_zones": p["preferred_zones"],
            "description": p["description"],
            "target_hourly_income": p["target_hourly_income"]
        })
    return result

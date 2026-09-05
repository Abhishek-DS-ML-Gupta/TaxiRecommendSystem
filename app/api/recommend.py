"""
recommend.py — POST /api/recommend
Accepts a taxi start position (lat/lon), time limit, and number of taxis.
Returns the recommended route(s) with expected earning.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

router = APIRouter()


# ─── Request / Response Models ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    lat: float = Field(..., description="Starting latitude", example=39.9042)
    lon: float = Field(..., description="Starting longitude", example=116.4074)
    time_limit: float = Field(10.0, ge=1.0, le=60.0, description="Max searching time in minutes")
    num_taxis: int = Field(1, ge=1, le=20, description="Number of taxis to recommend routes for")
    scenario: str = Field("normal", description="Scenario: 'normal', 'peak', 'rainy'")


class RoutePoint(BaseModel):
    lat: float
    lon: float
    segment_id: int
    pickup_prob: float
    expected_earning: float


class TaxiRecommendation(BaseModel):
    taxi_id: str
    route: List[RoutePoint]
    total_expected_earning: float
    route_length_segments: int
    estimated_time_minutes: float


class RecommendResponse(BaseModel):
    recommendations: List[TaxiRecommendation]
    start_segment_id: int
    start_lat: float
    start_lon: float
    algorithm: str


# ─── Helper: Snap lat/lon to nearest road segment ─────────────────────────────

def _snap_to_nearest_segment(lat: float, lon: float, app_state: dict) -> int:
    """
    Finds the nearest road segment (edge) to the given lat/lon.
    Uses the edge_coords midpoint lookup.
    """
    edge_coords = app_state["edge_coords"]
    valid_nodes = set(app_state["valid_start_nodes"])

    best_eid = None
    best_dist = float('inf')

    for eid, (elat, elon) in edge_coords.items():
        if eid not in valid_nodes:
            continue
        dist = (lat - elat) ** 2 + (lon - elon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_eid = eid

    return best_eid


def _build_route_points(route: list, app_state: dict) -> List[RoutePoint]:
    """Converts a list of segment IDs into RoutePoint objects with coordinates."""
    edge_coords = app_state["edge_coords"]
    pickup_probs = app_state["pickup_probs"]
    earnings = app_state["earnings"]

    points = []
    for seg_id in route:
        coords = edge_coords.get(seg_id, (39.9042, 116.4074))
        points.append(RoutePoint(
            lat=coords[0],
            lon=coords[1],
            segment_id=seg_id,
            pickup_prob=round(pickup_probs.get(seg_id, 0.0), 4),
            expected_earning=round(earnings.get(seg_id, 0.0), 2)
        ))
    return points


def _apply_scenario_modifier(pickup_probs: dict, scenario: str) -> dict:
    """Modifies pickup probabilities based on scenario (peak hour, rainy, etc.)."""
    if scenario == "peak":
        return {k: min(v * 1.4, 1.0) for k, v in pickup_probs.items()}
    elif scenario == "rainy":
        return {k: min(v * 1.6, 1.0) for k, v in pickup_probs.items()}
    elif scenario == "night":
        return {k: v * 0.6 for k, v in pickup_probs.items()}
    return pickup_probs


# ─── Endpoint ────────────────────────────────────────────────────────────────

@router.post("/recommend", response_model=RecommendResponse)
async def recommend_route(req: RecommendRequest):
    """
    Main recommendation endpoint.
    Snaps the taxi's lat/lon to the nearest road segment, then runs
    RRS (single taxi) or SRRS (multiple taxis) to find optimal routes.
    """
    from app.core.startup import app_state

    if not app_state["loaded"]:
        raise HTTPException(status_code=503, detail="System is still loading road network data. Please retry in a moment.")

    # Apply scenario modifier to pickup probs
    modified_probs = _apply_scenario_modifier(
        app_state["pickup_probs"].copy(), req.scenario
    )

    # Update optimizer with new time limit
    optimizer = app_state["optimizer"]
    original_tau = optimizer.tau
    optimizer.tau = req.time_limit

    # Snap to nearest road segment
    start_segment = _snap_to_nearest_segment(req.lat, req.lon, app_state)
    if start_segment is None:
        raise HTTPException(status_code=400, detail="Could not find a nearby road segment. Please click within Beijing's road network.")

    start_coords = app_state["edge_coords"].get(start_segment, (req.lat, req.lon))
    earnings = app_state["earnings"]
    travel_times = app_state["travel_times"]

    recommendations_out = []

    if req.num_taxis == 1:
        # Single taxi: use RRS
        route, expected_profit = optimizer.RRS(
            start_segment, 0.0, modified_probs, earnings, travel_times
        )
        route_points = _build_route_points(route, app_state)

        # Estimate total time
        total_time = sum(
            travel_times.get((route[i], route[i+1]), 1.0)
            for i in range(len(route)-1)
        ) if len(route) > 1 else 0.0

        recommendations_out.append(TaxiRecommendation(
            taxi_id="Taxi_1",
            route=route_points,
            total_expected_earning=round(expected_profit, 2),
            route_length_segments=len(route),
            estimated_time_minutes=round(total_time, 1)
        ))
        algo = "RRS (Recursive Route Searching)"
    else:
        # Multiple taxis: use SRRS to prevent recommendation overload
        taxis_queue = [(f"Taxi_{i+1}", start_segment) for i in range(req.num_taxis)]
        srrs_recs = optimizer.SRRS(taxis_queue, modified_probs, earnings, travel_times)

        for taxi_id, route in srrs_recs.items():
            route_points = _build_route_points(route, app_state)
            total_time = sum(
                travel_times.get((route[j], route[j+1]), 1.0)
                for j in range(len(route)-1)
            ) if len(route) > 1 else 0.0

            # Recalculate expected profit for display
            _, ep = optimizer.RRS(route[0] if route else start_segment, 0.0, modified_probs, earnings, travel_times)

            recommendations_out.append(TaxiRecommendation(
                taxi_id=taxi_id,
                route=route_points,
                total_expected_earning=round(ep, 2),
                route_length_segments=len(route),
                estimated_time_minutes=round(total_time, 1)
            ))
        algo = "SRRS (Sequential Route Recommendation System)"

    # Restore original tau
    optimizer.tau = original_tau

    return RecommendResponse(
        recommendations=recommendations_out,
        start_segment_id=start_segment,
        start_lat=start_coords[0],
        start_lon=start_coords[1],
        algorithm=algo
    )

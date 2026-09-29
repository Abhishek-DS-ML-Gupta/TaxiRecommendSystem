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
    scenario: str = Field("normal", description="Scenario: 'normal', 'peak', 'rainy', 'night'")
    algorithm: str = Field("mora_llm", description="Algorithm: 'mora_llm', 'srrs', 'rrs', 'mora_wollm', 'mora_wom', 'mora_woq'")
    driver_profile_id: str = Field("new_driver_cold_start", description="Driver Persona ID for MORA-LLM")
    alpha: float = Field(0.2, ge=0.0, le=1.0, description="Dynamic score correction parameter alpha (Eq. 8)")


class RoutePoint(BaseModel):
    lat: float
    lon: float
    segment_id: int
    pickup_prob: float
    expected_earning: float
    zone: Optional[str] = None
    step_reason: Optional[str] = None


class TaxiRecommendation(BaseModel):
    taxi_id: str
    route: List[RoutePoint]
    total_expected_earning: float
    route_length_segments: int
    estimated_time_minutes: float
    novelty_score: Optional[float] = None
    topic_diversity_score: Optional[float] = None
    sde_fitness: Optional[float] = None


class RecommendResponse(BaseModel):
    recommendations: List[TaxiRecommendation]
    start_segment_id: int
    start_lat: float
    start_lon: float
    algorithm: str
    pareto_alternatives: Optional[dict] = None
    llm_explanation: Optional[dict] = None
    driver_profile: Optional[dict] = None
    multi_objective_summary: Optional[dict] = None


class AccuracyRequest(BaseModel):
    recommendations: List[TaxiRecommendation]


class AccuracyResponse(BaseModel):
    available: bool
    message: str
    segments_evaluated: int = 0
    mean_absolute_error: Optional[float] = None
    brier_score: Optional[float] = None
    predicted_pickup_rate: Optional[float] = None
    historical_pickup_rate: Optional[float] = None
    estimated_route_success_rate: Optional[float] = None


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


from mora_llm import get_driver_profile, list_all_driver_profiles, get_zone_for_segment

@router.get("/mora/profiles", tags=["MORA-LLM"])
async def get_driver_profiles():
    """Returns available driver personas and cold-start profiles for MORA-LLM."""
    return list_all_driver_profiles()


def _build_route_points(route: list, app_state: dict, segment_reasons: Optional[dict] = None) -> List[RoutePoint]:
    """Converts a list of segment IDs into RoutePoint objects with coordinates, zones, and reasons."""
    edge_coords = app_state["edge_coords"]
    pickup_probs = app_state["pickup_probs"]
    earnings = app_state["earnings"]
    segment_reasons = segment_reasons or {}

    points = []
    for seg_id in route:
        coords = edge_coords.get(seg_id, (39.9042, 116.4074))
        zone = get_zone_for_segment(seg_id, coords[0], coords[1])
        reason = segment_reasons.get(seg_id)
        points.append(RoutePoint(
            lat=coords[0],
            lon=coords[1],
            segment_id=seg_id,
            pickup_prob=round(pickup_probs.get(seg_id, 0.0), 4),
            expected_earning=round(earnings.get(seg_id, 0.0), 2),
            zone=zone,
            step_reason=reason
        ))
    return points


def _apply_scenario_modifier(pickup_probs: dict, scenario: str) -> dict:
    """Modifies pickup probabilities based on scenario (peak hour, rainy, night, etc.)."""
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
    Supports:
      1. MORA-LLM (Multi-Objective Evolutionary Search + LLM Reasoning + LEPS Cold Start)
      2. SRRS (Sequential Route Recommendation System for multiple taxis)
      3. RRS (Recursive Route Searching for single taxi)
    """
    from app.core.startup import app_state

    if not app_state["loaded"]:
        raise HTTPException(status_code=503, detail="System is still loading road network data. Please retry in a moment.")

    # Snap to nearest road segment
    start_segment = _snap_to_nearest_segment(req.lat, req.lon, app_state)
    if start_segment is None:
        raise HTTPException(status_code=400, detail="Could not find a nearby road segment. Please click within Beijing's road network.")

    start_coords = app_state["edge_coords"].get(start_segment, (req.lat, req.lon))
    earnings = app_state["earnings"]
    travel_times = app_state["travel_times"]

    # Apply scenario modifier to pickup probs
    modified_probs = _apply_scenario_modifier(
        app_state["pickup_probs"].copy(), req.scenario
    )

    algo_choice = req.algorithm.lower().strip()

    # ─── Branch A: MORA-LLM (Multi-Objective + LLM Enhancement) ─────────────
    if algo_choice.startswith("mora"):
        mora_opt = app_state.get("mora_optimizer")
        if mora_opt is None:
            raise HTTPException(status_code=500, detail="MORA-LLM optimizer not initialized.")

        profile = get_driver_profile(req.driver_profile_id)
        mora_opt.tau = req.time_limit
        mora_opt.alpha = req.alpha
        
        # Set variant for ablations
        if algo_choice == "mora_wollm":
            mora_opt.variant = "MORA-LLM(-WOLLM)"
        elif algo_choice == "mora_wom":
            mora_opt.variant = "MORA-LLM(-WOM)"
        elif algo_choice == "mora_woq":
            mora_opt.variant = "MORA-LLM(-WOQ)"
        else:
            mora_opt.variant = "MORA-LLM"

        # Multi-taxi support: sequentially allocate routes avoiding competition
        working_taxi_counts = app_state.get("taxi_counts", {}).copy()
        all_profiles = list_all_driver_profiles()
        recommendations_out = []
        
        selected_profiles = [profile]
        for p in all_profiles:
            if len(selected_profiles) >= req.num_taxis:
                break
            if p["id"] != profile["id"]:
                selected_profiles.append(p)
        while len(selected_profiles) < req.num_taxis:
            selected_profiles.append(profile)

        first_result = None

        for t_idx in range(req.num_taxis):
            cur_profile = selected_profiles[t_idx]
            mora_res = mora_opt.optimize(
                start_segment=start_segment,
                driver_profile=cur_profile,
                base_pickup_probs=modified_probs,
                base_earnings=earnings,
                travel_times=travel_times,
                edge_coords=app_state["edge_coords"],
                taxi_counts=working_taxi_counts,
                scenario=req.scenario
            )
            if first_result is None:
                first_result = mora_res

            # update working_taxi_counts to simulate sequential competition avoidance
            for seg in mora_res["best_route"]:
                working_taxi_counts[seg] = working_taxi_counts.get(seg, 0) + 1

            reasons_by_seg = {}
            if mora_res.get("llm_explanation") and "segment_reasons" in mora_res["llm_explanation"]:
                for s in mora_res["llm_explanation"]["segment_reasons"]:
                    reasons_by_seg[s["segment_id"]] = s["reason"]

            route_pts = _build_route_points(mora_res["best_route"], app_state, reasons_by_seg)
            label = f"Taxi_{t_idx+1} ({cur_profile.get('name', 'Driver')})" if req.num_taxis > 1 else f"MORA_{cur_profile.get('name', 'Driver')}"

            recommendations_out.append(
                TaxiRecommendation(
                    taxi_id=label,
                    route=route_pts,
                    total_expected_earning=mora_res["expected_earning"],
                    route_length_segments=len(mora_res["best_route"]),
                    estimated_time_minutes=mora_res["travel_time_minutes"],
                    novelty_score=mora_res["novelty_score"],
                    topic_diversity_score=mora_res["topic_diversity_score"],
                    sde_fitness=mora_res["sde_fitness"]
                )
            )

        # Build route points for Pareto alternatives from first_result
        pareto_alts = {}
        for alt_key, alt_data in first_result.get("pareto_alternatives", {}).items():
            alt_points = _build_route_points(alt_data["route"], app_state)
            pareto_alts[alt_key] = {
                "label": alt_data["label"],
                "profit": alt_data["profit"],
                "novelty": alt_data["novelty"],
                "diversity": alt_data["diversity"],
                "route_points": [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in alt_points]
            }

        return RecommendResponse(
            recommendations=recommendations_out,
            start_segment_id=start_segment,
            start_lat=start_coords[0],
            start_lon=start_coords[1],
            algorithm=f"{first_result['algorithm']} (Multi-Objective Evolutionary + LLM)",
            pareto_alternatives=pareto_alts,
            llm_explanation=first_result.get("llm_explanation"),
            driver_profile=profile,
            multi_objective_summary={
                "expected_profit": first_result["expected_earning"],
                "novelty": first_result["novelty_score"],
                "diversity": first_result["topic_diversity_score"],
                "sde_fitness": first_result["sde_fitness"],
                "is_cold_start": profile.get("is_cold_start", False),
                "alpha_correction": req.alpha
            }
        )

    # ─── Branch B: Classical RRS / SRRS ───────────────────────────────────────
    optimizer = app_state["optimizer"]
    original_tau = optimizer.tau
    optimizer.tau = req.time_limit

    recommendations_out = []

    if req.num_taxis == 1:
        # Single taxi: use RRS
        route, expected_profit = optimizer.RRS(
            start_segment, 0.0, modified_probs, earnings, travel_times
        )
        route_points = _build_route_points(route, app_state)

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
        # Multiple taxis: use SRRS
        taxis_queue = [(f"Taxi_{i+1}", start_segment) for i in range(req.num_taxis)]
        srrs_recs = optimizer.SRRS(taxis_queue, modified_probs, earnings, travel_times)

        for taxi_id, route in srrs_recs.items():
            route_points = _build_route_points(route, app_state)
            total_time = sum(
                travel_times.get((route[j], route[j+1]), 1.0)
                for j in range(len(route)-1)
            ) if len(route) > 1 else 0.0

            cum_p = 1.0
            ep = 0.0
            for seg in route:
                pr = modified_probs.get(seg, 0.2)
                er = earnings.get(seg, 25.0)
                ep += cum_p * pr * er
                cum_p *= max(1.0 - pr, 0.0)

            recommendations_out.append(TaxiRecommendation(
                taxi_id=taxi_id,
                route=route_points,
                total_expected_earning=round(ep, 2),
                route_length_segments=len(route),
                estimated_time_minutes=round(total_time, 1)
            ))
        algo = "SRRS (Sequential Route Recommendation System)"

    optimizer.tau = original_tau

    return RecommendResponse(
        recommendations=recommendations_out,
        start_segment_id=start_segment,
        start_lat=start_coords[0],
        start_lon=start_coords[1],
        algorithm=algo
    )


@router.post("/recommend/accuracy", response_model=AccuracyResponse)
async def check_recommendation_accuracy(req: AccuracyRequest):
    """Compare route probabilities with historical segment pickup rates."""
    from app.core.startup import app_state

    network_props = app_state.get("network_props")
    if network_props is None or network_props.empty:
        return AccuracyResponse(
            available=False,
            message="Historical pickup data is unavailable for this recommendation.",
        )

    edges_df = app_state["edges_df"]
    observed_by_edge = {}
    for _, row in network_props.iterrows():
        matches = edges_df[
            (edges_df["source_id"] == int(row["source_id"])) &
            (edges_df["target_id"] == int(row["target_id"]))
        ]
        if len(matches) == 0:
            continue
        taxi_count = float(row["taxi_count"])
        observed_by_edge[int(matches.iloc[0]["edge_id"])] = (
            float(row["pickup_count"]) / taxi_count if taxi_count > 0 else 0.0
        )

    pairs = []
    for recommendation in req.recommendations:
        for point in recommendation.route:
            observed = observed_by_edge.get(point.segment_id)
            if observed is not None:
                pairs.append((min(max(point.pickup_prob, 0.0), 1.0), min(max(observed, 0.0), 1.0)))

    if not pairs:
        return AccuracyResponse(
            available=False,
            message="No recommended route segments have matching historical pickup records.",
        )

    predicted, observed = zip(*pairs)
    predicted_arr = np.array(predicted)
    observed_arr = np.array(observed)
    mae = float(np.mean(np.abs(predicted_arr - observed_arr)))
    brier = float(np.mean((predicted_arr - observed_arr) ** 2))
    success = float(np.mean([1.0 - np.prod(1.0 - predicted_arr)]))
    return AccuracyResponse(
        available=True,
        message="Compared with historical pickup rates from network_properties.csv.",
        segments_evaluated=len(pairs),
        mean_absolute_error=round(mae, 4),
        brier_score=round(brier, 4),
        predicted_pickup_rate=round(float(np.mean(predicted_arr)), 4),
        historical_pickup_rate=round(float(np.mean(observed_arr)), 4),
        estimated_route_success_rate=round(success, 4),
    )

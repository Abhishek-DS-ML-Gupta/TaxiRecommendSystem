"""
network.py — GET /api/network and GET /api/hotspots
Serves road network geometry for map display and hotspot heatmap data.
"""

from fastapi import APIRouter, Query
from typing import List, Dict, Any
import random

router = APIRouter()


@router.get("/network")
async def get_network(
    max_edges: int = Query(3000, description="Max road segments to return (for performance)")
) -> Dict[str, Any]:
    """
    Returns a subsampled road network as a list of edges with coordinates.
    The frontend uses this to draw the road network on the map.
    """
    from app.core.startup import app_state

    if not app_state["loaded"]:
        return {"edges": [], "nodes": [], "loaded": False}

    edges_df = app_state["edges_df"]
    node_coords = app_state["node_coords"]

    # Subsample for performance (38K edges → too heavy for browser)
    sample_df = edges_df.sample(min(max_edges, len(edges_df)), random_state=42)

    edges_out = []
    for _, row in sample_df.iterrows():
        src = int(row['source_id'])
        tgt = int(row['target_id'])
        src_c = node_coords.get(src)
        tgt_c = node_coords.get(tgt)
        if src_c and tgt_c:
            edges_out.append({
                "edge_id": int(row['edge_id']),
                "src_lat": src_c[0],
                "src_lon": src_c[1],
                "tgt_lat": tgt_c[0],
                "tgt_lon": tgt_c[1],
            })

    return {
        "edges": edges_out,
        "total_edges": len(edges_df),
        "displayed_edges": len(edges_out),
        "loaded": True
    }


@router.get("/hotspots")
async def get_hotspots(
    top_n: int = Query(150, description="Number of top hotspots to return")
) -> Dict[str, Any]:
    """
    Returns the top-N road segments by pickup probability as heatmap data.
    Each point: [lat, lon, intensity (0–1)]
    """
    from app.core.startup import app_state

    if not app_state["loaded"]:
        return {"points": [], "loaded": False}

    top_hotspots = app_state["top_hotspots"][:top_n]

    points = []
    for eid, prob, (lat, lon) in top_hotspots:
        if lat != 0 and lon != 0:
            points.append({
                "lat": lat,
                "lon": lon,
                "intensity": round(prob, 4),
                "segment_id": eid,
                "earning": round(app_state["earnings"].get(eid, 0.0), 2)
            })

    return {
        "points": points,
        "loaded": True,
        "total_segments": len(app_state["pickup_probs"])
    }


@router.get("/network/summary")
async def get_network_summary() -> Dict[str, Any]:
    """Returns high-level statistics about the loaded road network."""
    from app.core.startup import app_state

    if not app_state["loaded"]:
        return {"loaded": False}

    pickup_probs = app_state["pickup_probs"]
    earnings = app_state["earnings"]

    import numpy as np
    prob_values = list(pickup_probs.values())
    earn_values = [e for e in earnings.values() if e > 0]

    return {
        "loaded": True,
        "total_intersections": app_state["G"].number_of_nodes(),
        "total_road_segments": app_state["G"].number_of_edges(),
        "routing_graph_nodes": app_state["route_G"].number_of_nodes(),
        "routing_graph_edges": app_state["route_G"].number_of_edges(),
        "pickup_prob_stats": {
            "mean": round(float(np.mean(prob_values)), 4),
            "max": round(float(np.max(prob_values)), 4),
            "min": round(float(np.min(prob_values)), 4),
        },
        "earning_stats": {
            "mean": round(float(np.mean(earn_values)), 2) if earn_values else 0,
            "max": round(float(np.max(earn_values)), 2) if earn_values else 0,
        },
        "city": "Beijing, China",
        "area": "3rd Ring Road area",
        "data_source": "INFORMS JoC 2021.0112"
    }

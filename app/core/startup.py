"""
startup.py — Pre-loads all heavy data (road network, model, routing graph)
at application startup so API requests have zero data-loading latency.

Data sources used (real data from INFORMS JoC paper repository):
  - data/refined_edges.csv       → road segments (38,723 segments)
  - data/refined_vertice.csv     → intersection coordinates (13,287 nodes)
  - data/network_properties.csv  → pre-computed pickup probs & earnings per segment
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse

# Make src importable
_SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from data_processor import DataProcessor
from route_optimizer import RouteOptimizer

# ─── Global app state (populated at startup) ────────────────────────────────

app_state = {
    "G": None,               # NetworkX DiGraph (intersections)
    "edges_df": None,        # road segments DataFrame
    "vertices_df": None,     # intersections DataFrame
    "route_G": None,         # routing graph (segment→segment)
    "optimizer": None,       # RouteOptimizer instance
    "pickup_probs": None,    # {edge_id → float}
    "earnings": None,        # {edge_id → float}
    "travel_times": None,    # {(u,v) → float}
    "node_coords": None,     # {node_id → (lat, lon)}
    "edge_coords": None,     # {edge_id → (lat, lon)} midpoint of each segment
    "network_props": None,   # raw network_properties DataFrame
    "valid_start_nodes": [],
    "top_hotspots": [],      # top-100 pickup probability hotspots
    "loaded": False,
}


def _load_network_properties(data_dir: str) -> pd.DataFrame:
    """
    Loads network_properties.csv which has columns:
    source_id, target_id, src_lat, src_lon, tgt_lat, tgt_lon,
    pickup_count, taxi_count, pickup_prob, expected_earning
    """
    props_path = os.path.join(data_dir, 'network_properties.csv')
    if not os.path.exists(props_path):
        return None
    # The file has no header; infer from README / paper description
    df = pd.read_csv(props_path, header=None, names=[
        'source_id', 'target_id',
        'src_lat', 'src_lon',
        'tgt_lat', 'tgt_lon',
        'pickup_count', 'taxi_count',
        'pickup_prob', 'expected_earning'
    ])
    return df


def load_all(data_dir: str = None, time_limit_tau: float = 10.0):
    """
    Main startup function. Call once when the FastAPI app starts.
    Populates the global `app_state` dictionary.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    data_dir = os.path.abspath(data_dir)

    print(f"[startup] Loading data from: {data_dir}")

    # 1. Load base road network
    processor = DataProcessor(data_dir=data_dir)
    G, edges_df, vertices_df = processor.load_network()
    print(f"[startup] Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. Build node coordinate lookup
    node_coords = {}
    for _, row in vertices_df.iterrows():
        node_coords[int(row['node_id'])] = (float(row['lat']), float(row['lon']))

    # 3. Load network properties (real pickup probs from paper data)
    network_props = _load_network_properties(data_dir)

    # 4. Build pickup_probs and earnings per road segment
    pickup_probs = {}
    earnings = {}
    edge_coords = {}  # midpoint of each edge

    if network_props is not None:
        print(f"[startup] Using real pickup probabilities from network_properties.csv ({len(network_props)} rows)")
        for _, row in network_props.iterrows():
            # Key by (source_id, target_id) → we'll use edge_id later
            src = int(row['source_id'])
            tgt = int(row['target_id'])
            prob = float(row['pickup_prob']) if not pd.isna(row['pickup_prob']) else 0.0
            earn = float(row['expected_earning']) if not pd.isna(row['expected_earning']) else 0.0
            # mid-lat/lon
            mid_lat = (float(row['src_lat']) + float(row['tgt_lat'])) / 2.0
            mid_lon = (float(row['src_lon']) + float(row['tgt_lon'])) / 2.0

            # Map this (src, tgt) to edge_id
            # edges_df has: edge_id, source_id, target_id
            mask = (edges_df['source_id'] == src) & (edges_df['target_id'] == tgt)
            matched = edges_df.loc[mask, 'edge_id']
            if len(matched) > 0:
                eid = int(matched.iloc[0])
                pickup_probs[eid] = min(max(prob, 0.0), 1.0)
                earnings[eid] = max(earn, 0.0)
                edge_coords[eid] = (mid_lat, mid_lon)
    else:
        print("[startup] network_properties.csv not found, using simulated probabilities")

    # Fill any missing edge_ids from edges_df with random values and coords from vertices
    edge_ids = edges_df['edge_id'].unique()
    for _, row in edges_df.iterrows():
        eid = int(row['edge_id'])
        src = int(row['source_id'])
        tgt = int(row['target_id'])
        if eid not in pickup_probs:
            pickup_probs[eid] = float(np.random.uniform(0.05, 0.4))
            earnings[eid] = float(np.random.uniform(10.0, 50.0))
        if eid not in edge_coords:
            src_c = node_coords.get(src, (39.9, 116.4))
            tgt_c = node_coords.get(tgt, (39.9, 116.4))
            edge_coords[eid] = (
                (src_c[0] + tgt_c[0]) / 2.0,
                (src_c[1] + tgt_c[1]) / 2.0
            )

    # 5. Build routing graph (segment-to-segment transitions)
    print("[startup] Building routing graph...")
    route_G = nx.DiGraph()
    travel_times = {}

    source_group = edges_df.groupby('source_id')
    for eid in edge_ids:
        route_G.add_node(int(eid))

    for _, row in edges_df.iterrows():
        target_node = int(row['target_id'])
        current_edge = int(row['edge_id'])
        src = int(row['source_id'])

        # Travel time: Euclidean distance proxy (degrees → minutes, rough)
        src_c = node_coords.get(src, (39.9, 116.4))
        tgt_c = node_coords.get(target_node, (39.9, 116.4))
        dist_deg = ((src_c[0] - tgt_c[0])**2 + (src_c[1] - tgt_c[1])**2)**0.5
        t_minutes = max(dist_deg * 100.0, 0.5)  # rough conversion

        if target_node in source_group.groups:
            next_edges = source_group.get_group(target_node)
            for _, next_row in next_edges.iterrows():
                next_edge = int(next_row['edge_id'])
                route_G.add_edge(current_edge, next_edge)
                travel_times[(current_edge, next_edge)] = t_minutes

    print(f"[startup] Routing graph: {route_G.number_of_nodes()} segment nodes, {route_G.number_of_edges()} transitions")

    # 6. Build optimizer
    optimizer = RouteOptimizer(route_G, time_limit_tau=time_limit_tau)

    # 7. Valid starting nodes (have outgoing edges)
    valid_start_nodes = [n for n, deg in route_G.out_degree() if deg > 0]

    # 8. Top hotspots for heatmap (top 200 by pickup probability)
    top_hotspots = sorted(
        [(eid, pickup_probs[eid], edge_coords.get(eid, (0, 0))) for eid in pickup_probs],
        key=lambda x: x[1],
        reverse=True
    )[:200]

    # Populate global state
    app_state.update({
        "G": G,
        "edges_df": edges_df,
        "vertices_df": vertices_df,
        "route_G": route_G,
        "optimizer": optimizer,
        "pickup_probs": pickup_probs,
        "earnings": earnings,
        "travel_times": travel_times,
        "node_coords": node_coords,
        "edge_coords": edge_coords,
        "network_props": network_props,
        "valid_start_nodes": valid_start_nodes,
        "top_hotspots": top_hotspots,
        "loaded": True,
    })

    print("[startup] All data loaded and ready.")
    return app_state

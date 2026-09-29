"""
main_mora_recommend.py — Demonstration of MORA-LLM for Taxi Route Recommendation.

Demonstrates the implementation of:
"MORA-LLM: Enhancing Multiobjective Optimization Recommendation Algorithm by Integrating Large Language Models"
(Ge, Wu, Yang, Cheng, Zhao, Zhang, IEEE Transactions on Evolutionary Computation, 2026).

Applies:
  1. LEPS Strategy: Cold-start alleviation via driver background embeddings and CF similarity.
  2. LES Strategy: Multi-objective evolutionary search balancing Profit, Novelty, and Diversity.
  3. LLM Reasoning: Dynamic score correction (alpha=0.2) and Explainable Recommendation reasons.
"""

import os
import sys
import io

# Force UTF-8 stdout on Windows so emojis in print() don't crash
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from mora_llm import (
    MORARouteOptimizer,
    DRIVER_PROFILES,
    list_all_driver_profiles,
    get_driver_profile
)

def run_mora_demo():
    print("=" * 75)
    print("  🚕 MORA-LLM: MULTI-OBJECTIVE TAXI ROUTE RECOMMENDER WITH LLM REASONING")
    print("  Based on IEEE Transactions on Evolutionary Computation (2026)")
    print("=" * 75)

    processor = DataProcessor(data_dir='data')
    print("\n[1/5] Loading Beijing road network data...")
    G, edges_df, vertices_df = processor.load_network()

    edge_ids = edges_df['edge_id'].unique()
    print(f"      Loaded {len(edge_ids)} road segments, {len(vertices_df)} intersections.")

    # Node coordinates
    node_coords = {int(r['node_id']): (float(r['lat']), float(r['lon'])) for _, r in vertices_df.iterrows()}

    # Construct routing graph
    print("\n[2/5] Constructing segment transition graph...")
    route_G = nx.DiGraph()
    travel_times = {}
    edge_coords = {}

    source_group = edges_df.groupby('source_id')
    for eid in edge_ids:
        route_G.add_node(int(eid))

    for _, row in edges_df.iterrows():
        tgt = int(row['target_id'])
        curr = int(row['edge_id'])
        src = int(row['source_id'])

        src_c = node_coords.get(src, (39.9, 116.4))
        tgt_c = node_coords.get(tgt, (39.9, 116.4))
        mid = ((src_c[0] + tgt_c[0]) / 2.0, (src_c[1] + tgt_c[1]) / 2.0)
        edge_coords[curr] = mid

        dist = ((src_c[0] - tgt_c[0]) ** 2 + (src_c[1] - tgt_c[1]) ** 2) ** 0.5
        t = max(dist * 100.0, 0.5)

        if tgt in source_group.groups:
            for _, next_row in source_group.get_group(tgt).iterrows():
                nxt = int(next_row['edge_id'])
                route_G.add_edge(curr, nxt)
                travel_times[(curr, nxt)] = t

    valid_starts = [n for n, deg in route_G.out_degree() if deg > 0]
    start_seg = int(np.random.choice(valid_starts))
    print(f"      Graph ready: {route_G.number_of_nodes()} segments, {route_G.number_of_edges()} transitions.")
    print(f"      Selected Start Segment ID: {start_seg}")

    # Simulated base probabilities & taxi counts
    pickup_probs = {int(eid): float(np.random.uniform(0.1, 0.8)) for eid in edge_ids}
    earnings = {int(eid): float(np.random.uniform(15.0, 60.0)) for eid in edge_ids}
    taxi_counts = {int(eid): int(np.random.poisson(3)) for eid in edge_ids}

    # Test with a Cold-Start Driver Profile
    print("\n[3/5] Setting up Driver Profile (Simulating Cold-Start Scenario)...")
    profile = get_driver_profile("new_driver_cold_start")
    print(f"      Driver: {profile['name']}")
    print(f"      Status: {'COLD START (0 prior interactions)' if profile['is_cold_start'] else 'Experienced'}")
    print(f"      Preferred Zones: {profile['preferred_zones']}")
    print(f"      Background: {profile['description']}")

    # Initialize MORA-LLM Optimizer
    print("\n[4/5] Initializing MORA-LLM Multi-Objective Evolutionary Engine...")
    print("      Parameters: PopSize=25, MaxGen=20, LLM_Cycle=5, Alpha=0.2 (Score correction factor)")
    optimizer = MORARouteOptimizer(
        route_G=route_G,
        time_limit_tau=12.0,
        pop_size=25,
        max_gen=20,
        k_length=8,
        llm_cycle=5,
        alpha=0.2,
        variant="MORA-LLM"
    )

    print("\n[5/5] Executing MORA-LLM Optimization (LEPS + LES + SDE + LLM Reasoning)...")
    result = optimizer.optimize(
        start_segment=start_seg,
        driver_profile=profile,
        base_pickup_probs=pickup_probs,
        base_earnings=earnings,
        travel_times=travel_times,
        edge_coords=edge_coords,
        taxi_counts=taxi_counts,
        scenario="normal"
    )

    print("\n" + "=" * 75)
    print("  🏆 MORA-LLM OPTIMIZATION RESULTS")
    print("=" * 75)
    print(f"  Recommended Route: {result['best_route']}")
    print(f"  Expected Earning (Accuracy f1):   ¥{result['expected_earning']:.2f}")
    print(f"  Novelty Score (f2 - Low crowded):  {result['novelty_score']:.4f}")
    print(f"  Topic Diversity (f3 - Entropy):    {result['topic_diversity_score']:.4f}")
    print(f"  SDE Pareto Fitness:                {result['sde_fitness']:.4f}")
    print(f"  Estimated Cruising Time:           {result['travel_time_minutes']} min")

    print("\n  📊 PARETO FRONT ALTERNATIVES:")
    for key, alt in result["pareto_alternatives"].items():
        print(f"    • [{alt['label']}]: Earning=¥{alt['profit']:.2f}, Novelty={alt['novelty']:.3f}, Diversity={alt['diversity']:.3f}")
        print(f"      Path: {alt['route']}")

    print("\n  🧠 LLM NATURAL LANGUAGE EXPLANATION (EXPLAINABILITY):")
    explanation = result["llm_explanation"]
    print(f"  {explanation['summary_explanation']}\n")
    print("  Step-by-Step AI Rationale:")
    for step in explanation["segment_reasons"][:5]:
        print(f"    Step {step['step']} [Seg {step['segment_id']} | {step['zone']} | p={step['pickup_prob']*100:.1f}% | ¥{step['expected_earning']:.1f}]:")
        print(f"      → {step['reason']}")

    print("\n" + "=" * 75)
    print("  ✅ MORA-LLM DEMO COMPLETE!")
    print("=" * 75)

if __name__ == '__main__':
    run_mora_demo()

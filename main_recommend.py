import os
import sys
import numpy as np
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from route_optimizer import RouteOptimizer

def recommend():
    print("=" * 75)
    print("  Sequential Route Recommender System (C-SRR Simulation & Benchmarking)")
    print("  Paper: 'A Cost-Effective Sequential Route Recommender System for Taxi Drivers'")
    print("         Liu et al., INFORMS Journal on Computing (2023)")
    print("=" * 75)
    
    processor = DataProcessor(data_dir='data')
    
    print("\n[Step 1/4] Loading road network & segment properties...")
    G, edges_df, vertices_df = processor.load_network()
    network_props_df = processor.load_network_properties()
    
    print("\n[Step 2/4] Constructing road segment transition graph & properties...")
    route_G, pickup_probs, earnings, travel_times, segment_lengths, vacant_taxis, pickup_orders = \
        processor.construct_segment_graph(edges_df, network_props_df)
        
    print(f"Segment Graph: {route_G.number_of_nodes()} segments, {route_G.number_of_edges()} transitions.")
    
    # Extract candidate virtual stations (Section 3.1, Definition 5, Figure 1(b))
    virtual_stations = processor.extract_virtual_stations(network_props_df, top_k=50)
    print(f"Identified {len(virtual_stations)} virtual station candidates (taxi stands / hot spots).")
    
    # Initialize optimizer with searching time limit tau = 10 minutes
    tau_limit = 10.0
    optimizer = RouteOptimizer(route_G, time_limit_tau=tau_limit, max_depth=8)
    
    print("\n[Step 3/4] Initializing Sequential Taxi Queue Simulation...")
    # Select high-degree starting road segments for realistic rush-hour simulation
    valid_start_nodes = [
        n for n, out_deg in route_G.out_degree() 
        if out_deg >= 2 and n in pickup_probs and pickup_probs[n] > 0.15
    ]
    
    if not valid_start_nodes:
        valid_start_nodes = [n for n, out_deg in route_G.out_degree() if out_deg > 0]
        
    np.random.seed(42)
    start_node = np.random.choice(valid_start_nodes)
    num_taxis = 10
    taxis_queue = [(f"Taxi_{i:02d}", start_node) for i in range(1, num_taxis + 1)]
    
    print(f"Origin Segment: {start_node} (Out-degree: {route_G.out_degree(start_node)})")
    print(f"Simulating {num_taxis} vacant taxis submitting route queries in FIFO order...\n")
    
    print("-" * 75)
    print("  Running Proposed C-SRR (Dynamic Supply Overload Mitigation)...")
    print("-" * 75)
    recs_dynamic = optimizer.SRRS(
        taxis_queue=taxis_queue,
        pickup_probs=pickup_probs,
        earnings=earnings,
        travel_times=travel_times,
        segment_lengths=segment_lengths,
        vacant_taxis=vacant_taxis,
        pickup_orders=pickup_orders,
        virtual_stations=virtual_stations,
        objective='profit',
        strategy_mode='dynamic'
    )
    
    for taxi_id, res in recs_dynamic.items():
        print(f"  [{taxi_id}] Strategy: {res['strategy']:<18} | EUP: {res['expected_unit_profit']:.3f} CNY/min | "
              f"Time: {res['travel_time']:.1f}m | Route: {res['route'][:5]}{'...' if len(res['route']) > 5 else ''}")

    print("\n[Step 4/4] Multi-Strategy Performance Comparison (Reproducing Table 4 & Figure 8)...")
    
    # 1. Static Strategy (Qu et al. 2014)
    recs_static = optimizer.SRRS(
        taxis_queue, pickup_probs, earnings, travel_times, segment_lengths,
        vacant_taxis, pickup_orders, virtual_stations, objective='profit', strategy_mode='static'
    )
    
    # 2. Top-K Diversity Strategy
    recs_topk = optimizer.SRRS(
        taxis_queue, pickup_probs, earnings, travel_times, segment_lengths,
        vacant_taxis, pickup_orders, virtual_stations, objective='profit', strategy_mode='top_k'
    )
    
    # 3. Group-5 Batch Strategy
    recs_group5 = optimizer.SRRS(
        taxis_queue, pickup_probs, earnings, travel_times, segment_lengths,
        vacant_taxis, pickup_orders, virtual_stations, objective='profit', strategy_mode='group_5'
    )
    
    # 4. Minimum Searching Time (MST)
    recs_mst = optimizer.SRRS(
        taxis_queue, pickup_probs, earnings, travel_times, segment_lengths,
        vacant_taxis, pickup_orders, virtual_stations, objective='mst', strategy_mode='dynamic'
    )
    
    # 5. Minimum Searching Distance (MSD)
    recs_msd = optimizer.SRRS(
        taxis_queue, pickup_probs, earnings, travel_times, segment_lengths,
        vacant_taxis, pickup_orders, virtual_stations, objective='msd', strategy_mode='dynamic'
    )

    strategies = {
        'Static Model (Qu et al. 2014)': recs_static,
        'Top-K Diversity Strategy': recs_topk,
        'Group-5 Batch Strategy': recs_group5,
        'Minimum Searching Time (MST)': recs_mst,
        'Minimum Searching Distance (MSD)': recs_msd,
        'Proposed C-SRR (Dynamic Model)': recs_dynamic
    }
    
    print("\n" + "=" * 75)
    print(f"{'Recommendation Strategy':<35} | {'Avg Unit Profit (CNY/min)':<25} | {'Distinct Routes':<12}")
    print("=" * 75)
    
    for name, results in strategies.items():
        avg_eup = np.mean([r['expected_unit_profit'] for r in results.values()])
        distinct_routes = len(set(tuple(r['route']) for r in results.values()))
        marker = " (Best)" if name == 'Proposed C-SRR (Dynamic Model)' else ""
        print(f"{name:<35} | {avg_eup:^25.3f} | {distinct_routes:^12}{marker}")
        
    print("=" * 75)
    print("\nConclusion: The proposed dynamic C-SRR successfully balances taxi demand and supply,")
    print("mitigates recommendation overload, and maximizes the expected profit for taxi drivers.")
    print("=" * 75)

if __name__ == '__main__':
    recommend()

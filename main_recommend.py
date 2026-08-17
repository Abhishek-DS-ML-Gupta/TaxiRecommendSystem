import os
import sys
import numpy as np
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from route_optimizer import RouteOptimizer

def recommend():
    print("Initializing Data Processor...")
    processor = DataProcessor(data_dir='data')
    
    print("Loading network data...")
    G, edges_df, vertices_df = processor.load_network()
    
    # In a real scenario, we would use the GCN-LSTM model to predict these probabilities.
    # Here we will simulate the predictions for the sake of demonstration.
    print("Simulating pickup probabilities and earnings...")
    edge_ids = edges_df['edge_id'].unique()
    pickup_probs = {edge_id: np.random.uniform(0.1, 0.9) for edge_id in edge_ids}
    earnings = {edge_id: np.random.uniform(10.0, 50.0) for edge_id in edge_ids}
    travel_times = {} # default will be 1.0 in optimizer if not provided
    
    # We need to construct a graph where nodes are road segments (edges in original graph)
    # for the route optimizer to search through.
    print("Constructing road segment graph for routing...")
    route_G = nx.DiGraph()
    
    # Add nodes (road segments)
    for edge_id in edge_ids:
        route_G.add_node(edge_id)
        
    # Add edges between adjacent road segments
    source_group = edges_df.groupby('source_id')
    for _, row in edges_df.iterrows():
        target_node = row['target_id']
        current_edge = row['edge_id']
        
        if target_node in source_group.groups:
            next_edges = source_group.get_group(target_node)
            for _, next_row in next_edges.iterrows():
                next_edge = next_row['edge_id']
                route_G.add_edge(current_edge, next_edge)
                travel_times[(current_edge, next_edge)] = 1.0 # 1 minute travel time assumption
    
    print(f"Routing graph constructed with {route_G.number_of_nodes()} segments and {route_G.number_of_edges()} transitions.")
    
    # Initialize optimizer
    optimizer = RouteOptimizer(route_G, time_limit_tau=10.0) # 10 minutes limit
    
    # Simulate a queue of taxis querying at the same time
    # We pick some random starting segments that actually have outgoing edges
    valid_start_nodes = [n for n, out_degree in route_G.out_degree() if out_degree > 0]
    
    if not valid_start_nodes:
        print("No valid starting nodes found to search from.")
        return
        
    start_node = np.random.choice(valid_start_nodes)
    print(f"\nSimulating recommendation overload scenario at segment {start_node}...")
    
    taxis_queue = [
        (f'Taxi_{i}', start_node) for i in range(1, 6) # 5 taxis
    ]
    
    print("Generating sequential recommendations...")
    recommendations = optimizer.SRRS(taxis_queue, pickup_probs, earnings, travel_times)
    
    for taxi_id, route in recommendations.items():
        print(f"{taxi_id} recommended route (segments): {route}")

if __name__ == '__main__':
    recommend()

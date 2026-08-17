import networkx as nx
import numpy as np

class RouteOptimizer:
    def __init__(self, G, time_limit_tau=10.0):
        """
        Initializes the RouteOptimizer.
        G: A networkx DiGraph where nodes are road segments (or intersections).
           For the paper's formulation, it's easier to think of navigating a graph 
           where we transition from road segment to road segment.
           In our implementation, we can use the original graph where nodes are intersections 
           and edges are road segments.
        time_limit_tau: The maximum searching time limit (e.g., 10 minutes).
        """
        self.G = G
        self.tau = time_limit_tau

    def RRS(self, current_node, current_time, pickup_probs, earnings, travel_times, depth=0):
        """
        Recursive Route Searching algorithm (Algorithm 1 in paper)
        Returns the optimal route R* and expected profit G*
        
        current_node: the current road segment (or intersection)
        current_time: cumulative time spent on the route so far
        pickup_probs: dictionary mapping node -> pickup probability (p_r)
        earnings: dictionary mapping node -> expected earning (e_r)
        travel_times: dictionary mapping (u, v) -> travel time (t_r)
        """
        # Base case: if we exceed time limit
        if current_time >= self.tau or depth > 8: # Adding a depth limit to prevent infinite recursion
            return [current_node], 0.0

        p_r = pickup_probs.get(current_node, 0.0)
        e_r = earnings.get(current_node, 0.0)
        
        best_route = []
        best_future_profit = 0.0
        
        # Get neighbors
        neighbors = list(self.G.successors(current_node))
        
        if not neighbors:
            return [current_node], p_r * e_r
            
        for neighbor in neighbors:
            t_r = travel_times.get((current_node, neighbor), 1.0) # default travel time 1.0
            
            # Recursive call
            child_route, child_profit = self.RRS(
                neighbor, 
                current_time + t_r, 
                pickup_probs, 
                earnings, 
                travel_times, 
                depth + 1
            )
            
            if child_profit > best_future_profit:
                best_future_profit = child_profit
                best_route = child_route
                
        # Bellman equation: G*(r) = p_r * e_r + (1 - p_r) * max G*(r')
        expected_profit = p_r * e_r + (1.0 - p_r) * best_future_profit
        optimal_route = [current_node] + best_route
        
        return optimal_route, expected_profit

    def SRRS(self, taxis_queue, pickup_probs, earnings, travel_times):
        """
        Sequential Route Recommendation System (Algorithm 2 in paper)
        taxis_queue: list of tuples (taxi_id, start_node)
        
        Returns dictionary mapping taxi_id to optimal route
        """
        recommendations = {}
        
        # Working copy of pickup probabilities since they get updated dynamically
        current_pickup_probs = pickup_probs.copy()
        
        for taxi_id, start_node in taxis_queue:
            # 1. Find optimal searching route
            optimal_route, expected_profit = self.RRS(
                start_node, 0.0, current_pickup_probs, earnings, travel_times
            )
            
            # 2. Find optimal virtual station (simplified for this implementation)
            # In a full implementation, we would compare with virtual station waiting profit
            # Here we just use the searching route
            
            recommendations[taxi_id] = optimal_route
            
            # 3. Update taxi supply / pickup probabilities along route to avoid recommendation overload
            for node in optimal_route:
                # Decrease pickup probability as a simple heuristic for increased supply
                current_pickup_probs[node] = current_pickup_probs.get(node, 0.0) * 0.9
                
        return recommendations

if __name__ == '__main__':
    # Simple test
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    
    probs = {1: 0.1, 2: 0.5, 3: 0.2, 4: 0.8}
    earnings = {1: 10, 2: 20, 3: 15, 4: 30}
    travel_times = {(1, 2): 2, (1, 3): 5, (2, 4): 3, (3, 4): 2}
    
    optimizer = RouteOptimizer(G, time_limit_tau=10.0)
    route, profit = optimizer.RRS(1, 0.0, probs, earnings, travel_times)
    
    print(f"Optimal Route from node 1: {route}")
    print(f"Expected Profit: {profit}")
    
    # Test sequential
    taxis = [('taxi1', 1), ('taxi2', 1)]
    recs = optimizer.SRRS(taxis, probs, earnings, travel_times)
    print(f"Sequential Recommendations: {recs}")

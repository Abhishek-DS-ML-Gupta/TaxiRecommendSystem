import networkx as nx
import numpy as np

class RouteOptimizer:
    """
    Cost-Effective Sequential Route Recommender System (C-SRR)
    Implements:
    1. Recursive Route Searching Algorithm (RRS - Algorithm 1 in Liu et al. 2023)
    2. Virtual Station Waiting Strategy (Definition 5, Eq 2)
    3. Sequential Route Recommender System (SRRS - Algorithm 2) with Dynamic Supply Overload Mitigation
    4. Alternative objectives: Minimum Searching Time (MST), Minimum Searching Distance (MSD)
    5. Benchmark strategies: Static, Top-K diversity, Group-5 batch recommendation
    """
    def __init__(self, G, time_limit_tau=10.0, max_depth=10):
        """
        G: NetworkX DiGraph where nodes are road segments (edges in road network)
           and edges are valid transitions between consecutive road segments.
        time_limit_tau: Maximum searching time limit in minutes (default 10.0 mins per paper).
        max_depth: Maximum tree recursion depth.
        """
        self.G = G
        self.tau = time_limit_tau
        self.max_depth = max_depth

    def RRS(self, current_node, current_time, pickup_probs, earnings, travel_times, 
            segment_lengths=None, objective='profit', visited=None, depth=0):
        """
        Recursive Route Searching Algorithm (Algorithm 1)
        Solves Bellman's equation on the recursion tree:
        G*(r) = p_r * e_r + (1 - p_r) * max_{r' in A_r} G*(r')
        
        Supports objectives:
        - 'profit': Maximizes expected trip earning (Equation 1 & 12)
        - 'mst': Minimum Searching Time (Equation 13)
        - 'msd': Minimum Searching Distance (Equation 14)
        """
        if visited is None:
            visited = set()
            
        p_r = pickup_probs.get(current_node, 0.2)
        e_r = earnings.get(current_node, 25.0)
        t_r = travel_times.get(current_node, 1.0)
        l_r = segment_lengths.get(current_node, 300.0) if segment_lengths else 300.0
        
        # Determine reward weight based on objective
        if objective == 'profit':
            reward_r = e_r
        elif objective == 'mst':
            reward_r = t_r
        elif objective == 'msd':
            reward_r = l_r
        else:
            reward_r = e_r
            
        # Base case: Searching time reaches limit tau or max depth
        remaining_time = self.tau - current_time
        if remaining_time <= 0 or depth >= self.max_depth:
            return [current_node], p_r * reward_r
            
        # If segment exceeds remaining time, discount pickup probability proportionally (partial visit per Section 5.1)
        if t_r > remaining_time:
            discount = max(0.1, remaining_time / t_r)
            return [current_node], (p_r * discount) * reward_r

        # Recursion step
        new_visited = visited.copy()
        new_visited.add(current_node)
        
        neighbors = [nbr for nbr in self.G.successors(current_node) if nbr not in new_visited]
        
        # If no unvisited successors, terminate branch
        if not neighbors:
            return [current_node], p_r * reward_r
            
        best_future_val = -1.0
        best_route = []
        
        for neighbor in neighbors:
            child_route, child_val = self.RRS(
                current_node=neighbor,
                current_time=current_time + t_r,
                pickup_probs=pickup_probs,
                earnings=earnings,
                travel_times=travel_times,
                segment_lengths=segment_lengths,
                objective=objective,
                visited=new_visited,
                depth=depth + 1
            )
            
            if child_val > best_future_val:
                best_future_val = child_val
                best_route = child_route
                
        # Bellman equation
        if best_future_val >= 0:
            expected_val = p_r * reward_r + (1.0 - p_r) * best_future_val
            optimal_route = [int(current_node)] + [int(x) for x in best_route]
        else:
            expected_val = p_r * reward_r
            optimal_route = [int(current_node)]
            
        return optimal_route, expected_val

    def find_optimal_virtual_station(self, start_node, virtual_stations, travel_times):
        """
        Finds the optimal virtual station and fastest driving route (Algorithm 2, Steps 6-7).
        Evaluates waiting unit profit: G_w = e_v / (T_R + T_v)
        """
        if not virtual_stations:
            return None, [], 0.0
            
        best_vs_id = None
        best_route = []
        best_waiting_profit = 0.0
        
        for vs_id, vs_data in virtual_stations.items():
            if vs_id == start_node:
                t_route = 0.0
                route = [start_node]
            else:
                try:
                    # Shortest path in segment graph (by travel time)
                    route = nx.shortest_path(self.G, source=start_node, target=vs_id)
                    t_route = sum(travel_times.get(node, 1.0) for node in route[:-1])
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                    
            t_waiting = vs_data.get('waiting_time', 5.0)
            earning_v = vs_data.get('expected_earning', 30.0)
            
            # Unit profit of waiting strategy: e_v / (T_R + T_v)
            waiting_unit_profit = earning_v / (t_route + t_waiting + 1e-6)
            
            if waiting_unit_profit > best_waiting_profit:
                best_waiting_profit = waiting_unit_profit
                best_vs_id = vs_id
                best_route = route
                
        return best_vs_id, best_route, best_waiting_profit

    def calculate_expected_profit(self, route, pickup_probs, earnings):
        """
        Calculates expected profit along a given route according to Equation (1):
        G_s(R) = p_{r_0} e_{r_0} + sum_{i=1}^n [ prod_{j=0}^{i-1} (1 - p_{r_j}) ] p_{r_i} e_{r_i}
        """
        if not route:
            return 0.0
            
        expected_profit = 0.0
        prob_not_picked_up = 1.0
        
        for r in route:
            p_r = pickup_probs.get(r, 0.2)
            e_r = earnings.get(r, 25.0)
            expected_profit += prob_not_picked_up * p_r * e_r
            prob_not_picked_up *= (1.0 - p_r)
            
        return expected_profit

    def update_supply_and_probabilities(self, route, vacant_taxis, pickup_orders, current_probs):
        """
        Dynamically updates taxi supply along recommended route to avoid recommendation overload
        (Section 4 & Section 5.2, Algorithm 2 Step 10).
        Active supply v_r <- v_r + 1
        Pickup probability p_r(v_r) = o_r / (v_r + 1)
        """
        updated_probs = current_probs.copy()
        for r in route:
            v_curr = vacant_taxis.get(r, 5.0) + 1.0
            vacant_taxis[r] = v_curr
            o_curr = pickup_orders.get(r, 2.0)
            # Dynamic pickup probability update with supply saturation
            new_prob = min(0.95, max(0.01, o_curr / (v_curr + 1.0)))
            updated_probs[r] = new_prob
        return updated_probs

    def SRRS(self, taxis_queue, pickup_probs, earnings, travel_times, 
             segment_lengths=None, vacant_taxis=None, pickup_orders=None, 
             virtual_stations=None, objective='profit', strategy_mode='dynamic'):
        """
        Sequential Route Recommendation System (SRRS - Algorithm 2)
        
        Parameters:
        - taxis_queue: List of (taxi_id, start_node, query_time)
        - strategy_mode:
            - 'dynamic': C-SRR with dynamic supply updates to avoid overload
            - 'static': Static historical baseline (Qu et al. 2014)
            - 'top_k': Diversity penalty on route overlap
            - 'group_5': Batch recommendation of top 5 routes per 5 queries
        """
        recommendations = {}
        history_routes = []
        
        current_probs = pickup_probs.copy()
        current_vacant = vacant_taxis.copy() if vacant_taxis else {n: 5.0 for n in self.G.nodes()}
        current_orders = pickup_orders.copy() if pickup_orders else {n: 2.0 for n in self.G.nodes()}
        
        for idx, (taxi_id, start_node) in enumerate(taxis_queue):
            # Step 5: Find optimal searching route
            r_s, g_s = self.RRS(
                current_node=start_node,
                current_time=0.0,
                pickup_probs=current_probs,
                earnings=earnings,
                travel_times=travel_times,
                segment_lengths=segment_lengths,
                objective=objective
            )
            
            # Step 6 & 7: Find optimal virtual station
            vs_id, r_w, g_w = self.find_optimal_virtual_station(
                start_node=start_node,
                virtual_stations=virtual_stations,
                travel_times=travel_times
            )
            
            # Calculate unit profit (profit per minute) for searching route
            t_searching = sum(travel_times.get(node, 1.0) for node in r_s)
            eup_searching = g_s / (t_searching + 1e-6)
            eup_waiting = g_w  # already unit profit
            
            # Step 8 & 9: Compare searching vs waiting strategy
            if eup_waiting > eup_searching and len(r_w) > 0:
                chosen_route = r_w
                chosen_strategy = f"virtual_station_{vs_id}"
                expected_unit_profit = eup_waiting
            else:
                chosen_route = r_s
                chosen_strategy = "searching_route"
                expected_unit_profit = eup_searching
                
            # Mode-specific routing adjustments
            if strategy_mode == 'top_k' and history_routes:
                # Penalize overlapping routes
                for prev_route in history_routes[-3:]:
                    overlap = len(set(chosen_route) & set(prev_route))
                    if overlap > 1 and len(list(self.G.successors(start_node))) > 1:
                        # Pick alternative branch
                        alt_neighbors = [n for n in self.G.successors(start_node) if n not in prev_route]
                        if alt_neighbors:
                            alt_route, alt_val = self.RRS(
                                current_node=alt_neighbors[0],
                                current_time=travel_times.get(start_node, 1.0),
                                pickup_probs=current_probs,
                                earnings=earnings,
                                travel_times=travel_times
                            )
                            chosen_route = [start_node] + alt_route
                            expected_unit_profit = alt_val / max(1.0, sum(travel_times.get(n, 1.0) for n in chosen_route))
                            break

            # Record recommendation
            actual_exp_profit = self.calculate_expected_profit(chosen_route, current_probs, earnings)
            total_duration = sum(travel_times.get(node, 1.0) for node in chosen_route)
            
            recommendations[taxi_id] = {
                'route': chosen_route,
                'strategy': chosen_strategy,
                'expected_profit': actual_exp_profit,
                'expected_unit_profit': actual_exp_profit / max(1.0, total_duration),
                'travel_time': total_duration
            }
            
            history_routes.append(chosen_route)
            
            # Step 10: Update taxi supply dynamically if mode is dynamic
            if strategy_mode == 'dynamic':
                current_probs = self.update_supply_and_probabilities(
                    chosen_route, current_vacant, current_orders, current_probs
                )
            elif strategy_mode == 'group_5' and (idx + 1) % 5 == 0:
                # Batch update every 5 taxis
                for r in history_routes[-5:]:
                    current_probs = self.update_supply_and_probabilities(
                        r, current_vacant, current_orders, current_probs
                    )

        return recommendations

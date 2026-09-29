"""
mora_optimizer.py — Complete MORA-LLM Multi-Objective Evolutionary Algorithm.

Implements Algorithm 1, Algorithm 2 (Personalized Search), and Algorithm 3 (Competitive Search)
from "MORA-LLM: Enhancing Multiobjective Optimization Recommendation Algorithm by Integrating
Large Language Models" (Ge, Wu, Yang, Cheng, Zhao, Zhang, IEEE TEVC, 2026).
"""

import math
import random
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from .leps_predictor import LEPSPredictor
from .sde_fitness import ShiftBasedDensityEstimation
from .llm_reasoner import LLMRecommendationReasoner
from .driver_profiles import get_zone_for_segment, ZONE_TOPICS, DRIVER_PROFILES

class MORARouteOptimizer:
    """
    Multi-Objective Recommendation Algorithm integrating Large Language Models (MORA-LLM).
    Optimizes sequential route recommendations over three competing objectives:
      1. Accuracy / Expected Profit (Eq. 2)
      2. Novelty / Low Taxi Competition Exploration (Eq. 3)
      3. Topic / Urban Zone Diversity (Eq. 4 & 5)
    """
    def __init__(
        self,
        route_G: nx.DiGraph,
        time_limit_tau: float = 10.0,
        pop_size: int = 30,
        max_gen: int = 40,
        k_length: int = 8,
        llm_cycle: int = 10,
        alpha: float = 0.2,
        variant: str = "MORA-LLM"  # "MORA-LLM", "MORA-LLM(-WOLLM)", "MORA-LLM(-WOM)", "MORA-LLM(-WOQ)"
    ):
        self.route_G = route_G
        self.tau = time_limit_tau
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.k = k_length
        self.cycle = llm_cycle
        self.alpha = alpha
        self.variant = variant

        # Core sub-modules
        self.leps = LEPSPredictor()
        self.sde = ShiftBasedDensityEstimation()
        self.reasoner = LLMRecommendationReasoner(alpha=alpha)

    # ─── Multi-Objective Functions (Equations 2, 3, 4, 5) ─────────────────────

    def evaluate_objectives(
        self,
        route: List[int],
        predicted_scores: Dict[int, float],
        taxi_counts: Dict[int, int],
        edge_coords: Dict[int, Tuple[float, float]],
        travel_times: Dict[Tuple[int, int], float]
    ) -> Tuple[float, float, float]:
        """
        Computes the three core objective values for a given route Lu:
          f1 (Accuracy / Profit): sum R^_u,i / K
          f2 (Novelty): sum (1 / (taxi_count + 1)) / K
          f3 (Topic Diversity): H(Lu) Shannon entropy over urban functional zones
        """
        if not route:
            return 0.0, 0.0, 0.0

        K = float(len(route))

        # 1. Accuracy / Expected Profit (Eq. 2)
        f1_accuracy = sum(predicted_scores.get(eid, 0.0) for eid in route) / K

        # 2. Novelty (Eq. 3) - Proportion of unpopular / underserved items
        # Pop_u,i is calculated by the inverse of taxi density
        novelty_sum = 0.0
        for eid in route:
            cnt = taxi_counts.get(eid, 0)
            pop = 1.0 / (float(cnt) + 1.0)
            novelty_sum += pop
        f2_novelty = novelty_sum / K

        # 3. Topic Diversity (Eq. 4 & 5)
        # H(Lu) = sum_{j in sLu} - q_j log(q_j) * (|sLu| / Nt)
        zone_counts = {}
        for eid in route:
            lat, lon = edge_coords.get(eid, (39.9, 116.4))
            z = get_zone_for_segment(eid, lat, lon)
            zone_counts[z] = zone_counts.get(z, 0) + 1

        total_topics = len(ZONE_TOPICS)
        unique_topics_in_route = len(zone_counts)
        entropy = 0.0
        for z, count in zone_counts.items():
            q_j = count / K
            entropy += - q_j * math.log(q_j + 1e-9)

        # Scale by topic coverage ratio (|sLu| / Nt)
        f3_diversity = entropy * (unique_topics_in_route / float(total_topics))

        # Penalty if route exceeds time limit tau
        time_spent = 0.0
        for i in range(len(route) - 1):
            time_spent += travel_times.get((route[i], route[i+1]), 1.0)
        if time_spent > self.tau:
            overage = time_spent - self.tau
            penalty = math.exp(-0.15 * overage)
            f1_accuracy *= penalty

        return float(f1_accuracy), float(f2_novelty), float(f3_diversity)

    # ─── Graph-Constrained Route Generation & Variation Operators ──────────────

    def _generate_valid_route(self, start_segment: int, length: int) -> List[int]:
        """Generates a connected walk starting from start_segment on route_G."""
        route = [start_segment]
        curr = start_segment
        for _ in range(length - 1):
            neighbors = list(self.route_G.successors(curr))
            if not neighbors:
                break
            # Prefer unvisited neighbor if available
            unvisited = [n for n in neighbors if n not in route]
            next_seg = random.choice(unvisited) if unvisited else random.choice(neighbors)
            route.append(next_seg)
            curr = next_seg
        return route

    def _crossover_routes(self, parent_a: List[int], parent_b: List[int]) -> List[int]:
        """
        Graph-aware crossover: Loser learns from Winner (Algorithm 3, Step 1).
        Finds overlapping segments and splices them.
        """
        common_nodes = set(parent_a).intersection(set(parent_b))
        if common_nodes:
            pivot = random.choice(list(common_nodes))
            idx_a = parent_a.index(pivot)
            idx_b = parent_b.index(pivot)
            child = parent_a[:idx_a] + parent_b[idx_b:]
            # Ensure proper length
            if len(child) > self.k:
                child = child[:self.k]
            elif len(child) < self.k:
                # Extend from end
                extra = self._generate_valid_route(child[-1], self.k - len(child) + 1)
                child = child + extra[1:]
            return child
        # If no common node, return winner or mutated loser
        return list(parent_b)

    def _mutate_route(self, route: List[int]) -> List[int]:
        """Graph-aware mutation (Algorithm 3, Step 2). Mutates tail of route."""
        if len(route) < 3:
            return route
        cut_point = random.randint(1, len(route) - 2)
        prefix = route[:cut_point]
        suffix = self._generate_valid_route(prefix[-1], len(route) - cut_point + 1)
        mutated = prefix + suffix[1:]
        return mutated[:self.k]

    def _deduplicate_route(self, route: List[int]) -> List[int]:
        """Deduplicates cycles while maintaining connectivity (Algorithm 3, Step 3)."""
        seen = set()
        dedup = []
        for seg in route:
            if seg not in seen:
                seen.add(seg)
                dedup.append(seg)
        # If too short after dedup, extend
        if len(dedup) < 3 and len(dedup) > 0:
            ext = self._generate_valid_route(dedup[-1], self.k - len(dedup) + 1)
            dedup = dedup + ext[1:]
        return dedup

    # ─── Algorithm 2: PersonalizedSearch(P, R^, gen) ──────────────────────────

    def personalized_search(
        self,
        population: List[List[int]],
        predicted_scores: Dict[int, float],
        driver_profile: Dict[str, Any],
        top_candidates: List[int],
        taxi_counts: Dict[int, int],
        edge_coords: Dict[int, Tuple[float, float]],
        travel_times: Dict[Tuple[int, int], float],
        gen: int,
        scenario: str = "normal"
    ) -> Tuple[List[List[int]], Optional[Dict[str, Any]], Dict[int, float]]:
        """
        Implements Algorithm 2: Personalized Search.
        Selects optimal lists for Accuracy (S1), Novelty (S2), Diversity (S3).
        Periodically invokes LLM reasoning every `cycle` iterations and corrects R^ via Eq. 8.
        """
        # Evaluate all solutions in current population
        obj_evals = [
            self.evaluate_objectives(sol, predicted_scores, taxi_counts, edge_coords, travel_times)
            for sol in population
        ]

        # Select S1 (max Accuracy), S2 (max Novelty), S3 (max Diversity)
        s1_idx = int(np.argmax([ev[0] for ev in obj_evals]))
        s2_idx = int(np.argmax([ev[1] for ev in obj_evals]))
        s3_idx = int(np.argmax([ev[2] for ev in obj_evals]))

        s1_route = population[s1_idx]
        s2_route = population[s2_idx]
        s3_route = population[s3_idx]

        Pp = [s1_route, s2_route, s3_route]
        reason_output = None
        updated_scores = predicted_scores.copy()

        # Check if LLM cycle is triggered (line 6 in Algorithm 2)
        use_llm = (self.variant in ["MORA-LLM", "MORA-LLM(-WOM)"])
        if use_llm and (gen % self.cycle == 0 or gen == self.max_gen):
            # Construct prompt as in Fig. 5
            prompt = self.reasoner.construct_prompt(
                driver_profile=driver_profile,
                historical_samples=driver_profile.get("historical_interactions", []),
                candidate_pool=top_candidates,
                s1_optimal_route=s1_route,
                k=self.k
            )

            # LLM Reasoning produces candidate route S' and explanations
            # In our implementation, we generate high-utility personalized exploration route
            # combining S1 strengths with profile-preferred candidate segments
            llm_route = list(s1_route)
            pref_zones = set(driver_profile.get("preferred_zones", []))
            
            # Inject candidate segments matching driver profile
            for cand_id in top_candidates:
                lat, lon = edge_coords.get(cand_id, (39.9, 116.4))
                z = get_zone_for_segment(cand_id, lat, lon)
                if z in pref_zones and cand_id not in llm_route:
                    # Replace a lower utility step
                    if len(llm_route) > 2:
                        llm_route[random.randint(1, len(llm_route) - 1)] = cand_id
                    break

            # Dynamic Score Correction (Equation 8)
            updated_scores = self.reasoner.correct_prediction_scores(
                predicted_scores=predicted_scores,
                s1_route=s1_route,
                llm_recommended_route=llm_route
            )

            # Generate natural language explanation (NLG)
            safe_probs = {eid: min(max(getattr(self, 'base_pickup_probs', {}).get(eid, 0.2), 0.0), 1.0) for eid in llm_route}
            safe_earnings = {eid: max(getattr(self, 'base_earnings', {}).get(eid, 25.0), 0.0) for eid in llm_route}
            reason_output = self.reasoner.generate_personalized_reasoning(
                driver_profile=driver_profile,
                route=llm_route,
                pickup_probs=safe_probs,
                earnings=safe_earnings,
                edge_coords=edge_coords,
                scenario=scenario
            )

            Pp.append(llm_route)

        return Pp, reason_output, updated_scores

    # ─── Algorithm 3: CompetitiveSearch(P ∪ Pp) ───────────────────────────────

    def competitive_search(
        self,
        combined_population: List[List[int]],
        predicted_scores: Dict[int, float],
        taxi_counts: Dict[int, int],
        edge_coords: Dict[int, Tuple[float, float]],
        travel_times: Dict[Tuple[int, int], float]
    ) -> List[List[int]]:
        """
        Implements Algorithm 3: Competitive Swarm Optimization (CSO).
        Calculates fitness via SDE strategy. Losers learn from Winners via
        crossover, mutation, and deduplication.
        """
        N = len(combined_population)
        if N < 2:
            return list(combined_population)

        # 1. Compute objective matrix
        obj_matrix = np.zeros((N, 3), dtype=np.float32)
        for i, sol in enumerate(combined_population):
            f1, f2, f3 = self.evaluate_objectives(sol, predicted_scores, taxi_counts, edge_coords, travel_times)
            obj_matrix[i, :] = [f1, f2, f3]

        # 2. Fitness evaluation using SDE strategy
        fitness = self.sde.calculate_fitness(obj_matrix)

        Pc = []
        # 3. Pairwise competition
        for i in range(N):
            # Select random opponent Pj != Pi
            j_indices = [idx for idx in range(N) if idx != i]
            j = random.choice(j_indices)

            fit_i = fitness[i]
            fit_j = fitness[j]

            if fit_i >= fit_j:
                winner = combined_population[i]
                loser = combined_population[j]
            else:
                winner = combined_population[j]
                loser = combined_population[i]

            # Loser learns from Winner: Crossover -> Mutation -> Deduplication
            offspring = self._crossover_routes(loser, winner)
            if random.random() < 0.25:
                offspring = self._mutate_route(offspring)
            offspring = self._deduplicate_route(offspring)

            Pc.append(offspring)

        return Pc

    # ─── Environmental Selection via SDE Strategy ─────────────────────────────

    def environmental_selection(
        self,
        full_pool: List[List[int]],
        predicted_scores: Dict[int, float],
        taxi_counts: Dict[int, int],
        edge_coords: Dict[int, Tuple[float, float]],
        travel_times: Dict[Tuple[int, int], float],
        target_size: int
    ) -> List[List[int]]:
        """Selects top target_size solutions from pool using SDE fitness."""
        if len(full_pool) <= target_size:
            return full_pool

        # Deduplicate identical routes
        unique_pool = []
        seen = set()
        for sol in full_pool:
            t = tuple(sol)
            if t not in seen:
                seen.add(t)
                unique_pool.append(sol)

        if len(unique_pool) <= target_size:
            # Fill with mutated variations if needed
            while len(unique_pool) < target_size:
                unique_pool.append(self._mutate_route(unique_pool[0]))
            return unique_pool[:target_size]

        N = len(unique_pool)
        obj_matrix = np.zeros((N, 3), dtype=np.float32)
        for i, sol in enumerate(unique_pool):
            f1, f2, f3 = self.evaluate_objectives(sol, predicted_scores, taxi_counts, edge_coords, travel_times)
            obj_matrix[i, :] = [f1, f2, f3]

        fitness = self.sde.calculate_fitness(obj_matrix)
        sorted_indices = np.argsort(fitness)[::-1]  # Highest fitness first

        selected = [unique_pool[idx] for idx in sorted_indices[:target_size]]
        return selected

    # ─── Algorithm 1: General Framework of MORA-LLM ───────────────────────────

    def optimize(
        self,
        start_segment: int,
        driver_profile: Dict[str, Any],
        base_pickup_probs: Dict[int, float],
        base_earnings: Dict[int, float],
        travel_times: Dict[Tuple[int, int], float],
        edge_coords: Dict[int, Tuple[float, float]],
        taxi_counts: Optional[Dict[int, int]] = None,
        scenario: str = "normal"
    ) -> Dict[str, Any]:
        """
        Executes full MORA-LLM workflow (Algorithm 1):
          1. LEPS Strategy: Predict scores R^ and extract top 100 candidate items.
          2. Population initialization.
          3. Evolutionary optimization with PersonalizedSearch & CompetitiveSearch.
          4. Pareto front extraction and Explainable Recommendation generation.
        """
        taxi_counts = taxi_counts or {}
        self.base_pickup_probs = base_pickup_probs
        self.base_earnings = base_earnings

        # ─── Step 1: Prediction Score Stage (LEPS Strategy) ───────────────────
        use_leps = (self.variant != "MORA-LLM(-WOLLM)" and self.variant != "MORA-LLM(-WOM)")
        if use_leps:
            predicted_scores, top_candidates = self.leps.predict_scores(
                target_profile=driver_profile,
                base_pickup_probs=base_pickup_probs,
                base_earnings=base_earnings,
                edge_coords=edge_coords,
                top_k_candidates=100
            )
        else:
            # Traditional non-LLM score (raw expected utility)
            predicted_scores = {eid: base_pickup_probs.get(eid, 0.1) * base_earnings.get(eid, 25.0)
                                for eid in base_pickup_probs}
            top_candidates = sorted(predicted_scores.keys(), key=lambda x: predicted_scores[x], reverse=True)[:100]

        # ─── Step 2: Population Initialization ────────────────────────────────
        population: List[List[int]] = []
        # Seed initial population with walks connecting from start_segment
        for _ in range(self.pop_size):
            walk = self._generate_valid_route(start_segment, self.k)
            population.append(walk)

        latest_reason = None
        current_scores = predicted_scores

        # ─── Step 3: Evolutionary Optimization Loop (maxGen) ──────────────────
        for gen in range(1, self.max_gen + 1):
            # Line 4: Personalized Search
            Pp, reason_out, current_scores = self.personalized_search(
                population=population,
                predicted_scores=current_scores,
                driver_profile=driver_profile,
                top_candidates=top_candidates,
                taxi_counts=taxi_counts,
                edge_coords=edge_coords,
                travel_times=travel_times,
                gen=gen,
                scenario=scenario
            )
            if reason_out:
                latest_reason = reason_out

            # Line 5: Competitive Search (CSO)
            combined_P_Pp = population + Pp
            Pc = self.competitive_search(
                combined_population=combined_P_Pp,
                predicted_scores=current_scores,
                taxi_counts=taxi_counts,
                edge_coords=edge_coords,
                travel_times=travel_times
            )

            # Line 6: Environmental Selection via SDE
            population = self.environmental_selection(
                full_pool=combined_P_Pp + Pc,
                predicted_scores=current_scores,
                taxi_counts=taxi_counts,
                edge_coords=edge_coords,
                travel_times=travel_times,
                target_size=self.pop_size
            )

        # ─── Step 4: Extract Pareto Front & Recommendations ───────────────────
        evals = [
            self.evaluate_objectives(sol, current_scores, taxi_counts, edge_coords, travel_times)
            for sol in population
        ]
        obj_matrix = np.array(evals, dtype=np.float32)
        fitness = self.sde.calculate_fitness(obj_matrix)

        # Best overall solution (highest SDE fitness)
        best_overall_idx = int(np.argmax(fitness))
        best_route = population[best_overall_idx]

        # Best profit route (S1)
        best_profit_idx = int(np.argmax(obj_matrix[:, 0]))
        max_profit_route = population[best_profit_idx]

        # Best novelty route (S2)
        best_novelty_idx = int(np.argmax(obj_matrix[:, 1]))
        max_novelty_route = population[best_novelty_idx]

        # Best diversity route (S3)
        best_diversity_idx = int(np.argmax(obj_matrix[:, 2]))
        max_diversity_route = population[best_diversity_idx]

        # If reason wasn't generated during last cycle, generate for best route
        if not latest_reason:
            safe_probs = {eid: min(max(base_pickup_probs.get(eid, 0.2), 0.0), 1.0) for eid in best_route}
            safe_earnings = {eid: max(base_earnings.get(eid, 25.0), 0.0) for eid in best_route}
            latest_reason = self.reasoner.generate_personalized_reasoning(
                driver_profile=driver_profile,
                route=best_route,
                pickup_probs=safe_probs,
                earnings=safe_earnings,
                edge_coords=edge_coords,
                scenario=scenario
            )

        # Compute travel time for best route
        total_time = sum(travel_times.get((best_route[i], best_route[i+1]), 1.0) for i in range(len(best_route)-1))

        return {
            "algorithm": self.variant,
            "driver_name": driver_profile.get("name"),
            "is_cold_start": driver_profile.get("is_cold_start", False),
            "best_route": best_route,
            "expected_earning": round(float(obj_matrix[best_overall_idx, 0]), 2),
            "novelty_score": round(float(obj_matrix[best_overall_idx, 1]), 4),
            "topic_diversity_score": round(float(obj_matrix[best_overall_idx, 2]), 4),
            "sde_fitness": round(float(fitness[best_overall_idx]), 4),
            "travel_time_minutes": round(total_time, 1),
            "llm_explanation": latest_reason,
            "pareto_alternatives": {
                "balanced": {
                    "route": best_route,
                    "profit": round(float(obj_matrix[best_overall_idx, 0]), 2),
                    "novelty": round(float(obj_matrix[best_overall_idx, 1]), 4),
                    "diversity": round(float(obj_matrix[best_overall_idx, 2]), 4),
                    "label": "Balanced Pareto Best"
                },
                "max_profit": {
                    "route": max_profit_route,
                    "profit": round(float(obj_matrix[best_profit_idx, 0]), 2),
                    "novelty": round(float(obj_matrix[best_profit_idx, 1]), 4),
                    "diversity": round(float(obj_matrix[best_profit_idx, 2]), 4),
                    "label": "Max Profit / Revenue"
                },
                "max_exploration": {
                    "route": max_novelty_route,
                    "profit": round(float(obj_matrix[best_novelty_idx, 0]), 2),
                    "novelty": round(float(obj_matrix[best_novelty_idx, 1]), 4),
                    "diversity": round(float(obj_matrix[best_novelty_idx, 2]), 4),
                    "label": "Max Novelty (Underserved Demand)"
                }
            }
        }

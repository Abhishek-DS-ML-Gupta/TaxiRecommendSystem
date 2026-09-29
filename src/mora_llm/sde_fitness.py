"""
sde_fitness.py — Shift-based Density Estimation (SDE) Strategy for MORA-LLM.

Implements the SDE fitness assignment algorithm (M. Li et al., IEEE TEVC 2014) as used in:
  - Algorithm 1 Line 6 (Environmental Selection)
  - Algorithm 3 Line 3 (Competitive Search fitness calculation)

SDE captures both convergence (shifting inferior solutions toward superior ones)
and diversity (crowding density in objective space), resolving the objective conflict
between Accuracy/Profit, Novelty, and Topic Diversity.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

class ShiftBasedDensityEstimation:
    """
    Computes Pareto non-dominated ranks and SDE fitness values
    for multi-objective recommendation populations.
    Objectives: [f1: Profit/Accuracy, f2: Novelty, f3: Topic Diversity]
    All three objectives are to be MAXIMIZED.
    """
    def __init__(self, k_neighbor: int = 1):
        self.k = k_neighbor

    def calculate_fitness(self, objective_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates SDE fitness for N solutions with M objectives.
        Args:
            objective_matrix: np.ndarray of shape (N, M). All objectives assume maximization.
        Returns:
            fitness: np.ndarray of shape (N,). Higher fitness = superior candidate.
        """
        N, M = objective_matrix.shape
        if N <= 1:
            return np.ones(N, dtype=np.float32)

        # 1. Normalize objectives to [0, 1] range to avoid scale bias
        min_vals = np.min(objective_matrix, axis=0)
        max_vals = np.max(objective_matrix, axis=0)
        ranges = np.where((max_vals - min_vals) > 1e-8, max_vals - min_vals, 1.0)
        norm_obj = (objective_matrix - min_vals) / ranges

        # 2. Compute Shifted Distance Matrix D_shift
        # For maximization: if solution q is strictly worse than p in objective m,
        # its position is shifted to p's position: shifted_q_m = min(norm_obj[q, m], norm_obj[p, m])
        # The shifted Euclidean distance d(p, q) penalizes q if it is dominated by p.
        dist_matrix = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(N):
                if i == j:
                    dist_matrix[i, j] = float('inf')
                    continue
                # Shifted coordinate for j relative to i
                # If j has worse value than i (j < i), shift j to i's coordinate (diff = 0)
                # If j has better value than i (j > i), keep actual diff
                diffs = np.maximum(0.0, norm_obj[i] - norm_obj[j])
                # Shifted Euclidean distance
                dist_matrix[i, j] = np.sqrt(np.sum(diffs ** 2) + 1e-8)

        # 3. Density estimation: distance to k-th nearest neighbor in shifted space
        # A solution with larger shifted distance to other solutions is more isolated/diverse
        # and less dominated.
        sorted_dists = np.sort(dist_matrix, axis=1)
        k_idx = min(self.k - 1, N - 2)
        sde_density = sorted_dists[:, k_idx]

        # 4. Non-dominated ranking layer
        ranks = self.fast_non_dominated_sort(objective_matrix)
        
        # Combine Pareto rank and SDE density:
        # Lower rank (Rank 0 is Pareto front) + high SDE density = highest fitness
        max_rank = max(ranks) if len(ranks) > 0 else 1
        fitness = (max_rank - np.array(ranks, dtype=np.float32) + 1.0) * 10.0 + sde_density

        return fitness

    @staticmethod
    def fast_non_dominated_sort(objective_matrix: np.ndarray) -> List[int]:
        """
        Fast non-dominated sorting (Deb et al., NSGA-II).
        Maximizes all objectives.
        Returns array of ranks for each solution (0 = Pareto front).
        """
        N = objective_matrix.shape[0]
        domination_count = [0] * N
        dominated_solutions = [[] for _ in range(N)]
        fronts: List[List[int]] = [[]]

        for p in range(N):
            for q in range(p + 1, N):
                # Check if p dominates q (p >= q for all, and p > q for at least one)
                p_dominates_q = (
                    np.all(objective_matrix[p] >= objective_matrix[q]) and
                    np.any(objective_matrix[p] > objective_matrix[q])
                )
                q_dominates_p = (
                    np.all(objective_matrix[q] >= objective_matrix[p]) and
                    np.any(objective_matrix[q] > objective_matrix[p])
                )

                if p_dominates_q:
                    dominated_solutions[p].append(q)
                    domination_count[q] += 1
                elif q_dominates_p:
                    dominated_solutions[q].append(p)
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        ranks = [0] * N
        current_front_idx = 0
        while len(fronts[current_front_idx]) > 0:
            next_front = []
            for p in fronts[current_front_idx]:
                ranks[p] = current_front_idx
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            current_front_idx += 1
            fronts.append(next_front)

        return ranks

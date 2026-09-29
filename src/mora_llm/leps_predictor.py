"""
leps_predictor.py — LLM-Enhancing Prediction Score (LEPS) Strategy.

Implements Stage 1 of MORA-LLM (Ge et al., 2026, Section III-B):
    R^_u,j = sum_{v in Nu} sim(u, v) * R_{v,j} / sum_{v in Nu} sim(u, v)   [Equation 7]

Extracts interaction knowledge from similar experienced drivers to compensate for missing
interaction data of new drivers (cold start alleviation), and outputs the top-100 candidate items.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .embeddings import DriverEmbeddingGenerator
from .driver_profiles import DRIVER_PROFILES, get_driver_profile, get_zone_for_segment

class LEPSPredictor:
    """
    LLM-Enhanced Prediction Score model.
    Transforms raw network probabilities and driver profiles into personalized
    predicted utility scores R^, resolving the cold-start problem.
    """
    def __init__(self, embedder: Optional[DriverEmbeddingGenerator] = None):
        self.embedder = embedder or DriverEmbeddingGenerator()
        self.profile_embeddings: Dict[str, np.ndarray] = {}
        self._init_profiles()

    def _init_profiles(self):
        """Pre-computes embeddings for all reference driver personas."""
        for pid, profile in DRIVER_PROFILES.items():
            self.profile_embeddings[pid] = self.embedder.compute_driver_embedding(profile)

    def calculate_similarities(self, target_profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes cosine similarities between the target driver and all reference drivers:
            sim(u, v) = (eu · ev) / (||eu|| · ||ev||)   [Equation 6]
        """
        target_vec = self.embedder.compute_driver_embedding(target_profile)
        sims = {}
        for pid, ref_vec in self.profile_embeddings.items():
            if pid == target_profile.get("id"):
                continue
            sims[pid] = self.embedder.cosine_similarity(target_vec, ref_vec)
        return sims

    def predict_scores(
        self,
        target_profile: Dict[str, Any],
        base_pickup_probs: Dict[int, float],
        base_earnings: Dict[int, float],
        edge_coords: Optional[Dict[int, Tuple[float, float]]] = None,
        top_k_candidates: int = 100
    ) -> Tuple[Dict[int, float], List[int]]:
        """
        Computes predicted scores R^_u,j for all segments using LEPS (Equation 7),
        blending base spatial-temporal predictions with semantic user similarity.

        Returns:
            predicted_scores: {segment_id: score}
            top_candidates: List of top_k_candidates segment IDs sorted by score
        """
        edge_coords = edge_coords or {}
        sims = self.calculate_similarities(target_profile)
        preferred_zones = set(target_profile.get("preferred_zones", []))
        avoid_zones = set(target_profile.get("avoid_zones", []))
        is_cold = target_profile.get("is_cold_start", False)

        # Baseline expected utility = p * e
        predicted_scores: Dict[int, float] = {}

        for eid, prob in base_pickup_probs.items():
            earn = base_earnings.get(eid, 25.0)
            base_utility = prob * earn

            # Identify zone topic for segment
            lat, lon = edge_coords.get(eid, (39.9, 116.4))
            zone = get_zone_for_segment(eid, lat, lon)

            # Preference alignment factor
            if zone in preferred_zones:
                pref_mult = 1.35
            elif zone in avoid_zones:
                pref_mult = 0.65
            else:
                pref_mult = 1.0

            # Collaborative filtering aggregation from similar drivers
            # If cold start, peer knowledge has heavier weighting to compensate for lack of history
            cf_weight = 0.0
            total_sim = 0.0
            for pid, sim in sims.items():
                ref = DRIVER_PROFILES.get(pid, {})
                ref_pref = set(ref.get("preferred_zones", []))
                # Peer rating proxy based on peer preference alignment with zone
                peer_rating = 1.4 if zone in ref_pref else 0.8
                cf_weight += sim * peer_rating
                total_sim += sim

            cf_factor = (cf_weight / max(total_sim, 1e-6)) if total_sim > 0 else 1.0

            if is_cold:
                # Cold start: rely 50% on CF semantic peer alignment + 50% on spatial base
                final_score = base_utility * (0.5 * pref_mult + 0.5 * cf_factor)
            else:
                # Experienced driver: rely 75% on base + 25% on profile adjustment
                final_score = base_utility * (0.75 * pref_mult + 0.25 * cf_factor)

            predicted_scores[eid] = float(final_score)

        # Sort candidate segments by predicted score descending
        sorted_candidates = sorted(
            predicted_scores.keys(),
            key=lambda x: predicted_scores[x],
            reverse=True
        )

        top_candidates = sorted_candidates[:top_k_candidates]
        return predicted_scores, top_candidates

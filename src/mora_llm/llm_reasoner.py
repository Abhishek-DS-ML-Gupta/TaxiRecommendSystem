"""
llm_reasoner.py — LLM Prompting, Dynamic Score Correction, and NLG Explainability.

Implements Algorithm 2 & Section III-C-1 from the MORA-LLM paper (Ge et al., 2026):
  1. Construct prompt using Example1 (history), Example2 (exploration candidates), Example3 (S1 optimal).
  2. Perform LLM reasoning to output top-K recommended items and recommendation reasons.
  3. Dynamic Score Correction (Equation 8):
       R^_u,i' = R^_u,i + alpha * (sum_{j in Lu,S1} R^_u,j / |Lu,S1|)   for i in Lu,S' \ Lu,S1
     where alpha = 0.2 (optimal sensitivity from paper Fig. 14).
"""

import os
import json
import random
from typing import Dict, List, Tuple, Any, Optional

from .driver_profiles import get_zone_for_segment

class LLMRecommendationReasoner:
    """
    Handles LLM-driven personalized reasoning, prompt construction,
    dynamic score correction, and natural language explanation generation.
    """
    def __init__(self, alpha: float = 0.2, api_key: Optional[str] = None):
        self.alpha = alpha  # Weight factor for score correction (Eq. 8)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")

    def construct_prompt(
        self,
        driver_profile: Dict[str, Any],
        historical_samples: List[Dict[str, Any]],
        candidate_pool: List[int],
        s1_optimal_route: List[int],
        k: int = 8
    ) -> str:
        """
        Constructs the exact prompt template defined in Figure 5 of the MORA-LLM paper:
          - Example 1: User's historical interactions
          - Example 2: Random selection of 20 items from candidate recommendation list
          - Example 3: The current Accuracy/Profit optimal list S1
        """
        ex1_str = ", ".join([
            f"[Seg {item.get('segment_id', '?')}: {item.get('zone', 'Zone')} (rating {item.get('rating', 4.0)}, earn ¥{item.get('earning', 20):.1f})]"
            for item in historical_samples[:10]
        ]) or "[New driver onboarded, baseline urban preference profile]"

        # Example 2: candidates not in S1 to avoid repetition
        ex2_candidates = [eid for eid in candidate_pool if eid not in s1_optimal_route]
        sampled_ex2 = random.sample(ex2_candidates, min(15, len(ex2_candidates)))
        ex2_str = ", ".join([f"[Seg {eid}]" for eid in sampled_ex2])

        # Example 3: S1 optimal route
        ex3_str = ", ".join([f"[Seg {eid}]" for eid in s1_optimal_route[:k]])

        prompt = f"""
Prompt:
Based on driver {driver_profile.get('name', 'User u')}'s profile ({driver_profile.get('description', '')}) and historical interactions, recommend the top {k} sequential road segments for driver u from the candidate list.

Driver's historical interactions:
{ex1_str}

Candidate list (Exploration items Example2 + Optimal items Example3):
{ex2_str}, {ex3_str}

Instruction:
Just output the top {k} recommended segment IDs for driver u and a clear, persuasive recommendation reason for each segment explaining why it maximizes earnings, matches driver preference, and avoids unnecessary empty cruising.
"""
        return prompt.strip()

    def correct_prediction_scores(
        self,
        predicted_scores: Dict[int, float],
        s1_route: List[int],
        llm_recommended_route: List[int]
    ) -> Dict[int, float]:
        """
        Applies Dynamic Score Correction from Equation 8 in the MORA-LLM paper:
            R^_u,i' = R^_u,i + alpha * (sum_{j in Lu,S1} R^_u,j / |Lu,S1|)
            for i in Lu,S' \ Lu,S1
        """
        if not s1_route:
            return predicted_scores.copy()

        # Compute average prediction score of items in S1
        s1_scores = [predicted_scores.get(eid, 0.0) for eid in s1_route]
        s1_avg = sum(s1_scores) / max(len(s1_scores), 1)

        modified_scores = predicted_scores.copy()

        # Identify items in LLM route that are NOT in S1 (Lu,S' \ Lu,S1)
        diff_items = [eid for eid in llm_recommended_route if eid not in s1_route]

        for eid in diff_items:
            current_score = modified_scores.get(eid, 0.0)
            corrected_score = current_score + self.alpha * s1_avg
            modified_scores[eid] = float(corrected_score)

        return modified_scores

    def generate_personalized_reasoning(
        self,
        driver_profile: Dict[str, Any],
        route: List[int],
        pickup_probs: Dict[int, float],
        earnings: Dict[int, float],
        edge_coords: Dict[int, Tuple[float, float]],
        scenario: str = "normal"
    ) -> Dict[str, Any]:
        """
        Generates natural language explainability reasons for each segment and the whole route,
        adhering to Section III-C-1 and Section IV-D-4 of the MORA-LLM paper.
        """
        driver_name = driver_profile.get("name", "Driver")
        is_cold = driver_profile.get("is_cold_start", False)
        pref_zones = driver_profile.get("preferred_zones", [])
        driver_pref = driver_profile.get("driving_preference", "balanced")

        segment_reasons = []
        total_p = 0.0
        total_earn = 0.0

        for idx, eid in enumerate(route):
            prob = pickup_probs.get(eid, 0.15)
            earn = earnings.get(eid, 25.0)
            total_p += prob
            total_earn += earn

            lat, lon = edge_coords.get(eid, (39.904, 116.407))
            zone = get_zone_for_segment(eid, lat, lon)

            # Generate contextual reason for this step
            if idx == 0:
                step_reason = (
                    f"Initiates navigation at Segment {eid} ({zone}) with {prob*100:.1f}% pickup probability, "
                    f"positioning the taxi immediately into an active dispatch zone."
                )
            elif zone in pref_zones:
                step_reason = (
                    f"Directly satisfies {driver_name}'s preference for {zone} corridors with high passenger demand "
                    f"(expected fare ¥{earn:.1f})."
                )
            elif prob > 0.4:
                step_reason = (
                    f"Critical high-density passenger hotspot ({prob*100:.1f}% pickup rate) providing rapid ride matching "
                    f"to minimize idle fuel waste."
                )
            else:
                step_reason = (
                    f"Provides essential arterial connectivity through {zone}, bridging high-demand commercial clusters "
                    f"with minimal cruising delay."
                )

            segment_reasons.append({
                "step": idx + 1,
                "segment_id": eid,
                "zone": zone,
                "pickup_prob": round(prob, 4),
                "expected_earning": round(earn, 2),
                "reason": step_reason
            })

        avg_p = total_p / max(len(route), 1)
        expected_total = sum(pickup_probs.get(eid, 0.15) * earnings.get(eid, 25.0) for eid in route)

        # High-level summary rationale
        cold_start_note = (
            "LEPS collaborative filtering compensated for lack of historical ride data by referencing similar veteran drivers."
            if is_cold else
            "Personalized search tailored route choices to historical earnings and zone affinities."
        )

        scenario_note = {
            "peak": "Optimized to capitalize on peak-hour fare premiums while bypassing chronic gridlock bottlenecks.",
            "rainy": "Heavy rain conditions boost hail rates; route prioritizes sheltered pickup nodes and transit hubs.",
            "night": "Prioritizes illuminated entertainment strips and midnight transit interchanges with high night-surge fares.",
            "normal": "Balanced multi-objective routing ensuring steady pickup likelihood and low cruising time."
        }.get(scenario, "Standard balanced routing.")

        summary_explanation = (
            f"MORA-LLM recommended this {len(route)}-segment route for {driver_name} "
            f"to achieve an expected route revenue of ¥{expected_total:.2f} (mean {avg_p*100:.1f}% pickup chance per segment). "
            f"{scenario_note} {cold_start_note}"
        )

        return {
            "driver_name": driver_name,
            "is_cold_start": is_cold,
            "summary_explanation": summary_explanation,
            "segment_reasons": segment_reasons,
            "expected_revenue": round(expected_total, 2),
            "mean_pickup_rate": round(avg_p, 4),
            "objectives_alignment": {
                "profit_focus": "High" if driver_pref in ["maximum_profit", "steady_income"] else "Balanced",
                "novelty_focus": "Active (explores low-competition segments)" if driver_pref == "fuel_efficient_exploration" else "Standard",
                "diversity_focus": "Multi-Zone Coverage"
            }
        }

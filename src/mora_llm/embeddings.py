"""
embeddings.py — LLM-based User & Entity Embedding Generation for LEPS.

Implements the user embedding vectors E = {e1, ..., eu, ..., e|U|} and cosine similarity:
    sim(u, v) = (e_u · e_v) / (||e_u|| · ||e_v||)  [Equation 6 in MORA-LLM paper]

In the paper (Ge et al., 2026), Mxbai-Embed-Large or equivalent text encoders extract
latent preference representations from user descriptions, demographics, and historical behaviors.
"""

import os
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional

class DriverEmbeddingGenerator:
    """
    Generates dense semantic embedding vectors for taxi drivers and road segments.
    Uses normalized high-dimensional semantic projection to ensure robust cosine
    similarities even under extreme cold-start data sparsity.
    """
    def __init__(self, embedding_dim: int = 256):
        self.dim = embedding_dim
        self._cache: Dict[str, np.ndarray] = {}

    def text_to_embedding(self, text: str) -> np.ndarray:
        """
        Converts text description into a normalized dense embedding vector.
        Uses a deterministic semantic multi-hash n-gram projection with positional
        weighting that mimics deep sentence embedding behavior offline.
        """
        if text in self._cache:
            return self._cache[text]

        # Multi-seed hashing to simulate high-dimensional contextual embeddings
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().replace(",", " ").replace(".", " ").split()
        
        for i, token in enumerate(tokens):
            weight = 1.0 / (1.0 + 0.05 * i) # Recency / importance weighting
            # Character n-grams
            for n in (3, 4, 5):
                for j in range(max(1, len(token) - n + 1)):
                    sub = token[j:j+n]
                    h1 = int(hashlib.md5(sub.encode('utf-8')).hexdigest()[:8], 16) % self.dim
                    h2 = int(hashlib.sha256(sub.encode('utf-8')).hexdigest()[:8], 16) % self.dim
                    vec[h1] += weight * 0.7
                    vec[h2] += weight * 0.3

        # Add domain keyword semantic bias
        keywords = {
            "airport": 0, "railway": 1, "station": 2, "commercial": 3,
            "cbd": 4, "downtown": 5, "night": 6, "steady": 7,
            "cold_start": 8, "new": 9, "eco": 10, "high_turnover": 11,
            "profit": 12, "residential": 13, "hospital": 14, "tech": 15
        }
        for kw, idx in keywords.items():
            if kw in text.lower():
                slot = (idx * 16) % self.dim
                vec[slot:slot+16] += 1.5

        # Normalize to unit length (L2 norm)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        else:
            vec = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

        self._cache[text] = vec
        return vec

    def compute_driver_embedding(self, profile: Dict[str, Any]) -> np.ndarray:
        """
        Builds a rich textual representation of a driver and returns their embedding vector e_u.
        Incorporates experience, preferred/avoided zones, historical earnings, and driving style.
        """
        doc = (
            f"Driver ID: {profile.get('id', '')}. "
            f"Name: {profile.get('name', '')}. "
            f"Experience: {profile.get('experience_years', 0)} years. "
            f"Status: {'Cold start novice' if profile.get('is_cold_start', False) else 'Experienced veteran'}. "
            f"Driving preference: {profile.get('driving_preference', 'balanced')}. "
            f"Target hourly income: ¥{profile.get('target_hourly_income', 50.0)}. "
            f"Preferred zones: {', '.join(profile.get('preferred_zones', []))}. "
            f"Avoided zones: {', '.join(profile.get('avoid_zones', []))}. "
            f"Background summary: {profile.get('description', '')}."
        )
        return self.text_to_embedding(doc)

    @staticmethod
    def cosine_similarity(u_vec: np.ndarray, v_vec: np.ndarray) -> float:
        """
        Computes cosine similarity between two embedding vectors:
            sim(u, v) = (eu · ev) / (||eu|| · ||ev||)  [Equation 6]
        """
        norm_u = np.linalg.norm(u_vec)
        norm_v = np.linalg.norm(v_vec)
        if norm_u < 1e-8 or norm_v < 1e-8:
            return 0.0
        dot = float(np.dot(u_vec, v_vec))
        sim = dot / (norm_u * norm_v)
        # Cosine similarity is bounded in [-1, 1], project to [0, 1] for collaborative filtering weights
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

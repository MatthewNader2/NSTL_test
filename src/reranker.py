"""
src/reranker.py - Neuro-Symbolic Topological Lattice (NSTL)
Cross-Encoder Joint Scorer for High-Precision Candidate Reranking.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Any, Optional
from log_config import get_logger

logger = get_logger("reranker")


class CrossEncoderReranker:
    """
    Precision Cross-Encoder: Evaluates (prompt, cell_doc) pairs jointly
    to eliminate false-positive vector recall before graph traversal.
    """
    _instance: Optional[CrossEncoderReranker] = None

    @classmethod
    def get_instance(cls) -> CrossEncoderReranker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._model = None
        self._initialized = False
        self._init_model()

    def _init_model(self):
        if self._initialized:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            logger.info("[RERANKER] Loaded CrossEncoder model on CPU.")
        except Exception as e:
            logger.warning(f"[RERANKER WARNING] Could not load CrossEncoder: {e}. Operating in pass-through mode.")
            self._model = None
        self._initialized = True

    def rerank(
        self,
        prompt: str,
        candidates: List[Tuple[float, Any]],
        top_k: int = 20
    ) -> List[Tuple[float, Any]]:
        """Reranks candidate tuples (score, cell) using cross-encoder scores."""
        if not candidates or self._model is None:
            return candidates

        to_rank = candidates[:top_k]
        remaining = candidates[top_k:]

        try:
            pairs = []
            for bi_score, cell in to_rank:
                cid = getattr(cell, "cell_id", "")
                kws = " ".join(getattr(cell, "keywords", []))
                in_sig = getattr(cell, "primary_input", "")
                out_sig = getattr(cell, "primary_output", "")
                doc = f"Cell: {cid}. Keywords: {kws}. Input: {in_sig}, Output: {out_sig}."
                pairs.append([prompt, doc])

            cross_scores = self._model.predict(pairs)

            reranked = []
            for idx, (bi_score, cell) in enumerate(to_rank):
                score_val = float(cross_scores[idx])
                # Sigmoid normalization
                prob = 1.0 / (1.0 + math.exp(-max(min(score_val, 20.0), -20.0)))
                # Harmonic blend of vector retrieval and joint cross-encoder probability
                combined = (bi_score * 0.4) + (prob * 0.6)
                reranked.append((combined, cell))

            reranked.sort(key=lambda x: x[0], reverse=True)
            return reranked + remaining
        except Exception as e:
            logger.error(f"[RERANKER ERROR] Reranking failed: {e}")
            return candidates

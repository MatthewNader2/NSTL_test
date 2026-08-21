import logging
import re
from typing import List, Tuple, Any, Optional

logger = logging.getLogger("CrossEncoderReranker")

class CrossEncoderReranker:
    """
    Lightweight Precision Cross-Encoder Reranker.
    Jointly scores (prompt, candidate_description) pairs over the top recall candidates.
    Bypasses heavy GPU workloads by executing lightweight scoring on CPU.
    """
    def __init__(self):
        self._model = None
        self._initialized = False

    def _init_model(self):
        if self._initialized:
            return
        try:
            from sentence_transformers import CrossEncoder
            # Lightweight ms-marco / MiniLM cross encoder for rapid precision scoring
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            self._initialized = True
            logger.info("[RERANKER INIT] Loaded local cross-encoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2' on CPU.")
        except Exception as e:
            logger.warning(f"[RERANKER INIT WARNING] Could not load CrossEncoder model: {e}. Fallback to lexical alignment.")
            self._model = None
            self._initialized = True

    def rerank(self, prompt: str, scored_candidates: List[Tuple[float, Any]], top_k: int = 20) -> List[Tuple[float, Any]]:
        """
        Reranks top candidate tuples (score, cell) using cross-encoder joint scoring.
        """
        if not scored_candidates:
            return []

        if not self._initialized:
            self._init_model()

        candidates_to_rank = scored_candidates[:top_k]
        remaining_candidates = scored_candidates[top_k:]

        if self._model is not None:
            try:
                pairs = []
                for score, cell in candidates_to_rank:
                    cid = getattr(cell, "cell_id", "")
                    desc = getattr(cell, "description", "") or cid
                    in_t = getattr(getattr(cell, "inputs", {}), "type_name", "")
                    out_t = getattr(getattr(cell, "outputs", {}), "type_name", "")
                    doc = f"ID: {cid}. {desc}. Input: {in_t}, Output: {out_t}."
                    pairs.append([prompt, doc])

                cross_scores = self._model.predict(pairs)
                
                reranked = []
                for idx, (bi_score, cell) in enumerate(candidates_to_rank):
                    c_score = float(cross_scores[idx])
                    # Combine bi-encoder score and cross-encoder score
                    combined_score = (bi_score * 0.4) + (c_score * 0.6)
                    reranked.append((combined_score, cell))

                reranked.sort(key=lambda x: x[0], reverse=True)
                return reranked + remaining_candidates
            except Exception as e:
                logger.warning(f"[RERANKER RUN WARNING] Cross-encoder inference failed: {e}. Returning bi-encoder ranking.")
                return scored_candidates

        # Fallback precision alignment scoring if model is absent
        reranked = []
        prompt_words = set(re.findall(r"[a-zA-Z_]+", prompt.lower()))
        for score, cell in candidates_to_rank:
            cid = getattr(cell, "cell_id", "").lower()
            desc = (getattr(cell, "description", "") or "").lower()
            hits = sum(1 for w in prompt_words if len(w) > 2 and (w in cid or w in desc))
            boost = (hits / max(len(prompt_words), 1)) * 0.3
            reranked.append((score + boost, cell))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked + remaining_candidates

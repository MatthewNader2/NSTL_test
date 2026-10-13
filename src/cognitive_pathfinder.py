"""
src/cognitive_pathfinder.py - Neuro-Symbolic Topological Lattice (NSTL)
Pillar 3: Cognitive Latent Pathfinding.

Extracts continuous latent representations from local sub-1B models
(EmbeddingGemma-300M, Qwen2.5-Coder-0.5B, Jina-Nano) and guides A* / beam
search over the lattice DAG via latent geodesic alignment and directional
progress projection.
"""

from __future__ import annotations
import os
import threading
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

try:
    from log_config import get_logger
except ImportError:
    import logging
    get_logger = logging.getLogger

try:
    from .config import MODELS_DIR
except (ImportError, ValueError):
    from config import MODELS_DIR

logger = get_logger("cognitive_pathfinder")


class LatentVectorExtractor:
    """
    Thread-safe continuous latent vector extractor using local sub-1B models.
    Supports SentenceTransformer (EmbeddingGemma-300M, Jina-Nano) and llama.cpp GGUF.
    """
    _instance: Optional[LatentVectorExtractor] = None
    _lock = threading.RLock()

    def __init__(self, model_name: Optional[str] = None):
        self._lock = threading.RLock()
        self._cache: Dict[str, np.ndarray] = {}
        self._model = None
        self._backend = None
        self._dim = 768
        self._init_model(model_name)

    @classmethod
    def get_instance(cls) -> LatentVectorExtractor:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _init_model(self, model_name: Optional[str] = None):
        emb_dir = os.path.join(MODELS_DIR, "embeddings")
        llm_dir = os.path.join(MODELS_DIR, "llms")

        # 1. Try SentenceTransformer with local sub-1B models
        candidates = [
            os.path.join(emb_dir, "embeddinggemma-300m"),
            os.path.join(emb_dir, "jina-embeddings-v5-text-nano"),
            os.path.join(emb_dir, "jina-embeddings-v5-text-small"),
        ]

        if model_name:
            candidates.insert(0, os.path.join(emb_dir, model_name))
            candidates.insert(0, model_name)

        for path in candidates:
            if os.path.isdir(path):
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._model = SentenceTransformer(path, device=device)
                    self._backend = "sentence_transformers"
                    sample = self._model.encode("probe", normalize_embeddings=True)
                    self._dim = int(sample.shape[-1])
                    logger.info(f"[COGNITIVE PATHFINDER] Loaded '{os.path.basename(path)}' on {device} (dim={self._dim})")
                    return
                except Exception as e:
                    logger.debug(f"[COGNITIVE PATHFINDER] Failed loading {path}: {e}")

        # 2. Fallback to llama.cpp GGUF embedding
        qwen_gguf = os.path.join(llm_dir, "qwen2.5-coder-0.5b-instruct", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
        if os.path.isfile(qwen_gguf):
            try:
                from llama_cpp import Llama
                self._model = Llama(model_path=qwen_gguf, embedding=True, verbose=False, n_gpu_layers=0)
                self._backend = "llama_cpp"
                self._dim = 896
                logger.info(f"[COGNITIVE PATHFINDER] Loaded Qwen-0.5B GGUF embedding on CPU (dim={self._dim})")
                return
            except Exception as e:
                logger.debug(f"[COGNITIVE PATHFINDER] Failed loading GGUF: {e}")

        # 3. Deterministic hash-based projection fallback
        self._backend = "deterministic_hash"
        self._dim = 384
        logger.warning("[COGNITIVE PATHFINDER] No neural models available; using deterministic pseudo-latent fallback")

    def encode(self, text: str) -> np.ndarray:
        """Encodes text to a unit-normalized latent vector in R^d."""
        if not text:
            return np.zeros(self._dim, dtype=np.float32)

        with self._lock:
            if text in self._cache:
                return self._cache[text]

            if self._backend == "sentence_transformers" and self._model is not None:
                vec = self._model.encode(text, normalize_embeddings=True)
                norm_vec = np.array(vec, dtype=np.float32)

            elif self._backend == "llama_cpp" and self._model is not None:
                res = self._model.create_embedding(text)
                raw = np.array(res["data"][0]["embedding"], dtype=np.float32)
                pooled = np.mean(raw, axis=0) if raw.ndim > 1 else raw
                norm = float(np.linalg.norm(pooled))
                norm_vec = pooled / (norm + 1e-9)

            else:
                import hashlib
                h = hashlib.sha256(text.encode("utf-8")).digest()
                rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
                rnd = rng.randn(self._dim).astype(np.float32)
                norm_vec = rnd / (float(np.linalg.norm(rnd)) + 1e-9)

            self._cache[text] = norm_vec
            return norm_vec

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encodes a batch of texts to unit-normalized latent vectors."""
        if not texts:
            return []

        uncached_indices = []
        uncached_texts = []
        results: List[Optional[np.ndarray]] = [None] * len(texts)

        with self._lock:
            for i, t in enumerate(texts):
                if t in self._cache:
                    results[i] = self._cache[t]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(t)

            if uncached_texts:
                if self._backend == "sentence_transformers" and self._model is not None:
                    vecs = self._model.encode(uncached_texts, normalize_embeddings=True)
                    for idx, vec in zip(uncached_indices, vecs):
                        norm_vec = np.array(vec, dtype=np.float32)
                        self._cache[texts[idx]] = norm_vec
                        results[idx] = norm_vec
                else:
                    for idx in uncached_indices:
                        results[idx] = self.encode(texts[idx])

        return [r if r is not None else np.zeros(self._dim, dtype=np.float32) for r in results]


class CognitivePathfinder:
    """
    Cognitive Latent Pathfinder.
    Guides search across the lattice morphism DAG using latent space geodesic alignment.
    """

    def __init__(self, extractor: Optional[LatentVectorExtractor] = None):
        self.extractor = extractor or LatentVectorExtractor.get_instance()
        self._cell_vec_cache: Dict[str, np.ndarray] = {}

    def get_cell_representation_text(self, cell: Any) -> str:
        """Derives a rich semantic string for a cell covering identity, keywords, inputs, and outputs."""
        cid = getattr(cell, "cell_id", "")
        keywords = " ".join(getattr(cell, "keywords", []) or [])
        summary = getattr(cell, "summary", "") or ""
        doc = getattr(cell, "documentation", "") or getattr(cell, "docstring", "") or ""
        code = (getattr(cell, "code_template", "") or "")[:120]
        in_types = " ".join(
            str(getattr(getattr(p, "signature", None), "type_name", "") or getattr(p, "type_name", ""))
            for p in getattr(cell, "inputs", {}).values()
        )
        out_types = str(
            getattr(getattr(getattr(cell, "primary_output", None), "signature", None), "type_name", "")
            or getattr(getattr(cell, "primary_output", None), "type_name", "")
        )
        return f"{cid} {keywords} {summary} {doc} in:{in_types} out:{out_types} {code}".strip()

    def get_cell_vector(self, cell: Any) -> np.ndarray:
        """Retrieves or computes the unit-normalized latent vector for a cell."""
        cid = getattr(cell, "cell_id", id(cell))
        if cid in self._cell_vec_cache:
            return self._cell_vec_cache[cid]

        text = self.get_cell_representation_text(cell)
        vec = self.extractor.encode(text)
        self._cell_vec_cache[cid] = vec
        return vec

    def compute_latent_heuristic(
        self,
        current_vec: np.ndarray,
        candidate_vec: np.ndarray,
        goal_vec: np.ndarray
    ) -> float:
        """
        Computes the latent geodesic heuristic H(current, candidate, goal).
        Combines:
          1. Direct cosine similarity with goal: sim(cand, goal)
          2. Directional delta progress: projection of (cand - current) along (goal - current).
        """
        sim = float(np.dot(candidate_vec, goal_vec))

        delta_goal = goal_vec - current_vec
        delta_step = candidate_vec - current_vec
        norm_dg = float(np.linalg.norm(delta_goal))

        if norm_dg > 1e-6:
            unit_goal = delta_goal / norm_dg
            progress = float(np.dot(delta_step, unit_goal))
        else:
            progress = sim

        clamped_sim = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        clamped_prog = max(0.0, min(1.0, (progress + 1.0) / 2.0))

        return 0.5 * clamped_sim + 0.5 * clamped_prog

    def rank_candidates(
        self,
        candidates: List[Any],
        current_vec: np.ndarray,
        goal_vec: np.ndarray,
        relevance_map: Optional[Dict[str, float]] = None,
        alpha: float = 0.7,
        beta: float = 0.3
    ) -> List[Tuple[Any, float]]:
        """
        Ranks candidate morphism cells by combining the latent geodesic heuristic
        with prior semantic relevance scores.
        """
        scored: List[Tuple[Any, float]] = []
        for cell in candidates:
            c_vec = self.get_cell_vector(cell)
            h_score = self.compute_latent_heuristic(current_vec, c_vec, goal_vec)
            r_score = (relevance_map or {}).get(getattr(cell, "cell_id", ""), 0.0)
            composite = alpha * h_score + beta * r_score
            scored.append((cell, composite))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


__all__ = [
    "LatentVectorExtractor",
    "CognitivePathfinder",
]

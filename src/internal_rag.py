"""
src/internal_rag.py - Neuro-Symbolic Topological Lattice (NSTL)
Local Vector Index (FAISS) and Incremental Embedding Cache.
"""

from __future__ import annotations
import os
import json
import hashlib
import pickle
import threading
from typing import Optional, Dict, Any, List, Tuple

import faiss
import numpy as np
from log_config import get_logger
from inference import ModelManager

logger = get_logger("internal_rag")
_CACHE_DIR_NAME = ".rag_cache"


class LocalRAG:
    """
    Maintains a dense vector index over all lattice cells for sub-millisecond
    semantic tunneling and context retrieval.
    """
    def __init__(self, trees_dir: str, orchestrator=None):
        self.trees_dir = trees_dir
        self.orchestrator = orchestrator
        self._cache_dir = os.path.join(os.path.dirname(trees_dir), _CACHE_DIR_NAME)

        model_mgr = ModelManager.get_instance()
        if model_mgr.profile is None:
            raise RuntimeError(
                "LocalRAG requires ModelManager to be initialized with a profile before construction."
            )

        self.dimension = model_mgr.embedding_dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_to_schema: Dict[int, Dict[str, Any]] = {}
        self.cell_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        self.build_index()

    def _get_cache_path(self) -> str:
        profile = ModelManager.get_instance().active_profile
        emb_name = getattr(profile, "embedder_name", "default")
        dim = getattr(profile, "_dim", self.dimension)
        safe_name = "".join(c for c in f"{emb_name}__dim{dim}" if c.isalnum() or c in "._-")
        return os.path.join(self._cache_dir, f"{safe_name}_cache.pkl")

    def _load_cache(self):
        path = self._get_cache_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.cell_cache = pickle.load(f)
                logger.info(f"[RAG CACHE] Loaded {len(self.cell_cache)} records from {path}")
            except Exception as e:
                logger.warning(f"[RAG CACHE] Failed to load cache: {e}")
                self.cell_cache = {}
        else:
            self.cell_cache = {}

    def _save_cache(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        path = self._get_cache_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(self.cell_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"[RAG CACHE] Failed to save cache: {e}")

    def build_index(self):
        """Constructs or updates the FAISS index incrementally from loaded cells."""
        with self._lock:
            self._load_cache()
            if self.orchestrator is None:
                return

            cells = list(self.orchestrator.loaded_cells.values())
            seen_ids = set()
            new_or_changed = []

            for cell in cells:
                cid = cell.cell_id
                seen_ids.add(cid)

                kws = " ".join(sorted(cell.keywords))
                in_sig = f"{cell.primary_input.type_name}[{cell.primary_input.state}]"
                out_sig = f"{cell.primary_output.type_name}[{cell.primary_output.state}]"
                text_repr = f"ID: {cid} | Keywords: {kws} | Flow: {in_sig} -> {out_sig} | Domain: {cell.domain_name}"
                content_hash = hashlib.sha256(text_repr.encode("utf-8")).hexdigest()

                schema = {
                    "cell_id": cid,
                    "type": cell.cell_type,
                    "stage": cell.stage,
                    "keywords": sorted(cell.keywords),
                    "domain": cell.domain_name,
                    "primary_input": cell.primary_input.type_name,
                    "primary_output": cell.primary_output.type_name
                }

                if cid in self.cell_cache and self.cell_cache[cid].get("hash") == content_hash:
                    self.cell_cache[cid]["schema"] = schema
                else:
                    new_or_changed.append({"cell_id": cid, "text": text_repr, "hash": content_hash, "schema": schema})

            # Evict removed cells
            for cid in list(self.cell_cache.keys()):
                if cid not in seen_ids:
                    del self.cell_cache[cid]

            # Embed new/modified cells
            if new_or_changed:
                logger.info(f"[RAG] Batch-embedding {len(new_or_changed)} new/changed cells...")
                texts = [item["text"] for item in new_or_changed]
                embeddings = ModelManager.get_instance().get_embeddings(texts)

                for item, emb in zip(new_or_changed, embeddings):
                    self.cell_cache[item["cell_id"]] = {
                        "hash": item["hash"],
                        "embedding": emb,
                        "schema": item["schema"]
                    }
                self._save_cache()

            if not self.cell_cache:
                self.index = None
                self.id_to_schema.clear()
                return

            # Build FAISS normalized flat inner-product index
            all_embs = []
            self.id_to_schema.clear()
            for idx, (cid, data) in enumerate(self.cell_cache.items()):
                all_embs.append(data["embedding"])
                self.id_to_schema[idx] = data["schema"]

            matrix = np.array(all_embs, dtype=np.float32)
            self.dimension = matrix.shape[1]

            # L2 Normalization (Cosine Similarity)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            matrix = matrix / norms

            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(matrix)
            logger.info(f"[RAG] FAISS Index ready with {self.index.ntotal} vectors.")

    def add_dynamic_cell(self, cell_dict: Dict[str, Any]):
        """Appends a newly synthesized cell dynamically to the active FAISS index."""
        with self._lock:
            if self.index is None:
                return

            cid = cell_dict.get("cell_id", "dynamic_cell")
            kws = " ".join(cell_dict.get("keywords", []))
            in_t = cell_dict.get("inputs", {}).get("type_name", "any")
            out_t = cell_dict.get("outputs", {}).get("type_name", "any")
            text_repr = f"ID: {cid} | Keywords: {kws} | Flow: {in_t} -> {out_t}"

            raw_emb = np.array([ModelManager.get_instance().get_embedding(text_repr)], dtype=np.float32)
            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                raw_emb = raw_emb / norm

            self.index.add(raw_emb)
            new_idx = len(self.id_to_schema)
            self.id_to_schema[new_idx] = cell_dict
            logger.info(f"[RAG] Dynamically indexed synthesized cell: {cid}")

    def get_relevant_context(self, prompt: str, top_k: int = 25) -> str:
        """Retrieves formatted context lines for the top-k most semantically aligned cells."""
        if self.index is None or self.index.ntotal == 0:
            return "No verified cells loaded."

        raw_emb = np.array([ModelManager.get_instance().get_embedding(prompt)], dtype=np.float32)
        norm = np.linalg.norm(raw_emb)
        if norm == 0:
            return "Embedding failure."
        raw_emb = raw_emb / norm

        search_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(raw_emb, search_k)

        lines = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx not in self.id_to_schema:
                continue
            schema = self.id_to_schema[idx]
            lines.append(
                f"- ID: {schema.get('cell_id')} | "
                f"In: {schema.get('primary_input', 'any')} -> "
                f"Out: {schema.get('primary_output', 'any')} | "
                f"Domain: {schema.get('domain', 'generic')}"
            )
        return "\n".join(lines)

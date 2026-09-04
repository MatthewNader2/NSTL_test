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


def build_cell_embedding_text(cell: Any) -> str:
    """Builds a rich semantic representation of a cell for dense vector embedding.

    Includes:
    - Natural language description / docstring
    - Human-readable operation name from cell ID
    - Canonical code template / callable API
    - Combined semantic tags and keywords (synonyms)
    - Input & output typestate data flow contract
    - Input parameter names and descriptions
    - Domain & library context
    """
    parts = []

    # 1. Functional description / docstring
    desc = (getattr(cell, "docstring", "") or "").strip() if not isinstance(cell, dict) else (cell.get("docstring", "") or "").strip()
    if desc:
        parts.append(f"Description: {desc}")

    # 2. Human-readable operation name from cell_id
    cid = getattr(cell, "cell_id", "") if not isinstance(cell, dict) else cell.get("cell_id", "")
    clean_name = cid.replace("_", " ").lower()
    if clean_name:
        parts.append(f"Operation: {clean_name}")

    # 3. Canonical code template (gives exact API call, e.g. .dropna(), pd.read_csv)
    code = (getattr(cell, "code_template", "") or "").strip() if not isinstance(cell, dict) else (cell.get("code_template", "") or "").strip()
    if code:
        parts.append(f"Code: {code}")

    # 4. Semantic tags and keywords (synonym vocabulary: clean, remove, drop, null, filter, etc.)
    all_terms = set()
    raw_tags = getattr(cell, "semantic_tags", []) if not isinstance(cell, dict) else cell.get("semantic_tags", [])
    for t in raw_tags or []:
        if t:
            all_terms.add(str(t).lower())
    raw_kws = getattr(cell, "keywords", []) if not isinstance(cell, dict) else cell.get("keywords", [])
    for k in raw_kws or []:
        if k:
            all_terms.add(str(k).lower())
    if all_terms:
        parts.append(f"Keywords: {', '.join(sorted(all_terms))}")

    # 5. Typestate data flow
    if hasattr(cell, "primary_input") and hasattr(cell, "primary_output"):
        in_sig = f"{cell.primary_input.type_name}[{cell.primary_input.state}]"
        out_sig = f"{cell.primary_output.type_name}[{cell.primary_output.state}]"
        parts.append(f"Flow: {in_sig} -> {out_sig}")
    elif isinstance(cell, dict):
        in_t = cell.get("inputs", {}).get("type_name", "any") if isinstance(cell.get("inputs"), dict) else "any"
        out_t = cell.get("outputs", {}).get("type_name", "any") if isinstance(cell.get("outputs"), dict) else "any"
        parts.append(f"Flow: {in_t} -> {out_t}")

    # 6. Inputs parameter details
    inputs = getattr(cell, "inputs", {}) if not isinstance(cell, dict) else cell.get("inputs", {})
    if isinstance(inputs, dict) and inputs:
        in_descs = []
        for p_name, p_sig in inputs.items():
            if isinstance(p_sig, dict):
                t_name = p_sig.get("type_name", "any")
            else:
                t_name = getattr(p_sig, "type_name", "any")
            in_descs.append(f"{p_name}: {t_name}")
        parts.append(f"Inputs: {', '.join(in_descs)}")

    # 7. Domain name
    domain = getattr(cell, "domain_name", "") if not isinstance(cell, dict) else (cell.get("domain_name") or cell.get("domain", ""))
    if domain:
        parts.append(f"Domain: {domain}")

    return " | ".join(parts) if parts else cid


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
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self.cell_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning(f"[RAG CACHE] Failed to save cache: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

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

                desc = (getattr(cell, "docstring", "") or "").strip()
                text_repr = build_cell_embedding_text(cell)
                content_hash = hashlib.sha256(text_repr.encode("utf-8")).hexdigest()

                schema = {
                    "cell_id": cid,
                    "type": cell.cell_type,
                    "stage": cell.stage,
                    "keywords": sorted(cell.keywords),
                    "domain": cell.domain_name,
                    "docstring": desc,
                    "enrichment_source": getattr(cell, "enrichment_source", None),
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

            # Embed new/modified cells incrementally to avoid VRAM exhaustion
            if new_or_changed:
                logger.info(f"[RAG] Batch-embedding {len(new_or_changed)} new/changed cells in chunks...")
                chunk_size = 1000
                total_cnt = len(new_or_changed)
                for i in range(0, total_cnt, chunk_size):
                    chunk = new_or_changed[i:i + chunk_size]
                    texts = [item["text"] for item in chunk]
                    embeddings = ModelManager.get_instance().get_embeddings(texts)
                    for item, emb in zip(chunk, embeddings):
                        self.cell_cache[item["cell_id"]] = {
                            "hash": item["hash"],
                            "embedding": emb,
                            "schema": item["schema"]
                        }
                    self._save_cache()
                    print(f"[*] RAG Embedding Progress: {min(i + chunk_size, total_cnt)} / {total_cnt} ({min(i + chunk_size, total_cnt)/total_cnt*100:.1f}%)")
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

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
            text_repr = build_cell_embedding_text(cell_dict)

            raw_emb = np.array([ModelManager.get_instance().get_embedding(text_repr)], dtype=np.float32)
            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                raw_emb = raw_emb / norm

            self.index.add(raw_emb)
            new_idx = len(self.id_to_schema)
            self.id_to_schema[new_idx] = cell_dict
            logger.info(f"[RAG] Dynamically indexed synthesized cell: {cid}")

    def get_relevant_context(self, prompt: str, top_k: int = 25) -> List[Dict[str, Any]]:
        """Retrieves structured context list for the top-k most semantically aligned cells."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
    
            raw_emb = np.array([ModelManager.get_instance().get_embedding(prompt)], dtype=np.float32)
            norm = np.linalg.norm(raw_emb)
            if norm == 0:
                return []
            raw_emb = raw_emb / norm
    
            search_k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(raw_emb, search_k)
    
            results: List[Dict[str, Any]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx not in self.id_to_schema:
                    continue
                schema = self.id_to_schema[idx]
                cid = schema.get("cell_id", "")
                results.append({
                    "cell_id": cid,
                    "score": float(dist),
                    "schema": schema,
                    "domain": schema.get("domain", "generic"),
                    "primary_input": schema.get("primary_input", "any"),
                    "primary_output": schema.get("primary_output", "any"),
                    "text": f"ID: {cid} | In: {schema.get('primary_input', 'any')} -> Out: {schema.get('primary_output', 'any')} | Domain: {schema.get('domain', 'generic')}"
                })
            return results

    def format_context_for_prompt(self, context_items: List[Dict[str, Any]]) -> str:
        """Formats structured context list into a prompt-friendly string for LLMs."""
        if not context_items:
            return "No verified cells available."
        lines = []
        for item in context_items:
            lines.append(
                f"- ID: {item.get('cell_id')} | "
                f"In: {item.get('primary_input', 'any')} -> "
                f"Out: {item.get('primary_output', 'any')} | "
                f"Domain: {item.get('domain', 'generic')}"
            )
        return "\n".join(lines)

    def find_closest_cell_by_embedding(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Finds the single closest cell by cosine similarity."""
        with self._lock:
            results = self.get_relevant_context(prompt, top_k=1)
            return results[0] if results else None

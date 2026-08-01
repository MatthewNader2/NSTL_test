import os
import json
import hashlib
import logging
import pickle
import threading
import faiss
import numpy as np
from inference import ModelManager
from router import HardwareProfiler

# Cache lives next to the trees directory, inside a hidden .rag_cache folder
_CACHE_DIR_NAME = ".rag_cache"

class LocalRAG:
    def __init__(self, trees_dir: str):
        self.trees_dir = trees_dir
        self.logger = logging.getLogger("LocalRAG")
        self._cache_dir = os.path.join(os.path.dirname(trees_dir), _CACHE_DIR_NAME)

        # BUG 20 FIX: Explicitly guard that ModelManager has a profile before building
        # the index. The dimension depends on the active profile, and get_embedding()
        # raises RuntimeError if no profile is initialized.
        if ModelManager.get_instance().profile is None:
            raise RuntimeError(
                "LocalRAG requires ModelManager to be initialized with a profile before construction. "
                "Call ModelManager.get_instance().initialize_profile(...) first."
            )

        self.dimension = ModelManager.get_instance().embedding_dimension
        self.index = None
        self.id_to_schema = {}
        self.cell_cache = {}
        self._index_lock = threading.Lock()  # Protects concurrent FAISS index mutations

        self.build_index()

    # ------------------------------------------------------------------
    # Incremental Cell Cache
    # ------------------------------------------------------------------
    def _get_model_cache_path(self):
        try:
            model_name = ModelManager.get_instance().active_profile.embedder_name
        except Exception:
            model_name = "unknown_model"
        # Sanitize model name
        model_name = "".join(c for c in str(model_name) if c.isalnum() or c in "._-")
        return os.path.join(self._cache_dir, f"{model_name}_rag_cache.pkl")

    def _load_cell_cache(self):
        cache_path = self._get_model_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.cell_cache = pickle.load(f)
                self.logger.info(f"[RAG CACHE] Loaded cell cache from {cache_path}")
            except Exception as e:
                self.logger.warning(f"[RAG CACHE] Failed to load cell cache: {e}")
                self.cell_cache = {}
        else:
            self.cell_cache = {}

    def _save_cell_cache(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        cache_path = self._get_model_cache_path()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.cell_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"[RAG CACHE] Saved cell cache to {cache_path}")
        except Exception as e:
            self.logger.warning(f"[RAG CACHE] Failed to save cell cache: {e}")

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------
    def build_index(self):
        self.logger.info("Building FAISS index for local RAG (incremental)...")
        self._load_cell_cache()

        dirs_to_scan = []
        if os.path.exists(self.trees_dir):
            dirs_to_scan.append(self.trees_dir)
        for subdir in ["macro", "micro"]:
            sub_path = os.path.join(self.trees_dir, subdir)
            if os.path.exists(sub_path):
                dirs_to_scan.append(sub_path)

        seen_cell_ids = set()
        new_or_changed_cells = []
        valid_cached_cells = []

        for dir_path in dirs_to_scan:
            for root, subdirs, files in os.walk(dir_path):
                if dir_path == self.trees_dir:
                    subdirs.clear()   # Don't recurse into subdirs here (handled above)
                
                # Sort files to prioritize _domain.json over _auto.json, etc.
                sorted_files = sorted(files, key=lambda f: (not f.endswith('_domain.json'), f))
                
                for file in sorted_files:
                    if not file.endswith(".json"):
                        continue
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        cells = data.get("cells", [data]) if isinstance(data, dict) else data
                        if not isinstance(cells, list):
                            cells = [cells]
                        for cell in cells:
                            if not isinstance(cell, dict):
                                continue
                            cell_id = cell.get("cell_id", "")
                            if cell_id in seen_cell_ids:
                                continue
                            seen_cell_ids.add(cell_id)

                            # Semantic string representation
                            kws     = " ".join(sorted(cell.get("keywords", [])))
                            inputs  = cell.get("inputs", {})
                            outputs = cell.get("outputs", {})
                            
                            in_str  = (
                                f"{inputs.get('type_name', inputs.get('input_type', ''))}"
                                if isinstance(inputs, dict) else str(inputs)[:50]
                            )
                            out_str = (
                                f"{outputs.get('type_name', outputs.get('output_type', ''))}"
                                if isinstance(outputs, dict) else str(outputs)[:50]
                            )
                            description = cell.get("description", "")
                            text_repr = f"ID: {cell_id} | Keywords: {kws} | Flow: {in_str} -> {out_str} | Description: {description}"
                            
                            # Hash for change detection
                            cell_hash = hashlib.md5(text_repr.encode('utf-8')).hexdigest()

                            if cell_id in self.cell_cache and self.cell_cache[cell_id].get("hash") == cell_hash:
                                valid_cached_cells.append(cell_id)
                                self.cell_cache[cell_id]["schema"] = cell # Update schema in case non-indexed fields changed
                            else:
                                new_or_changed_cells.append({
                                    "cell_id": cell_id,
                                    "text": text_repr,
                                    "hash": cell_hash,
                                    "schema": cell
                                })
                    except Exception as e:
                        self.logger.error(f"Error parsing {file}: {e}")

        # Remove deleted cells from cache
        cache_keys = list(self.cell_cache.keys())
        for cid in cache_keys:
            if cid not in seen_cell_ids:
                del self.cell_cache[cid]

        # Embed new or changed cells
        if new_or_changed_cells:
            self.logger.info(f"[RAG] Embedding {len(new_or_changed_cells)} new/changed cells...")
            print(f"  [RAG] Embedding {len(new_or_changed_cells)} new/changed cells...")
            texts_to_embed = [item["text"] for item in new_or_changed_cells]
            embeddings_list = ModelManager.get_instance().get_embeddings(texts_to_embed)
            
            for i, item in enumerate(new_or_changed_cells):
                cid = item["cell_id"]
                self.cell_cache[cid] = {
                    "hash": item["hash"],
                    "embedding": embeddings_list[i],
                    "schema": item["schema"]
                }
            self._save_cell_cache()
        else:
            self.logger.info(f"[RAG] All {len(valid_cached_cells)} cells loaded from cache.")
            print(f"  [RAG] Cache hit — all {len(valid_cached_cells)} cells loaded from cache instantly.")

        if not self.cell_cache:
            self.logger.warning("No nodes found to build FAISS index.")
            self.index = None
            self.id_to_schema = {}
            return

        # Build FAISS index from cell_cache
        all_embeddings = []
        self.id_to_schema = {}
        for idx, (cid, data) in enumerate(self.cell_cache.items()):
            all_embeddings.append(data["embedding"])
            self.id_to_schema[idx] = data["schema"]

        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # Unconditionally sync the RAG dimension to whatever the model actually produced
        self.dimension = embeddings.shape[1]

        # Normalize for cosine similarity via L2 index
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # Initialize flat Inner Product index with the correct dynamic dimension
        cpu_index = faiss.IndexFlatIP(self.dimension)
        
        # Bypass GPU mapping because it causes StackDeviceMemory assertions in concurrent runs
        self.index = cpu_index
        self.logger.info("FAISS index built and kept on CPU for stability.")
            
        self.index.add(embeddings)

        self.logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def add_dynamic_cell(self, cell_dict: dict):
        """Dynamically embeds and adds a single synthesized cell to the active FAISS index."""
        if self.index is None:
            return
            
        cell_id = cell_dict.get("cell_id", "")
        kws = " ".join(cell_dict.get("keywords", []))
        inputs = cell_dict.get("inputs", {})
        outputs = cell_dict.get("outputs", {})

        in_str = f"{inputs.get('type_name', inputs.get('input_type', ''))}" if isinstance(inputs, dict) else str(inputs)[:50]
        out_str = f"{outputs.get('type_name', outputs.get('output_type', ''))}" if isinstance(outputs, dict) else str(outputs)[:50]
        schema_text = f"ID: {cell_id} | Keywords: {kws} | Flow: {in_str} -> {out_str}"
        cell_hash = hashlib.md5(schema_text.encode('utf-8')).hexdigest()
        
        # 2. Get embedding
        raw_emb = np.array(
            [ModelManager.get_instance().get_embedding(schema_text)], dtype=np.float32
        )
        
        # 3. Normalize
        norm = np.linalg.norm(raw_emb, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        raw_emb = raw_emb / norm
        
        # 4. Add to index (thread-safe)
        with self._index_lock:
            self.index.add(raw_emb)
            
            # 5. Add to schema lookup
            new_idx = len(self.id_to_schema)
            self.id_to_schema[new_idx] = cell_dict
            
            # Update cell_cache
            self.cell_cache[cell_id] = {
                "hash": cell_hash,
                "embedding": raw_emb[0].tolist(),
                "schema": cell_dict
            }
        
        # 6. Save cache to persist
        self._save_cell_cache()
        self.logger.info(f"Dynamically added synthesized node {cell_id} to FAISS index.")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_relevant_context(self, prompt: str, top_k: int = 15) -> str:
        if self.index is None or self.index.ntotal == 0:
            return "No available micro-nodes in trees directory."

        raw_emb = np.array(
            [ModelManager.get_instance().get_embedding(prompt)], dtype=np.float32
        )
        norm = np.linalg.norm(raw_emb)
        if norm == 0:
            self.logger.warning(f"Embedding returned zero vector for prompt: {prompt[:80]}... — skipping FAISS query.")
            return "Embedding failed for the given prompt. Cannot retrieve context."
        raw_emb = raw_emb / norm

        # Search deeper for semantic gravity re-ranking
        search_k = min(top_k * 10, self.index.ntotal)
        distances, indices = self.index.search(raw_emb, search_k)
        
        import re
        prompt_tokens = set(re.findall(r"[a-zA-Z_]+", prompt.lower()))

        scored_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx not in self.id_to_schema:
                continue
            cell = self.id_to_schema[idx]
            
            # Semantic Gravity: Boost by adding to IP distance based on keyword matches
            kws = {kw.lower() for kw in cell.get("keywords", [])}
            id_parts = {p for p in re.split(r"[_\W]+", cell.get("cell_id", "").lower()) if p}
            
            overlap = len(prompt_tokens.intersection(kws)) * 0.2 + len(prompt_tokens.intersection(id_parts)) * 0.1
            adjusted_dist = dist + overlap
            
            scored_results.append((adjusted_dist, cell))
            
        # Re-sort by adjusted distance (larger is better for IP)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for _, cell in scored_results[:top_k]:
            in_dict   = cell.get("inputs", {})
            out_dict  = cell.get("outputs", {})
            in_type   = in_dict.get("type_name",  in_dict.get("input_type",  ""))
            in_state  = in_dict.get("state",       in_dict.get("expected_state", ""))
            out_type  = out_dict.get("type_name",  out_dict.get("output_type",  ""))
            out_state = out_dict.get("state",       out_dict.get("resulting_state", ""))
            results.append(
                f"- ID: {cell.get('cell_id')} | "
                f"Inputs: {in_type}[{in_state}] -> "
                f"Outputs: {out_type}[{out_state}]"
            )

        return "\n".join(results)

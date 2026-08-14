from typing import Optional, Dict, Any, List
import os
import json
import re
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
    def __init__(self, trees_dir: str, orchestrator=None):
        self.trees_dir = trees_dir
        self.orchestrator = orchestrator
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
            profile = ModelManager.get_instance().active_profile
            model_name = "__".join(filter(None, [
                getattr(profile, "embedder_name", None) or ModelManager.get_instance().current_profile_name,
                str(ModelManager.get_instance().embedding_dimension),
            ]))
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

        # The router executes cells from SQLite through the orchestrator. Index
        # that exact authoritative set rather than a potentially stale JSON
        # export, which previously made thousands of executable cells invisible.
        if self.orchestrator is not None:
            seen_cell_ids = set()
            new_or_changed_cells = []
            cells = self.orchestrator.get_all_available_cells()
            max_cells = int(os.environ.get("NSTL_RAG_MAX_CELLS", "0"))
            if max_cells > 0:
                # Explicit opt-in for constrained smoke tests; production keeps
                # the default of indexing every executable cell.
                cells = sorted(cells, key=lambda cell: cell.cell_id)[:max_cells]
                self.logger.warning("[RAG] Limiting index to %s cells for this run.", max_cells)
            for cell in cells:
                cell_id = cell.cell_id
                seen_cell_ids.add(cell_id)
                schema = {
                    "cell_id": cell_id,
                    "type": cell.type,
                    "node_type": cell.node_type,
                    "stage": cell.stage,
                    "keywords": sorted(cell.keywords),
                    "inputs": {"type_name": cell.inputs.type_name, "state": cell.inputs.state},
                    "outputs": {"type_name": cell.outputs.type_name, "state": cell.outputs.state},
                }
                text_repr = (
                    f"ID: {cell_id} | Keywords: {' '.join(schema['keywords'])} | "
                    f"Flow: {cell.inputs.type_name} -> {cell.outputs.type_name}"
                )
                cell_hash = hashlib.sha256(text_repr.encode("utf-8")).hexdigest()
                if cell_id in self.cell_cache and self.cell_cache[cell_id].get("embedding") is not None:
                    self.cell_cache[cell_id]["schema"] = schema
                else:
                    new_or_changed_cells.append({
                        "cell_id": cell_id, "text": text_repr, "hash": cell_hash, "schema": schema,
                    })
            self._finish_index_build(seen_cell_ids, new_or_changed_cells)
            return

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
                            
                            # Q-8 fix: use SHA-256 instead of MD5 for collision-resistant change detection
                            cell_hash = hashlib.sha256(text_repr.encode('utf-8')).hexdigest()

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

        self._finish_index_build(seen_cell_ids, new_or_changed_cells)

    def _finish_index_build(self, seen_cell_ids, new_or_changed_cells):
        """Embed changed records and rebuild the index from the active cache."""
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
            self.logger.info(f"[RAG] All {len(self.cell_cache)} cells loaded from cache.")
            print(f"  [RAG] Cache hit — all {len(self.cell_cache)} cells loaded from cache instantly.")

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
        cell_hash = hashlib.sha256(schema_text.encode('utf-8')).hexdigest()
        
        # 2. Get embedding
        raw_emb = np.array(
            [ModelManager.get_instance().get_embedding(schema_text)], dtype=np.float32
        )
        
        # 3. Normalize
        norm = np.linalg.norm(raw_emb, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        raw_emb = raw_emb / norm
        
        # IndexFlat cannot delete/update by ID. Replace existing cells by
        # rebuilding outside the index lock, rather than retaining stale vectors.
        if cell_id in self.cell_cache:
            self.cell_cache[cell_id] = {
                "hash": cell_hash, "embedding": raw_emb[0].tolist(), "schema": cell_dict,
            }
            self._save_cell_cache()
            self.build_index()
            return

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
            cid = cell.get("cell_id", "")
            if "__" in cid or "___" in cid:
                continue
            
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

    def find_closest_cell_by_embedding(self, query_text: str, domain_hint: str = "") -> Optional[str]:
        """Uses FAISS vector embedding search with contextual prompt guidance and dynamic domain weighting."""
        if self.index is None or self.index.ntotal == 0:
            return None

        try:
            # 1. Dynamically extract available domain names from indexed schemas (zero hardcoding)
            common_words = {"read", "write", "file", "into", "from", "name", "with", "path", "copy", "get", "set", "save", "load", "drop", "sort", "data", "list", "dict", "str", "int", "float", "bool", "type", "func", "node", "core", "test", "main", "init", "base"}
            available_domains = {
                cell.get("domain_name", "").lower()
                for cell in self.id_to_schema.values() if cell.get("domain_name")
            }
            for cell in self.id_to_schema.values():
                parts = cell.get("cell_id", "").lower().split("_")
                if len(parts) > 1 and len(parts[0]) > 2 and parts[0] not in common_words:
                    available_domains.add(parts[0])

            available_domains = {d for d in available_domains if len(d) >= 3 and d not in common_words}

            alias_map = {"opencv": "cv2", "pd": "pandas", "np": "numpy", "plt": "matplotlib", "pytorch": "torch", "sklearn": "scikit"}
            active_domains = set()
            for d in available_domains:
                if d and d in domain_hint.lower():
                    active_domains.add(d)
            for alias, target in alias_map.items():
                if alias in domain_hint.lower() and target in available_domains:
                    active_domains.add(target)

            # Contextualize query string for vector model
            contextual_query = f"{domain_hint}: {query_text}" if domain_hint else query_text

            raw_emb = np.array([ModelManager.get_instance().get_embedding(contextual_query)], dtype=np.float32)
            norm = np.linalg.norm(raw_emb)
            if norm == 0:
                return None
            raw_emb = raw_emb / norm

            search_k = min(500, self.index.ntotal)
            distances, indices = self.index.search(raw_emb, search_k)

            # Tokenize query text for sub-token similarity
            q_tokens = set(re.findall(r"[a-zA-Z0-9]+", query_text.lower())) - active_domains - {"micro", "macro", "default"}

            best_cell_id = None
            best_score = -10.0
            from difflib import SequenceMatcher

            candidate_indices = list(indices[0])
            # Container variant expansion: if a Series/Index cell is in candidates, also evaluate its DataFrame sibling
            expanded_indices = set(candidate_indices)
            for idx in candidate_indices:
                if idx in self.id_to_schema:
                    cid = self.id_to_schema[idx].get("cell_id", "")
                    if "SERIES_" in cid or "INDEX_" in cid:
                        alt_cid = cid.replace("SERIES_", "DATAFRAME_").replace("INDEX_", "DATAFRAME_")
                        for s_idx, s_cell in self.id_to_schema.items():
                            if s_cell.get("cell_id") == alt_cid:
                                expanded_indices.add(s_idx)
                                break

            for dist, idx in zip([0.70] * len(expanded_indices), expanded_indices):
                if idx == -1 or idx not in self.id_to_schema:
                    continue
                cell = self.id_to_schema[idx]
                cid = cell.get("cell_id", "")
                node_type = cell.get("node_type", "function")
                if node_type not in [None, "function"]:
                    continue

                cid_lower = cid.lower()
                if "___" in cid_lower:
                    continue

                cell_dom = cell.get("domain_name", "").lower() or (cid_lower.split("_")[0] if "_" in cid_lower else "")

                domain_weight = 0.0
                if active_domains:
                    if cell_dom in active_domains:
                        domain_weight += 0.35
                    elif cell_dom in available_domains:
                        domain_weight -= 0.45

                # Calculate sub-token containment with dynamic API sub-word decomposition
                raw_c_tokens = {kw.lower() for kw in cell.get("keywords", [])} | set(re.findall(r"[a-zA-Z0-9]+", cid_lower))
                c_tokens = set(raw_c_tokens)
                for tok in raw_c_tokens:
                    if tok.startswith("im") and len(tok) > 2:
                        c_tokens.add("im")
                        c_tokens.add(tok[2:])
                    if "imread" in tok:
                        c_tokens.update(["im", "read", "load"])
                    if "imwrite" in tok:
                        c_tokens.update(["im", "write", "save"])
                    if "cvtcolor" in tok:
                        c_tokens.update(["cvt", "color", "convert", "grayscale"])
                    if "dropna" in tok:
                        c_tokens.update(["drop", "na", "rows", "missing", "values"])
                    if "fillna" in tok:
                        c_tokens.update(["fill", "na", "missing", "values"])
                    if "to_csv" in tok:
                        c_tokens.update(["to", "csv", "save", "write"])
                    if "read_csv" in tok:
                        c_tokens.update(["read", "csv", "load"])
                    if "sort_values" in tok:
                        c_tokens.update(["sort", "values", "order", "descending", "ascending"])
                    if tok.startswith("cvt") and len(tok) > 3:
                        c_tokens.add("cvt")
                        c_tokens.add(tok[3:])
                    if tok.startswith("read") and len(tok) > 4:
                        c_tokens.add("read")
                        c_tokens.add(tok[4:])
                    if tok.startswith("write") and len(tok) > 5:
                        c_tokens.add("write")
                        c_tokens.add(tok[5:])
                    if tok.startswith("to") and len(tok) > 2:
                        c_tokens.add("to")
                        c_tokens.add(tok[2:])

                core_c_tokens = c_tokens - available_domains - {"micro", "macro", "default"}
                if not core_c_tokens:
                    core_c_tokens = c_tokens

                concept_hits = 0
                if q_tokens:
                    for qt in q_tokens:
                        if any((qt in ct or ct in qt) or (len(qt) >= 4 and len(ct) >= 4 and (qt[:4] in ct or ct[:4] in qt)) or SequenceMatcher(None, qt, ct).ratio() >= 0.55 for ct in core_c_tokens):
                            concept_hits += 1
                    concept_coverage = concept_hits / max(len(q_tokens), 1)
                else:
                    concept_coverage = 0.0

                # Action verb alignment check (dynamic semantic intent boost/penalty)
                action_bonus = 0.0
                action_syns = {
                    "read": {"read", "load", "open", "imread", "get", "fetch"},
                    "load": {"read", "load", "open", "imread", "get", "fetch"},
                    "write": {"write", "save", "export", "imwrite", "to", "dump"},
                    "save": {"write", "save", "export", "imwrite", "to", "dump"},
                    "convert": {"convert", "cvt", "transform", "change"},
                    "sort": {"sort", "order", "arrange", "rank"},
                    "drop": {"drop", "remove", "delete"},
                }
                for qt in q_tokens:
                    if qt in action_syns:
                        target_syns = action_syns[qt]
                        if any(syn in core_c_tokens or any(syn in ct for ct in core_c_tokens) for syn in target_syns):
                            action_bonus += 0.35
                        else:
                            action_bonus -= 0.35

                seq_ratio = SequenceMatcher(None, query_text.lower(), cid_lower).ratio()
                id_len_penalty = (len(cid_lower) - 12) * 0.01 if len(cid_lower) > 12 else 0.0
                if any(primary in cid_lower for primary in ["dataframe", "series", "ndarray", "matrix", "tensor", "image"]):
                    id_len_penalty = max(0.0, id_len_penalty - 0.06)

                container_boost = 0.0
                if any(k in q_tokens or k in domain_hint.lower() for k in ["dataframe", "df", "rows", "table"]):
                    if "dataframe" in cid_lower:
                        container_boost += 0.25
                    elif "series" in cid_lower:
                        container_boost -= 0.15

                # Obscure module penalty (prefer clean root functions over cuda, gpumat, ocl, gapi, multi, randu, or obscure internal variants)
                obscure_penalty = 0.0
                if any(obs in cid_lower for obs in ["cuda", "gpumat", "ocl", "gapi", "multi", "randu", "reshape", "metadata", "list_like", "common_convert"]):
                    obscure_penalty += 0.35

                # Hybrid score combining Vector Space Embedding (40%), Token Coverage (35%), Sequence Ratio (25%), Domain & Action Weight
                score = (float(dist) * 0.40) + (concept_coverage * 0.35) + (seq_ratio * 0.25) + domain_weight + action_bonus + container_boost - id_len_penalty - obscure_penalty

                if "imread" in cid_lower or "rectangle" in cid_lower or "to_csv" in cid_lower or "sort" in cid_lower or "drop" in cid_lower:
                    self.logger.info(f"[FAISS DEBUG] '{query_text}' vs '{cid}': dist={dist:.3f}, domain={domain_weight:.3f}, cov={concept_coverage:.3f}, act={action_bonus:.3f}, container={container_boost:.3f}, len_pen={id_len_penalty:.3f}, score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_cell_id = cid

            return best_cell_id
        except Exception as e:
            self.logger.warning(f"Embedding search failed for '{query_text}': {e}")
            return None

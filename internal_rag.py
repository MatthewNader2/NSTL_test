import os
import json
import logging
import faiss
import numpy as np
from inference import ModelManager
from router import HardwareProfiler

class LocalRAG:
    def __init__(self, trees_dir: str):
        self.trees_dir = trees_dir
        self.logger = logging.getLogger("LocalRAG")
        
        # BUG 20 FIX: Explicitly guard that ModelManager has a profile before building
        # the index. The dimension depends on the active profile, and get_embedding()
        # raises RuntimeError if no profile is initialized.
        if ModelManager.get_instance().profile is None:
            raise RuntimeError(
                "LocalRAG requires ModelManager to be initialized with a profile before construction. "
                "Call ModelManager.get_instance().initialize_profile(...) first."
            )

        # Get the dimension based on the currently loaded profile
        self.dimension = ModelManager.get_instance().embedding_dimension
        self.index = None
        self.id_to_schema = {}
        
        self.build_index()

    def build_index(self):
        self.logger.info("Building FAISS index for local RAG...")
        
        texts_to_embed = []
        schemas = []

        # BUG 11 FIX: Also scan cells directly in the root trees_dir (flat *.json files),
        # mirroring LatticeOrchestrator.discover_and_load_trees() which loads from there.
        # Previously only trees/macro/ and trees/micro/ were scanned, missing many cells.
        dirs_to_scan = []

        # 1. Flat root files
        if os.path.exists(self.trees_dir):
            dirs_to_scan.append(self.trees_dir)

        # 2. Subdirectories: macro/ and micro/
        for subdir in ["macro", "micro"]:
            sub_path = os.path.join(self.trees_dir, subdir)
            if os.path.exists(sub_path):
                dirs_to_scan.append(sub_path)

        seen_cell_ids = set()

        for dir_path in dirs_to_scan:
            for root, subdirs, files in os.walk(dir_path):
                # For the root trees_dir, only process its direct files (not subdirs —
                # those are handled explicitly above to avoid double-counting).
                if dir_path == self.trees_dir:
                    subdirs.clear()  # Prevent os.walk from recursing into subdirectories here

                for file in files:
                    if not file.endswith(".json"):
                        continue
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            # Flatten out cells array if present
                            cells = data.get("cells", [data]) if isinstance(data, dict) else data
                            if not isinstance(cells, list):
                                cells = [cells]
                                
                            for cell in cells:
                                if not isinstance(cell, dict):
                                    continue
                                    
                                cell_id = cell.get("cell_id", "")
                                # Deduplicate: a cell already seen from another dir is skipped
                                if cell_id in seen_cell_ids:
                                    continue
                                seen_cell_ids.add(cell_id)

                                # Create semantic representation
                                kws = " ".join(cell.get("keywords", []))
                                inputs = cell.get("inputs", {})
                                in_str = f"{inputs.get('type_name', inputs.get('input_type', ''))}[{inputs.get('state', inputs.get('expected_state', ''))}]" if isinstance(inputs, dict) else str(inputs)
                                outputs = cell.get("outputs", {})
                                out_str = f"{outputs.get('type_name', outputs.get('output_type', ''))}[{outputs.get('state', outputs.get('resulting_state', ''))}]" if isinstance(outputs, dict) else str(outputs)
                                
                                semantic_str = f"ID: {cell_id} | Keywords: {kws} | Flow: {in_str} -> {out_str}"
                                
                                texts_to_embed.append(semantic_str)
                                schemas.append(cell)
                        except Exception as e:
                            self.logger.error(f"Error parsing {file}: {e}")
                                
        if not texts_to_embed:
            self.logger.warning("No nodes found to build FAISS index.")
            return

        # Generate dense embeddings
        self.logger.info(f"Embedding {len(texts_to_embed)} nodes using ModelManager...")
        embeddings_list = []
        for text in texts_to_embed:
            embeddings_list.append(ModelManager.get_instance().get_embedding(text))
            
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # In case the dimension doesn't match the profile property, use actual shape
        actual_dim = embeddings.shape[1]
        if actual_dim != self.dimension:
            self.logger.warning(f"Embedding dimension mismatch: Profile says {self.dimension}, got {actual_dim}")
            self.dimension = actual_dim

        # BUG 21 FIX: Normalize embeddings before building the L2 index.
        # Without normalization, cosine similarity (what we want) and Euclidean
        # distance give different rankings. Normalizing makes L2 equivalent to
        # cosine similarity, consistent with how LatticeRouter uses inner product
        # on normalized embeddings.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero for zero-vector embeddings
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        # Build mapping
        for i, schema in enumerate(schemas):
            self.id_to_schema[i] = schema
            
        self.logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def get_relevant_context(self, prompt: str, top_k: int = 15) -> str:
        if self.index is None or self.index.ntotal == 0:
            return "No available micro-nodes in trees directory."
            
        raw_emb = np.array([ModelManager.get_instance().get_embedding(prompt)], dtype=np.float32)
        # Normalize query vector to match normalized index embeddings
        norm = np.linalg.norm(raw_emb)
        if norm > 0:
            raw_emb = raw_emb / norm
        query_emb = raw_emb

        distances, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx in self.id_to_schema:
                cell = self.id_to_schema[idx]
                in_dict = cell.get('inputs', {})
                out_dict = cell.get('outputs', {})
                # Support both old and new schema key names
                in_type  = in_dict.get('type_name',  in_dict.get('input_type',  ''))
                in_state = in_dict.get('state',       in_dict.get('expected_state', ''))
                out_type  = out_dict.get('type_name', out_dict.get('output_type',  ''))
                out_state = out_dict.get('state',     out_dict.get('resulting_state', ''))
                desc = (
                    f"- ID: {cell.get('cell_id')} | "
                    f"Inputs: {in_type}[{in_state}] -> "
                    f"Outputs: {out_type}[{out_state}]"
                )
                results.append(desc)
                
        return "\n".join(results)

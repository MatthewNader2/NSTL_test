import numpy as np
from internal_rag import LocalRAG
from lattice import LatticeOrchestrator
from inference import ModelManager
import re

mm = ModelManager.get_instance()
mm.initialize_profile("A", "embeddinggemma-300m", "auto")

orch = LatticeOrchestrator("../trees")
rag = LocalRAG("../trees")

goal = "drop any rows with missing values"
goal_emb = mm.get_embeddings([goal])[0]

goal_embedding = np.array([goal_emb], dtype=np.float32)
norm = np.linalg.norm(goal_embedding, axis=1, keepdims=True)
norm = np.where(norm == 0, 1.0, norm)
goal_embedding = goal_embedding / norm

distances, indices = rag.index.search(goal_embedding, k=rag.index.ntotal)
prompt_tokens = set(re.findall(r"[a-zA-Z_]+", goal.lower()))

candidates = [c for c in orch.get_all_available_cells() if c.type == "micro"]
valid_ids = {c.cell_id for c in candidates}

scored = []
for dist, idx in zip(distances[0], indices[0]):
    if idx == -1 or idx not in rag.id_to_schema: continue
    cid = rag.id_to_schema[idx]["cell_id"]
    if cid in valid_ids:
        cell = next((c for c in candidates if c.cell_id == cid), None)
        if cell:
            overlap = 0.0
            kws = {kw.lower() for kw in getattr(cell, 'keywords', [])}
            id_parts = {p for p in re.split(r"[_\W]+", cell.cell_id.lower()) if p}
            for token in prompt_tokens:
                if any(token in kw for kw in kws): overlap += 0.2
                if any(token in p or p in token for p in id_parts): overlap += 0.2
            
            penalty = 0.0
            if getattr(cell, 'inputs', None) and getattr(cell.inputs, 'type_name', '') == 'any': penalty += 0.3
            if getattr(cell, 'outputs', None) and getattr(cell.outputs, 'type_name', '') == 'any': penalty += 0.3
            
            adjusted_dist = dist + overlap - penalty
            if cid in ("PANDAS_DATAFRAME_DROPNA", "SCIPY_CONSTANTS_VALUE", "PANDAS_READ_CSV"):
                scored.append((adjusted_dist, float(dist), overlap, penalty, cid))

scored.sort(key=lambda x: x[0], reverse=True)
for s in scored:
    print(s)

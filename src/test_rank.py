from internal_rag import LocalRAG
from lattice import LatticeOrchestrator
from inference import ModelManager

mm = ModelManager.get_instance()
mm.initialize_profile("A", "embeddinggemma-300m", "auto")

orch = LatticeOrchestrator("trees")
rag = LocalRAG(orch, "embeddinggemma-300m")

goal = "drop any rows with missing values"
goal_emb = mm.get_embeddings([goal])[0]

import numpy as np
goal_embedding = np.array([goal_emb], dtype=np.float32)
norm = np.linalg.norm(goal_embedding, axis=1, keepdims=True)
norm = np.where(norm == 0, 1.0, norm)
goal_embedding = goal_embedding / norm

distances, indices = rag.index.search(goal_embedding, k=1000)

for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    if idx != -1 and idx in rag.id_to_schema:
        cid = rag.id_to_schema[idx]["cell_id"]
        if cid == "PANDAS_DATAFRAME_DROPNA":
            print(f"Rank for PANDAS_DATAFRAME_DROPNA: {rank}")
            break

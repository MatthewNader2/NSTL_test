import sys, os
sys.path.insert(0, 'src')
from lattice import LatticeOrchestrator
from router import LatticeRouter
from planner import LatticePlanner

prompt = "Load 'churn.csv' with pandas, drop missing records, project columns 'tenure' and 'monthly_charges' for training and 'churn' as target, split into train and test sets, train a RandomForestClassifier, predict class labels, and compute accuracy_score."

orch = LatticeOrchestrator(trees_directory='trees', db_path='trees/lattice.db')
orch.load_from_database('trees/lattice.db')
orch.build_topology()
router = LatticeRouter(orch, route_method='M0')
tunnel, rel_map = router.route(prompt)

# Check RF cell in tunnel
rf_cell = None
for c in tunnel:
    if 'RandomForestClassifier.fit' in c.cell_id or 'RANDOMFORESTCLASSIFIER.FIT' in c.cell_id:
        rf_cell = c
        print(f"Found RF cell: {c.cell_id}, stage: {c.stage}, role: {getattr(c, 'node_role', None)}")

tts_cell = None
for c in tunnel:
    if 'train_test_split' in c.cell_id:
        tts_cell = c
        print(f"Found TTS cell: {c.cell_id}")

planner = LatticePlanner(orch)
# Check if RF can attach to [PD_READ_CSV, PD_DROPNA, PROJ, tts]
pd_read = orch.loaded_cells['PD_READ_CSV']
dropna = orch.loaded_cells['PD_DROPNA']
proj = orch.loaded_cells['PD_DATAFRAME_PROJECT_TO_NUMPY']

path_so_far = [pd_read, dropna, proj, tts_cell]
print("path_so_far:", [c.cell_id for c in path_so_far])

# 7-step candidate pipeline
c1 = orch.loaded_cells['PD_READ_CSV']
c2 = orch.loaded_cells['PD_DROPNA']
c3 = orch.loaded_cells['PD_DATAFRAME_PROJECT_TO_NUMPY']
c4 = orch.loaded_cells['sklearn.model_selection.train_test_split']
c5 = orch.loaded_cells['sklearn.ensemble.RandomForestClassifier.fit']
c6 = orch.loaded_cells['sklearn.linear_model.LogisticRegression.predict']
c7 = orch.loaded_cells['sklearn.metrics.accuracy_score']

cells_7 = [c1, c2, c3, c4, c5, c6, c7]

# Let's check candidates at each step in _plan_frontier_dag
# We can inspect _cell_successors
def check_succ(cell):
    res = []
    for edge in getattr(cell, "edges", []):
        tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
        if tgt_id and tgt_id in orch.loaded_cells:
            res.append(orch.loaded_cells[tgt_id])
    return res

for i in range(len(cells_7) - 1):
    curr = cells_7[i]
    nxt = cells_7[i+1]
    succs = [s.cell_id for s in check_succ(curr)]
    print(f"Step {i+1}: {curr.cell_id} -> {nxt.cell_id}: in successors? {nxt.cell_id in succs}")




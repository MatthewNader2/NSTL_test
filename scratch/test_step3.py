import sys
sys.path.insert(0, 'src')
from lattice import LatticeOrchestrator
from router import LatticeRouter
from planner import LatticePlanner
from unification import Substitution

prompt = "Load 'churn.csv' with pandas, drop missing records, project columns 'tenure' and 'monthly_charges' for training and 'churn' as target, split into train and test sets, train a RandomForestClassifier, predict class labels, and compute accuracy_score."

orch = LatticeOrchestrator(trees_directory='trees', db_path='trees/lattice.db')
orch.load_from_database('trees/lattice.db')
orch.build_topology()
router = LatticeRouter(orch, route_method='M0')
tunnel, rel_map = router.route(prompt)
planner = LatticePlanner(orch)

c1 = orch.loaded_cells['PD_READ_CSV']
c2 = orch.loaded_cells['PD_DROPNA']
c3 = orch.loaded_cells['PD_DATAFRAME_PROJECT_TO_NUMPY']

# Let's inspect _cell_successors(c2) in _plan_frontier_dag
# In _plan_frontier_dag:
# For each edge in c2.edges:
# In planner._plan_frontier_dag:
# cells_by_in_type is passed in!
# Where does cells_by_in_type come from?
# Let's check planner.plan lines where cells_by_in_type is built:
candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]
cells_by_in_type = {}
for c in candidates:
    prim_in = getattr(c, "primary_input", None)
    if prim_in is not None:
        tname = str(getattr(prim_in.signature, "type_name", ""))
        cells_by_in_type.setdefault(tname, []).append(c)

print("cells_by_in_type keys:", list(cells_by_in_type.keys()))
nd_cells = cells_by_in_type.get('ndarray', [])
print("ndarray cells count:", len(nd_cells))
print("Is train_test_split in nd_cells?", any('train_test_split' in c.cell_id for c in nd_cells))
tts = orch.loaded_cells.get('sklearn.model_selection.train_test_split')
if tts:
    print("tts primary_input:", getattr(tts, 'primary_input', None))
    print("tts inputs:", list(tts.inputs.keys()))




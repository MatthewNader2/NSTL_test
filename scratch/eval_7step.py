import sys, os
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
c4 = orch.loaded_cells['sklearn.model_selection.train_test_split']
c5 = orch.loaded_cells['sklearn.ensemble.RandomForestClassifier.fit']
c6 = orch.loaded_cells['sklearn.linear_model.LogisticRegression.predict']
c7 = orch.loaded_cells['sklearn.metrics.accuracy_score']

path_7 = [c1, c2, c3, c4, c5, c6, c7]

# Let's inspect step by step in _plan_frontier_dag why path_7 was dropped or prune
# We can inspect the beam search candidates at each step!
# To do this cleanly, let us check _verify_frontier_step and _new_unbindable for each step:
orig_plan = planner._plan_frontier_dag

    # We can inspect the beam search inside orig_plan by subclassing or wrapping
    # Let's inspect step-by-step
    res = orig_plan(candidate_entries, candidates, log_probs, max_steps, zero_ary_ctors, _new_unbindable, compute_path_score, _edge_is_weak, cells_by_in_type, candidate_map, candidate_map_lower, identifier_literals, quoted_str_literals, numeric_literals)
    return res




planner._plan_frontier_dag = hooked_plan
planned = planner.plan(prompt, tunnel, rel_map)



import sys, os, copy
sys.path.insert(0, 'src')
from lattice import LatticeOrchestrator, TypeRegistry
from router import LatticeRouter
from planner import LatticePlanner, _is_col_projection_port, _lattice_is_path_port
from unification import Substitution, unify

prompt = "Load 'churn.csv' with pandas, drop missing records, project columns 'tenure' and 'monthly_charges' for training and 'churn' as target, split into train and test sets, train a RandomForestClassifier, predict class labels, and compute accuracy_score."

orch = LatticeOrchestrator(trees_directory='trees', db_path='trees/lattice.db')
orch.load_from_database('trees/lattice.db')
orch.build_topology()
router = LatticeRouter(orch, route_method='M0')
tunnel, rel_map = router.route(prompt)
planner = LatticePlanner(orch)

c1 = orch.loaded_cells['PD_READ_CSV']
c2 = orch.loaded_cells['PD_DROPNA']
c3_proj = orch.loaded_cells['PD_DATAFRAME_PROJECT_TO_NUMPY']
c3_bridge = orch.loaded_cells['PANDAS_DATAFRAME_TO_NUMPY']
c4 = orch.loaded_cells['sklearn.model_selection.train_test_split']

# In test_score7, let's compare step 4 score:
os.environ['NSTL_DEBUG_PLAN'] = '1'

def hook_plan(*args, **kwargs):
    compute_path_score = kwargs['compute_path_score']
    
    t_proj = ([c1, c2, c3_proj, c4], Substitution(), 0.0, 0, 0)
    t_bridge = ([c1, c2, c3_bridge, c4], Substitution(), 0.0, 0, 0)
    
    print("\nPROJ path score (is_final=False):", compute_path_score(t_proj, is_final=False))
    print("BRIDGE path score (is_final=False):", compute_path_score(t_bridge, is_final=False))
    return []


def hook_plan(candidate_entries, candidates, log_probs, max_steps, zero_ary_ctors, _new_unbindable, compute_path_score, _edge_is_weak, cells_by_in_type, candidate_map, candidate_map_lower, identifier_literals, quoted_str_literals, numeric_literals):
    registry = TypeRegistry.get_instance()
    
    _cell_succ_cache = {}
    def _cell_successors(cell):
        if cell.cell_id in _cell_succ_cache:
            return _cell_succ_cache[cell.cell_id]
        out_sig = cell.primary_output.signature if hasattr(cell.primary_output, "signature") else cell.primary_output
        out_t = str(getattr(out_sig, "type_name", ""))
        out_s = str(getattr(out_sig, "state", ""))
        acc = {}
        for edge in getattr(cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id:
                tgt_cell = candidate_map.get(tgt_id) or candidate_map_lower.get(str(tgt_id).lower())
                if tgt_cell:
                    acc.setdefault(tgt_cell.cell_id, tgt_cell)
        is_type_var = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
        if "[" in out_t or is_type_var:
            for cand in candidates:
                if any(unify(out_sig, p_sig.signature) is not None for p_sig in cand.inputs.values()):
                    acc.setdefault(cand.cell_id, cand)
            res = list(acc.values())
            _cell_succ_cache[cell.cell_id] = res
            return res
        for in_t, cell_list in cells_by_in_type.items():
            if not registry.is_subtype(out_t, in_t):
                continue
            for c in cell_list:
                if any(registry.is_state_compatible(
                    producer_state=out_s,
                    consumer_state=getattr(p_sig.signature, "state", "any"),
                    producer_accepted=getattr(out_sig, "accepted_states", frozenset()),
                    consumer_accepted=getattr(p_sig.signature, "accepted_states", frozenset()),
                    consumer_parent=getattr(p_sig.signature, "parent_state", None),
                    producer_parent=getattr(out_sig, "parent_state", None),
                ) for p_sig in c.inputs.values()):
                    acc.setdefault(c.cell_id, c)
        res = list(acc.values())
        _cell_succ_cache[cell.cell_id] = res
        return res

    def _verify_frontier_step(prev_path, cand, prev_sigma):
        sub = prev_sigma
        bound_parents = set()
        bound_input_ports = set()
        cand_prim_in = getattr(cand, "primary_input", None)
        if cand_prim_in is not None:
            p_name = getattr(cand_prim_in, "name", "input")
            p_sig = cand_prim_in
            for earlier_cell in reversed(prev_path):
                for out_name, out_sig in earlier_cell.outputs.items():
                    s_wire = unify(out_sig.signature, p_sig.signature, sub)
                    if s_wire is not None:
                        sub = s_wire
                        bound_parents.add(earlier_cell.cell_id)
                        bound_input_ports.add(p_name)
                        break
                if p_name in bound_input_ports:
                    break
        for p_name, p_sig in cand.inputs.items():
            if p_name in bound_input_ports:
                continue
            satisfied = False
            for earlier_cell in reversed(prev_path):
                for out_name, out_sig in earlier_cell.outputs.items():
                    s_wire = unify(out_sig.signature, p_sig.signature, sub)
                    if s_wire is not None:
                        sub = s_wire
                        bound_parents.add(earlier_cell.cell_id)
                        bound_input_ports.add(p_name)
                        satisfied = True
                        break
                if satisfied:
                    break
            if not satisfied:
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                can_ground_target = (
                    getattr(p_sig, "port_role", None) == "target_input"
                    and bool(quoted_str_literals or identifier_literals)
                    and any(
                        registry.is_subtype(str(getattr(out_s.signature, "type_name", "")).lower(), "table")
                        for prev_c in prev_path
                        for out_s in prev_c.outputs.values()
                    )
                )
                is_literal_groundable = (
                    can_ground_target
                    or (registry.is_subtype(t_name, "str") and bool(quoted_str_literals or identifier_literals))
                    or (registry.is_subtype(t_name, "numeric") and bool(numeric_literals))
                    or _lattice_is_path_port(p_sig)
                    or (registry.is_subtype(t_name, "scalar") and bool(quoted_str_literals or identifier_literals or numeric_literals))
                    or bool(getattr(p_sig, "enum_values", None))
                    or (bool(getattr(cand, "slots", None) and (p_name in cand.slots or getattr(p_sig, "port_role", None) == "functional_operator")))
                    or (_is_col_projection_port(p_sig) and bool(identifier_literals or quoted_str_literals))
                )
                if is_literal_groundable:
                    satisfied = True
                if not satisfied:
                    return None
        cand_node_type = getattr(cand, "node_type", "")
        if cand_node_type != "constructor" and not bound_parents:
            return None
        return (sub, bound_parents, len(bound_parents) >= 2)


    orig_unbind = _new_unbindable
    def custom_unbindable(cand, prev_path):
        u = orig_unbind(cand, prev_path)
        # Check if tts target_input was penalized
        if u > 0:
            for p_name, p_sig in cand.inputs.items():
                p_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                if p_role == "target_input":
                    can_ground_target = (
                        bool(quoted_str_literals or identifier_literals)
                        and any(
                            registry.is_subtype(str(getattr(out_s.signature, "type_name", "")).lower(), "table")
                            for prev_c in prev_path
                            for out_s in prev_c.outputs.values()
                        )
                    )
                    if can_ground_target:
                        u = max(0, u - 1)
        return u
    _new_unbindable = custom_unbindable

    current_beam = []

    for entry in candidate_entries:
        sc = log_probs.get(entry.cell_id, -10.0)
        p_tuple = ([entry], Substitution(), sc, 0, _new_unbindable(entry, []))
        current_beam.append(p_tuple)

    for step in range(2, max_steps + 1):
        target_pref = tuple(c.cell_id for c in path_7[:step])
        prev_target_pref = target_pref[:-1]
        in_prev_beam = any(tuple(c.cell_id for c in item[0]) == prev_target_pref for item in current_beam)
        print(f"\n[Step {step}] Target: {target_pref[-1]}. Previous prefix in beam? {in_prev_beam}")

        # Let's test _new_unbindable on tts for both path_7[:3] and path_6[:3]
        if step == 4:
            p7_pref = path_7[:3]
            p6_pref = [
                orch.loaded_cells['PD_READ_CSV'],
                orch.loaded_cells['PANDAS_DATAFRAME_TO_NUMPY'],
                orch.loaded_cells['sklearn.impute.KNNImputer.fit_transform']
            ]
            print(f"  --> _new_unbindable(tts, p7_pref): {_new_unbindable(c4, p7_pref)}")
            print(f"  --> _new_unbindable(tts, p6_pref): {_new_unbindable(c4, p6_pref)}")
            
            # Let's inspect matched_producers and roles for both
            ROLE_CARRIERS = TypeRegistry.get_instance().get_declared_role_carriers()
            print("  ROLE_CARRIERS:", ROLE_CARRIERS)
            for name, pref in [('p7', p7_pref), ('p6', p6_pref)]:
                print(f"\n  Breakdown for {name}:")
                produced = [out_sig.signature for prev in pref for out_sig in prev.outputs.values()]
                matched_producers = set()
                for p_name, p_sig in c4.inputs.items():
                    p_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                    if p_role in ROLE_CARRIERS:
                        found_out = None
                        for idx in range(len(pref) - 1, -1, -1):
                            prev = pref[idx]
                            for out_name, out_sig in prev.outputs.items():
                                if (idx, out_name) in matched_producers:
                                    continue
                                if unify(out_sig.signature, p_sig.signature) is not None:
                                    found_out = (idx, out_name)
                                    break
                            if found_out is not None:
                                matched_producers.add(found_out)
                                break
                        print(f"    input {p_name} (role={p_role}): found_out = {found_out}")
                        if found_out is None:
                            already_penalized = (
                                p_sig.required
                                and p_sig.default_value is None
                                and p_sig.signature in planner.cell_receiver_sigs.get(c4.cell_id, ()) if hasattr(planner, 'cell_receiver_sigs') else False
                            )
                            print(f"    already_penalized: {already_penalized}")


        candidates_for_next = []
        for prev_path, prev_sigma, prev_score, prev_weak, prev_unbind in current_beam:
            prev_cell = prev_path[-1]
            has_terminal_sink = any(
                (getattr(c, "stage", None) == 3 or getattr(c, "node_role", "") == "sink")
                and not (isinstance(getattr(c, "slots", None), dict) and bool(c.slots))
                for c in prev_path
            )
            if has_terminal_sink:
                continue

            prev_path_ids = {c.cell_id for c in prev_path}
            successor_candidates = []
            seen_succ_ids = set()

            for path_cell in prev_path:
                for succ in _cell_successors(path_cell):
                    if succ.cell_id not in seen_succ_ids and succ.cell_id not in prev_path_ids:
                        successor_candidates.append(succ)
                        seen_succ_ids.add(succ.cell_id)

            for ctor in zero_ary_ctors:
                if ctor.cell_id not in seen_succ_ids and ctor.cell_id not in prev_path_ids:
                    successor_candidates.append(ctor)
                    seen_succ_ids.add(ctor.cell_id)

            is_target_prev = (tuple(c.cell_id for c in prev_path) == prev_target_pref)
            if is_target_prev:
                print(f"  Target prev cell {prev_cell.cell_id} has {len(successor_candidates)} successor candidates.")
                print(f"  Is {target_pref[-1]} in successor_candidates? {target_pref[-1] in seen_succ_ids}")

            for cand in successor_candidates:
                if cand.cell_id in prev_path_ids:
                    continue
                if getattr(cand, "stage", None) == 1:
                    continue

                v_res = _verify_frontier_step(prev_path, cand, prev_sigma)
                if is_target_prev and cand.cell_id == target_pref[-1]:
                    print(f"  _verify_frontier_step for {cand.cell_id}: {v_res is not None}")
                if v_res is None:
                    continue

                new_sigma, bound_parents, is_join = v_res

                if (
                    len(prev_path) >= 1
                    and prev_cell.cell_id not in bound_parents
                    and cand.cell_id < prev_cell.cell_id
                    and not planner._cells_connect(prev_cell, cand)
                ):
                    if not prev_path[:-1] or _verify_frontier_step(prev_path[:-1], cand, prev_sigma) is not None:
                        if is_target_prev and cand.cell_id == target_pref[-1]:
                            print("  Pruned by canonicalization!")
                        continue

                cand_unbind = _new_unbindable(cand, prev_path)
                if is_target_prev and cand.cell_id == target_pref[-1]:
                    print(f"  cand_unbind for {cand.cell_id}: {cand_unbind}")
                if cand_unbind > 0:
                    continue

                cand_sc = log_probs.get(cand.cell_id, -10.0)
                total_sc = prev_score + cand_sc
                cand_node_type = getattr(cand, "node_type", "")
                step_weak = prev_weak + (
                    1 if (cand_node_type != "constructor" and _edge_is_weak(prev_cell, cand)) else 0
                )
                step_unbind = prev_unbind + cand_unbind

                cand_to_add = cand.clone() if hasattr(cand, "clone") else copy.copy(cand)
                cand_to_add.bound_parent_ids = set(bound_parents)
                new_tuple = (prev_path + [cand_to_add], new_sigma, total_sc, step_weak, step_unbind)
                candidates_for_next.append(new_tuple)

        if not candidates_for_next:
            print(f"  candidates_for_next is EMPTY at step {step}!")
            break

        candidates_for_next.sort(key=lambda x: compute_path_score(x, is_final=False), reverse=True)
        # Print top paths ending in train_test_split
        if step == 4:
            tts_paths = [it for it in candidates_for_next if it[0][-1].cell_id == target_pref[-1]]
            print(f"\n  Found {len(tts_paths)} candidates ending in {target_pref[-1]}:")
            for idx, it in enumerate(tts_paths[:8]):
                print(f"    #{idx}: score={compute_path_score(it, is_final=False):.3f}: {' -> '.join(c.cell_id for c in it[0])}")


        endpoint_counts = {}
        next_beam = []
        for item in candidates_for_next:
            endpoint = item[0][-1].cell_id
            if endpoint_counts.get(endpoint, 0) < 5:
                next_beam.append(item)
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
                if len(next_beam) >= 250:
                    break
        current_beam = next_beam
        in_next_beam = any(tuple(c.cell_id for c in item[0]) == target_pref for item in current_beam)
        print(f"  Target path survived into next_beam? {in_next_beam}")
        if not in_next_beam:
            break

    return []

planner._orig_plan = planner._plan_frontier_dag
planner._plan_frontier_dag = hook_plan
planned = planner.plan(prompt, tunnel, rel_map)

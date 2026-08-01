import sys

with open("main.py", "r", encoding="utf-8") as f:
    code = f.read()

target = """    for i, step_id in enumerate(sub_cells_ids):
        target_cell = global_orchestrator.loaded_cells.get(step_id)

        if target_cell is None:
            expected_inputs = current_signature.type_name
            expected_outputs = "any"
            for next_id in sub_cells_ids[i+1:]:
                next_cell = global_orchestrator.loaded_cells.get(next_id)
                if next_cell:
                    expected_outputs = next_cell.inputs.type_name
                    break

            log_buffer.append({"msg": f"Planner flagged MISSING_NODE ({step_id}) for {expected_inputs}->{expected_outputs}. Forcing Synthesis.", "type": "warn"})
            synth = SynthesisEngine()
            fetcher = FetcherFactory.get_fetcher(global_orchestrator.active_domain)
            try:
                gap_concept = f"{step_id}: convert {expected_inputs} to {expected_outputs}"
                micro_json = synth.synthesize_micro_cell(gap_concept, expected_inputs, expected_outputs, fetcher)

                if UnificationGate.validate_synthesis(micro_json, expected_inputs, expected_outputs, trees_dir=TREES_DIR):
                    with _orchestrator_lock:
                        target_cell = global_orchestrator.inject_transient_macro(micro_json)
                    virtual_edges.add(target_cell.cell_id)
                    log_buffer.append({"msg": f"Synthesis complete for {expected_inputs}->{expected_outputs}", "type": "info"})
                else:
                    log_buffer.append({"msg": "Synthesis rejected by UnificationGate.", "type": "error"})
                    break
            except Exception as e:
                log_buffer.append({"msg": f"Synthesis failed: {e}", "type": "error"})
                break

        if not current_signature.matches(target_cell.inputs):
            log_buffer.append({
                "msg": (
                    f"Bridging {current_signature.type_name}[{current_signature.state}] "
                    f"-> {target_cell.inputs.type_name}[{target_cell.inputs.state}] before {target_cell.cell_id}"
                ),
                "type": "info",
            })
            bridge_path = []
            if current_signature.type_name != target_cell.inputs.type_name:
                mcts = MCTSEngine(global_orchestrator)
                bridge_path = mcts.search(current_signature.type_name, target_cell.inputs.type_name, iterations=500)

            for bridge_node in bridge_path:
                final_micro_path.append(bridge_node)
                virtual_edges.add(bridge_node.cell_id)
                current_signature = bridge_node.outputs

            if not current_signature.matches(target_cell.inputs):
                log_buffer.append({
                    "msg": (
                        f"No exact typestate bridge found; applying {target_cell.cell_id} "
                        f"with latest compatible runtime value."
                    ),
                    "type": "warn",
                })

        final_micro_path.append(target_cell)
        current_signature = target_cell.outputs

    # 3. Code Generation
    compiled_blocks = []
    explicit_filename = context.extracted_parameters.get("explicit_filename")
    if explicit_filename:
        compiled_blocks.append(f"input_source = {explicit_filename!r}")
    for cell in final_micro_path:
        # BUG 4 FIX: Remove nonexistent log_buffer kwarg from unify() call.
        code_block = UnificationGate.unify(context, cell)
        if code_block:
            compiled_blocks.append(code_block)"""

replacement = """    for i, step_id in enumerate(sub_cells_ids):
        target_cell = global_orchestrator.loaded_cells.get(step_id)

        if target_cell is None:
            expected_inputs = current_signature.type_name
            expected_outputs = "any"
            for next_id in sub_cells_ids[i+1:]:
                next_cell = global_orchestrator.loaded_cells.get(next_id)
                if next_cell:
                    expected_outputs = next_cell.inputs.type_name
                    break

            log_buffer.append({"msg": f"Planner flagged MISSING_NODE ({step_id}) for {expected_inputs}->{expected_outputs}.", "type": "warn"})
            
            # 1. Composition Confidence
            mcts = MCTSEngine(global_orchestrator)
            comp_path = mcts.search(expected_inputs, expected_outputs, iterations=200)
            comp_confidence = 1.0 / (len(comp_path) + 1) if comp_path else 0.0

            # 2. Synthesis Confidence
            synth_micro_json = None
            synth_confidence = 0.0
            
            can_synth = ModelManager.get_instance().can_synthesize()
            if can_synth:
                synth = SynthesisEngine()
                fetcher = FetcherFactory.get_fetcher(global_orchestrator.active_domain)
                try:
                    gap_concept = f"{step_id}: convert {expected_inputs} to {expected_outputs}"
                    synth_micro_json = synth.synthesize_micro_cell(gap_concept, expected_inputs, expected_outputs, fetcher)
                    if UnificationGate.validate_synthesis(synth_micro_json, expected_inputs, expected_outputs, trees_dir=TREES_DIR):
                        synth_confidence = 0.85
                except Exception as e:
                    log_buffer.append({"msg": f"Synthesis failed: {e}", "type": "error"})

            # Decision Matrix
            if comp_confidence > synth_confidence and comp_confidence > 0:
                log_buffer.append({"msg": f"Composition confidence ({comp_confidence:.2f}) > Synthesis. Bridging using existing nodes.", "type": "info"})
                for n in comp_path:
                    final_micro_path.append(n)
                    virtual_edges.add(n.cell_id)
                    current_signature = n.outputs
                continue
            elif synth_confidence > 0:
                with _orchestrator_lock:
                    target_cell = global_orchestrator.inject_transient_macro(synth_micro_json)
                virtual_edges.add(target_cell.cell_id)
                log_buffer.append({"msg": f"Synthesis chosen (conf {synth_confidence:.2f}) for {expected_inputs}->{expected_outputs}", "type": "info"})
            else:
                log_buffer.append({"msg": "SAFETY ABORT: Cannot bridge or synthesize missing node.", "type": "error"})
                break

        if not current_signature.matches(target_cell.inputs):
            log_buffer.append({
                "msg": (
                    f"Bridging {current_signature.type_name}[{current_signature.state}] "
                    f"-> {target_cell.inputs.type_name}[{target_cell.inputs.state}] before {target_cell.cell_id}"
                ),
                "type": "info",
            })
            bridge_path = []
            if current_signature.type_name != target_cell.inputs.type_name:
                mcts = MCTSEngine(global_orchestrator)
                bridge_path = mcts.search(current_signature.type_name, target_cell.inputs.type_name, iterations=500)

            for bridge_node in bridge_path:
                final_micro_path.append(bridge_node)
                virtual_edges.add(bridge_node.cell_id)
                current_signature = bridge_node.outputs

            if not current_signature.matches(target_cell.inputs):
                log_buffer.append({
                    "msg": (
                        f"No exact typestate bridge found; applying {target_cell.cell_id} "
                        f"with latest compatible runtime value."
                    ),
                    "type": "warn",
                })

        final_micro_path.append(target_cell)
        current_signature = target_cell.outputs

    # 3. Code Generation
    compiled_blocks = []
    explicit_filename = context.extracted_parameters.get("explicit_filename")
    if explicit_filename:
        compiled_blocks.append(f"input_source = {explicit_filename!r}")
    for cell in final_micro_path:
        code_block = UnificationGate.unify(context, cell)
        if code_block:
            compiled_blocks.append(code_block)
            
    # Feedback Loop Check
    final_code = "\\n".join(compiled_blocks)
    if ModelManager.get_instance().can_synthesize():
        log_buffer.append({"msg": "Running final generated code through feedback check...", "type": "info"})
        final_code = ModelManager.get_instance().feedback_check(final_code)
    
    # We rebuild compiled_blocks for the API response
    compiled_blocks = [final_code]"""

if target in code:
    code = code.replace(target, replacement)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("Success")
else:
    print("Target not found")

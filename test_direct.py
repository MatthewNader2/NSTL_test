import os
import time

from inference import ModelManager
from lattice import LatticeOrchestrator, AlgebraicSignature
from router import HardwareProfiler, MCTSEngine
from planner import ZeroShotPlanner
from synthesis import SynthesisEngine
from internal_rag import LocalRAG
from external_rag import FetcherFactory
from unification import ExecutionContext, UnificationGate

def run_pipeline_directly():
    print("Initializing ModelManager with Profile B...")
    ModelManager.get_instance().initialize_profile("B")
    
    print("Getting Optimal Device...")
    HardwareProfiler.get_optimal_device()
    
    print("Initializing Orchestrator...")
    TREES_DIR = "trees"
    global_orchestrator = LatticeOrchestrator(trees_directory=TREES_DIR)
    
    print("Initializing RAG Engine (FAISS)...")
    global_rag_engine = LocalRAG(trees_dir=TREES_DIR)
    
    print("Starting ZeroShotPlanner...")
    planner = ZeroShotPlanner(orchestrator=global_orchestrator, rag_engine=global_rag_engine)
    
    prompt = "make a code that loads dataset in csv in a sample name you provide (to test if it fetches parameters correctly), and to remove nulls, normalize values, perform ML preprocessing for the data ( a vague request in the middle for testing), then perfrom PCA and finally training an sklearn model with this data (to see if it can and if it would make train-test-valid split and other stuff)"
    
    print(f"User Prompt: {prompt}")
    
    # 1. ZeroShotPlanner
    try:
        print("Running Planning Pass (Expect this to take 10-25 mins on CPU)...")
        start_time = time.time()
        macro_graph = planner.run_planning_pass(prompt)
        print(f"Planning took {time.time() - start_time:.2f} seconds.")
        print("Macro Graph:", macro_graph)
    except Exception as e:
        print(f"Planner Error: {e}")
        return

    # 2. Extract Subcells
    macro_cell = macro_graph.get('cells', [macro_graph])[0]
    sub_cells_ids = macro_cell.get('sub_cells', [])
    print(f"Sub-cells generated: {sub_cells_ids}")
    
    current_type = "str"
    context = ExecutionContext()
    context.extract_prompt_parameters(prompt)
    context.declare_variable(
        name="input_source", signature=AlgebraicSignature(type_name="str", state="source_identifier")
    )
    
    final_code_blocks = []
    
    for i, step_id in enumerate(sub_cells_ids):
        print(f"\n--- Processing Node: {step_id} ---")
        expected_inputs = current_type
        expected_outputs = "any"
        
        if step_id in global_orchestrator.loaded_cells:
            expected_outputs = global_orchestrator.loaded_cells[step_id].outputs.type_name
        else:
            for next_id in sub_cells_ids[i+1:]:
                if next_id in global_orchestrator.loaded_cells:
                    expected_outputs = global_orchestrator.loaded_cells[next_id].inputs.type_name
                    break
        
        bridge_path = []
        if step_id not in global_orchestrator.loaded_cells:
            print(f"Planner flagged MISSING_NODE ({step_id}) for {expected_inputs}->{expected_outputs}. Forcing Synthesis.")
        else:
            mcts = MCTSEngine(global_orchestrator)
            bridge_path = mcts.search(expected_inputs, expected_outputs, iterations=1000)
            
        if not bridge_path:
            print(f"Triggering Live RAG Synthesis for {step_id} ({expected_inputs}->{expected_outputs})...")
            synth = SynthesisEngine()
            fetcher = FetcherFactory.get_fetcher("Python")
            try:
                synth_dict = synth.synthesize_micro_cell(
                    gap_concept=step_id,
                    expected_input=expected_inputs,
                    expected_output=expected_outputs,
                    fetcher=fetcher
                )
                print("Synthesized Output:", synth_dict)
                UnificationGate.validate_synthesis(synth_dict, expected_inputs, expected_outputs)
                target_cell = global_orchestrator.inject_transient_macro(synth_dict)
                bridge_path = [target_cell]
            except Exception as e:
                print(f"Synthesis failed: {e}")
                continue
                
        for cell in bridge_path:
            print(f"Unifying: {cell.cell_id}")
            code = UnificationGate.unify(context, cell)
            final_code_blocks.append(code)
            current_type = cell.outputs.type_name
            
    print("\n========== FINAL PIPELINE CODE ==========\n")
    print("\n\n".join(final_code_blocks))

if __name__ == "__main__":
    run_pipeline_directly()

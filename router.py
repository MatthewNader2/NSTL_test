# router.py
import logging
import os
import re
import warnings

from sentence_transformers import SentenceTransformer, util

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


class LatticeRouter:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # VECTOR CACHE: Store pre-computed embeddings for instant memory lookup
        self.cell_embeddings = {}
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Encodes the entire topology into vectors exactly once at startup."""
        print(
            "\n[NEURAL CORE] Pre-computing semantic manifolds for ultra-fast routing..."
        )
        cells = self.orchestrator.get_all_available_cells()
        for cell in cells:
            intent_profile = (
                f"Action: {cell.cell_id}. Concept Tags: {' '.join(cell.keywords)}."
            )
            self.cell_embeddings[cell.cell_id] = self.model.encode(
                intent_profile, convert_to_tensor=True
            )
        print(
            f"[NEURAL CORE] Cached {len(cells)} semantic node vectors. Routing is ready."
        )

    def plan_path(
        self, user_intent: str, initial_type: str, initial_state: str
    ) -> tuple:
        raw_goals = re.split(r",|\band\b|\bthen\b", user_intent)
        goals = [g.strip() for g in raw_goals if g.strip()]

        final_path = []
        virtual_edges = set()

        current_type = initial_type
        current_state = initial_state
        current_node = None

        MIN_CONFIDENCE = 0.25
        TUNNELING_MARGIN = 0.15
        MACRO_THRESHOLD = 0.40

        step = 0
        while goals:
            goal = goals.pop(0)
            goal_embedding = self.model.encode(goal, convert_to_tensor=True)

            # 1. Evaluate the best GLOBAL Micro-Cell first to see if we have an exact tool
            global_micro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro"
                and c.cell_id != (current_node.cell_id if current_node else "")
            ]
            best_global_micro, global_micro_score = self._score_and_select_best(
                global_micro_candidates, goal_embedding
            )

            # 2. Evaluate Macro-Cells
            macro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "macro"
            ]
            best_macro, macro_score = self._score_and_select_best(
                macro_candidates, goal_embedding
            )

            # 3. MICRO-PRIORITY OVERWRITE: Unfold Macro ONLY if we don't have a highly-confident Micro tool
            if (
                best_macro
                and macro_score > MACRO_THRESHOLD
                and global_micro_score < 0.70
            ):
                print(
                    f"[FRACTAL UNFOLDING] Abstract goal '{goal}' expanded into {len(best_macro.intent_expansion)} sub-operations."
                )
                # Inject the macro array directly into the front of the queue
                goals = best_macro.intent_expansion + goals
                continue

            if step == 0:
                candidates = [
                    c
                    for c in self.orchestrator.get_all_available_cells()
                    if c.type == "micro"
                    and c.inputs.get("input_type") == current_type
                    and c.inputs.get("expected_state") == current_state
                ]
                best_node, best_score = self._score_and_select_best(
                    candidates, goal_embedding
                )

                if best_score < MIN_CONFIDENCE:
                    print(
                        f"[ROUTER HALT] Entry confidence too low ({best_score:.2f}) for goal: '{goal}'."
                    )
                    break

                final_path.append(best_node)
                current_node = best_node
                current_type = current_node.outputs.get("output_type")
                current_state = current_node.outputs.get("resulting_state")
                step += 1
                continue

            strict_candidates = self.orchestrator.get_neighbors(current_node.cell_id)
            best_local_node, best_local_score = self._score_and_select_best(
                strict_candidates, goal_embedding
            )

            global_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro" and c.cell_id != current_node.cell_id
            ]
            best_global_node, best_global_score = self._score_and_select_best(
                global_candidates, goal_embedding
            )

            if best_global_score > MIN_CONFIDENCE and (
                best_local_score < MIN_CONFIDENCE
                or (best_global_score - best_local_score > TUNNELING_MARGIN)
            ):
                print(
                    f"[ROUTER] Semantic gravity exceeded local bounds! Goal: '{goal}'"
                )
                target_type = best_global_node.inputs.get("input_type")

                if target_type == current_type:
                    print(
                        f"  [+] VIRTUAL EDGE COMPILED! Tunneling to -> {best_global_node.cell_id} (Score: {best_global_score:.2f})"
                    )
                    best_node = best_global_node
                    virtual_edges.add(best_node.cell_id)
                else:
                    print(
                        f"  [!] TYPE MISMATCH: Current '{current_type}' cannot flow into '{target_type}'. Searching for bridge..."
                    )
                    bridge_node = self.orchestrator.find_type_bridge(
                        current_type, target_type
                    )
                    if bridge_node:
                        print(
                            f"  [+] COERCION BRIDGE FOUND! Injecting -> {bridge_node.cell_id}"
                        )
                        final_path.append(bridge_node)
                        virtual_edges.add(bridge_node.cell_id)
                        print(
                            f"  [+] TUNNELING COMPLETED to -> {best_global_node.cell_id} (Score: {best_global_score:.2f})"
                        )
                        best_node = best_global_node
                        virtual_edges.add(best_node.cell_id)
                    else:
                        print(
                            f"  [-] FATAL: No coercion bridge exists between '{current_type}' and '{target_type}'. Path blocked."
                        )
                        break
            elif best_local_score >= MIN_CONFIDENCE:
                best_node = best_local_node
            else:
                print(f"[ROUTER HALT] Pathfinding failed for goal: '{goal}'.")
                break

            final_path.append(best_node)
            current_node = best_node
            current_type = current_node.outputs.get("output_type")
            current_state = current_node.outputs.get("resulting_state")
            step += 1

        print(
            f"\n[PATHFINDER COMPLETE] Route Generated: {[c.cell_id for c in final_path]}"
        )
        return final_path, virtual_edges

    def _score_and_select_best(self, candidates: list, prompt_embedding) -> tuple:
        best_node = None
        best_score = -1.0

        if not candidates:
            return None, -1.0

        for cell in candidates:
            # OPTIMIZATION: Pull from cache instead of re-running the model
            intent_embedding = self.cell_embeddings[cell.cell_id]
            score = util.cos_sim(prompt_embedding, intent_embedding).item()

            if score > best_score:
                best_score = score
                best_node = cell

        return best_node, best_score

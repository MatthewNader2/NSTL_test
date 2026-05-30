# main.py — NSTL Engine (Fixed & Enhanced)
# Fixes: (1) Infinite macro-expansion loop, (2) Micro-priority override, (3) Graceful bridge fallback
# Visual: (1) Cylindrical node layout, (2) Dynamic axis/bounds, (3) Performance-limited edges,
#         (4) Triple-glow active nodes, (5) Smooth 72-frame orbit, (6) Stage-colored zone planes

import json
import logging
import math
import os
import queue
import re
import sys
import threading
import warnings
from collections import defaultdict

import plotly.graph_objects as go
import plotly.io as pio
import webview
from sentence_transformers import SentenceTransformer, util

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# =====================================================================
#  UNIFICATION LAYER (Execution context, variable linking, code gen)
# =====================================================================
class ExecutionContext:
    """Manages variables and literal extraction arguments at runtime using type-state keys."""

    def __init__(self):
        self.registry = {}
        self.extracted_parameters = {}

    def extract_prompt_parameters(self, user_prompt: str):
        self.extracted_parameters = {}
        quoted_items = re.findall(r'["\']([^"\']+)["\']', user_prompt)
        if quoted_items:
            self.extracted_parameters["explicit_filename"] = quoted_items[0]
        else:
            file_match = re.search(
                r"\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|feather|html))\b",
                user_prompt.lower(),
            )
            if file_match:
                self.extracted_parameters["explicit_filename"] = file_match.group(1)

    def declare_variable(self, name: str, var_type: str, state: str) -> str:
        base_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
        sanitized_name = base_name
        counter = 2
        while sanitized_name in self.registry:
            sanitized_name = f"{base_name}_v{counter}"
            counter += 1
        self.registry[sanitized_name] = {"type": var_type, "state": state}
        return sanitized_name

    def find_compatible_variable(self, expected_type: str, expected_state: str) -> str:
        expected_type_clean = expected_type.lower().strip()
        expected_state_clean = (
            expected_state.lower().replace("_", "").replace("-", "").strip()
        )
        for var_name, tracking_info in reversed(list(self.registry.items())):
            current_type = tracking_info["type"].lower().strip()
            current_state = (
                tracking_info["state"].lower().replace("_", "").replace("-", "").strip()
            )
            if current_type == expected_type_clean and (
                expected_state_clean in current_state
                or current_state in expected_state_clean
            ):
                return var_name
        return None


class UnificationGate:
    """Performs dynamic monadic structural unification across cell signatures."""

    @staticmethod
    def unify(context: ExecutionContext, target_cell) -> str:
        matching_input_var = context.find_compatible_variable(
            expected_type=target_cell.inputs["input_type"],
            expected_state=target_cell.inputs["expected_state"],
        )
        if not matching_input_var:
            if context.registry:
                matching_input_var = list(context.registry.keys())[-1]
            else:
                matching_input_var = "input_source"

        raw_output_name = target_cell.outputs["resulting_state"].lower().strip()
        output_var_name = context.declare_variable(
            name=raw_output_name,
            var_type=target_cell.outputs["output_type"],
            state=target_cell.outputs["resulting_state"],
        )
        compiled_snippet = target_cell.code_template
        compiled_snippet = compiled_snippet.replace("{input_var}", matching_input_var)
        compiled_snippet = compiled_snippet.replace("{output_var}", output_var_name)
        if "explicit_filename" in context.extracted_parameters:
            user_assigned_name = context.extracted_parameters["explicit_filename"]
            compiled_snippet = re.sub(
                r"['\"]export\.(?:csv|json|html|feather|parquet)['\"]",
                f"'{user_assigned_name}'",
                compiled_snippet,
            )
            compiled_snippet = compiled_snippet.replace(
                "export.csv", user_assigned_name
            )
        print(
            f"[UNIFICATION SUCCESS] Linked {matching_input_var} -> "
            f"{target_cell.cell_id} -> {output_var_name}"
        )
        return compiled_snippet


# =====================================================================
#  LATTICE DATA STRUCTURES & ORCHESTRATOR
# =====================================================================
class MicroCell:
    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        code_template: str,
        inputs: dict,
        outputs: dict,
        cell_type: str = "micro",
        intent_expansion: list = None,
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(keywords)
        self.type = cell_type
        self.intent_expansion = intent_expansion or []
        self.code_template = code_template
        self.inputs = inputs or {}
        self.outputs = outputs or {}


class LatticeOrchestrator:
    def __init__(self, trees_directory="trees"):
        self.trees_directory = trees_directory
        self.loaded_cells = {}
        self.topology = {}
        self.discover_and_load_trees()
        self.build_topology()

    def discover_and_load_trees(self):
        if not os.path.exists(self.trees_directory):
            return
        for file_name in os.listdir(self.trees_directory):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.trees_directory, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        tree_data = json.load(f)
                    for raw_cell in tree_data.get("cells", []):
                        cell = MicroCell(
                            cell_id=raw_cell.get("cell_id", "UNKNOWN"),
                            stage=raw_cell.get("stage", 0),
                            keywords=raw_cell.get("keywords", []),
                            code_template=raw_cell.get("code_template", ""),
                            inputs=raw_cell.get("inputs", {}),
                            outputs=raw_cell.get("outputs", {}),
                            cell_type=raw_cell.get("type", "micro"),
                            intent_expansion=raw_cell.get("intent_expansion", []),
                        )
                        self.loaded_cells[cell.cell_id] = cell
                except Exception:
                    pass

    def build_topology(self):
        for cell_id in self.loaded_cells:
            self.topology[cell_id] = []
        for cell_a in self.loaded_cells.values():
            if cell_a.type == "macro":
                continue
            for cell_b in self.loaded_cells.values():
                if cell_b.type == "macro":
                    continue
                if cell_a.outputs.get("output_type") == cell_b.inputs.get(
                    "input_type"
                ) and cell_a.outputs.get("resulting_state") == cell_b.inputs.get(
                    "expected_state"
                ):
                    self.topology[cell_a.cell_id].append(cell_b.cell_id)

    def get_all_available_cells(self) -> list:
        return list(self.loaded_cells.values())

    def get_neighbors(self, cell_id: str) -> list:
        neighbor_ids = self.topology.get(cell_id, [])
        return [self.loaded_cells[nid] for nid in neighbor_ids]

    def find_type_bridge(self, from_type: str, to_type: str):
        for cell in self.loaded_cells.values():
            if (
                cell.type == "micro"
                and cell.inputs.get("input_type") == from_type
                and cell.outputs.get("output_type") == to_type
            ):
                return cell
        return None


# =====================================================================
#  NEURAL ROUTER — FIXED
#  Fix 1: expanded_macro_goals set prevents infinite macro re-expansion
#  Fix 2: MAX_GOAL_QUEUE ceiling stops runaway queue growth
#  Fix 3: MICRO-PRIORITY OVERWRITE (ported from router.py)
#  Fix 4: Bridge failure → graceful local fallback instead of hard break
# =====================================================================
class LatticeRouter:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cell_embeddings = {}
        self._precompute_embeddings()

    def _precompute_embeddings(self):
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

        # FIX 1: Track which goal texts have already been macro-expanded.
        # If the same goal text re-enters the queue (circular reference in
        # intent_expansion), it is treated as a micro task instead of re-expanding.
        expanded_macro_goals: set = set()

        # FIX 2: Hard ceiling on the goal queue. Without this, a macro whose
        # intent_expansion list indirectly references itself grows the queue
        # unboundedly, causing the infinite loop seen in the 2nd prompt.
        MAX_GOAL_QUEUE = 60

        current_type = initial_type
        current_state = initial_state
        current_node = None

        MIN_CONFIDENCE = 0.25
        TUNNELING_MARGIN = 0.15
        MACRO_THRESHOLD = 0.40

        step = 0
        while goals:
            # FIX 2: Enforce ceiling before popping
            if len(goals) > MAX_GOAL_QUEUE:
                print(
                    f"[SAFETY] Goal queue ({len(goals)} items) exceeded {MAX_GOAL_QUEUE}. "
                    f"Truncating to prevent infinite expansion."
                )
                goals = goals[:MAX_GOAL_QUEUE]

            goal = goals.pop(0)
            goal_key = goal.lower().strip()
            goal_embedding = self.model.encode(goal, convert_to_tensor=True)

            # FIX 3: MICRO-PRIORITY OVERWRITE — evaluate best global micro FIRST.
            # Only expand a macro when no highly-confident micro tool exists.
            global_micro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro"
                and c.cell_id != (current_node.cell_id if current_node else "")
            ]
            best_global_micro, global_micro_score = self._score_and_select_best(
                global_micro_candidates, goal_embedding
            )

            macro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "macro"
            ]
            best_macro, macro_score = self._score_and_select_best(
                macro_candidates, goal_embedding
            )

            # FIX 1 + 3: Expand macro only if:
            #   • a macro actually won, AND
            #   • no highly-confident micro tool (score >= 0.70) already exists, AND
            #   • this exact goal text has NOT been expanded before (cycle guard)
            if (
                best_macro
                and macro_score > MACRO_THRESHOLD
                and global_micro_score < 0.70
                and goal_key not in expanded_macro_goals
            ):
                expanded_macro_goals.add(goal_key)
                print(
                    f"[FRACTAL UNFOLDING] Abstract goal '{goal}' expanded into "
                    f"{len(best_macro.intent_expansion)} sub-operations."
                )
                goals = best_macro.intent_expansion + goals
                continue

            # ── Step 0: Entry node (must match initial type/state) ─────────
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
                        f"[ROUTER HALT] Entry confidence too low "
                        f"({best_score:.2f}) for goal: '{goal}'."
                    )
                    break
                final_path.append(best_node)
                current_node = best_node
                current_type = current_node.outputs.get("output_type")
                current_state = current_node.outputs.get("resulting_state")
                step += 1
                continue

            # ── Step N: Local neighbours vs global tunneling ───────────────
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
                        f"  [+] VIRTUAL EDGE COMPILED! Tunneling to -> "
                        f"{best_global_node.cell_id} (Score: {best_global_score:.2f})"
                    )
                    best_node = best_global_node
                    virtual_edges.add(best_node.cell_id)
                else:
                    print(
                        f"  [!] TYPE MISMATCH: '{current_type}' cannot flow into "
                        f"'{target_type}'. Searching for bridge..."
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
                            f"  [+] TUNNELING COMPLETED to -> "
                            f"{best_global_node.cell_id} (Score: {best_global_score:.2f})"
                        )
                        best_node = best_global_node
                        virtual_edges.add(best_node.cell_id)
                    else:
                        # FIX 4: Graceful fallback to local path instead of hard break.
                        # The original code did `break` here, which discarded ALL remaining
                        # goals. Now we try the local neighbour, or skip this goal only.
                        print(
                            f"  [-] No bridge between '{current_type}' → '{target_type}'. "
                            f"Falling back to local path..."
                        )
                        if best_local_score >= MIN_CONFIDENCE:
                            best_node = best_local_node
                            print(
                                f"  [~] Local fallback: {best_local_node.cell_id} "
                                f"(Score: {best_local_score:.2f})"
                            )
                        else:
                            print(f"  [-] No viable path for goal: '{goal}'. Skipping.")
                            step += 1
                            continue

            elif best_local_score >= MIN_CONFIDENCE:
                best_node = best_local_node
            else:
                # FIX 4: Skip instead of halt — remaining goals may still be routable
                print(f"[ROUTER SKIP] No path found for goal: '{goal}'. Continuing...")
                step += 1
                continue

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
            intent_embedding = self.cell_embeddings[cell.cell_id]
            score = util.cos_sim(prompt_embedding, intent_embedding).item()
            if score > best_score:
                best_score = score
                best_node = cell
        return best_node, best_score


# =====================================================================
#  INTERACTIVE VISUALIZATION — WebGL 3D Lattice (ENHANCED)
# =====================================================================
class Api:
    """Exposed to JavaScript via pywebview. Provides plot HTML via polling."""

    def __init__(self, update_queue):
        self._queue = update_queue

    def get_new_html(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class VisualizerApp:
    """Manages the standalone native window for the WebGL visualization."""

    def __init__(self):
        self.html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                * { box-sizing: border-box; }
                body {
                    margin: 0;
                    background: #05070f;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    overflow: hidden;
                }
                iframe {
                    width: 100%;
                    height: 100%;
                    border: none;
                    background: #05070f;
                }
            </style>
        </head>
        <body>
            <iframe id="plotframe" src="about:blank"></iframe>
            <script>
                async function pollPlot() {
                    try {
                        const newHtml = await window.pywebview.api.get_new_html();
                        if (newHtml) {
                            document.getElementById('plotframe').srcdoc = newHtml;
                        }
                    } catch (e) { /* pywebview not ready yet */ }
                }
                setInterval(pollPlot, 500);
            </script>
        </body>
        </html>
        """
        self._plot_queue = queue.Queue()
        self.window = webview.create_window(
            "NSTL Cyber-Lattice Engine",
            html=self.html,
            width=1400,
            height=900,
            resizable=True,
            min_size=(800, 600),
            text_select=True,
            js_api=Api(self._plot_queue),
        )

    def enqueue_visualization(self, fig):
        html_str = pio.to_html(
            fig, full_html=True, include_plotlyjs="cdn", validate=False
        )
        self._plot_queue.put(html_str)

    def start_ui(self):
        webview.start(gui="edgechromium", debug=False)


class WebGLGraphVisualizer:
    """
    ENHANCED: High-quality, performance-optimised dark-mode WebGL 3D network.

    Changes vs original:
    • Cylindrical node layout per stage (concentric rings) — cleaner than flat grid
    • Dynamic axis tick values and stage zone bounds from actual data (no more hardcoded [8,16,24,32,40])
    • Performance cap on base-lattice edges (MAX_BASE_EDGES=500, active-stage filter)
      — fixes O(n²) slowdown with 1262 nodes
    • Triple-layer glow on active/tunnel nodes (outer halo → mid ring → solid core)
    • Tunnel nodes rendered as diamonds instead of circles for instant visual distinction
    • 72-frame camera orbit (5° per frame) instead of 60-frame (6° per frame) for smoother spin
    • Stage zone plane bounds computed from actual min/max Y/Z instead of hard -5..25
    • Stage zones colour-coded along a deep-space palette gradient
    • Degree (connectivity count) shown in hover tooltips
    """

    # Deep-space palette: one shade per stage, cycling if more stages exist
    _ZONE_COLORS = [
        "#051830",
        "#052e2e",
        "#062a10",
        "#1a2a06",
        "#2a1806",
        "#2a0618",
        "#18062a",
        "#060e2a",
    ]
    # Readable stage name fallbacks
    _STAGE_LABELS = [
        "Ingest",
        "Clean",
        "Transform",
        "Aggregate",
        "Export",
        "Analyze",
        "Encode",
        "Output",
    ]

    @staticmethod
    def generate_3d_lattice(all_cells: list, active_path: list, virtual_edges: set):

        # ── 1. GROUP MICRO CELLS BY STAGE ────────────────────────────────
        stage_groups: dict = defaultdict(list)
        for cell in sorted(all_cells, key=lambda c: (c.stage, c.cell_id)):
            if cell.type == "micro":
                stage_groups[cell.stage].append(cell)

        # ── 2. OUTBOUND CONNECTIVITY DEGREE ──────────────────────────────
        degree: dict = defaultdict(int)
        for ca in all_cells:
            if ca.type != "micro":
                continue
            for cb in all_cells:
                if cb.type != "micro":
                    continue
                if ca.outputs.get("output_type") == cb.inputs.get(
                    "input_type"
                ) and ca.outputs.get("resulting_state") == cb.inputs.get(
                    "expected_state"
                ):
                    degree[ca.cell_id] += 1

        # ── 3. CYLINDRICAL 3D COORDINATES ────────────────────────────────
        # Nodes arranged in concentric rings around each stage's X axis position.
        # Ring 0 holds RING_CAP nodes at radius R_BASE; each successive ring
        # expands outward by R_STEP and is rotated by half a slot to stagger.
        node_coords: dict = {}
        STAGE_X_GAP = 12.0
        RING_CAP = 8
        R_BASE = 5.0
        R_STEP = 5.5

        for stage, cells in stage_groups.items():
            n = len(cells)
            x_pos = stage * STAGE_X_GAP
            for idx, cell in enumerate(cells):
                if n == 1:
                    yc, zc = 0.0, 0.0
                else:
                    ring = idx // RING_CAP
                    slot = idx % RING_CAP
                    ring_n = min(RING_CAP, n - ring * RING_CAP)
                    angle = (slot / ring_n) * 2.0 * math.pi
                    angle += ring * (math.pi / RING_CAP)  # stagger rings
                    radius = R_BASE + ring * R_STEP
                    yc = radius * math.cos(angle)
                    zc = radius * math.sin(angle)
                node_coords[cell.cell_id] = (x_pos, yc, zc)

        # ── 4. CLASSIFY ACTIVE / TUNNEL SETS ─────────────────────────────
        active_ids = {c.cell_id for c in active_path}
        path_indices = {c.cell_id: i + 1 for i, c in enumerate(active_path)}
        active_stages = {c.stage for c in active_path}

        # ── 5. DYNAMIC AXIS BOUNDS FROM REAL DATA ─────────────────────────
        all_pts = list(node_coords.values())
        if all_pts:
            PAD = 4.0
            min_y = min(p[1] for p in all_pts) - PAD
            max_y = max(p[1] for p in all_pts) + PAD
            min_z = min(p[2] for p in all_pts) - PAD
            max_z = max(p[2] for p in all_pts) + PAD
        else:
            min_y, max_y, min_z, max_z = -10.0, 10.0, -10.0, 10.0

        # ── 6. BASE LATTICE EDGES — PERFORMANCE-LIMITED ───────────────────
        # Only draw edges for stages near the active path.  Without this cap,
        # 1262 nodes produce ≈1.6 M iteration attempts, crashing the render.
        MAX_BASE_EDGES = 500
        edge_count = 0
        base_ex, base_ey, base_ez = [], [], []

        near_stages = set()
        for s in active_stages:
            near_stages.update([s - 1, s, s + 1])
        if not near_stages:
            near_stages = set(stage_groups.keys())

        for ca in all_cells:
            if edge_count >= MAX_BASE_EDGES:
                break
            if ca.type != "micro" or ca.stage not in near_stages:
                continue
            if ca.cell_id not in node_coords:
                continue
            for cb in all_cells:
                if cb.type != "micro" or cb.cell_id not in node_coords:
                    continue
                if ca.outputs.get("output_type") == cb.inputs.get(
                    "input_type"
                ) and ca.outputs.get("resulting_state") == cb.inputs.get(
                    "expected_state"
                ):
                    x1, y1, z1 = node_coords[ca.cell_id]
                    x2, y2, z2 = node_coords[cb.cell_id]
                    base_ex.extend([x1, x2, None])
                    base_ey.extend([y1, y2, None])
                    base_ez.extend([z1, z2, None])
                    edge_count += 1
                    if edge_count >= MAX_BASE_EDGES:
                        break

        # ── 7. ACTIVE PATH EDGES ──────────────────────────────────────────
        act_ex, act_ey, act_ez = [], [], []
        tun_ex, tun_ey, tun_ez = [], [], []

        for i in range(len(active_path) - 1):
            ca, cb = active_path[i], active_path[i + 1]
            if ca.cell_id not in node_coords or cb.cell_id not in node_coords:
                continue
            x1, y1, z1 = node_coords[ca.cell_id]
            x2, y2, z2 = node_coords[cb.cell_id]
            if cb.cell_id in virtual_edges:
                tun_ex.extend([x1, x2, None])
                tun_ey.extend([y1, y2, None])
                tun_ez.extend([z1, z2, None])
            else:
                act_ex.extend([x1, x2, None])
                act_ey.extend([y1, y2, None])
                act_ez.extend([z1, z2, None])

        # ── 8. FLOW CONES ─────────────────────────────────────────────────
        cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
        for i in range(len(active_path) - 1):
            ca, cb = active_path[i], active_path[i + 1]
            if ca.cell_id not in node_coords or cb.cell_id not in node_coords:
                continue
            x1, y1, z1 = node_coords[ca.cell_id]
            x2, y2, z2 = node_coords[cb.cell_id]
            dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
            mag = math.sqrt(dx**2 + dy**2 + dz**2)
            if mag > 0:
                dx, dy, dz = dx / mag, dy / mag, dz / mag
                cx.append(x1 + dx * mag * 0.65)
                cy.append(y1 + dy * mag * 0.65)
                cz.append(z1 + dz * mag * 0.65)
                cu.append(dx)
                cv.append(dy)
                cw.append(dz)

        # ── 9. SEPARATE NODE COORDINATE LISTS ────────────────────────────
        in_x, in_y, in_z = [], [], []
        ac_x, ac_y, ac_z, ac_lbl, ac_hvr = [], [], [], [], []
        tu_x, tu_y, tu_z, tu_lbl, tu_hvr = [], [], [], [], []

        for cell in all_cells:
            if cell.cell_id not in node_coords:
                continue
            xc, yc, zc = node_coords[cell.cell_id]
            hvr = (
                f"<b>{cell.cell_id}</b><br>"
                f"Stage: {cell.stage}<br>"
                f"In:  {cell.inputs.get('input_type', '?')} "
                f"[{cell.inputs.get('expected_state', '?')}]<br>"
                f"Out: {cell.outputs.get('output_type', '?')} "
                f"[{cell.outputs.get('resulting_state', '?')}]<br>"
                f"Connections: {degree.get(cell.cell_id, 0)}"
            )
            if cell.cell_id in active_ids:
                n = path_indices[cell.cell_id]
                lbl = f"[{n}] {cell.cell_id}"
                if cell.cell_id in virtual_edges:
                    tu_x.append(xc)
                    tu_y.append(yc)
                    tu_z.append(zc)
                    tu_lbl.append(lbl)
                    tu_hvr.append(hvr)
                else:
                    ac_x.append(xc)
                    ac_y.append(yc)
                    ac_z.append(zc)
                    ac_lbl.append(lbl)
                    ac_hvr.append(hvr)
            else:
                in_x.append(xc)
                in_y.append(yc)
                in_z.append(zc)

        # ── 10. ASSEMBLE ALL PLOTLY TRACES ────────────────────────────────
        traces = []
        sorted_stages = sorted(stage_groups.keys())

        # Stage zone planes — dynamic bounds, unique colour per stage
        for i, stage in enumerate(sorted_stages):
            xp = stage * STAGE_X_GAP
            col = WebGLGraphVisualizer._ZONE_COLORS[
                i % len(WebGLGraphVisualizer._ZONE_COLORS)
            ]
            traces.append(
                go.Mesh3d(
                    x=[xp, xp, xp, xp],
                    y=[min_y, max_y, max_y, min_y],
                    z=[min_z, min_z, max_z, max_z],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color=col,
                    opacity=0.07,
                    flatshading=True,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Base lattice (faint, nearby-stage only)
        if base_ex:
            traces.append(
                go.Scatter3d(
                    x=base_ex,
                    y=base_ey,
                    z=base_ez,
                    mode="lines",
                    line=dict(color="#0d1f4a", width=1),
                    opacity=0.30,
                    name="Lattice Edges",
                    hoverinfo="skip",
                )
            )

        # Active path — outer glow + core line
        if act_ex:
            traces.append(
                go.Scatter3d(  # glow halo
                    x=act_ex,
                    y=act_ey,
                    z=act_ez,
                    mode="lines",
                    line=dict(color="#00e5ff", width=20),
                    opacity=0.06,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            traces.append(
                go.Scatter3d(  # solid core
                    x=act_ex,
                    y=act_ey,
                    z=act_ez,
                    mode="lines",
                    line=dict(color="#00e5ff", width=6),
                    name="Execution Path",
                    hoverinfo="skip",
                )
            )

        # Tunnel edges
        if tun_ex:
            traces.append(
                go.Scatter3d(
                    x=tun_ex,
                    y=tun_ey,
                    z=tun_ez,
                    mode="lines",
                    line=dict(color="#ff00aa", width=5, dash="dash"),
                    name="Semantic Tunnel",
                    hoverinfo="skip",
                )
            )

        # Flow cones
        if cx:
            traces.append(
                go.Cone(
                    x=cx,
                    y=cy,
                    z=cz,
                    u=cu,
                    v=cv,
                    w=cw,
                    colorscale=[[0, "#00e5ff"], [1, "#00e5ff"]],
                    sizemode="absolute",
                    sizeref=0.45,
                    showscale=False,
                    name="Flow Direction",
                    hoverinfo="skip",
                )
            )

        # Dormant nodes
        if in_x:
            traces.append(
                go.Scatter3d(
                    x=in_x,
                    y=in_y,
                    z=in_z,
                    mode="markers",
                    marker=dict(
                        color="#192850",
                        size=4,
                        opacity=0.40,
                        line=dict(color="#2a4070", width=0.5),
                    ),
                    name="Dormant Nodes",
                    hoverinfo="skip",
                )
            )

        # Active nodes — triple glow + label
        if ac_x:
            for sz, op in [(40, 0.04), (24, 0.10)]:  # two halo layers
                traces.append(
                    go.Scatter3d(
                        x=ac_x,
                        y=ac_y,
                        z=ac_z,
                        mode="markers",
                        marker=dict(color="#00e5ff", size=sz, opacity=op),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            traces.append(
                go.Scatter3d(  # solid core + label
                    x=ac_x,
                    y=ac_y,
                    z=ac_z,
                    mode="markers+text",
                    text=ac_lbl,
                    textposition="top center",
                    textfont=dict(
                        color="#e0f7ff", size=10, family="'Courier New', monospace"
                    ),
                    hovertext=ac_hvr,
                    hoverinfo="text",
                    marker=dict(
                        color="#00e5ff",
                        size=12,
                        opacity=1.0,
                        line=dict(color="#ffffff", width=2),
                        symbol="circle",
                    ),
                    name="Active Nodes",
                )
            )

        # Tunnel nodes — triple glow + diamond marker
        if tu_x:
            for sz, op in [(36, 0.05), (22, 0.12)]:
                traces.append(
                    go.Scatter3d(
                        x=tu_x,
                        y=tu_y,
                        z=tu_z,
                        mode="markers",
                        marker=dict(color="#ff00aa", size=sz, opacity=op),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            traces.append(
                go.Scatter3d(
                    x=tu_x,
                    y=tu_y,
                    z=tu_z,
                    mode="markers+text",
                    text=tu_lbl,
                    textposition="top center",
                    textfont=dict(
                        color="#ffd6f0", size=10, family="'Courier New', monospace"
                    ),
                    hovertext=tu_hvr,
                    hoverinfo="text",
                    marker=dict(
                        color="#ff00aa",
                        size=11,
                        opacity=1.0,
                        line=dict(color="#ffffff", width=2),
                        symbol="diamond",  # visually distinct from active nodes
                    ),
                    name="Tunnel Nodes",
                )
            )

        # ── 11. SMOOTH 72-FRAME CAMERA ORBIT ─────────────────────────────
        frames = [
            go.Frame(
                layout=dict(
                    scene=dict(
                        camera=dict(
                            eye=dict(
                                x=1.8 * math.cos(math.radians(i * 5)),
                                y=1.8 * math.sin(math.radians(i * 5)),
                                z=0.75,
                            )
                        )
                    )
                )
            )
            for i in range(72)  # 5° per frame → smooth 360° orbit
        ]

        # ── 12. DYNAMIC AXIS TICKS ────────────────────────────────────────
        tick_vals = [s * STAGE_X_GAP for s in sorted_stages]
        lbl_pool = WebGLGraphVisualizer._STAGE_LABELS
        tick_texts = [
            f"S{s}: {lbl_pool[i] if i < len(lbl_pool) else 'Step'}"
            for i, s in enumerate(sorted_stages)
        ]

        # ── 13. LAYOUT ────────────────────────────────────────────────────
        ax_common = dict(
            backgroundcolor="#05070f",
            gridcolor="#0a1828",
            showbackground=True,
            zeroline=False,
        )
        layout = go.Layout(
            paper_bgcolor="#05070f",
            plot_bgcolor="#05070f",
            font=dict(family="'Courier New', Courier, monospace", color="#8892b0"),
            title=dict(
                text=(
                    "<b style='color:#ccd6f6;'>NSTL CYBER-LATTICE 3D ENGINE</b>"
                    "<br><span style='font-size:11px;color:#00e5ff;'>"
                    "Matthew Nader · ID: 202300103 · Real-Time Execution Visualizer"
                    "</span>"
                ),
                font=dict(size=18),
                x=0.5,
                xanchor="center",
            ),
            scene=dict(
                xaxis=dict(
                    **ax_common,
                    title=dict(
                        text="EXECUTION STAGE", font=dict(color="#4a6080", size=10)
                    ),
                    tickvals=tick_vals,
                    ticktext=tick_texts,
                    tickfont=dict(color="#4a5568", size=8),
                ),
                yaxis=dict(
                    **ax_common,
                    title=dict(text="", font=dict(color="#4a6080")),
                    tickfont=dict(color="#2a3a5e"),
                    showticklabels=False,
                ),
                zaxis=dict(
                    **ax_common,
                    title=dict(text="", font=dict(color="#4a6080")),
                    tickfont=dict(color="#2a3a5e"),
                    showticklabels=False,
                ),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
                aspectratio=dict(x=1.8, y=1, z=1),
            ),
            margin=dict(l=0, r=0, b=0, t=90),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.97,
                font=dict(color="#8892b0", family="'Courier New', monospace", size=10),
                bgcolor="rgba(5, 7, 16, 0.88)",
                bordercolor="#00e5ff",
                borderwidth=1,
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.12,
                    x=0.01,
                    direction="right",
                    buttons=[
                        dict(
                            label="▶ Orbit",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=80, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )

        return go.Figure(data=traces, layout=layout, frames=frames)


# =====================================================================
#  MAIN EXECUTION LOGIC
# =====================================================================
def run_interactive_engine(user_prompt: str, router, orchestrator, app: VisualizerApp):
    all_available = orchestrator.get_all_available_cells()
    if not all_available:
        print("[ABORT] Engine failed to parse files from the trees/ folder database.")
        return

    context = ExecutionContext()
    context.extract_prompt_parameters(user_prompt)
    context.declare_variable(
        name="input_source", var_type="str", state="source_identifier"
    )

    execution_path, virtual_edges = router.plan_path(
        user_prompt, initial_type="str", initial_state="source_identifier"
    )

    compiled_blocks = []
    for cell in execution_path:
        code_block = UnificationGate.unify(context, cell)
        if code_block is not None:
            compiled_blocks.append(code_block)

    print("\n" + "=" * 80)
    print(" UNIFIED REASONING SCRIPT SYSTEM PRODUCTION OUTPUT")
    print("=" * 80)
    if compiled_blocks:
        print("\n".join(compiled_blocks))
    else:
        print(
            "[SYSTEM ALERT] Empty assembly pipeline generated. Constraints mismatched."
        )
    print("=" * 80 + "\n")

    print("[3D VIEW] Rendering updated GPU-accelerated WebGL 3D Lattice...")
    physical_cells = [
        c for c in all_available if getattr(c, "type", "micro") == "micro"
    ]
    physical_path = [
        c for c in execution_path if getattr(c, "type", "micro") == "micro"
    ]

    fig = WebGLGraphVisualizer.generate_3d_lattice(
        physical_cells, physical_path, virtual_edges
    )
    app.enqueue_visualization(fig)


def cli_loop(router, orchestrator, app):
    print("System initialization stable. Ready for user inquiries.")
    print("Type 'exit' to close the interface loop.\n")
    while True:
        try:
            user_input = input("NSTL-Client-Prompt > ")
        except EOFError:
            break
        if user_input.strip().lower() == "exit":
            print("Shutting down engine core systems.")
            if app.window:
                app.window.destroy()
            break
        if not user_input.strip():
            continue
        run_interactive_engine(user_input, router, orchestrator, app)


if __name__ == "__main__":
    print("======================================================================")
    print("|  NSTL SOTA PERSISTENT NEURAL INTENT INTERFACE ONLINE               |")
    print("======================================================================")
    print("Initializing engine architecture core subsystems exactly once...")

    global_orchestrator = LatticeOrchestrator()
    global_router = LatticeRouter(global_orchestrator)
    app = VisualizerApp()

    cli_thread = threading.Thread(
        target=cli_loop, args=(global_router, global_orchestrator, app), daemon=True
    )
    cli_thread.start()

    app.start_ui()

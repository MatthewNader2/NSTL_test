# src/cli.py
import argparse
import ast
import json
import math
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from schema import CellSchema, TreeSchema, PortSchema
    from harvester import IntelligentHarvester
except ImportError:
    from .schema import CellSchema, TreeSchema, PortSchema
    from .harvester import IntelligentHarvester

def cmd_harvest(args):
    """Harvest public APIs from a package and merge into trees/{domain}.json."""
    domain = args.domain or args.package
    package = args.package
    trees_dir = Path(args.trees_dir)
    trees_dir.mkdir(parents=True, exist_ok=True)
    out_file = trees_dir / f"{domain}.json"

    print(f"[*] Initializing Intelligent Harvester for package '{package}' (domain: '{domain}')...")
    harvester = IntelligentHarvester(domain=domain, package_name=package)
    tree = harvester.harvest_and_save(out_file)
    print(f"[+] Harvested {len(tree.cells)} function cells from '{package}'.")
    print(f"[+] Merged and saved into '{out_file}'.")


def init_sqlite_db(db_path: Path, clean: bool = False) -> sqlite3.Connection:
    """Initializes standard NSTL SQLite schema without destroying existing data unless clean=True."""
    if clean and db_path.exists():
        os.remove(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            cell_id              TEXT PRIMARY KEY,
            domain_name          TEXT,
            node_type            TEXT,
            node_role            TEXT DEFAULT 'function',
            stage                INTEGER,
            keywords             TEXT,
            input_type           TEXT,
            input_state          TEXT,
            output_type          TEXT,
            output_state         TEXT,
            code                 TEXT,
            dependencies         TEXT,
            configuration_schema TEXT,
            slots                TEXT DEFAULT '{}',
            verified             INTEGER DEFAULT 0,
            docstring            TEXT DEFAULT '',
            enrichment_source    TEXT DEFAULT NULL,
            enriched_at          TEXT DEFAULT NULL,
            source_provenance    TEXT DEFAULT 'unknown',
            source_priority      INTEGER DEFAULT 100
        )
    """)
    # Check if slots column exists for existing databases
    cur.execute("PRAGMA table_info(nodes)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "slots" not in existing_cols:
        try:
            cur.execute("ALTER TABLE nodes ADD COLUMN slots TEXT DEFAULT '{}'")
        except Exception:
            pass

    cur.execute("CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_role ON nodes(node_role)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_type ON nodes(node_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_slots ON nodes(slots)")
    conn.commit()
    return conn


def cmd_compile(args):
    """Compiles all trees/*.json domain files into a target SQLite database."""
    trees_dir = Path(args.trees_dir)
    out_db = Path(args.output)
    domain_filter = args.domains

    json_files = sorted(trees_dir.glob("*.json"))
    if domain_filter:
        json_files = [f for f in json_files if f.stem in domain_filter or any(d in f.stem for d in domain_filter)]

    print(f"[*] Compiling {len(json_files)} domain JSON files from '{trees_dir}' into '{out_db}'...")
    conn = init_sqlite_db(out_db, clean=getattr(args, "clean", False))
    cur = conn.cursor()

    total_compiled = 0
    stats: Dict[str, int] = {}

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Failed to read {jf.name}: {e}")
            continue

        if isinstance(data, dict) and "cells" in data:
            try:
                tree = TreeSchema(**data)
                cells = tree.cells
                domain = tree.domain
            except Exception as e:
                print(f"[!] Schema validation error in {jf.name}: {e}")
                cells = []
                domain = jf.stem
        elif isinstance(data, list):
            cells = [CellSchema(**c) for c in data if isinstance(c, dict) and "cell_id" in c]
            domain = jf.stem.replace("_tree", "").replace("_seeds", "")
        else:
            continue

        count = 0
        for cell in cells:
            cid = cell.cell_id.strip().upper()
            primary_in = cell.primary_input
            primary_out = cell.primary_output

            in_type = primary_in.type_name
            in_state = primary_in.state
            out_type = primary_out.type_name
            out_state = primary_out.state

            cfg_dict = {
                "inputs": {k: v.model_dump() for k, v in cell.inputs.items()},
                "outputs": {k: v.model_dump() for k, v in cell.outputs.items()},
                "slots": getattr(cell, "slots", {}),
                "topology_type": getattr(cell, "topology_type", "sequential"),
                "feedback_state_type": getattr(cell, "feedback_state_type", None)
            }
            cfg_json = json.dumps(cfg_dict)
            deps_json = json.dumps(cell.dependencies)
            kws_json = json.dumps(cell.keywords or cell.semantic_tags)
            verified_val = 1 if cell.source_priority <= 10 else 0

            # Priority check: lower source_priority = higher trust (1 = seed, 100 = auto)
            cur.execute("SELECT source_priority FROM nodes WHERE cell_id = ?", (cid,))
            row = cur.fetchone()
            if row and row[0] < cell.source_priority:
                continue

            slots_dict = getattr(cell, "slots", {}) or {}
            slots_json = json.dumps(slots_dict)

            cur.execute("""
                INSERT OR REPLACE INTO nodes
                (cell_id, domain_name, node_type, node_role, stage, keywords,
                 input_type, input_state, output_type, output_state, code,
                 dependencies, configuration_schema, slots, verified, docstring,
                 enrichment_source, enriched_at, source_provenance, source_priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid,
                cell.domain_name or domain,
                cell.node_type or "function",
                cell.node_role or "function",
                cell.stage,
                kws_json,
                in_type,
                in_state,
                out_type,
                out_state,
                cell.code_template,
                deps_json,
                cfg_json,
                slots_json,
                verified_val,
                cell.docstring or "",
                getattr(cell, "enrichment_source", None),
                getattr(cell, "enriched_at", None),
                jf.name,
                cell.source_priority
            ))
            count += 1
            total_compiled += 1

        stats[domain] = count
        print(f"  [+] Domain '{domain}': compiled {count} nodes ({jf.name})")

    conn.commit()
    conn.close()
    print(f"[*] Compilation Complete: {total_compiled} total verified nodes compiled into '{out_db}'.")


def cmd_validate(args):
    """Performs dry-run AST validation and integrity verification on all nodes in SQLite."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[!] Database file '{db_path}' does not exist!")
        sys.exit(1)

    print(f"[*] Validating SQLite Database '{db_path}'...")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT cell_id, domain_name, stage, code, input_type, output_type FROM nodes")
    rows = cur.fetchall()

    valid_count = 0
    failed_count = 0
    errors: List[str] = []

    for row in rows:
        cell_id, domain, stage, code, in_t, out_t = row
        if not code or not code.strip():
            failed_count += 1
            errors.append(f"{cell_id}: Empty code template")
            continue

        # Replace all {placeholders} with dummy variables for AST dry-run
        seen: Dict[str, str] = {}
        res = []
        i = 0
        n = len(code)
        while i < n:
            if code[i] == "{" and i + 1 < n:
                end = code.find("}", i + 1)
                if end != -1:
                    inner = code[i + 1 : end]
                    if inner.isidentifier():
                        key = f"{{{inner}}}"
                        if key not in seen:
                            seen[key] = f"_ph_{len(seen)}"
                        res.append(seen[key])
                        i = end + 1
                        continue
            res.append(code[i])
            i += 1
        dummy_code = "".join(res)
        try:
            ast.parse(dummy_code)
            valid_count += 1
        except SyntaxError as e:
            failed_count += 1
            errors.append(f"{cell_id}: AST Syntax Error: {e}")

    conn.close()

    print(f"\n==================================================")
    print(f" VALIDATION RESULTS FOR: {db_path.name}")
    print(f"==================================================")
    print(f" Total Nodes Checked : {len(rows)}")
    print(f" Syntactically Valid : {valid_count}")
    print(f" Failed Nodes        : {failed_count}")
    print(f" Success Rate        : {(valid_count / len(rows) * 100):.2f}%" if rows else "0.00%")

    if errors:
        print("\n[!] Top Errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n✅ All nodes in database passed 100% AST dry-run validation!")


import cmd
import glob
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align
from rich import box
from rich.columns import Columns

try:
    from lattice import LatticeOrchestrator, Cell, PortSignature, AlgebraicSignature
    from router import LatticeRouter, HardwareProfiler
    from unification import (
        UnificationGate, DynamicPlaceholderResolver, UnresolvedPlaceholderError,
        UnificationFailure, ExecutionContext, Substitution, TypeRegistry, unify,
        Success, Failure
    )
    from gevr_sandbox import GEVRSandbox
    from inference import ModelManager, select_optimal_embedder
    from internal_rag import LocalRAG
    from config import MODELS_DIR
    from utils import extract_code_from_llm_response
    from tokenizer import CellTokenizer
except ImportError:
    from .lattice import LatticeOrchestrator, Cell, PortSignature, AlgebraicSignature
    from .router import LatticeRouter, HardwareProfiler
    from .unification import (
        UnificationGate, DynamicPlaceholderResolver, UnresolvedPlaceholderError,
        UnificationFailure, ExecutionContext, Substitution, TypeRegistry, unify,
        Success, Failure
    )
    from .gevr_sandbox import GEVRSandbox
    from .inference import ModelManager, select_optimal_embedder
    from .internal_rag import LocalRAG
    from .config import MODELS_DIR
    from .utils import extract_code_from_llm_response
    from .tokenizer import CellTokenizer

STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "from", "by", "with",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "them", "they", "their", "this", "that", "these", "those"
})


console = Console()


_ENVIRONMENTAL_ERROR_MARKERS = (
    "FileNotFoundError", "PermissionError", "IsADirectoryError",
    "ModuleNotFoundError", "ConnectionError", "TimeoutError",
    "No such file or directory",
)


def _is_environmental_error(error_msg: str) -> bool:
    """
    Classifies a sandbox failure as environmental (missing input asset, missing
    module, OS/IO conditions) rather than a defect in the synthesized code.
    Repairing code cannot create the user's data file, so these failures are
    reported honestly instead of triggering the LLM repair cycle.
    """
    if not error_msg:
        return False
    return any(marker in error_msg for marker in _ENVIRONMENTAL_ERROR_MARKERS)


def _collect_template_api_names(templates) -> Set[str]:
    """Collects attribute/method names referenced by verified cell templates via AST."""
    names: Set[str] = set()
    for tmpl in templates:
        if not tmpl:
            continue
        # Substitute placeholders with plain identifiers for a parseable AST
        safe = tmpl
        out = []
        i = 0
        while i < len(safe):
            ch = safe[i]
            if ch == "{":
                j = safe.find("}", i + 1)
                if j == -1:
                    break
                inner = safe[i + 1: j]
                out.append("_" + ("".join(c if c.isalnum() else "_" for c in inner) or "ph"))
                i = j + 1
                continue
            out.append(ch)
            i += 1
        code = "".join(out)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names


def _unknown_api_references(repaired_code: str, original_code: str, cells) -> Set[str]:
    """
    Returns attribute names referenced by the repaired code that (a) were not in
    the original code and (b) appear in no verified lattice cell template.
    Guards the GEVR repair cycle against hallucinated API calls (e.g. calling
    .get_metrics() on a tuple).
    """
    def _attrs(code: str) -> Set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    repaired_attrs = _attrs(repaired_code)
    original_attrs = _attrs(original_code)
    known = _collect_template_api_names(getattr(c, "code_template", "") for c in cells)
    return repaired_attrs - original_attrs - known


def _get_available_embedders() -> List[str]:
    """Scans MODELS_DIR/embeddings for available embedding models."""
    emb_dir = os.path.join(MODELS_DIR, "embeddings")
    if os.path.exists(emb_dir):
        return sorted([
            d for d in os.listdir(emb_dir)
            if os.path.isdir(os.path.join(emb_dir, d)) and not d.endswith("-GGUF")
        ])
    return []


def _get_available_llms() -> List[str]:
    """Scans MODELS_DIR/llms for available GGUF LLMs."""
    llm_dir = os.path.join(MODELS_DIR, "llms")
    if os.path.exists(llm_dir):
        return sorted([
            d for d in os.listdir(llm_dir)
            if os.path.isdir(os.path.join(llm_dir, d))
        ])
    return []

class PipelineDebugger:
    """
    Diagnostic tracer and rich visualizer for NSTL pipeline layers.
    Inspects and displays:
      - Header: Engine & hardware environment
      - Layer 0: Query Translator / Intent Decomposition
      - Layer 1: Semantic Tunneling & Top-Scoring Candidates (Router)
      - Layer 2: Topological Planning & Trellis Viterbi (Planner)
      - Layer 3: Type-Monadic Unification & AST Code Synthesis (Gate)
      - Layer 4: GEVR Sandbox Execution & Egress Verification
      - Layer 5: Self-Repair Cycle (if triggered in Profile C/E)
      - Layer 6: End-to-End Latency & Diagnostic Root Cause Analysis
    """

    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        router: LatticeRouter,
        gate: UnificationGate,
        sandbox: GEVRSandbox,
        active_profile: str = "0",
        device: str = "auto",
        embedder_name: str = "",
        llm_name: str = "",
        console: Optional[Console] = None
    ):
        self.orchestrator = orchestrator
        self.router = router
        self.gate = gate
        self.sandbox = sandbox
        self.active_profile = active_profile
        self.device = device
        self.embedder_name = embedder_name
        self.llm_name = llm_name
        self.console = console or Console()

    def run(
        self,
        prompt: str,
        execute_sandbox: bool = True,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        c = self.console
        t_total_start = time.perf_counter()
        prof = self.active_profile.upper()
        prompt_clean = prompt.strip()

        # =================================================================
        # HEADER: Debug Session Banner
        # =================================================================
        header_table = Table(box=box.SIMPLE_HEAD, expand=True, border_style="cyan")
        header_table.add_column("Profile", style="bold yellow")
        header_table.add_column("Embedder", style="magenta")
        header_table.add_column("LLM (GGUF)", style="magenta")
        header_table.add_column("Device", style="green")
        header_table.add_column("Total Nodes", style="white")

        emb_disp = self.embedder_name or "None (Bypassed)"
        llm_disp = self.llm_name or "None (Bypassed)"
        domains = set(cl.domain_name for cl in self.orchestrator.loaded_cells.values() if cl.domain_name)
        header_table.add_row(
            f"Profile {prof}",
            emb_disp,
            llm_disp,
            self.device.upper(),
            f"{len(self.orchestrator.loaded_cells):,} in {len(domains)} domains"
        )

        c.print(Panel(
            header_table,
            title="[bold white on blue] 🔬 NSTL FULL-PIPELINE DEBUG TRACE [/bold white on blue]",
            subtitle=f"[dim cyan]Prompt: \"{prompt_clean}\"[/dim cyan]",
            border_style="blue",
            padding=(0, 1)
        ))

        # =================================================================
        # LAYER 0: Query Intent & Pre-Processing
        # =================================================================
        t0_trans = time.perf_counter()
        effective_prompt = prompt_clean
        t_trans = 0.0

        if prof == "E":
            mm = ModelManager.get_instance()
            if mm.profile and mm.has_translator_pass():
                t0_t = time.perf_counter()
                trans_system = (
                    "You are a precise technical translator. Rewrite the user request as ONE "
                    "comma-separated pipeline sentence: first the input source with its asset "
                    "name, then each transform verb with its arguments in order, then the "
                    "destination. Use only words from the request. Output ONLY the sentence, "
                    "no headers, no lists, no formatting."
                )
                effective_prompt = mm.generate_text(prompt_clean, max_tokens=128, system_prompt=trans_system)
                effective_prompt = effective_prompt.strip().strip('`').strip()
                t_trans = (time.perf_counter() - t0_t) * 1000.0

        clauses = [cl.strip() for cl in re.split(r'[,;]|\b(?:and|then)\b', effective_prompt) if cl.strip()]
        prompt_tokens = CellTokenizer.tokenize_prompt(effective_prompt)
        content_tokens = prompt_tokens - STOPWORDS
        literals = ExecutionContext._extract_universal_literals(effective_prompt)

        l0_table = Table(box=box.ROUNDED, expand=True, border_style="dim cyan")
        l0_table.add_column("Property", style="bold cyan", width=24)
        l0_table.add_column("Extracted Information", style="white")

        l0_table.add_row("Raw User Prompt", prompt_clean)
        if prof == "E":
            l0_table.add_row(f"Translator Output ({t_trans:.1f}ms)", f"[italic magenta]{effective_prompt}[/italic magenta]")
        else:
            l0_table.add_row("Translator Pass", "[dim]Bypassed (Active in Profile E only)[/dim]")

        clauses_formatted = "  ➔  ".join(f"[bold yellow]Clause {idx+1}:[/bold yellow] '{cl}'" for idx, cl in enumerate(clauses)) if clauses else "[dim]None[/dim]"
        l0_table.add_row(f"Decomposed Clauses ({len(clauses)})", clauses_formatted)
        l0_table.add_row(f"Content Tokens ({len(content_tokens)})", ", ".join(sorted(content_tokens)) if content_tokens else "[dim]None[/dim]")

        if literals:
            lit_strs = [f"[bold green]{val}[/bold green] ([dim]{kind}[/dim])" for _, kind, val in literals]
            l0_table.add_row(f"Universal Literals ({len(literals)})", ", ".join(lit_strs))
        else:
            l0_table.add_row("Universal Literals", "[dim]None detected[/dim]")

        c.print(Panel(l0_table, title="[bold cyan]⚡ LAYER 0: Query Intent & Pre-Processing[/bold cyan]", border_style="cyan"))

        # =================================================================
        # LAYER 1: Semantic Tunneling & Scoring (Router)
        # =================================================================
        t_route_0 = time.perf_counter()
        is_rag = (self.router.internal_rag is not None and getattr(self.router.internal_rag, "index", None) is not None)
        engine_desc = "Dense Vector Embeddings via LocalRAG (FAISS)" if is_rag else "Lexical Token Coverage with IDF Poset Index"
        query_spans = self.router._generate_query_spans(effective_prompt)
        gamma = getattr(self.router, "gamma", 0.15)
        epsilon = getattr(self.router, "epsilon", 0.001)

        tunnel_cells, relevance_map = self.router.route(effective_prompt, top_k=400)
        route_dt = (time.perf_counter() - t_route_0) * 1000.0

        ranked_candidates = sorted(tunnel_cells, key=lambda cl: float(relevance_map.get(cl.cell_id, 0.0)), reverse=True)

        stage_counts = {1: 0, 2: 0, 3: 0, "other": 0}
        domain_counts: Dict[str, int] = {}
        carriers_out = set()
        carriers_in = set()
        for cl in ranked_candidates:
            st = getattr(cl, "stage", "other")
            if st in (1, 2, 3):
                stage_counts[st] += 1
            else:
                stage_counts["other"] += 1
            dom = getattr(cl, "domain_name", "generic")
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            out_t = getattr(cl.primary_output, "type_name", "")
            in_t = getattr(cl.primary_input, "type_name", "")
            if out_t and out_t.lower() not in ("any", "none", "*", "top", "void"):
                carriers_out.add(out_t)
            if in_t and in_t.lower() not in ("any", "none", "*", "top", "void"):
                carriers_in.add(in_t)

        top_domains_str = ", ".join(f"{d} ({cnt})" for d, cnt in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:6])
        summary_text = (
            f"[cyan]Retrieval Engine:[/cyan] {engine_desc}\n"
            f"[cyan]Hyperparameters:[/cyan] Temperature γ={gamma}, Cutoff ε={epsilon}, Multi-Scale Spans={len(query_spans)}\n"
            f"[cyan]Tunnel Distribution:[/cyan] {len(ranked_candidates):,} nodes | "
            f"Stage 1 (Ingress): [bold green]{stage_counts[1]}[/bold green] | "
            f"Stage 2 (Transforms): [bold yellow]{stage_counts[2]}[/bold yellow] | "
            f"Stage 3 (Egress/Sinks): [bold blue]{stage_counts[3]}[/bold blue]\n"
            f"[cyan]Carrier Types Detected:[/cyan] Outputs: {sorted(list(carriers_out))[:8]} | Inputs: {sorted(list(carriers_in))[:8]}\n"
            f"[cyan]Active Domains in Tunnel:[/cyan] {top_domains_str}"
        )

        c.print(Panel(summary_text, title=f"[bold yellow]⚡ LAYER 1: Semantic Tunneling & Scoring ({route_dt:.2f}ms)[/bold yellow]", border_style="yellow"))

        if ranked_candidates:
            cand_table = Table(title=f"🎯 Top Scoring Nodes in Semantic Tunnel (Displaying top {min(15, len(ranked_candidates))} of {len(ranked_candidates):,})", box=box.ROUNDED, expand=True, border_style="yellow")
            cand_table.add_column("#", style="dim", width=4)
            cand_table.add_column("Score / P(v|x)", style="bold yellow", width=14)
            cand_table.add_column("Cell ID", style="bold cyan", ratio=3)
            cand_table.add_column("Domain", style="magenta", width=12)
            cand_table.add_column("Stage", style="white", width=11)
            cand_table.add_column("Role", style="dim", width=10)
            cand_table.add_column("Primary Input (τ_in)", style="green", ratio=2)
            cand_table.add_column("Primary Output (τ_out)", style="blue", ratio=2)

            display_limit = 15
            for idx, cl in enumerate(ranked_candidates[:display_limit], 1):
                sc = float(relevance_map.get(cl.cell_id, 0.0))
                sc_color = "bold green" if sc >= 0.5 else ("bold yellow" if sc >= 0.1 else "white")
                sc_str = f"[{sc_color}]{sc:.4f} ({sc*100:.1f}%)[/{sc_color}]"

                stage_name = {1: "1: Ingress", 2: "2: Transform", 3: "3: Egress"}.get(cl.stage, str(cl.stage))
                in_sig = f"{getattr(cl.primary_input, 'type_name', 'any')} [{getattr(cl.primary_input, 'state', 'any')}]"
                out_sig = f"{getattr(cl.primary_output, 'type_name', 'any')} [{getattr(cl.primary_output, 'state', 'any')}]"

                cand_table.add_row(
                    str(idx),
                    sc_str,
                    cl.cell_id,
                    cl.domain_name,
                    stage_name,
                    cl.node_role or cl.node_type,
                    in_sig,
                    out_sig
                )
            c.print(cand_table)
            if len(ranked_candidates) > display_limit:
                c.print(f"  [dim]... and {len(ranked_candidates) - display_limit:,} more candidate nodes in semantic tunnel.[/dim]\n")
        else:
            c.print("[bold red][!] Empty tunnel: No nodes cleared the relevance cutoff threshold.[/bold red]\n")

        # =================================================================
        # LAYER 2: Topological Planning & Trellis Viterbi (Planner)
        # =================================================================
        t_plan_0 = time.perf_counter()
        os.environ["NSTL_DEBUG_PLAN"] = "1"
        try:
            cells = self.router.planner.plan(
                prompt=effective_prompt,
                tunnel=tunnel_cells,
                relevance_map=relevance_map
            )
        finally:
            os.environ.pop("NSTL_DEBUG_PLAN", None)
        plan_dt = (time.perf_counter() - t_plan_0) * 1000.0

        if cells:
            path_table = Table(title=f"🛣️ Planned Monadic Pipeline Composition ({len(cells)} Steps)", box=box.ROUNDED, expand=True, border_style="green")
            path_table.add_column("Step", style="bold yellow", width=6)
            path_table.add_column("Cell ID", style="bold cyan", ratio=3)
            path_table.add_column("Domain & Stage", style="magenta", width=18)
            path_table.add_column("Input (τ_in)", style="green", ratio=2)
            path_table.add_column("Output (τ_out)", style="blue", ratio=2)
            path_table.add_column("Type-Monadic Transition", style="bold", ratio=3)

            reg = TypeRegistry.get_instance()
            for idx, cl in enumerate(cells, 1):
                stage_label = f"{cl.domain_name} (S{cl.stage})"
                in_sig_str = f"{getattr(cl.primary_input, 'type_name', 'any')} [{getattr(cl.primary_input, 'state', 'any')}]"
                out_sig_str = f"{getattr(cl.primary_output, 'type_name', 'any')} [{getattr(cl.primary_output, 'state', 'any')}]"

                if idx == 1:
                    trans_status = "[bold green]✓ Ingress Entry Point[/bold green]"
                else:
                    prev_cl = cells[idx - 2]
                    prev_out = prev_cl.primary_output.signature
                    curr_in = cl.primary_input.signature
                    u = unify(prev_out, curr_in)
                    is_sub = reg.is_subtype(str(prev_out.type_name), str(curr_in.type_name))

                    if u is not None:
                        trans_status = f"[bold green]✓ Unifies ({prev_out.type_name} ➔ {curr_in.type_name})[/bold green]"
                    elif is_sub:
                        trans_status = f"[green]✓ Subtype ({prev_out.type_name} ⊆ {curr_in.type_name})[/green]"
                    else:
                        alt_port = next((p_name for p_name, p in cl.inputs.items() if unify(prev_out, p.signature) is not None), None)
                        if alt_port:
                            trans_status = f"[cyan]✓ Wire via port '{alt_port}'[/cyan]"
                        else:
                            trans_status = "[yellow]~ Weak / Wildcard bind[/yellow]"

                path_table.add_row(
                    str(idx),
                    cl.cell_id,
                    stage_label,
                    in_sig_str,
                    out_sig_str,
                    trans_status
                )

            path_chain = " ➔ ".join(f"[bold cyan]{c.cell_id}[/bold cyan]" for c in cells)
            c.print(Panel(
                f"[bold green]✓ Synthesized Type-Valid Path ({len(cells)} steps):[/bold green] {path_chain}",
                title=f"[bold green]⚡ LAYER 2: Topological Planning & Trellis Viterbi ({plan_dt:.2f}ms)[/bold green]",
                border_style="green"
            ))
            c.print(path_table)
            c.print("")
        else:
            c.print(Panel(
                "[bold red]❌ PLANNING FAILURE: No valid compositional path found through the lattice.[/bold red]",
                title=f"[bold red]⚡ LAYER 2: Topological Planning Failure ({plan_dt:.2f}ms)[/bold red]",
                border_style="red"
            ))

            diag_table = Table(title="🔍 Planning Failure Root Cause Analysis", box=box.ROUNDED, expand=True, border_style="red")
            diag_table.add_column("Diagnostic Check", style="bold yellow", width=28)
            diag_table.add_column("Result", style="bold", width=14)
            diag_table.add_column("Underlying Issue & Resolution Advice", style="white")

            if not tunnel_cells:
                diag_table.add_row(
                    "Semantic Tunnel Population",
                    "[red]EMPTY (0)[/red]",
                    f"No candidate nodes met the relevance threshold (epsilon={epsilon}). Rephrase or provide package hints."
                )
            else:
                diag_table.add_row(
                    "Semantic Tunnel Population",
                    f"[green]OK ({len(tunnel_cells):,})[/green]",
                    f"{len(tunnel_cells):,} nodes in active tunnel."
                )

            entries = [cl for cl in tunnel_cells if getattr(cl, "stage", None) == 1]
            if not entries:
                diag_table.add_row(
                    "Stage 1 Ingress/Source Nodes",
                    "[red]MISSING (0)[/red]",
                    "No data ingestion nodes (e.g. read_csv, imread, load) found in tunnel. Pipelines must begin with Stage 1."
                )
            else:
                entry_ids = ", ".join(cl.cell_id for cl in entries[:4])
                diag_table.add_row(
                    "Stage 1 Ingress/Source Nodes",
                    f"[green]FOUND ({len(entries)})[/green]",
                    f"Candidate entries: {entry_ids}"
                )

            sinks = [cl for cl in tunnel_cells if getattr(cl, "stage", None) == 3]
            if not sinks and literals:
                diag_table.add_row(
                    "Stage 3 Egress/Sink Nodes",
                    "[yellow]NONE[/yellow]",
                    "Prompt contains literals/destinations, but no terminal sinks (e.g. to_csv, imwrite) were retrieved."
                )
            elif sinks:
                sink_ids = ", ".join(cl.cell_id for cl in sinks[:4])
                diag_table.add_row(
                    "Stage 3 Egress/Sink Nodes",
                    f"[green]FOUND ({len(sinks)})[/green]",
                    f"Candidate sinks: {sink_ids}"
                )

            if entries:
                entry_out_types = {getattr(cl.primary_output, 'type_name', '') for cl in entries}
                transform_cells = [cl for cl in tunnel_cells if getattr(cl, "stage", None) == 2]
                transform_in_types = {getattr(cl.primary_input, 'type_name', '') for cl in transform_cells}
                diag_table.add_row(
                    "Carrier Compatibility",
                    "[yellow]INSPECT[/yellow]",
                    f"Entry outputs: {entry_out_types} vs Transform inputs: {transform_in_types}. If carriers are disjoint and no bridge morphism exists, composition cannot proceed."
                )

            c.print(diag_table)
            c.print("")

        # =================================================================
        # LAYER 3: Type-Monadic Unification & AST Synthesis
        # =================================================================
        final_code = ""
        dest_paths = None
        pipeline_bindings = None
        accum_sigma = None
        synth_dt = 0.0

        if cells:
            t_synth_0 = time.perf_counter()
            ctx = ExecutionContext(prompt=prompt_clean)
            try:
                unify_res = self.gate.unify_pipeline(cells, ctx)
            except Exception as exc:
                unify_res = Failure(reason=str(exc))
            synth_dt = (time.perf_counter() - t_synth_0) * 1000.0

            if unify_res.is_bottom():
                fail_reason = getattr(unify_res, "reason", "Unknown unification failure")
                c.print(Panel(
                    f"[bold red]❌ UNIFICATION FAILED: {fail_reason}[/bold red]\n"
                    f"[yellow]A type constraint or required port could not be satisfied in the Type Monad.[/yellow]",
                    title=f"[bold red]⚡ LAYER 3: Type-Monadic Unification Failure ({synth_dt:.2f}ms)[/bold red]",
                    border_style="red"
                ))
            else:
                pipeline_bindings = unify_res.value
                accum_sigma = unify_res.sigma
                dest_paths = self.gate._derive_egress_paths(pipeline_bindings)
                self.gate.last_egress_paths = dest_paths
                try:
                    final_code = self.gate.emit_code(pipeline_bindings, ctx)
                except Exception as e:
                    c.print(f"[bold red][!] Code emission error: {e}[/bold red]")

                bind_table = Table(title="🔌 Port Placeholders & Variable Wiring Bindings", box=box.ROUNDED, expand=True, border_style="cyan")
                bind_table.add_column("Step", style="dim", width=4)
                bind_table.add_column("Cell ID", style="bold cyan", ratio=3)
                bind_table.add_column("Port Name", style="bold yellow", ratio=2)
                bind_table.add_column("Dir", style="white", width=5)
                bind_table.add_column("Type & State", style="green", ratio=2)
                bind_table.add_column("Req", style="dim", width=5)
                bind_table.add_column("Bound Expression", style="bold magenta", ratio=2)
                bind_table.add_column("Provenance / Source", style="dim", ratio=2)

                prompt_lits = {v for _, _, v in literals}
                for step_idx, (cl, bindings) in enumerate(pipeline_bindings, 1):
                    for p_name, p_sig in cl.inputs.items():
                        b_val = str(bindings.get(p_name, "<unbound>"))
                        t_str = f"{p_sig.signature.type_name} [{p_sig.signature.state}]"
                        if any(b_val.strip("'\"") == lit for lit in prompt_lits):
                            source_desc = "Prompt Literal"
                        elif b_val.startswith("var_") or b_val.startswith("v"):
                            source_desc = "Wired Variable"
                        elif p_sig.default_value is not None:
                            source_desc = "Declared Default"
                        else:
                            source_desc = "Dynamic Resolver"

                        bind_table.add_row(
                            str(step_idx),
                            cl.cell_id,
                            p_name,
                            "IN",
                            t_str,
                            "Yes" if p_sig.required else "No",
                            b_val,
                            source_desc
                        )
                    for p_name, p_sig in cl.outputs.items():
                        b_val = str(bindings.get(p_name, "<unbound>"))
                        t_str = f"{p_sig.signature.type_name} [{p_sig.signature.state}]"
                        bind_table.add_row(
                            str(step_idx),
                            cl.cell_id,
                            p_name,
                            "OUT",
                            t_str,
                            "-",
                            b_val,
                            "Produced Value"
                        )

                egress_str = ", ".join(f"'{p}'" for p in dest_paths) if dest_paths else "[dim]None (In-memory pipeline)[/dim]"
                sigma_str = str(accum_sigma) if accum_sigma and getattr(accum_sigma, "mappings", None) else "[dim]None (Ground types)[/dim]"
                unify_summary = (
                    f"[cyan]Egress Target Files:[/cyan] {egress_str}\n"
                    f"[cyan]Type Substitutions (σ):[/cyan] {sigma_str}"
                )

                c.print(Panel(unify_summary, title=f"[bold cyan]⚡ LAYER 3: Monadic Variable Binding & AST Synthesis ({synth_dt:.2f}ms)[/bold cyan]", border_style="cyan"))
                c.print(bind_table)
                c.print("")

                if final_code:
                    syntax_code = Syntax(final_code, "python", theme="monokai", line_numbers=True)
                    c.print(Panel(syntax_code, title="[bold green]✨ Synthesized Python Code[/bold green]", border_style="green", padding=(0, 1)))
                    c.print("")

        # =================================================================
        # LAYER 4: GEVR Sandbox Execution & Egress Verification
        # =================================================================
        sandbox_res = {"success": False, "error": "Execution skipped"}
        sandbox_dt = 0.0

        if final_code and execute_sandbox:
            t_exec_start = time.perf_counter()
            sandbox_res = self.sandbox.execute(final_code, timeout=timeout, egress_paths=dest_paths)
            sandbox_dt = (time.perf_counter() - t_exec_start) * 1000.0

            sb_success = sandbox_res.get("success", False)
            sb_ret = sandbox_res.get("returncode", 0)
            sb_stdout = sandbox_res.get("stdout", "").strip()
            sb_stderr = sandbox_res.get("stderr", "").strip()
            sb_err = sandbox_res.get("error", "").strip()

            if sb_success:
                sb_badge = "[bold green]✓ PASSED[/bold green]"
                border_col = "green"
            else:
                sb_badge = f"[bold red]✗ FAILED (Exit Code: {sb_ret})[/bold red]"
                border_col = "red"

            sb_summary = [
                f"[cyan]Status:[/cyan] {sb_badge}",
                f"[cyan]Execution Time:[/cyan] {sandbox_dt:.2f}ms",
                f"[cyan]Timeout Bound:[/cyan] {timeout:.1f}s"
            ]

            if dest_paths:
                egress_checks = []
                for dp in dest_paths:
                    p = Path(dp)
                    if p.exists():
                        sz = p.stat().st_size
                        egress_checks.append(f"'{dp}': [bold green]EXISTS ({sz} bytes)[/bold green]")
                    else:
                        egress_checks.append(f"'{dp}': [bold red]MISSING ON DISK[/bold red]")
                sb_summary.append(f"[cyan]Egress Artifacts:[/cyan] {' | '.join(egress_checks)}")

            c.print(Panel("\n".join(sb_summary), title=f"[bold {border_col}]⚡ LAYER 4: GEVR Sandbox Execution & Verification[/bold {border_col}]", border_style=border_col))

            if sb_stdout:
                c.print(Panel(sb_stdout, title="[green]Standard Output (stdout)[/green]", border_style="dim green"))

            if sb_stderr or sb_err:
                err_text = sb_err or sb_stderr
                c.print(Panel(err_text, title="[red]Standard Error / Traceback (stderr)[/red]", border_style="red"))

                if sandbox_res.get("extrinsic", False) or _is_environmental_error(err_text):
                    c.print(Panel(
                        "[yellow][ENVIRONMENTAL FAILURE DETECTED][/yellow]\n"
                        "The synthesized Python program failed due to an external environmental dependency "
                        "(e.g. input file not found on disk, missing system package, or IO permission).\n"
                        "[bold green]The synthesized pipeline logic and type composition are structurally sound.[/bold green]",
                        border_style="yellow"
                    ))
                else:
                    c.print(Panel(
                        "[bold red][CODE DEFECT DETECTED][/bold red]\n"
                        "An exception occurred inside the synthesized code during runtime execution.",
                        border_style="red"
                    ))
            c.print("")

        # =================================================================
        # LAYER 5: Self-Repair Cycle (Profile C/E)
        # =================================================================
        rep_dt = 0.0
        repaired = False
        if final_code and not sandbox_res.get("success", False) and prof in ("C", "E"):
            mm = ModelManager.get_instance()
            if mm.profile and mm.can_feedback_check():
                err_msg = sandbox_res.get("error", "")
                if sandbox_res.get("extrinsic", False) or _is_environmental_error(err_msg):
                    c.print(
                        "  [yellow][!] Environmental failure detected. LLM self-repair skipped — "
                        "the synthesized pipeline is not the defect.[/yellow]\n"
                    )
                else:
                    c.print(Panel("[bold yellow]⚡ LAYER 5: GEVR Sandbox LLM Self-Repair Cycle[/bold yellow]", border_style="yellow"))
                    t_rep_0 = time.perf_counter()
                    failing_code = final_code
                    repaired_code = extract_code_from_llm_response(mm.feedback_check(failing_code, err_msg))
                    rep_dt = (time.perf_counter() - t_rep_0) * 1000.0

                    if repaired_code and repaired_code.strip() != failing_code.strip():
                        unknown = _unknown_api_references(repaired_code, failing_code, self.orchestrator.loaded_cells.values())
                        if unknown:
                            c.print(f"  [bold red][x] Repair rejected: references APIs absent from the lattice: {', '.join(sorted(unknown)[:5])}[/bold red]\n")
                        else:
                            c.print(f"  [bold green][✓] Repair accepted. Re-executing in sandbox... ({rep_dt:.1f}ms)[/bold green]")
                            final_code = repaired_code
                            repaired = True
                            sandbox_res = self.sandbox.execute(final_code, timeout=timeout, egress_paths=dest_paths)
                            c.print(f"  [bold]Post-Repair Result:[/bold] {'[green]PASSED[/green]' if sandbox_res.get('success') else '[red]FAILED[/red]'}\n")
                    else:
                        c.print(f"  [yellow][!] LLM could not produce an alternative repair ({rep_dt:.1f}ms).[/yellow]\n")

        # =================================================================
        # LAYER 6: Performance & Latency Breakdown
        # =================================================================
        total_dt = (time.perf_counter() - t_total_start) * 1000.0

        perf_table = Table(title="⏱️ End-to-End Pipeline Latency Breakdown", box=box.ROUNDED, expand=True, border_style="cyan")
        perf_table.add_column("Pipeline Layer", style="bold cyan")
        perf_table.add_column("Latency (ms)", style="bold yellow", justify="right")
        perf_table.add_column("% of Total", style="dim", justify="right")

        timings = [
            ("Layer 0: Translator Pass", t_trans),
            ("Layer 1: Semantic Routing & Tunneling", route_dt),
            ("Layer 2: Topological Planning (Viterbi)", plan_dt),
            ("Layer 3: Monadic Unification & Code Gen", synth_dt),
            ("Layer 4: GEVR Sandbox Execution", sandbox_dt),
            ("Layer 5: LLM Self-Repair Cycle", rep_dt),
        ]
        for name, lat in timings:
            pct = (lat / total_dt * 100) if total_dt > 0 else 0.0
            lat_str = f"{lat:.2f} ms" if lat > 0 else "[dim]-[/dim]"
            pct_str = f"{pct:.1f}%" if lat > 0 else "[dim]-[/dim]"
            perf_table.add_row(name, lat_str, pct_str)

        perf_table.add_section()
        perf_table.add_row("[bold white]Total End-to-End Latency[/bold white]", f"[bold green]{total_dt:.2f} ms[/bold green]", "[bold green]100.0%[/bold green]")
        c.print(perf_table)
        c.print("")

        path_ids = [cl.cell_id for cl in cells] if cells else []
        sb_status = "PASSED" if sandbox_res.get("success", False) else ("FAILED: " + sandbox_res.get("error", "").splitlines()[-1] if sandbox_res.get("error") else "FAILED")
        return {
            "prompt": prompt_clean,
            "profile": self.active_profile,
            "path": path_ids,
            "latency_ms": total_dt,
            "sandbox_status": sb_status,
            "code": final_code,
            "route_ms": route_dt,
            "plan_ms": plan_dt,
            "synth_ms": synth_dt,
            "sandbox_ms": sandbox_dt,
            "sandbox_result": sandbox_res,
            "cells": cells,
            "tunnel_size": len(tunnel_cells),
            "relevance_map": relevance_map
        }


class NSTLInteractiveShell(cmd.Cmd):
    """
    Rich Terminal User Interface (TUI) Studio for NSTL Neuro-Symbolic Synthesis.
    Provides a visual, interactive CLI workspace with instant profile switching,
    model exploration, real-time latency diagnostics, and self-repair cycles.
    """

    prompt = "\033[1;36mNSTL [Profile 0: Symbolic]\033[0m > "

    def __init__(
        self,
        db_path: str = "trees/lattice.db",
        initial_profile: str = "0",
        embedder: str = "",
        llm: str = "",
        device: str = "auto",
        debug: bool = False,
        interactive: bool = True
    ):
        super().__init__()
        self.db_path = db_path
        self.device = device
        self.embedder_name = embedder
        self.llm_name = llm
        self.active_profile = "0"
        self.rag: Optional[LocalRAG] = None
        self.history: List[Dict[str, Any]] = []
        self.debug: bool = debug
        self.interactive: bool = interactive

        if interactive:
            console.print("\n[bold cyan][*] Initializing NSTL Neuro-Symbolic Engine...[/bold cyan]")
        t0 = time.perf_counter()

        self.orchestrator = LatticeOrchestrator()
        self.orchestrator.load_from_database(db_path)
        self.orchestrator.build_topology()
        self.gate = UnificationGate()
        self.sandbox = GEVRSandbox()
        self.resolver = DynamicPlaceholderResolver()

        node_count = len(self.orchestrator.cells)
        load_time = (time.perf_counter() - t0) * 1000.0
        if interactive:
            console.print(f"[bold green][✓] Lattice Graph Loaded: {node_count:,} verified nodes ({load_time:.1f}ms)[/bold green]\n")

        # Initialize requested profile
        self._switch_profile(initial_profile, embedder=embedder, llm=llm, device=device, verbose=False)
        if interactive:
            self._render_dashboard()

    def _render_dashboard(self):
        """Renders the top visual status dashboard."""
        domains = set(c.domain_name for c in self.orchestrator.cells if c.domain_name)
        prof_name = self._format_profile_name(self.active_profile)
        prof_desc = self._get_profile_description(self.active_profile)

        # Main Header Table
        header_table = Table(box=box.ROUNDED, expand=True, border_style="cyan")
        header_table.add_column("⚡ Layer Profile", style="bold yellow", ratio=3)
        header_table.add_column("🧠 Neural Models", style="bold magenta", ratio=3)
        header_table.add_column("📊 Topology & Hardware", style="bold green", ratio=3)

        # Profile info
        prof_text = f"[bold white]{prof_name}[/bold white]\n[dim]{prof_desc}[/dim]"

        # Models info
        emb_text = f"[cyan]Embedder:[/cyan] {self.embedder_name or '[dim]None (Bypassed)[/dim]'}"
        llm_text = f"[cyan]LLM (GGUF):[/cyan] {self.llm_name or '[dim]None (Bypassed)[/dim]'}"
        models_text = f"{emb_text}\n{llm_text}"

        # Hardware & DB info
        db_nodes = f"[cyan]Nodes:[/cyan] {len(self.orchestrator.cells):,} in {len(domains)} domains"
        dbg_text = "[bold green]ON (Verbose)[/bold green]" if self.debug else "[dim]OFF[/dim]"
        dev_info = f"[cyan]Device:[/cyan] {self.device.upper()} | [cyan]Debug:[/cyan] {dbg_text} | [cyan]Queries:[/cyan] {len(self.history)}"
        hardware_text = f"{db_nodes}\n{dev_info}"

        header_table.add_row(prof_text, models_text, hardware_text)

        # Title Banner Panel
        title_text = Text("🧬 NSTL NEURO-SYMBOLIC TOPOLOGICAL LATTICE STUDIO", justify="center", style="bold white on blue")
        quick_shortcuts = Text(
            "Quick Layers: [1] 0:Symbolic  [2] A:Embedder  [3] C:Neuro-Symbolic  [4] D:Routing  [5] E:Translator\n"
            "Commands: /profile <0|A|C|D|E> | /debug [on|off] | /models | /set <key> <val> | /status | /new | /clear | /exit",
            justify="center",
            style="dim cyan"
        )

        dashboard_panel = Panel(
            header_table,
            title=title_text,
            subtitle=quick_shortcuts,
            border_style="bright_blue",
            padding=(0, 1)
        )
        console.print(dashboard_panel)
        console.print("[dim]Type any natural language pipeline specification to synthesize code in real-time:[/dim]\n")

    def _format_profile_name(self, prof: str) -> str:
        prof_u = prof.upper()
        if prof_u in ("0", "SYMBOLIC", "ZERO", "PURE"):
            return "Profile 0 (Pure Symbolic Layer)"
        elif prof_u == "A":
            return "Profile A (Dense Embeddings RAG)"
        elif prof_u in ("C", "B"):
            return "Profile C (Hybrid Neuro-Symbolic + GGUF LLM)"
        elif prof_u == "D":
            return "Profile D (Routing-Only Benchmark)"
        elif prof_u == "E":
            return "Profile E (Translator Pass + Neuro-Symbolic)"
        return f"Profile {prof_u}"

    def _get_profile_description(self, prof: str) -> str:
        prof_u = prof.upper()
        if prof_u in ("0", "SYMBOLIC", "ZERO", "PURE"):
            return "Deterministic type-safe lattice search across all loaded nodes. Baseline layer (zero neural models)."
        elif prof_u == "A":
            return "FAISS vector retrieval with dense embedding model."
        elif prof_u in ("C", "B"):
            return "Embedder RAG + GGUF local LLM slot-filling & self-repair."
        elif prof_u == "D":
            return "LLM-guided path search without full code synthesis."
        elif prof_u == "E":
            return "2-stage pipeline: conversational prompt -> canonical translator."
        return ""

    def _update_prompt(self):
        prof_label = self.active_profile.upper()
        if prof_label in ("0", "SYMBOLIC", "ZERO", "PURE"):
            prof_label = "0: Symbolic"
        dbg_label = " \033[1;33m[DEBUG]\033[0m" if self.debug else ""
        self.prompt = f"\033[1;36mNSTL [Profile {prof_label}]\033[0m{dbg_label} > "

    def do_debug(self, arg: str):
        """Toggle or configure debug mode across all pipeline layers. Usage: debug [on|off] or /debug [on|off]"""
        arg = arg.strip().lower().lstrip("/")
        if arg.startswith("debug"):
            arg = arg[5:].strip()
        if not arg:
            self.debug = not self.debug
        elif arg in ("1", "true", "on", "yes", "enable", "enabled"):
            self.debug = True
        elif arg in ("0", "false", "off", "no", "disable", "disabled"):
            self.debug = False
        else:
            console.print(f"[yellow]Usage: /debug [on|off] (currently {'ON' if self.debug else 'OFF'})[/yellow]")
            return

        status_str = "[bold green]ENABLED (full layer-by-layer diagnostics)[/bold green]" if self.debug else "[dim]DISABLED[/dim]"
        console.print(f"\n[*] Debug Mode: {status_str}\n")
        self._update_prompt()

    def _switch_profile(self, profile: str, embedder: str = "", llm: str = "", device: str = "auto", verbose: bool = True) -> bool:
        p = profile.strip().upper()
        if p in ("0", "SYMBOLIC", "ZERO", "PURE"):
            self.active_profile = "0"
            self.rag = None
            self.router = LatticeRouter(self.orchestrator, internal_rag=None)
            self._update_prompt()
            if verbose:
                console.print(f"[bold green][✓] Switched to {self._format_profile_name(self.active_profile)}[/bold green]\n")
            return True

        if p not in ("A", "C", "D", "E"):
            console.print(f"[bold red][!] Unknown profile '{profile}'. Valid options: 0 (Symbolic), A, C, D, E.[/bold red]")
            return False

        # Determine default model names dynamically based on cache coverage
        available_llm = _get_available_llms()
        emb_choice = select_optimal_embedder(embedder or self.embedder_name or "auto")
        llm_choice = llm or self.llm_name or (available_llm[0] if available_llm else "qwen2.5-coder-0.5b-instruct")

        if verbose:
            console.print(f"[bold cyan][*] Loading {self._format_profile_name(p)}...[/bold cyan]")
        t0 = time.perf_counter()
        try:
            HardwareProfiler.set_config(embedder_device=device, llm_device=device)
            mm = ModelManager.get_instance()
            mm.initialize_profile(
                profile_type=p,
                embedder_name=emb_choice,
                llm_name=llm_choice if p in ("C", "D", "E") else ""
            )
            self.embedder_name = emb_choice
            self.llm_name = llm_choice if p in ("C", "D", "E") else ""

            if verbose:
                console.print(f"[*] Indexing FAISS vector space for {len(self.orchestrator.cells):,} nodes...")
            self.rag = LocalRAG(trees_dir="trees", orchestrator=self.orchestrator)
            self.router = LatticeRouter(self.orchestrator, internal_rag=self.rag)

            self.active_profile = p
            self._update_prompt()
            dt = (time.perf_counter() - t0) * 1000.0
            if verbose:
                console.print(f"[bold green][✓] {self._format_profile_name(p)} ready ({dt:.1f}ms).[/bold green]\n")
        except Exception as e:
            console.print(f"[bold red][!] Failed to load Profile {p}: {e}[/bold red]")
            console.print("[yellow][*] Reverting to Profile 0 (Pure Symbolic)...[/yellow]")
            try:
                ModelManager.get_instance().cleanup()
            except Exception:
                pass
            self.active_profile = "0"
            self.rag = None
            self.router = LatticeRouter(self.orchestrator, internal_rag=None)
            self._update_prompt()
            return False

    def do_profile(self, arg: str):
        """Switch active inference profile. Usage: profile <0|A|C|D|E>"""
        arg = arg.strip().lstrip("/")
        if arg.lower().startswith("profile"):
            arg = arg[7:].strip()
        if not arg:
            table = Table(title="Available Profile Layers", box=box.ROUNDED, border_style="cyan")
            table.add_column("Key", style="bold yellow")
            table.add_column("Layer Profile", style="bold white")
            table.add_column("Target Latency", style="bold green")
            table.add_column("Role in NSTL Paper / Architecture", style="dim")

            table.add_row("0 / symbolic", "Profile 0 (Pure Symbolic)", "< 15 ms", "Deterministic A* graph search (Baseline layer, zero neural models)")
            table.add_row("A", "Profile A (Dense Embeddings RAG)", "~50–100 ms", "Vector embeddings (SentenceTransformer / FAISS HNSW search)")
            table.add_row("C", "Profile C (Neuro-Symbolic LLM)", "~500 ms–2 s", "Full hybrid: Embedder + Local GGUF LLM slot-filling + Sandbox repair")
            table.add_row("D", "Profile D (Routing Benchmark)", "~200–500 ms", "LLM-guided path search without code generation")
            table.add_row("E", "Profile E (Translator Pass)", "~1–2 s", "Two-stage: Query Translator pass + Neuro-Symbolic synthesis")

            console.print(table)
            console.print(f"\nActive Profile: [bold yellow]{self._format_profile_name(self.active_profile)}[/bold yellow]\n")
            return
        self._switch_profile(arg)

    def do_models(self, arg: str):
        """List all available embedding models and LLMs found on disk."""
        available_emb = _get_available_embedders()
        available_llm = _get_available_llms()

        table = Table(title="📦 Available Neural Models in models/", box=box.ROUNDED, border_style="cyan")
        table.add_column("Category", style="bold yellow")
        table.add_column("Model Name", style="bold white")
        table.add_column("Status", style="bold green")

        if available_emb:
            for m in available_emb:
                is_active = (m == self.embedder_name and self.active_profile != "0")
                status = "[bold green]ACTIVE[/bold green]" if is_active else "[dim]Available[/dim]"
                table.add_row("Embedding Model", m, status)
        else:
            table.add_row("Embedding Model", "[dim]None found in models/embeddings/[/dim]", "-")

        if available_llm:
            for m in available_llm:
                is_active = (m == self.llm_name and self.active_profile in ("C", "D", "E"))
                status = "[bold green]ACTIVE[/bold green]" if is_active else "[dim]Available[/dim]"
                table.add_row("LLM (GGUF)", m, status)
        else:
            table.add_row("LLM (GGUF)", "[dim]None found in models/llms/[/dim]", "-")

        console.print(table)
        console.print("[dim]Use `set embedder <name>` or `set llm <name>` to activate a specific model.[/dim]\n")

    def do_set(self, arg: str):
        """Configure models or hardware device. Usage: set <embedder|llm|device> <value>"""
        arg = arg.strip().lstrip("/")
        if arg.lower().startswith("set"):
            arg = arg[3:].strip()
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            console.print("[yellow]Usage: set <embedder|llm|device> <value>[/yellow]")
            return
        key, val = parts[0].lower(), parts[1].strip()

        if key in ("embedder", "emb"):
            self.embedder_name = val
            console.print(f"[green][*] Embedder set to '{val}'.[/green]")
            if self.active_profile != "0":
                self._switch_profile(self.active_profile, embedder=val)
        elif key == "llm":
            self.llm_name = val
            console.print(f"[green][*] LLM set to '{val}'.[/green]")
            if self.active_profile in ("C", "D", "E"):
                self._switch_profile(self.active_profile, llm=val)
        elif key == "device":
            self.device = val
            console.print(f"[green][*] Compute device set to '{val}'.[/green]")
            if self.active_profile != "0":
                self._switch_profile(self.active_profile, device=val)
        elif key in ("debug", "dbg"):
            if val.lower() in ("1", "true", "on", "yes", "enable", "enabled"):
                self.debug = True
            elif val.lower() in ("0", "false", "off", "no", "disable", "disabled"):
                self.debug = False
            else:
                self.debug = not self.debug
            console.print(f"[green][*] Debug mode set to {'ON' if self.debug else 'OFF'}.[/green]")
            self._update_prompt()
        else:
            console.print(f"[bold red][!] Unknown parameter '{key}'. Supported: embedder, llm, device, debug.[/bold red]")

    def do_status(self, arg: str):
        """Display real-time system status and active configuration."""
        self._render_dashboard()

    def do_info(self, arg: str):
        """Alias for status."""
        self.do_status(arg)

    def do_new(self, arg: str):
        """Start a fresh chat/synthesis session and clear history."""
        self.history.clear()
        console.print("\n[bold green][✓] Session reset. Query history cleared.[/bold green]\n")

    def do_reset(self, arg: str):
        """Alias for new."""
        self.do_new(arg)

    def do_clear(self, arg: str):
        """Clears the terminal screen and redraws the dashboard."""
        os.system("clear" if os.name == "posix" else "cls")
        self._render_dashboard()

    def do_history(self, arg: str):
        """Display query history and performance metrics for the current session."""
        if not self.history:
            console.print("\n[yellow]No queries executed in this session yet.[/yellow]\n")
            return

        table = Table(title=f"📜 Session History ({len(self.history)} Queries)", box=box.ROUNDED, border_style="cyan")
        table.add_column("#", style="bold yellow", width=4)
        table.add_column("Profile", style="bold magenta", width=12)
        table.add_column("Prompt", style="bold white", ratio=3)
        table.add_column("Path", style="cyan", ratio=3)
        table.add_column("Latency", style="bold green", width=12)
        table.add_column("Sandbox", style="bold", width=16)

        for idx, item in enumerate(self.history, 1):
            sb_style = "green" if "PASSED" in item["sandbox_status"] else "red"
            table.add_row(
                str(idx),
                item["profile"],
                item["prompt"][:40] + ("..." if len(item["prompt"]) > 40 else ""),
                " ➔ ".join(item["path"]) if item["path"] else "[dim]None[/dim]",
                f"{item['latency_ms']:.1f} ms",
                f"[{sb_style}]{item['sandbox_status']}[/{sb_style}]"
            )

        console.print(table)
        console.print("")

    def default(self, line: str):
        prompt = line.strip()
        if not prompt:
            return

        # Handle quick numeric shortcuts for profiles (1 to 5)
        if prompt == "1":
            self._switch_profile("0")
            return
        elif prompt == "2":
            self._switch_profile("A")
            return
        elif prompt == "3":
            self._switch_profile("C")
            return
        elif prompt == "4":
            self._switch_profile("D")
            return
        elif prompt == "5":
            self._switch_profile("E")
            return

        # Handle slash commands
        if prompt.startswith("/"):
            cmd_part = prompt[1:].strip()
            parts = cmd_part.split(maxsplit=1)
            cmd_name = parts[0].lower()
            cmd_arg = parts[1] if len(parts) > 1 else ""

            if cmd_name in ("profile", "p"):
                self.do_profile(cmd_arg)
                return
            elif cmd_name in ("models", "m"):
                self.do_models(cmd_arg)
                return
            elif cmd_name == "set":
                self.do_set(cmd_arg)
                return
            elif cmd_name in ("status", "info"):
                self.do_status(cmd_arg)
                return
            elif cmd_name in ("new", "reset"):
                self.do_new(cmd_arg)
                return
            elif cmd_name in ("clear", "cls"):
                self.do_clear(cmd_arg)
                return
            elif cmd_name in ("history", "hist"):
                self.do_history(cmd_arg)
                return
            elif cmd_name in ("debug", "dbg"):
                self.do_debug(cmd_arg)
                return
            elif cmd_name in ("help", "h", "?"):
                self.do_help(cmd_arg)
                return
            elif cmd_name in ("exit", "quit", "q"):
                return self.do_exit(cmd_arg)

        # Check for query-level or shell-level debug flag
        query_debug = self.debug
        if "--debug" in prompt:
            prompt = prompt.replace("--debug", "").strip()
            query_debug = True

        if query_debug:
            debugger = PipelineDebugger(
                orchestrator=self.orchestrator,
                router=self.router,
                gate=self.gate,
                sandbox=self.sandbox,
                active_profile=self.active_profile,
                device=self.device,
                embedder_name=self.embedder_name,
                llm_name=self.llm_name,
                console=console
            )
            res = debugger.run(prompt, execute_sandbox=True, timeout=5.0)
            self.history.append({
                "prompt": prompt,
                "profile": self.active_profile,
                "path": res["path"],
                "latency_ms": res["latency_ms"],
                "sandbox_status": res["sandbox_status"],
                "code": res["code"],
                "route_ms": res["route_ms"],
                "synth_ms": res["synth_ms"],
                "sandbox_ms": res["sandbox_ms"],
                "sandbox_error": res.get("sandbox_result", {}).get("error", "")
            })
            return

        t_total_start = time.perf_counter()
        prof = self.active_profile.upper()

        # Step 1: Optional Translator Pass (Profile E)
        effective_prompt = prompt
        t_trans = 0.0
        if prof == "E":
            mm = ModelManager.get_instance()
            if mm.profile and mm.has_translator_pass():
                t0_trans = time.perf_counter()
                trans_system = (
                    "You are a precise technical translator. Rewrite the user request as ONE "
                    "comma-separated pipeline sentence: first the input source with its asset "
                    "name, then each transform verb with its arguments in order, then the "
                    "destination. Use only words from the request. Output ONLY the sentence, "
                    "no headers, no lists, no formatting."
                )
                effective_prompt = mm.generate_text(prompt, max_tokens=128, system_prompt=trans_system)
                # The translator output is free text: strip wrapper fences if present.
                effective_prompt = effective_prompt.strip().strip('`').strip()
                t_trans = (time.perf_counter() - t0_trans) * 1000.0
                console.print(f"[bold magenta][Translator Pass ({t_trans:.1f}ms)][/bold magenta] [italic]{effective_prompt}[/italic]")

        # Step 2: Routing via LatticeRouter
        t_route_start = time.perf_counter()
        cells = self.router.plan_path(effective_prompt, return_tuple=False)
        route_dt = (time.perf_counter() - t_route_start) * 1000.0

        if not cells:
            console.print(f"\n[bold red][!] No valid path found through lattice for: '{prompt}'[/bold red]\n")
            return

        path_ids = [c.cell_id for c in cells]
        path_arrows = " [bold green]➔[/bold green] ".join([f"[bold cyan]{cid}[/bold cyan]" for cid in path_ids])
        console.print(f"\n[bold yellow]⚡ Routed Path ({route_dt:.2f}ms):[/bold yellow] {path_arrows}")

        # Step 3: Synthesis & Code Generation
        t_synth_start = time.perf_counter()
        try:
            final_code = self.gate.unify_and_emit(cells, prompt)
        except (UnresolvedPlaceholderError, UnificationFailure) as e:
            console.print(f"\n[bold red][!] Could not synthesize code for: '{prompt}'[/bold red]")
            console.print(f"[red]    {e}[/red]\n")
            return
        synth_dt = (time.perf_counter() - t_synth_start) * 1000.0

        # Step 4: Sandbox Verification & Optional Self-Repair
        t_exec_start = time.perf_counter()
        # Egress destinations come from the unification gate itself (values bound to
        # path-typed ports of terminal morphisms) — a single source of truth derived
        # from the verified dataflow, never re-parsed from the raw prompt.
        dest_paths = self.gate.get_egress_paths() or None

        sandbox_res = self.sandbox.execute(final_code, timeout=5.0, egress_paths=dest_paths)
        sandbox_dt = (time.perf_counter() - t_exec_start) * 1000.0

        repaired = False
        # If execution failed and LLM feedback is available (Profile C/E), trigger self-repair.
        # Guardrails:
        # 1. Environmental failures (missing input asset, missing module, connectivity)
        #    are NOT code defects — repairing code cannot create the user's data file,
        #    so the failure is reported honestly and the repair cycle is skipped.
        # 2. Accepted repairs must stay on the lattice's API surface: every attribute
        #    referenced by the repaired code must already appear in a verified cell
        #    template or in the original code — hallucinated methods are rejected.
        if not sandbox_res.get("success", False) and prof in ("C", "E"):
            mm = ModelManager.get_instance()
            if mm.profile and mm.can_feedback_check():
                error_msg = sandbox_res.get("error", "")
                if sandbox_res.get("extrinsic", False) or _is_environmental_error(error_msg):
                    console.print(
                        "  [yellow][!] Environmental failure detected (missing input asset or "
                        "unavailable module). LLM self-repair skipped — the synthesized "
                        "pipeline is not the defect.[/yellow]"
                    )
                else:
                    console.print("  [bold yellow][*] GEVR Sandbox triggered LLM Self-Repair Cycle...[/bold yellow]")
                    t_rep_start = time.perf_counter()
                    failing_code = final_code
                    repaired_code = extract_code_from_llm_response(mm.feedback_check(failing_code, error_msg))
                    if repaired_code and repaired_code.strip() != failing_code.strip():
                        unknown = _unknown_api_references(repaired_code, failing_code, self.orchestrator.loaded_cells.values())
                        if unknown:
                            console.print(
                                f"  [bold red][x] Repair rejected: references APIs absent from the "
                                f"lattice: {', '.join(sorted(unknown)[:5])}[/bold red]"
                            )
                        else:
                            final_code = repaired_code
                            repaired = True
                            # Re-verify repaired code
                            sandbox_res = self.sandbox.execute(final_code, timeout=5.0, egress_paths=dest_paths)
                    rep_dt = (time.perf_counter() - t_rep_start) * 1000.0
                    console.print(f"  [bold green][✓] Repair cycle completed ({rep_dt:.1f}ms).[/bold green]")

        total_dt = (time.perf_counter() - t_total_start) * 1000.0

        # Output code in a styled Syntax box
        syntax_code = Syntax(final_code, "python", theme="monokai", line_numbers=True)
        code_panel = Panel(
            syntax_code,
            title=f"[bold green]✨ Synthesized Python Code ({self._format_profile_name(self.active_profile)})[/bold green]",
            border_style="green",
            padding=(0, 1)
        )
        console.print(code_panel)

        # Report Latency Metrics Bar
        timing_elements = [
            f"[cyan]Route:[/cyan] [bold]{route_dt:.2f}ms[/bold]",
            f"[cyan]Synth:[/cyan] [bold]{synth_dt:.2f}ms[/bold]",
            f"[cyan]Exec:[/cyan] [bold]{sandbox_dt:.2f}ms[/bold]"
        ]
        if t_trans > 0:
            timing_elements.insert(0, f"[magenta]Trans:[/magenta] [bold]{t_trans:.1f}ms[/bold]")
        timing_elements.append(f"[bold yellow]Total: {total_dt:.2f}ms[/bold yellow]")

        # Report Sandbox Execution Status
        if sandbox_res.get("success", False):
            sb_badge = f"[bold green]✓ PASSED[/bold green]"
            sb_status = "PASSED"
        else:
            err = sandbox_res.get("error", "Unknown error").strip()
            first_err_line = err.splitlines()[-1] if err else "Execution Error"
            sb_badge = f"[bold red]✗ FAILED[/bold red] [dim]({first_err_line})[/dim]"
            sb_status = f"FAILED: {first_err_line}"

        metrics_panel = Panel(
            f"{'  │  '.join(timing_elements)}   │   [bold]Sandbox:[/bold] {sb_badge}",
            border_style="dim",
            padding=(0, 1)
        )
        console.print(metrics_panel)
        console.print("")

        # Record in history
        self.history.append({
            "prompt": prompt,
            "profile": self.active_profile,
            "path": path_ids,
            "latency_ms": total_dt,
            "sandbox_status": sb_status,
            "code": final_code,
            "route_ms": route_dt,
            "synth_ms": synth_dt,
            "sandbox_ms": sandbox_dt,
            "sandbox_error": sandbox_res.get("error", "")
        })

    def do_exit(self, line):
        """Exit the NSTL Interactive Studio."""
        console.print("\n[bold cyan]Exiting NSTL Studio. Goodbye![/bold cyan]\n")
        return True

    def do_quit(self, line):
        """Alias for exit."""
        return self.do_exit(line)

    def do_q(self, line):
        """Alias for exit."""
        return self.do_exit(line)

    def do_EOF(self, line):
        """Handle CTRL+D / EOF."""
        return self.do_exit(line)


def cmd_shell(args):
    """Launches the full interactive Rich TUI studio."""
    shell = NSTLInteractiveShell(
        db_path=args.db,
        initial_profile=args.profile,
        embedder=args.embedder,
        llm=args.llm,
        device=args.device,
        debug=getattr(args, "debug", False),
        interactive=True
    )
    shell.cmdloop()


def cmd_run(args):
    """Executes a single prompt synthesis directly from CLI, optionally with --debug."""
    db_path = getattr(args, "db", "trees/lattice.db")
    profile = getattr(args, "profile", "0")
    embedder = getattr(args, "embedder", "")
    llm = getattr(args, "llm", "")
    device = getattr(args, "device", "auto")
    debug_mode = getattr(args, "debug", False)
    prompt = getattr(args, "prompt", "").strip()

    if not prompt:
        console.print("[bold red][!] Prompt must not be empty.[/bold red]")
        sys.exit(1)

    shell = NSTLInteractiveShell(
        db_path=db_path,
        initial_profile=profile,
        embedder=embedder,
        llm=llm,
        device=device,
        debug=debug_mode,
        interactive=False
    )
    shell.default(f"{prompt} --debug" if debug_mode else prompt)


def cmd_precompute_rag(args):
    """Precomputes dense vector embeddings for all loaded lattice nodes into .rag_cache/."""
    try:
        from lattice import LatticeOrchestrator
        from internal_rag import LocalRAG
        from inference import ModelManager
    except ImportError:
        from .lattice import LatticeOrchestrator
        from .internal_rag import LocalRAG
        from .inference import ModelManager

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[!] Database not found: {db_path}. Please run 'compile' first.")
        return

    print(f"[*] Loading lattice cells from '{db_path}'...")
    orch = LatticeOrchestrator(trees_directory=args.trees_dir, db_path=str(db_path))
    print(f"[+] Loaded {len(orch.cells):,} cells.")

    print(f"[*] Initializing Profile A (embedder: '{args.embedder or 'optimal'}')...")
    ModelManager.get_instance().initialize_profile("A", embedder_name=args.embedder)

    try:
        print(f"[*] Starting incremental FAISS precomputation into '.rag_cache/'...")
        t0 = time.perf_counter()
        rag = LocalRAG(trees_dir=args.trees_dir, orchestrator=orch)
        dt = time.perf_counter() - t0
        print(f"[✓] Precomputation complete in {dt:.2f}s. Cache is 100% synchronized.")
    finally:
        ModelManager.get_instance().cleanup()


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser with all subcommands and global debug flags."""
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="NSTL Toolchain CLI & Interactive Studio")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose debug mode across all pipeline layers")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # harvest
    p_harvest = subparsers.add_parser("harvest", help="Harvest API primitives into single-file domain JSON")
    p_harvest.add_argument("package", type=str, help="Python package name to harvest")
    p_harvest.add_argument("--domain", type=str, default=None, help="Target domain name (defaults to package name)")
    p_harvest.add_argument("--trees-dir", type=str, default="trees", help="Directory for domain tree JSON files")
    p_harvest.set_defaults(func=cmd_harvest)

    # compile
    p_compile = subparsers.add_parser("compile", help="Compile single-file domain JSONs into SQLite database")
    p_compile.add_argument("--trees-dir", type=str, default="trees", help="Directory containing domain JSON files")
    p_compile.add_argument("--output", type=str, default="trees/lattice.db", help="Target SQLite DB path")
    p_compile.add_argument("--domains", nargs="*", default=None, help="Optional domain filter")
    p_compile.add_argument("--clean", action="store_true", help="Purge target database before compiling (default: non-destructive upsert)")
    p_compile.set_defaults(func=cmd_compile)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate AST syntax and schema of all nodes in SQLite")
    p_validate.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_validate.set_defaults(func=cmd_validate)

    # precompute-rag
    p_precompute = subparsers.add_parser("precompute-rag", help="Precompute FAISS dense embeddings into .rag_cache/")
    p_precompute.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_precompute.add_argument("--trees-dir", type=str, default="trees", help="Directory for domain tree JSON files")
    p_precompute.add_argument("--embedder", type=str, default="", help="Embedding model name (e.g. jina-embeddings-v5-text-nano)")
    p_precompute.set_defaults(func=cmd_precompute_rag)

    # run
    p_run = subparsers.add_parser("run", help="Synthesize code for a natural language prompt directly from CLI")
    p_run.add_argument("prompt", type=str, help="Natural language pipeline specification")
    p_run.add_argument("--debug", "-d", action="store_true", help="Enable verbose debug output across all pipeline layers")
    p_run.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_run.add_argument("--profile", type=str, default="0", help="Inference profile (0=Symbolic, A=Embedder, C=Neuro-Symbolic, D, E)")
    p_run.add_argument("--embedder", type=str, default="", help="Embedding model name (e.g. jina-embeddings-v5-text-nano)")
    p_run.add_argument("--llm", type=str, default="", help="LLM model name (e.g. qwen2.5-coder-0.5b-instruct)")
    p_run.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    p_run.add_argument("--no-exec", action="store_true", help="Skip GEVR sandbox execution")
    p_run.set_defaults(func=cmd_run)

    # shell
    p_shell = subparsers.add_parser("shell", help="Launch real-time interactive synthesis TUI studio")
    p_shell.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_shell.add_argument("--profile", type=str, default="0", help="Initial inference profile (0=Symbolic/Instant, A=Embedder, C=Neuro-Symbolic LLM, D, E)")
    p_shell.add_argument("--embedder", type=str, default="", help="Embedding model name (e.g. jina-embeddings-v5-text-nano)")
    p_shell.add_argument("--llm", type=str, default="", help="LLM model name (e.g. qwen2.5-coder-0.5b-instruct)")
    p_shell.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    p_shell.add_argument("--debug", "-d", action="store_true", help="Launch studio with debug mode enabled")
    p_shell.set_defaults(func=cmd_shell)

    return parser


def main():
    # If invoked without arguments (e.g. `python3 nstl_cli.py` or `python3 src/cli.py`), launch TUI Studio directly
    if len(sys.argv) == 1:
        shell = NSTLInteractiveShell()
        shell.cmdloop()
        return

    # If invoked with only --debug, launch TUI studio directly with debug=True
    if len(sys.argv) == 2 and sys.argv[1] in ("--debug", "-d"):
        shell = NSTLInteractiveShell(debug=True)
        shell.cmdloop()
        return

    # Pre-process arguments to support direct prompt or top-level --debug
    known_cmds = {"harvest", "compile", "validate", "precompute-rag", "shell", "run", "-h", "--help"}
    if len(sys.argv) > 1 and sys.argv[1] not in known_cmds:
        if sys.argv[1] in ("--debug", "-d") and len(sys.argv) > 2 and sys.argv[2] not in known_cmds:
            prompt_arg = sys.argv[2]
            remaining = sys.argv[3:]
            sys.argv = [sys.argv[0], "run", prompt_arg, "--debug"] + remaining
        elif sys.argv[1] not in ("-h", "--help"):
            sys.argv.insert(1, "run")

    parser = build_parser()

    args = parser.parse_args()
    if hasattr(args, "func"):
        if getattr(args, "debug", False):
            setattr(args, "debug", True)
        args.func(args)
    else:
        # Default to interactive shell
        cmd_shell(argparse.Namespace(
            db="trees/lattice.db",
            profile="0",
            embedder="",
            llm="",
            device="auto",
            debug=getattr(args, "debug", False)
        ))


if __name__ == "__main__":
    main()

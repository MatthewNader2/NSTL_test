# src/cli.py
import argparse
import ast
import json
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
    cells = harvester.harvest_all()
    print(f"[+] Harvested {len(cells)} function cells from '{package}'.")

    harvester.merge_and_save(cells, out_file)
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
            verified             INTEGER DEFAULT 0,
            docstring            TEXT DEFAULT '',
            enrichment_source    TEXT DEFAULT NULL,
            enriched_at          TEXT DEFAULT NULL,
            source_provenance    TEXT DEFAULT 'unknown',
            source_priority      INTEGER DEFAULT 100
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_role ON nodes(node_role)")
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
                "outputs": {k: v.model_dump() for k, v in cell.outputs.items()}
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

            cur.execute("""
                INSERT OR REPLACE INTO nodes
                (cell_id, domain_name, node_type, node_role, stage, keywords,
                 input_type, input_state, output_type, output_state, code,
                 dependencies, configuration_schema, verified, docstring,
                 enrichment_source, enriched_at, source_provenance, source_priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        dummy_code = re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", "dummy_var", code)
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
    from lattice import LatticeOrchestrator
    from router import LatticeRouter, HardwareProfiler
    from unification import UnificationGate, DynamicPlaceholderResolver, UnresolvedPlaceholderError, UnificationFailure
    from gevr_sandbox import GEVRSandbox
    from inference import ModelManager, select_optimal_embedder
    from internal_rag import LocalRAG
    from config import MODELS_DIR
    from utils import extract_code_from_llm_response
except ImportError:
    from .lattice import LatticeOrchestrator
    from .router import LatticeRouter, HardwareProfiler
    from .unification import UnificationGate, DynamicPlaceholderResolver
    from .gevr_sandbox import GEVRSandbox
    from .inference import ModelManager, select_optimal_embedder
    from .internal_rag import LocalRAG
    from .config import MODELS_DIR
    from .utils import extract_code_from_llm_response


console = Console()


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
        device: str = "auto"
    ):
        super().__init__()
        self.db_path = db_path
        self.device = device
        self.embedder_name = embedder
        self.llm_name = llm
        self.active_profile = "0"
        self.rag: Optional[LocalRAG] = None
        self.history: List[Dict[str, Any]] = []

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
        console.print(f"[bold green][✓] Lattice Graph Loaded: {node_count:,} verified nodes ({load_time:.1f}ms)[/bold green]\n")

        # Initialize requested profile
        self._switch_profile(initial_profile, embedder=embedder, llm=llm, device=device, verbose=False)
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
        dev_info = f"[cyan]Device:[/cyan] {self.device.upper()} | [cyan]Queries:[/cyan] {len(self.history)}"
        hardware_text = f"{db_nodes}\n{dev_info}"

        header_table.add_row(prof_text, models_text, hardware_text)

        # Title Banner Panel
        title_text = Text("🧬 NSTL NEURO-SYMBOLIC TOPOLOGICAL LATTICE STUDIO", justify="center", style="bold white on blue")
        quick_shortcuts = Text(
            "Quick Layers: [1] 0:Symbolic  [2] A:Embedder  [3] C:Neuro-Symbolic  [4] D:Routing  [5] E:Translator\n"
            "Commands: /profile <0|A|C|D|E> | /models | /set <key> <val> | /status | /new | /history | /clear | /exit",
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
            return "Sub-15ms deterministic A* search across 34K nodes. Baseline layer."
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
        self.prompt = f"\033[1;36mNSTL [Profile {prof_label}]\033[0m > "

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
            return True
        except Exception as e:
            console.print(f"[bold red][!] Failed to load Profile {p}: {e}[/bold red]")
            console.print("[yellow][*] Reverting to Profile 0 (Pure Symbolic)...[/yellow]")
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
        else:
            console.print(f"[bold red][!] Unknown parameter '{key}'. Supported: embedder, llm, device.[/bold red]")

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
            elif cmd_name in ("help", "h", "?"):
                self.do_help(cmd_arg)
                return
            elif cmd_name in ("exit", "quit", "q"):
                return self.do_exit(cmd_arg)

        t_total_start = time.perf_counter()
        prof = self.active_profile.upper()

        # Step 1: Optional Translator Pass (Profile E)
        effective_prompt = prompt
        t_trans = 0.0
        if prof == "E":
            mm = ModelManager.get_instance()
            if mm.profile and mm.has_translator_pass():
                t0_trans = time.perf_counter()
                trans_system = "You are a precise technical translator. Convert the following user request into a concise canonical pipeline specification stating input source, exact transforms, and destination sink. Output ONLY the canonical query."
                effective_prompt = mm.generate_text(prompt, max_tokens=128, system_prompt=trans_system)
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
        has_sink = any(
            getattr(c, "stage", None) == 3 or
            any(getattr(p, "state", None) in ("filepath_written", "saved", "exported") or
                getattr(getattr(p, "signature", None), "state", None) in ("filepath_written", "saved", "exported")
                for p in getattr(c, "outputs", {}).values())
            for c in cells
        )
        dest_paths = None
        if has_sink:
            if hasattr(self.gate, "context") and hasattr(self.gate.context, "dest_files") and self.gate.context.dest_files:
                dest_paths = [self.gate.context.dest_files[-1]]
            else:
                from router import extract_file_paths_and_extensions
                paths, _ = extract_file_paths_and_extensions(prompt)
                if paths:
                    dest_paths = [paths[-1]]

        sandbox_res = self.sandbox.execute(final_code, timeout=5.0, egress_paths=dest_paths)
        sandbox_dt = (time.perf_counter() - t_exec_start) * 1000.0

        repaired = False
        # If execution failed and LLM feedback is available (Profile C/E), trigger self-repair
        if not sandbox_res.get("success", False) and prof in ("C", "E"):
            mm = ModelManager.get_instance()
            if mm.profile and mm.can_feedback_check():
                console.print("  [bold yellow][*] GEVR Sandbox triggered LLM Self-Repair Cycle...[/bold yellow]")
                t_rep_start = time.perf_counter()
                failing_code = final_code
                error_msg = sandbox_res.get("error", "")
                repaired_code = extract_code_from_llm_response(mm.feedback_check(failing_code, error_msg))
                if repaired_code and repaired_code.strip() != failing_code.strip():
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
        device=args.device
    )
    shell.cmdloop()


def main():
    # If invoked without arguments (e.g. `python3 nstl_cli.py` or `python3 src/cli.py`), launch TUI Studio directly
    if len(sys.argv) == 1:
        shell = NSTLInteractiveShell()
        shell.cmdloop()
        return

    parser = argparse.ArgumentParser(prog="python -m src.cli", description="NSTL Toolchain CLI & Interactive Studio")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # harvest
    p_harvest = subparsers.add_parser("harvest", help="Harvest API primitives into single-file domain JSON")
    p_harvest.add_argument("package", type=str, help="Python package name to harvest (e.g. cv2, pandas)")
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

    # shell
    p_shell = subparsers.add_parser("shell", help="Launch real-time interactive synthesis TUI studio")
    p_shell.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_shell.add_argument("--profile", type=str, default="0", help="Initial inference profile (0=Symbolic/Instant, A=Embedder, C=Neuro-Symbolic LLM, D, E)")
    p_shell.add_argument("--embedder", type=str, default="", help="Embedding model name (e.g. jina-embeddings-v5-text-nano)")
    p_shell.add_argument("--llm", type=str, default="", help="LLM model name (e.g. qwen2.5-coder-0.5b-instruct)")
    p_shell.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    p_shell.set_defaults(func=cmd_shell)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        # Default to interactive shell
        cmd_shell(argparse.Namespace(db="trees/lattice.db", profile="0", embedder="", llm="", device="auto"))


if __name__ == "__main__":
    main()

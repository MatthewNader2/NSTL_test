# src/main.py
"""
src/main.py - Neuro-Symbolic Topological Lattice (NSTL)
Production REST API Server and Core Engine Lifecycle Manager.
Provides real-time endpoints for health monitoring, domain metadata,
multi-profile engine initialization, program synthesis, and benchmark automation.
"""

from __future__ import annotations
import os
import asyncio
import platform
import resource
import sys
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings, MODELS_DIR
from lattice import LatticeOrchestrator
from router import LatticeRouter
from unification import UnificationGate
from gevr_sandbox import GEVRSandbox
from inference import ModelManager, select_optimal_embedder
from internal_rag import LocalRAG
from log_config import get_logger

logger = get_logger("main")

DB_PATH = os.environ.get("NSTL_DB_PATH", str(settings.trees_dir / "lattice.db"))

# =====================================================================
# Global Engine State
# =====================================================================
_orchestrator: Optional[LatticeOrchestrator] = None
_router: Optional[LatticeRouter] = None
_gate: Optional[UnificationGate] = None
_sandbox: Optional[GEVRSandbox] = None
_internal_rag: Optional[LocalRAG] = None
_active_profile: str = "0"
_active_embedder: str = "none"
_active_llm: str = "none"
_engine_ready: bool = False
_engine_lock = threading.Lock()
_start_time = time.time()
_benchmark_active: bool = False


def _ensure_base_engine_loaded():
    """Initializes the base SQLite lattice orchestrator, gate, and sandbox."""
    global _orchestrator, _gate, _sandbox, _router
    if _orchestrator is None:
        logger.info(f"[*] Preloading NSTL Lattice Database from '{DB_PATH}'...")
        t0 = time.perf_counter()
        _orchestrator = LatticeOrchestrator()
        _orchestrator.load_from_database(DB_PATH)
        _orchestrator.build_topology()
        _gate = UnificationGate()
        _sandbox = GEVRSandbox()
        _router = LatticeRouter(orchestrator=_orchestrator, internal_rag=None)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"[✓] Base Lattice ready with {len(_orchestrator.loaded_cells)} nodes loaded in {elapsed:.1f}ms.")


# =====================================================================
# Request / Response Schemas
# =====================================================================
class InitRequest(BaseModel):
    profile: str = Field(default="0", description="Operational profile: 0, A, C, D, or E")
    embedder_model: Optional[str] = Field(default="auto", description="Embedding model identifier or 'auto'")
    llm_model: Optional[str] = Field(default="", description="GGUF LLM model identifier or 'auto'")
    embedder_device: Optional[str] = Field(default="auto", description="Target compute device: cpu, cuda, or auto")
    llm_device: Optional[str] = Field(default="auto", description="Target compute device: cpu, cuda, or auto")
    trees_storage: Optional[str] = Field(default="ram", description="Lattice storage strategy: ram or disk")


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000, description="Natural language specification")
    execute_in_sandbox: bool = Field(default=False, description="Whether to execute code in GEVR sandbox")
    timeout_seconds: float = Field(default=5.0, ge=0.5, le=60.0, description="Sandbox execution timeout")


class RunResponse(BaseModel):
    status: str = "success"
    prompt: str
    path: List[str]
    code: str
    latency_ms: float = 0.0
    route_latency_ms: float = 0.0
    synthesis_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    sandbox_result: Optional[Dict[str, Any]] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)


class HealthResponse(BaseModel):
    status: str
    nodes_count: int
    uptime_seconds: float
    memory_rss_mb: float


class DomainInfo(BaseModel):
    total_nodes: int
    curated_seeds: int


class DomainsResponse(BaseModel):
    domains: Dict[str, DomainInfo]
    total_domains: int
    total_nodes: int


# =====================================================================
# Core In-Process Engine Functions
# =====================================================================
def initialize_engine(
    req_or_profile: Union[InitRequest, Dict[str, Any], str] = "0",
    embedder_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    embedder_device: str = "auto",
    llm_device: str = "auto",
    trees_storage: str = "ram"
) -> Dict[str, Any]:
    """
    Initializes or reconfigures the NSTL synthesis engine for a specific profile.
    Can be called directly in-process or via REST endpoint.
    """
    global _orchestrator, _router, _gate, _sandbox, _internal_rag
    global _active_profile, _active_embedder, _active_llm, _engine_ready

    if isinstance(req_or_profile, InitRequest):
        prof = req_or_profile.profile
        emb_m = req_or_profile.embedder_model
        llm_m = req_or_profile.llm_model
        emb_dev = req_or_profile.embedder_device or embedder_device
        llm_dev = req_or_profile.llm_device or llm_device
    elif isinstance(req_or_profile, dict):
        prof = req_or_profile.get("profile", "0")
        emb_m = req_or_profile.get("embedder_model", embedder_model)
        llm_m = req_or_profile.get("llm_model", llm_model)
        emb_dev = req_or_profile.get("embedder_device", embedder_device)
        llm_dev = req_or_profile.get("llm_device", llm_device)
    else:
        prof = str(req_or_profile)
        emb_m = embedder_model
        llm_m = llm_model
        emb_dev = embedder_device
        llm_dev = llm_device

    p_norm = prof.strip().upper()
    with _engine_lock:
        _engine_ready = False
        _ensure_base_engine_loaded()

        if p_norm in ("0", "SYMBOLIC", "ZERO", "PURE"):
            _active_profile = "0"
            _active_embedder = "none"
            _active_llm = "none"
            _internal_rag = None
            _router = LatticeRouter(orchestrator=_orchestrator, internal_rag=None)
            _engine_ready = True
            logger.info("[✓] NSTL Engine configured for Profile 0 (Pure Symbolic).")
            return {
                "status": "ready",
                "profile": "0",
                "embedder": "none",
                "llm": "none",
                "device": "cpu",
                "nodes": len(_orchestrator.loaded_cells)
            }

        # Neural profiles: A, C, D, E
        from router import HardwareProfiler
        HardwareProfiler.set_config(embedder_device=emb_dev, llm_device=llm_dev)
        actual_device = HardwareProfiler.get_optimal_device()

        # Select optimal embedder dynamically
        optimal_emb = select_optimal_embedder(emb_m or "auto")
        _active_embedder = optimal_emb

        mm = ModelManager.get_instance()
        mm.initialize_profile(
            profile_type=p_norm,
            embedder_name=optimal_emb,
            llm_name=llm_m or ""
        )
        _active_llm = llm_m or getattr(mm.active_profile, "llm_name", "none")

        # Build / attach LocalRAG with precomputed embeddings
        _internal_rag = LocalRAG(trees_dir=str(settings.trees_dir), orchestrator=_orchestrator)
        _router = LatticeRouter(orchestrator=_orchestrator, internal_rag=_internal_rag)
        _active_profile = p_norm
        _engine_ready = True

        logger.info(f"[✓] NSTL Engine configured for Profile {p_norm} (Embedder: {optimal_emb}, LLM: {_active_llm}).")
        return {
            "status": "ready",
            "profile": p_norm,
            "embedder": optimal_emb,
            "llm": _active_llm,
            "device": actual_device,
            "nodes": len(_orchestrator.loaded_cells)
        }


def run_prompt(request: Union[RunRequest, Dict[str, Any], str]) -> RunResponse:
    """
    Executes end-to-end program synthesis for a natural language prompt.
    """
    _ensure_base_engine_loaded()
    if not _engine_ready:
        initialize_engine(_active_profile)

    if isinstance(request, RunRequest):
        prompt = request.prompt
        exec_sandbox = request.execute_in_sandbox
        timeout = request.timeout_seconds
    elif isinstance(request, dict):
        prompt = request.get("prompt", "")
        exec_sandbox = request.get("execute_in_sandbox", False)
        timeout = float(request.get("timeout_seconds", 5.0))
    else:
        prompt = str(request)
        exec_sandbox = False
        timeout = 5.0

    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    t_start = time.perf_counter()

    # 1. Routing
    t_route_0 = time.perf_counter()
    cells = _router.plan_path(prompt, return_tuple=False)
    route_dt = (time.perf_counter() - t_route_0) * 1000.0

    if not cells:
        return RunResponse(
            status="error",
            prompt=prompt,
            path=[],
            code="",
            latency_ms=round(route_dt, 2),
            route_latency_ms=round(route_dt, 2),
            synthesis_latency_ms=0.0,
            total_latency_ms=round(route_dt, 2),
            sandbox_result={"success": False, "error": f"No valid path found through lattice for prompt: {prompt}"}
        )

    # 2. Synthesis & Unification
    t_synth_0 = time.perf_counter()
    code = _gate.unify_and_emit(cells, prompt)
    synth_dt = (time.perf_counter() - t_synth_0) * 1000.0
    total_dt = (time.perf_counter() - t_start) * 1000.0

    # 3. Optional Sandbox Verification
    sandbox_result = None
    if exec_sandbox:
        dest_paths = getattr(_gate.context, "dest_files", None) if hasattr(_gate, "context") else None
        sandbox_result = _sandbox.execute(code, timeout=timeout, egress_paths=dest_paths)

    path_ids = [c.cell_id for c in cells]
    return RunResponse(
        status="success",
        prompt=prompt,
        path=path_ids,
        code=code,
        latency_ms=round(total_dt, 2),
        route_latency_ms=round(route_dt, 2),
        synthesis_latency_ms=round(synth_dt, 2),
        total_latency_ms=round(total_dt, 2),
        sandbox_result=sandbox_result
    )


# =====================================================================
# FastAPI Application & Lifespan
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes base lattice and engine on server boot."""
    _ensure_base_engine_loaded()
    initialize_engine("0")
    yield
    logger.info("[*] Shutting down NSTL engine...")
    if _sandbox and hasattr(_sandbox, "_pool") and _sandbox._pool is not None:
        try:
            _sandbox._pool.terminate()
            _sandbox._pool.join(timeout=3)
        except Exception:
            pass
    logger.info("[✓] NSTL engine shutdown complete.")


app = FastAPI(
    title="NSTL Neuro-Symbolic Topological Lattice API",
    description="High-throughput REST API for real-time neuro-symbolic program synthesis.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# API Endpoints
# =====================================================================
@app.get("/api/status", tags=["Status"])
async def get_status():
    """Engine status endpoint polled by frontend and evaluation runners."""
    _ensure_base_engine_loaded()
    nodes_cnt = len(_orchestrator.loaded_cells) if _orchestrator else 0
    return {
        "status": "ready" if _engine_ready else "initializing",
        "profile": _active_profile,
        "embedder": _active_embedder,
        "llm": _active_llm,
        "nodes_count": nodes_cnt,
        "uptime": round(time.time() - _start_time, 2),
        "benchmark_active": _benchmark_active
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Monitoring"])
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """System health check reporting uptime, node count, and memory footprint."""
    uptime = time.time() - _start_time
    nodes_count = len(_orchestrator.loaded_cells) if _orchestrator else 0
    try:
        import psutil
        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)

    return HealthResponse(
        status="healthy",
        nodes_count=nodes_count,
        uptime_seconds=round(uptime, 2),
        memory_rss_mb=round(rss_bytes / 1024 / 1024, 2)
    )


@app.post("/api/initialize", tags=["Lifecycle"])
async def api_initialize(req: InitRequest):
    """Dynamically reconfigures the synthesis profile, embedder, and LLM."""
    result = await asyncio.to_thread(initialize_engine, req)
    return result


@app.post("/api/run", response_model=RunResponse, tags=["Synthesis"])
@app.post("/synthesize", response_model=RunResponse, tags=["Synthesis"])
async def api_run(request: RunRequest):
    """Synthesizes executable code from a natural language prompt."""
    try:
        return await asyncio.to_thread(run_prompt, request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Backward-compatible API aliases
SynthesizeRequest = RunRequest
synthesize_pipeline = api_run


@app.get("/api/models", tags=["Metadata"])
async def get_models():
    """Scans and returns available local embedding models and GGUF LLMs."""
    emb_dir = os.path.join(MODELS_DIR, "embeddings")
    llm_dir = os.path.join(MODELS_DIR, "llms")

    embeddings = (
        sorted([d for d in os.listdir(emb_dir) if os.path.isdir(os.path.join(emb_dir, d)) and not d.endswith("-GGUF")])
        if os.path.exists(emb_dir) else []
    )
    llms = (
        sorted([d for d in os.listdir(llm_dir) if os.path.isdir(os.path.join(llm_dir, d))])
        if os.path.exists(llm_dir) else []
    )

    optimal_emb = select_optimal_embedder("auto")
    return {
        "embeddings": embeddings,
        "llms": llms,
        "optimal_embedder": optimal_emb,
        "active_profile": _active_profile,
        "active_embedder": _active_embedder,
        "active_llm": _active_llm
    }


@app.get("/api/cells", tags=["Metadata"])
async def get_cells(limit: int = 1000):
    """Returns cell node metadata for Three.js graph visualization."""
    _ensure_base_engine_loaded()
    if not _orchestrator:
        return {"cells": []}

    cells_out = []
    for c in list(_orchestrator.loaded_cells.values())[:limit]:
        in_sig = getattr(c, "primary_input", None)
        out_sig = getattr(c, "primary_output", None)
        cells_out.append({
            "id": c.cell_id,
            "domain": c.domain_name or "generic",
            "stage": c.stage,
            "input_type": in_sig.type_name if in_sig else "any",
            "output_type": out_sig.type_name if out_sig else "any",
            "verified": getattr(c, "verified", False)
        })
    return {"cells": cells_out, "total": len(_orchestrator.loaded_cells)}


@app.get("/domains", response_model=DomainsResponse, tags=["Metadata"])
async def get_domains():
    """Returns domain distribution across the active lattice."""
    _ensure_base_engine_loaded()
    domain_stats: Dict[str, Dict[str, int]] = {}

    for cell in _orchestrator.loaded_cells.values():
        dom = cell.domain_name or "generic"
        if dom not in domain_stats:
            domain_stats[dom] = {"total_nodes": 0, "curated_seeds": 0}
        domain_stats[dom]["total_nodes"] += 1
        if getattr(cell, "verified", False):
            domain_stats[dom]["curated_seeds"] += 1

    formatted_domains = {
        dom: DomainInfo(total_nodes=data["total_nodes"], curated_seeds=data["curated_seeds"])
        for dom, data in sorted(domain_stats.items())
    }

    return DomainsResponse(
        domains=formatted_domains,
        total_domains=len(formatted_domains),
        total_nodes=len(_orchestrator.loaded_cells)
    )


@app.post("/api/benchmark/toggle", tags=["Benchmark"])
async def toggle_benchmark():
    """Toggles active benchmark runner state."""
    global _benchmark_active
    _benchmark_active = not _benchmark_active
    return {"benchmark_active": _benchmark_active}


@app.get("/api/benchmark/status", tags=["Benchmark"])
async def get_benchmark_status():
    """Returns current benchmark execution state."""
    return {"benchmark_active": _benchmark_active}


# =====================================================================
# Server Entrypoint
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    host = settings.api_host
    port = int(os.environ.get("PORT", settings.api_port))
    logger.info(f"[*] Starting NSTL API Server on {host}:{port}...")
    uvicorn.run("main:app", host=host, port=port, reload=False)

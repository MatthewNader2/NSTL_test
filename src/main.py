# src/main.py
"""
src/main.py - Neuro-Symbolic Topological Lattice (NSTL)
Production REST API Server powered by FastAPI.
Provides real-time endpoints for health monitoring, domain metadata, and program synthesis.
"""

from __future__ import annotations
import os
import resource
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from lattice import LatticeOrchestrator
from router import LatticeRouter
from unification import UnificationGate
from gevr_sandbox import GEVRSandbox

DB_PATH = os.environ.get("NSTL_DB_PATH", "trees/lattice.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads the lattice orchestrator, router, unification gate, and sandbox pool on server startup."""
    print(f"[*] Preloading NSTL Lattice Database from '{DB_PATH}'...")
    t0 = time.perf_counter()
    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()
    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    gate = UnificationGate()
    sandbox = GEVRSandbox()

    app.state.orchestrator = orchestrator
    app.state.router = router
    app.state.gate = gate
    app.state.sandbox = sandbox
    app.state.start_time = time.time()

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[✓] NSTL API Server ready with {len(orchestrator.loaded_cells)} nodes loaded in {elapsed:.1f}ms.")
    yield


app = FastAPI(
    title="NSTL Neuro-Symbolic Topological Lattice API",
    description="High-throughput REST API for real-time sub-15ms neuro-symbolic program synthesis.",
    version="1.0.0",
    lifespan=lifespan
)


class SynthesizeRequest(BaseModel):
    prompt: str = Field(..., description="Natural language specification of the pipeline or task")
    execute_in_sandbox: bool = Field(default=False, description="Whether to execute synthesized code in isolated GEVR worker sandbox")
    timeout_seconds: float = Field(default=5.0, ge=0.5, le=30.0, description="Sandbox execution timeout limit")


class SynthesizeResponse(BaseModel):
    prompt: str
    path: List[str]
    code: str
    route_latency_ms: float
    synthesis_latency_ms: float
    total_latency_ms: float
    sandbox_result: Optional[Dict[str, Any]] = None


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


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint reporting uptime, lattice size, and memory footprint."""
    uptime = time.time() - getattr(app.state, "start_time", time.time())
    nodes_count = len(app.state.orchestrator.loaded_cells) if hasattr(app.state, "orchestrator") else 0
    try:
        import psutil
        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    return HealthResponse(
        status="healthy",
        nodes_count=nodes_count,
        uptime_seconds=round(uptime, 2),
        memory_rss_mb=round(rss_bytes / 1024 / 1024, 2)
    )


@app.get("/domains", response_model=DomainsResponse, tags=["Metadata"])
async def get_domains():
    """Returns domain distribution and curated seed counts across the active lattice."""
    if not hasattr(app.state, "orchestrator"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Lattice not initialized")

    orchestrator: LatticeOrchestrator = app.state.orchestrator
    domain_stats: Dict[str, Dict[str, int]] = {}

    for cell in orchestrator.loaded_cells.values():
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
        total_nodes=len(orchestrator.loaded_cells)
    )


@app.post("/synthesize", response_model=SynthesizeResponse, tags=["Synthesis"])
async def synthesize_pipeline(request: SynthesizeRequest):
    """
    Synthesizes executable code from a natural language prompt via A* topological routing and typestate unification.
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt must not be empty.")

    router: LatticeRouter = app.state.router
    gate: UnificationGate = app.state.gate
    sandbox: GEVRSandbox = app.state.sandbox

    # 1. Routing
    t_start = time.perf_counter()
    cells = router.plan_path(request.prompt, return_tuple=False)
    route_dt = (time.perf_counter() - t_start) * 1000.0

    if not cells:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No valid path found through lattice for prompt: {request.prompt}"
        )

    # 2. Synthesis
    t_synth = time.perf_counter()
    code = gate.unify_and_emit(cells, request.prompt)
    synth_dt = (time.perf_counter() - t_synth) * 1000.0
    total_dt = (time.perf_counter() - t_start) * 1000.0

    # 3. Optional Sandbox Execution
    sandbox_result = None
    if request.execute_in_sandbox:
        sandbox_result = sandbox.execute(code, timeout=request.timeout_seconds)

    path_ids = [c.cell_id for c in cells]
    return SynthesizeResponse(
        prompt=request.prompt,
        path=path_ids,
        code=code,
        route_latency_ms=round(route_dt, 2),
        synthesis_latency_ms=round(synth_dt, 2),
        total_latency_ms=round(total_dt, 2),
        sandbox_result=sandbox_result
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

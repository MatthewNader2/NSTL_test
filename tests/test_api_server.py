# tests/test_api_server.py
"""
tests/test_api_server.py - Neuro-Symbolic Topological Lattice (NSTL)
Phase 6 REST API Automated Verification Suite.
Validates /health, /domains, and /synthesize endpoints.
"""

import asyncio
import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from main import app, health_check, get_domains, synthesize_pipeline, SynthesizeRequest


@pytest.mark.anyio
async def test_api_endpoints_end_to_end():
    """Verify all REST API endpoints under async lifespan."""
    async with app.router.lifespan_context(app):
        # 1. Health endpoint
        health = await health_check()
        assert health.status == "healthy"
        assert health.nodes_count >= 34000
        assert health.uptime_seconds >= 0.0
        assert health.memory_rss_mb > 0.0

        # 2. Domains endpoint
        domains_resp = await get_domains()
        assert domains_resp.total_domains >= 7
        assert domains_resp.total_nodes >= 34000
        assert "pandas" in domains_resp.domains
        assert "cv2" in domains_resp.domains
        assert "sklearn" in domains_resp.domains
        assert "matplotlib" in domains_resp.domains

        # 3. Synthesize endpoint (Tabular)
        req1 = SynthesizeRequest(
            prompt="load input.csv and drop missing values then save to output.csv",
            execute_in_sandbox=False
        )
        synth1 = await synthesize_pipeline(req1)
        assert len(synth1.path) == 3
        assert "PANDAS_READ_CSV" in synth1.path
        assert "PANDAS_DROPNA" in synth1.path
        assert "PANDAS_TO_CSV" in synth1.path
        assert "import pandas as pd" in synth1.code
        assert synth1.total_latency_ms < 50.0

        # 4. Synthesize endpoint (Vision)
        req2 = SynthesizeRequest(
            prompt="read image input.jpg and convert to grayscale then save to output.jpg",
            execute_in_sandbox=False
        )
        synth2 = await synthesize_pipeline(req2)
        assert len(synth2.path) == 3
        assert "CV2_IMREAD" in synth2.path
        assert "CV2_CVTCOLOR" in synth2.path
        assert "CV2_IMWRITE" in synth2.path
        assert "import cv2" in synth2.code

        # 5. Synthesize endpoint (Cross-Domain)
        req3 = SynthesizeRequest(
            prompt="read input.csv, standardize features with sklearn StandardScaler, plot histogram of feature_1 with matplotlib, and save figure to plot.png",
            execute_in_sandbox=False
        )
        synth3 = await synthesize_pipeline(req3)
        assert len(synth3.path) == 4
        assert "SKLEARN_STANDARD_SCALER" in synth3.path
        assert "MATPLOTLIB_HISTOGRAM" in synth3.path
        assert "MATPLOTLIB_SAVEFIG" in synth3.path

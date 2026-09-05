# Neuro-Symbolic Topological Lattice (NSTL)

A category-theoretic neuro-symbolic program synthesis engine. NSTL synthesizes deterministic, verifiable, multi-domain code pipelines from natural language prompts using stage-stratified semantic tunneling, FAISS vector indexing, and formal monadic typestate unification over domain morphism trees.

---

## Architecture Overview

```
Natural Language Specification
              │
              ▼
   [ Semantic Routing ] ── Stage-Stratified Softmax Tunneling (T = T₁ ∪ T₂ ∪ T₃)
              │
              ▼
    [ Lattice Planner ] ── Monadic Morphism Composition & Curation Priority
              │
              ▼
   [ Robinson Unifier ] ── Dual Port Typestate Unification (τ_out ⊓ τ_in ≠ ⊥)
              │
              ▼
  [ Synthesizer & GEVR ] ── AST Assembly & Isolated Sandbox Execution
```

### Core Components (`src/`)

- **`src/lattice.py`**: Category-theoretic lattice representation, poset type hierarchy, and SQLite orchestrator.
- **`src/router.py`**: Semantic routing engine computing the active topological search tunnel $T$ via dense sentence embeddings and stage stratification.
- **`src/planner.py`**: Monadic composition planner constructing verified morphism chains with universal category-theoretic fitness.
- **`src/unification.py`**: Robinson first-order unification gate enforcing algebraic typestates, qualifier polarity projection, and in-scope variable resolution.
- **`src/synthesis.py`**: Deterministic Python code generation, import resolution, and AST variable renaming.
- **`src/gevr_sandbox.py`**: Isolated sandbox execution runtime validating pipeline output correctness.
- **`src/main.py`**: High-performance FastAPI REST server for programmatic pipeline synthesis.

---

## Directory Layout

```
nstl_prototype/
├── benchmarks/         # Benchmark suites, stress evaluators, and evaluation reports
│   └── reports/        # Benchmark result JSONs and summary audit reports
├── nstl_enrichment/    # LLM-guided domain enrichment checkpoints and progress logs
│   └── checkpoints/    # Enriched domain tree JSON checkpoints
├── packaging/          # PyInstaller specifications and application assets
├── scripts/            # Environment setup and launch automation
├── src/                # Core NSTL engine source code
├── tests/              # Unit tests, property tests, and end-to-end integration tests
│   └── fixtures/       # Test fixtures and reference datasets
├── tools/              # Tree compilation, sanitization, auditing, and repair utilities
├── trees/              # Canonical domain trees (*.json) and compiled SQLite lattice
├── run_phase0_smoke.py # Hardened end-to-end sandbox verification test suite
└── pytest.ini          # Pytest configuration
```

---

## Domain Knowledge Trees (`trees/`)

The repository contains 7 verified domain trees with **48,000+** compiled morphisms:
- **`cv2.json`**: Computer vision primitives (filtering, color space conversion, edge detection, I/O).
- **`pandas.json`**: Tabular data manipulation, aggregation, cleaning, and export.
- **`numpy.json`**: Array algebra, linear algebra, FFT, and tensor transformations.
- **`scipy.json`**: Scientific computing, interpolation, optimization, signal processing, and statistics.
- **`sklearn.json`**: Preprocessing, classification, regression, clustering, and model metrics.
- **`matplotlib.json`**: Data visualization, figure management, plotting, and image export.
- **`python_core.json`**: Graph algorithms, data structures, heap operations, and utilities.

To compile domain JSONs into the fast query database:
```bash
python3 tools/compile_trees.py
python3 tools/sanitize_lattice_db.py
```

---

## Verification & Testing

### 1. Phase 0 Hardened End-to-End Smoke Test
Executes real sandbox execution across Tabular, Vision, Algorithmic (Dijkstra), and Multi-Port Sort pipelines:
```bash
python3 run_phase0_smoke.py
```

### 2. Full Pytest Suite
```bash
pytest tests/test_mathematical_monadic_unification.py \
       tests/test_type_propagation_soundness.py \
       tests/test_phase0_fixes.py \
       tests/test_code_cleanliness.py \
       tests/test_api_server.py
```

### 3. Tree Structural QA Audit
```bash
python3 tools/audit_trees.py trees/*.json
```

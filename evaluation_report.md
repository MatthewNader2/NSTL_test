# NSTL Engine - Evaluation Report (Profiles A, C, D, E)

## 1. Executive Summary & Evaluation Matrix

This evaluation report documents the benchmark of the **Neural Syntax Tree Lattice (NSTL)** engine across target evaluation configurations for 1 representative model of each category (Embedder: `jina-embeddings-v5-text-small`, LLM: `qwen2.5-coder-1.5b-instruct`) across all 5 domain categories (excluding Profile B):

- **Profile A**: Deterministic / Embedding-Only (`jina-embeddings-v5-text-small`)

- **Profile C**: Full Pipeline with Zero-Shot Code Synthesis (`jina-embeddings-v5-text-small` + `qwen2.5-coder-1.5b-instruct`)

- **Profile D**: Synthesis Disabled (`jina-embeddings-v5-text-small` + `qwen2.5-coder-1.5b-instruct`)

- **Profile E**: Pre-Translation Pass + Code Synthesis (`jina-embeddings-v5-text-small` + `qwen2.5-coder-1.5b-instruct`)


---

## 2. Benchmark Summary Table

| Task ID | Domain Category | Profile A | Profile C | Profile D | Profile E | Overall Status |
|---|---|---|---|---|---|---|
| `pandas_csv_clean` | Data Engineering | 1.531s (FAILED) | 1.512s (PASSED) | 1.589s (FAILED) | 1.455s (PASSED) | **PARTIAL / FAILED** |
| `opencv_gray_convert` | Image Processing | 1.682s (FAILED) | 1.329s (FAILED) | 2.195s (FAILED) | 1.329s (FAILED) | **PARTIAL / FAILED** |
| `vague_data_transform` | Vague Human Prompt | 2.087s (FAILED) | 8.717s (FAILED) | 2.627s (FAILED) | 8.661s (FAILED) | **PARTIAL / FAILED** |
| `long_ml_pipeline` | Long ML/Data Pipeline | 2.972s (FAILED) | 8.278s (FAILED) | 4.043s (FAILED) | 8.294s (FAILED) | **PARTIAL / FAILED** |
| `dijkstra_algorithm` | Multi-Step Algorithm | 2.302s (FAILED) | 2.838s (FAILED) | 3.478s (FAILED) | 2.843s (FAILED) | **PARTIAL / FAILED** |

---

## 3. Performance & Accuracy Breakdown

### A. Pass Rate & Accuracy

- **Profile A**: 0/5 Passed (**0.0%**)
- **Profile C**: 1/5 Passed (**20.0%**)
- **Profile D**: 0/5 Passed (**0.0%**)
- **Profile E**: 1/5 Passed (**20.0%**)

### B. Mean Execution Latency per Profile

- **Profile A**: **2.115s** average latency
- **Profile C**: **4.535s** average latency
- **Profile D**: **2.786s** average latency
- **Profile E**: **4.516s** average latency

### C. AST Program Complexity

- **Profile A**: Average AST Node Count = **61.2 nodes**
- **Profile C**: Average AST Node Count = **44.0 nodes**
- **Profile D**: Average AST Node Count = **65.6 nodes**
- **Profile E**: Average AST Node Count = **44.0 nodes**
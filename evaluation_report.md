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
| `pandas_csv_clean` | Data Engineering | 3.191s (FAILED) | 1.835s (FAILED) | 3.615s (FAILED) | 1.731s (FAILED) | **PARTIAL / FAILED** |
| `opencv_gray_convert` | Image Processing | 35.897s (FAILED) | 3.191s (FAILED) | 37.481s (FAILED) | 3.091s (FAILED) | **PARTIAL / FAILED** |
| `vague_data_transform` | Vague Human Prompt | 1.360s (FAILED) | 3.040s (FAILED) | 1.945s (FAILED) | 3.000s (FAILED) | **PARTIAL / FAILED** |
| `long_ml_pipeline` | Long ML/Data Pipeline | 6.092s (FAILED) | 8.493s (FAILED) | 7.297s (FAILED) | 8.405s (FAILED) | **PARTIAL / FAILED** |
| `dijkstra_algorithm` | Multi-Step Algorithm | 3.431s (FAILED) | 7.331s (FAILED) | 3.859s (FAILED) | 7.361s (FAILED) | **PARTIAL / FAILED** |

---

## 3. Performance & Accuracy Breakdown

### A. Pass Rate & Accuracy

- **Profile A**: 0/5 Passed (**0.0%**)
- **Profile C**: 0/5 Passed (**0.0%**)
- **Profile D**: 0/5 Passed (**0.0%**)
- **Profile E**: 0/5 Passed (**0.0%**)

### B. Mean Execution Latency per Profile

- **Profile A**: **9.994s** average latency
- **Profile C**: **4.778s** average latency
- **Profile D**: **10.839s** average latency
- **Profile E**: **4.718s** average latency

### C. AST Program Complexity

- **Profile A**: Average AST Node Count = **51.6 nodes**
- **Profile C**: Average AST Node Count = **64.2 nodes**
- **Profile D**: Average AST Node Count = **49.2 nodes**
- **Profile E**: Average AST Node Count = **64.2 nodes**
# NSTL Engine - Evaluation Report (Profile A & Profile D)

## 1. Executive Summary & Evaluation Matrix

This paper evaluation report documents the benchmark of the hardened **Neural Syntax Tree Lattice (NSTL)** engine across target evaluation configurations on the verified 21,753-node lattice database:

- **Profile A**: Deterministic / No LLM (`jina-embeddings-v5-text-nano`)

- **Profile D**: Dedicated Embedder (`jina-embeddings-v5-text-small`) + 7B LLM (`Qwen2.5-Coder-7B-Instruct-GGUF`), zero-shot code synthesis disabled


---

## 2. Benchmark Summary Table

| Task ID | Domain | Profile A | Profile D | Validation Status |
|---|---|---|---|---|
| `pandas_csv_clean` | Data Engineering | 30.330s (PASSED) | 31.856s (PASSED) | **PASSED** |
| `opencv_gray_convert` | Image Processing | 29.138s (PASSED) | 33.680s (PASSED) | **PASSED** |
| `vague_data_transform` | Vague Human Prompt | 20.593s (PASSED) | 20.882s (PASSED) | **PASSED** |
| `long_ml_pipeline` | Long ML/Data Pipeline | 41.304s (PASSED) | 42.843s (PASSED) | **PASSED** |
| `dijkstra_algorithm` | Multi-Step Algorithm | 12.566s (PASSED) | 12.641s (PASSED) | **PASSED** |

---

## 3. Performance & Latency Breakdown

### A. Mean Execution Latency per Profile

- **Profile A**: **26.786s** average generation latency
- **Profile D**: **28.380s** average generation latency

### B. AST Node Program Complexity

- **Profile A**: Average AST Node Count = **106.8 nodes**
- **Profile D**: Average AST Node Count = **106.8 nodes**

---

## 4. Soundness & Structural Integrity Confirmation

1. **Zero Hardcoded Fallbacks & Task-Sniffing Rules**: Verification confirmed that 0 keyword task-sniffing shortcuts or benchmark fallbacks were triggered during execution.

2. **GEVR Sandboxed Execution & Verification**: All generated programs passed sandboxed execution with strict stdout/file assertion checks.

3. **Lattice Invariant Compliance**: The lattice database was compiled with 21,753 verified nodes, maintaining 0 self-named function type bugs (`cvtColor`, `Expanding`, `slice` mapped to canonical types).

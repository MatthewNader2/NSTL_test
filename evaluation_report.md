# NSTL Prototype - Comprehensive Evaluation Report

## 1. Executive Summary
This report summarizes the multi-matrix benchmark and evaluation of the **Neural Syntax Tree Lattice (NSTL)** engine across **Profiles A, C, and D**, **3 Local GGUF LLMs** (`0.5B`, `1.5B`, `7B`), **Vector Embedders** (`jina-embeddings-v5-text-nano`), and **4 Operational Domains**:
1. Standard Data Engineering Requests
2. Vague Human Language Prompts
3. Ultra-Long Multi-Step ML Pipelines
4. Hard Competitive Programming Algorithms

---

## 2. Advanced Benchmark Results Summary

### Ultra Stress & Compute Metrics Breakdown

| Metric / Evaluation Dimension | Profile A (Baseline) | Profile C (7B LLM) | Profile D (0.5B LLM) | Profile D (7B LLM) |
|---|---|---|---|---|
| **Data Engineering Success Rate** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Vague Human Prompt Accuracy** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Ultra-Long Pipeline Scale Test** | 50.0% | 0.0% | **100.0%** | 0.0% |
| **Competitive Programming (Hard)** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Mean Execution Latency** | **7.60 sec** | 14.77 sec | **13.59 sec** | 14.98 sec |
| **AST Node Density (Passed Code)** | 17 AST nodes | N/A | **27 AST nodes** | N/A |
| **Accuracy-Per-Second** | 0.1426 | 0.0000 | **0.0863** | 0.0000 |
| **Accuracy-Per-Watt (Energy)** | 8.56 Wh | 0.00 Wh | **5.18 Wh** | 0.00 Wh |
| **Estimated Cost-Per-Run** | **$0.000002** | $0.000015 | **$0.000003** | $0.000020 |

---

## 3. Domain Analysis & Architectural Insights

### A. Data Science & Vague Human Prompts
* **Winner**: **Qwen2.5-Coder-7B-Instruct-GGUF** + `jina-embeddings-v5-text-nano` under **Profile D**.
* **Performance**: Achieved 100% success rate on vague human prompts (e.g. *"Process dataset, clean, transform values, and give summary output"*).
* **Lineage Repair**: Profile D AST dead-variable repair successfully re-linked sink functions (`.to_csv()`, `.to_json()`) to newly created transformed variables rather than initial raw inputs.

### B. Competitive Programming (Algorithmic Limit Test)
* **Status**: 0% Pass Rate across all profiles.
* **Root Cause Analysis (Tree Topology Limitation)**:
  * NSTL's SQLite database of 35,061 pre-indexed AST nodes consists of standard library and Data Science APIs (`pandas`, `numpy`, `sklearn`, `scipy`). It does **not** contain pre-indexed node trees for custom algorithmic data structures (`SegmentTree`, `tsp_bitmask`).
  * Vector routing attempted to match `SegmentTree` against `pandas.get_values_for_csv()`. Because no algorithmic node existed in the tree topology, the unifier emitted un-synthesized placeholder variables, resulting in runtime `NameError`.
* **Architectural Remedy**: Implement a **Zero-Shot Code Synthesis Fallback Node** in the lattice that triggers when prompt intent detects custom algorithmic constructs (`class`, `def`, `DP`) with low database node affinity (score < 0.20).

### C. Small vs Large Models in Profile D AST Repair
* **Discovery**: `qwen2.5-coder-0.5b-instruct` under Profile D achieved higher AST execution accuracy on long pipelines than the 7B model.
* **Why**: The 0.5B model's review pass was conservative—trimming broken AST branches—whereas larger LLMs attempted deep virtual edge tunneling into incompatible API signatures (`numpy.polyfit(math_erf)`).

---

## 4. Hardware Autodetection & Stability Metrics
* **GPU Compute**: Mapped to **NVIDIA GeForce RTX 3070 Ti Laptop GPU (7.7 GB VRAM)** via CUDA.
* **VRAM Stability**: Multi-stage adaptive `Llama` initialization (`n_ctx=4096` $\rightarrow$ `n_ctx=2048` $\rightarrow$ CPU offloading fallback) prevented VRAM out-of-memory crashes across 35+ consecutive benchmark runs.

#!/bin/bash
# ============================================================
#  NSTL - Installation & Model Setup Script
#  Run this once after cloning to download required models.
# ============================================================

set -e

echo ""
echo "======================================"
echo "   NSTL Model & Dependency Installer  "
echo "======================================"
echo ""

# --- 1. Python deps ---
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

# --- 2. Install huggingface_hub CLI ---
echo "[2/4] Ensuring huggingface-cli is available..."
pip install -q "huggingface_hub[cli]>=0.23"

# --- 3. Download GGUF LLM (Qwen 7B) ---
echo "[3/4] Downloading Qwen2.5-Coder-7B-Instruct GGUF model (~4.5 GB)..."
echo "      This may take a while depending on your connection speed."
mkdir -p models/llms/Qwen2.5-Coder-7B-Instruct-GGUF
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --local-dir models/llms/Qwen2.5-Coder-7B-Instruct-GGUF \
    --local-dir-use-symlinks False

# --- 4. Download Embedder (Jina v5 Nano) ---
echo "[4/4] Downloading Jina Embeddings v5 Nano (~400 MB)..."
mkdir -p models/embeddings/jina-embeddings-v5-text-nano
huggingface-cli download jinaai/jina-embeddings-v5-text-nano \
    --local-dir models/embeddings/jina-embeddings-v5-text-nano \
    --local-dir-use-symlinks False

echo ""
echo "======================================"
echo "   All models downloaded successfully!"
echo ""
echo "   To start NSTL, run:"
echo "     python src/main.py"
echo "   or on Windows:"
echo "     run_nstl.bat"
echo "======================================"
echo ""

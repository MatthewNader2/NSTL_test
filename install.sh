#!/bin/bash

echo "======================================"
echo " NSTL Installation & Model Downloader"
echo "======================================"

echo "[1/3] Installing huggingface-cli..."
pip install -q huggingface_hub

echo "[2/3] Downloading Qwen2.5-Coder-7B-Instruct-GGUF..."
mkdir -p models/llms/Qwen2.5-Coder-7B-Instruct-GGUF
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --local-dir models/llms/Qwen2.5-Coder-7B-Instruct-GGUF \
    --local-dir-use-symlinks False

echo "[3/3] Downloading Embeddings (Jina Nano)..."
# Replace 'jinaai/jina-embeddings-v2-small-en' with your exact repo if it's different.
mkdir -p models/embeddings/jina-embeddings-v5-text-nano
huggingface-cli download jinaai/jina-embeddings-v2-small-en \
    --local-dir models/embeddings/jina-embeddings-v5-text-nano \
    --local-dir-use-symlinks False

echo "======================================"
echo " Setup complete! You can now start the project."
echo "======================================"

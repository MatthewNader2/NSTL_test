"""
colab_enrich.py — Tier 3 LLM enrichment for NSTL trees/*.json

Workflow this script assumes:
  1. You've committed & pushed the repo (post bug-fixes) from your machine.
  2. On Colab: clone the repo fresh, mount Drive, run this script.
  3. It reads trees/{domain}.json from the fresh clone, but reads/writes its
     *progress* against a checkpoint copy on Drive, so a Colab disconnect never
     loses more than one in-flight batch.
  4. Re-running this script (same session or a new one) picks up exactly where
     it left off — already-enriched cells are skipped, not reprocessed.

Only ever fills `docstring` when it is empty, and only ever on cells missing one.
Never touches code_template / inputs / outputs / stage / dependencies.

--- Model ---
Default: Qwen2.5-14B-Instruct-GGUF, Q4_K_M quant (~9GB), via llama-cpp-python.
Matches the quantized-GGUF approach the project already uses locally for its
GGUF LLM profile, so nothing new to learn. Q4_K_M 14B fits comfortably on a
free-tier T4 (16GB VRAM) with room for KV cache at the short context lengths
used here. If you're on a smaller GPU or want more throughput, drop
MODEL_REPO/MODEL_FILE to the 7B instruct GGUF (see MODEL OPTIONS below) — the
task here (short factual descriptions from already-given facts, not open-ended
generation) does not need the full 14B to do well, so 7B is a completely
reasonable choice if you want to move faster over ~15k+ cells.

Run in Colab:
    !pip install -q llama-cpp-python huggingface_hub
    !python colab_enrich.py
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================== CONFIG ======================================

# --- Repo / paths ---
REPO_URL = "https://github.com/MatthewNader2/NSTL_test.git"
IS_COLAB = Path("/content").exists()
LOCAL_REPO_DIR = Path(__file__).resolve().parent
LOCAL_CLONE_DIR = Path("/content/NSTL_test") if IS_COLAB else LOCAL_REPO_DIR
DRIVE_ROOT = Path("/content/drive/MyDrive/nstl_enrichment") if IS_COLAB else LOCAL_REPO_DIR / "enrichment_checkpoints"
DRIVE_CHECKPOINTS = DRIVE_ROOT / "checkpoints"        # working copies of trees/*.json
DRIVE_LOGS = DRIVE_ROOT / "logs"

# Which domain trees to enrich, in order. Add/remove as needed.
DOMAINS = ["pandas", "sklearn", "numpy", "cv2", "matplotlib", "scipy", "python_core"]

# --- Model options ---
# Qwen3.5-9B, Q4_K_M quant (~5.7GB), via llama-cpp-python. Dense, not MoE — a
# predictable ~6GB VRAM footprint on a free T4 with plenty of headroom for KV
# cache at these short context lengths.
#
# Checked directly against Qwen3-14B (older, larger, non-.5 generation) before
# picking this: on every benchmark both report (GPQA-Diamond, MMLU-Redux, IFEval),
# Qwen3.5-9B beats Qwen3-14B, including Qwen3-14B's thinking-mode scores
# (GPQA-Diamond 81.7 vs 64.0, MMLU-Redux 91.1 vs 88.6, IFEval 91.5 vs 85.4) despite
# being ~35% smaller — newer generation with an improved architecture, not just a
# smaller version of the same thing. "Qwen3.5-14B" genuinely doesn't exist as a
# released size; the "Small" tier stops at 9B, "Medium" jumps straight to 27B+ MoE.
#
# Note: Qwen3.5-9B is natively multimodal (vision-language). The GGUF text weights
# work standalone via llama-cpp-python with no mmproj file needed for this
# text-only enrichment task.
# Verified real, single-file, exact repo+filename as of writing:
MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"

# Fallback for higher throughput at some quality cost, still fits a T4 easily:
# MODEL_FILE = "Qwen3.5-9B-Q5_K_M.gguf"   # 6.58GB

MODEL_LOCAL_DIR = Path("/content/models")             # ephemeral cache, re-downloaded
N_CTX = 4096
N_GPU_LAYERS = -1   # offload everything to GPU; llama-cpp-python handles Colab's T4 fine

# --- Batching ---
# Cells per LLM call. Kept deliberately small: each cell's context (code_template,
# ports, tags) is short, but asking a 7-14B model to track and correctly index 50+
# independent items in one JSON array degrades reliably around output length. 20 is
# a comfortable margin below where you start seeing dropped/merged entries; raise it
# if you're validating cleanly and want more throughput, lower it if you see batches
# failing schema validation.
BATCH_SIZE = 20
MAX_RETRIES_PER_BATCH = 2

# ============================================================================


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def mount_drive() -> None:
    """Does NOT call drive.mount() itself — that requires the live notebook
    kernel to show the auth prompt, which isn't available when this script runs
    as a subprocess (e.g. `!python colab_enrich.py`). Mount Drive from an actual
    notebook cell first:

        from google.colab import drive
        drive.mount('/content/drive')

    Once mounted, it's a real filesystem mount — this script (or any subprocess)
    can read/write it directly with no further API calls needed. This function
    just checks it's actually mounted before proceeding."""
    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError(
            "Google Drive isn't mounted. Run this in its own Colab cell first, "
            "then re-run this script:\n\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
        )
    DRIVE_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    DRIVE_LOGS.mkdir(parents=True, exist_ok=True)


def clone_repo() -> None:
    if not IS_COLAB:
        log("Running in local environment — using local repository directly.")
        return
    if LOCAL_CLONE_DIR.exists():
        shutil.rmtree(LOCAL_CLONE_DIR)
    os.system(f"git clone --depth 1 {REPO_URL} {LOCAL_CLONE_DIR}")
    if not (LOCAL_CLONE_DIR / "trees").exists():
        raise RuntimeError("Clone failed or repo layout changed — no trees/ dir found.")


def get_hf_token() -> Optional[str]:
    """Colab's Secrets panel (userdata) is the safer place for this than a plain
    env var — it's not shown in cell output/logs and survives across sessions
    without retyping. Falls back to a plain HF_TOKEN env var for non-Colab runs."""
    if IS_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                return token
        except Exception:
            pass  # secret not set, or userdata unavailable — fall through
    return os.environ.get("HF_TOKEN")


def load_model():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    token = get_hf_token()
    if token:
        log("HF_TOKEN found — using authenticated download.")
    else:
        log("No HF_TOKEN found (checked Colab secrets and env var) — proceeding "
            "unauthenticated. Fine for public repos, but gated models or heavy "
            "rate-limiting will fail without it. To add one: Colab's left sidebar "
            "key icon -> add secret named HF_TOKEN -> enable notebook access.")

    MODEL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Ensuring model present: {MODEL_REPO}/{MODEL_FILE} (downloads if missing)...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_LOCAL_DIR),
        token=token,
    )
    log(f"Loading model from {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=True,  # first run: confirm "offloaded N/N layers to GPU" appears —
                       # if it doesn't, llama-cpp-python has no CUDA support and
                       # everything is running on CPU regardless of N_GPU_LAYERS
    )
    return llm


# --------------------------- checkpoint I/O ---------------------------------

def checkpoint_path(domain: str) -> Path:
    return DRIVE_CHECKPOINTS / f"{domain}.json"


def load_working_tree(domain: str) -> Dict[str, Any]:
    """Resume from Drive checkpoint if present, else seed from the fresh clone."""
    ckpt = checkpoint_path(domain)
    if ckpt.exists():
        log(f"[{domain}] Resuming from existing Drive checkpoint.")
        with open(ckpt, "r", encoding="utf-8") as f:
            return json.load(f)

    src = LOCAL_CLONE_DIR / "trees" / f"{domain}.json"
    if not src.exists():
        raise FileNotFoundError(f"No tree found for domain '{domain}' at {src}")
    log(f"[{domain}] No checkpoint yet — seeding from repo's trees/{domain}.json")
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save(domain: str, data: Dict[str, Any]) -> None:
    ckpt = checkpoint_path(domain)
    tmp = ckpt.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ckpt)  # atomic on same filesystem


# --------------------------- prompt construction -----------------------------

SYSTEM_PROMPT = """You are documenting a library's functions for a code-synthesis tool.
For each function given, write ONE short, plain, factual sentence describing what it
does — like a one-line docstring summary. Base it ONLY on the function name, its code
template, and its input/output types given to you. Do NOT invent parameter behavior,
defaults, or edge cases that aren't visible in what you're given. If you are not
confident what a function does from its name and signature alone, write a generic but
honest description (e.g. "Performs the <x> operation on <type>.") rather than guessing
specifics.

Respond with ONLY a JSON array, no prose before or after, with exactly one object per
input item, in the same order, using this shape:
[{"cell_id": "...", "docstring": "..."}, ...]
"""


def build_batch_prompt(cells: List[Dict[str, Any]]) -> str:
    items = []
    for c in cells:
        in_types = {k: v.get("type_name") for k, v in c.get("inputs", {}).items()}
        out_types = {k: v.get("type_name") for k, v in c.get("outputs", {}).items()}
        items.append({
            "cell_id": c["cell_id"],
            "code_template": c.get("code_template", ""),
            "inputs": in_types,
            "outputs": out_types,
            "existing_tags": c.get("semantic_tags", [])[:8],
            # Pass along any Tier-2 partial docstring fragment as grounding context,
            # even if it was judged too thin to count as "enriched" on its own.
            "partial_docs": c.get("docstring") or None,
        })
    return json.dumps(items, indent=2)


def extract_json_array(raw: str) -> Optional[List[Dict[str, str]]]:
    """Same spirit as src/utils.py::extract_json_from_llm — don't repeat the fence bug."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def call_model(llm, cells: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Returns {cell_id: docstring} for a validated batch, or None if it never validates."""
    prompt = build_batch_prompt(cells)
    expected_ids = [c["cell_id"] for c in cells]

    for attempt in range(1, MAX_RETRIES_PER_BATCH + 1):
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200 * len(cells),
        )
        raw = resp["choices"][0]["message"]["content"]
        parsed = extract_json_array(raw)

        if parsed is None:
            log(f"  attempt {attempt}: response wasn't valid JSON, retrying...")
            continue

        result_ids = [item.get("cell_id") for item in parsed if isinstance(item, dict)]
        if len(parsed) != len(expected_ids) or set(result_ids) != set(expected_ids):
            log(f"  attempt {attempt}: id/length mismatch "
                f"(expected {len(expected_ids)}, got {len(parsed)}), retrying...")
            continue

        return {item["cell_id"]: str(item.get("docstring", "")).strip() for item in parsed}

    return None


# --------------------------------- main --------------------------------------

def enrich_domain(llm, domain: str) -> None:
    data = load_working_tree(domain)
    cells = data.get("cells", [])

    pending = [c for c in cells if not c.get("docstring") and c.get("enrichment_source") != "llm"]
    already_done = len(cells) - len(pending)
    log(f"[{domain}] {len(cells)} total cells | {already_done} already enriched/native | "
        f"{len(pending)} remaining")

    if not pending:
        log(f"[{domain}] Nothing to do.")
        return

    by_id = {c["cell_id"]: c for c in cells}
    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log(f"[{domain}] batch {batch_num}/{n_batches} ({len(batch)} cells)...")

        result = call_model(llm, batch)
        if result is None:
            log(f"[{domain}] batch {batch_num} FAILED validation after retries — "
                f"skipping, will retry on next run. cell_ids: "
                f"{[c['cell_id'] for c in batch]}")
            with open(DRIVE_LOGS / "failed_batches.log", "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} {domain} "
                        f"{[c['cell_id'] for c in batch]}\n")
            continue

        now = datetime.now(timezone.utc).isoformat()
        for cid, doc in result.items():
            cell = by_id.get(cid)
            if cell is None or not doc:
                continue
            cell["docstring"] = doc
            cell["enrichment_source"] = "llm"
            cell["enriched_at"] = now

        # Checkpoint after every batch, not every tree — this is the resumability
        # guarantee for large trees (cv2 is ~11k cells; don't risk hours of work).
        atomic_save(domain, data)
        log(f"[{domain}] batch {batch_num}/{n_batches} done, checkpoint saved.")

    log(f"[{domain}] Finished. Checkpoint at {checkpoint_path(domain)}")


def main() -> None:
    mount_drive()
    clone_repo()
    llm = load_model()

    for domain in DOMAINS:
        try:
            enrich_domain(llm, domain)
        except Exception as e:
            log(f"[{domain}] ERROR: {e} — moving to next domain, this one is "
                f"resumable from its last checkpoint on next run.")

    log("All domains processed for this run. Copy the checkpoints in "
        f"{DRIVE_CHECKPOINTS} over trees/*.json in your repo and commit.")


if __name__ == "__main__":
    main()

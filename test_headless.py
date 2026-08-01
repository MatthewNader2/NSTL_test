"""
NSTL Headless Backend Test
Starts the FastAPI server via uvicorn (no GUI) and runs a sequence of
API calls to verify the backend engine works end-to-end.
"""
import sys
import os
import io
import time
import json
import threading

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import urllib.request
import urllib.error

# Ensure src/ is on the path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

BASE = "http://127.0.0.1:58102"
PASS = 0
FAIL = 0
ERRORS = []


def api_get(path, timeout=10):
    url = f"{BASE}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def api_post(path, payload, timeout=120):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


def wait_for_server(timeout=15):
    """Poll until the server is accepting connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 58102), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def main():
    global PASS, FAIL
    
    print("=" * 64)
    print("  NSTL HEADLESS BACKEND TEST")
    print("=" * 64)
    
    # ── 1. Start uvicorn in a daemon thread ──
    print("\n[1/7] Starting uvicorn server (headless)...")
    import uvicorn
    
    # Import main to get the FastAPI app
    import main as nstl_main
    
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(nstl_main.app, host="127.0.0.1", port=58102, log_level="warning"),
        daemon=True,
    )
    server_thread.start()
    
    if not wait_for_server():
        print("  [FAIL] Server failed to start within 15 seconds!")
        sys.exit(1)
    print("  [PASS] Server is listening on port 58102")
    
    # ── 2. Test /api/health ──
    print("\n[2/7] Testing /api/health ...")
    try:
        health = api_get("/api/health")
        check("Health endpoint returns 200", True)
        check("Health has 'status' key", "status" in health, str(health))
    except Exception as e:
        check("Health endpoint reachable", False, str(e))
    
    # ── 3. Test /api/status (before init) ──
    print("\n[3/7] Testing /api/status (pre-init) ...")
    try:
        status = api_get("/api/status")
        check("Status returns JSON", True)
        check("Status is 'uninitialized'", status.get("status") == "uninitialized",
              f"got: {status.get('status')}")
    except Exception as e:
        check("Status endpoint reachable", False, str(e))
    
    # ── 4. Test /api/models ──
    print("\n[4/7] Testing /api/models ...")
    try:
        models = api_get("/api/models")
        check("Models returns JSON", True)
        check("Models has 'embedders' key", "embedders" in models)
        check("Models has 'llms' key", "llms" in models)
        print(f"       Found: {len(models.get('embedders', []))} embedders, {len(models.get('llms', []))} LLMs")
        if models.get("embedders"):
            print(f"       Embedders: {models['embedders']}")
        if models.get("llms"):
            print(f"       LLMs: {models['llms']}")
    except Exception as e:
        check("Models endpoint reachable", False, str(e))
    
    # ── 5. Initialize the engine (Profile A = embedder only, fastest) ──
    print("\n[5/7] Testing /api/initialize (Profile A — embedder only) ...")
    init_ok = False
    try:
        # Use Profile A (embedder only) for fastest init
        init_result = api_post("/api/initialize", {
            "profile": "A",
            "embedder_model": "auto",
            "llm_model": "",
            "embedder_device": "auto",
            "llm_device": "auto",
            "trees_storage": "ram"
        }, timeout=120)
        check("Initialize returns JSON", True)
        check("Initialize has 'status' key", "status" in init_result, str(init_result))
        
        if init_result.get("status") == "error":
            check("Initialize succeeded", False, init_result.get("message", "unknown error"))
        else:
            check("Initialize succeeded", True)
            init_ok = True
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, 'read') else ""
        check("Initialize returned 200", False, f"HTTP {e.code}: {body[:300]}")
    except Exception as e:
        check("Initialize reachable", False, str(e))
    
    # Wait for engine to become ready
    if init_ok:
        print("       Waiting for engine to become ready...")
        ready = False
        for _ in range(120):  # up to 120 seconds for initial vector embedding / cache build
            try:
                s = api_get("/api/status")
                if s.get("status") == "ready":
                    ready = True
                    break
                elif s.get("status") == "error":
                    check("Engine reached ready state", False, s.get("message", "unknown"))
                    break
            except:
                pass
            time.sleep(1)
        
        if ready:
            check("Engine reached ready state", True)
            
            # ── 6. Test /api/cells ──
            print(f"\n[6/7] Testing /api/cells ...")
            try:
                cells_data = api_get("/api/cells")
                cell_count = cells_data.get("count", 0)
                check("Cells endpoint returns JSON", True)
                check("Cells count > 0", cell_count > 0, f"got {cell_count}")
                print(f"       Loaded: {cell_count} cells")
                
                # Sample a few cell IDs
                cells_list = cells_data.get("cells", [])
                if cells_list:
                    sample = cells_list[:3]
                    for c in sample:
                        print(f"       Sample: {c['cell_id']} | {c.get('type','?')} | {c['inputs'].get('type_name','?')} → {c['outputs'].get('type_name','?')}")
            except Exception as e:
                check("Cells endpoint reachable", False, str(e))
            
            # ── 7. Test /api/run ──
            print(f"\n[7/7] Testing /api/run (prompt execution) ...")
            test_prompt = 'Open "sample_input.txt" and strip whitespace from the text.'
            try:
                run_result = api_post("/api/run", {"prompt": test_prompt}, timeout=120)
                check("Run endpoint returns JSON", True)
                
                code = run_result.get("code", "")
                logs = run_result.get("logs", [])
                path = run_result.get("path", [])
                virtual = run_result.get("virtual_edges", [])
                
                check("Run returned generated code", len(code) > 10, f"code length: {len(code)}")
                check("Run returned execution logs", len(logs) > 0, f"log count: {len(logs)}")
                check("Run returned execution path", len(path) > 0, f"path length: {len(path)}")
                
                # Print execution path
                if path:
                    path_ids = [n.get("cell_id", "?") for n in path]
                    print(f"       Path: {' → '.join(path_ids)}")
                
                if virtual:
                    print(f"       Virtual edges (synthesized): {virtual}")
                
                # Print logs
                for log in logs:
                    log_type = log.get("type", "info")
                    icon = {"info": "[i]", "warn": "[!]", "error": "[X]", "debug": "[.]", "system": "[*]"}.get(log_type, "   ")
                    print(f"       {icon} [{log_type.upper()}] {log.get('msg', '')}")
                
                # Print generated code snippet
                if code:
                    print(f"\n       ── Generated Code ({len(code)} chars) ──")
                    for line in code.split("\n")[:15]:
                        print(f"       | {line}")
                    if code.count("\n") > 15:
                        print(f"       | ... ({code.count(chr(10)) - 15} more lines)")
                        
            except urllib.error.HTTPError as e:
                body = e.read().decode() if hasattr(e, 'read') else ""
                check("Run returned 200", False, f"HTTP {e.code}: {body[:300]}")
            except Exception as e:
                check("Run endpoint reachable", False, str(e))
        else:
            check("Engine reached ready state (timeout)", False, "Engine did not become ready within 60s")
            print("\n[6/7] Skipping /api/cells (engine not ready)")
            print("\n[7/7] Skipping /api/run (engine not ready)")
    else:
        print("\n[6/7] Skipping /api/cells (init failed)")
        print("\n[7/7] Skipping /api/run (init failed)")
    
    # ── Summary ──
    print("\n" + "=" * 64)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 64)
    
    if ERRORS:
        print("\n  FAILURES:")
        for err in ERRORS:
            print(f"  {err}")
    
    print("\n  Backend test complete. Exiting.\n")
    os._exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()

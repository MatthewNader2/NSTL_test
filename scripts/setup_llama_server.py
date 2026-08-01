import os
import sys
import json
import urllib.request
import zipfile
import tarfile
import stat
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_latest_release():
    url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    return data

def setup_server():
    target_dir = os.path.join(PROJECT_ROOT, "tools", "llama-cpp")
    os.makedirs(target_dir, exist_ok=True)
    
    print("[*] Fetching latest llama.cpp release metadata...")
    release = get_latest_release()
    assets = release.get("assets", [])
    
    llama_url = None
    cudart_url = None
    is_win = sys.platform.startswith("win")
    is_mac = sys.platform == "darwin"
    
    for asset in assets:
        name = asset["name"].lower()
        if is_win:
            if ("bin-win-cuda" in name or "bin-win-cu" in name or "bin-win-x64" in name) and name.endswith(".zip"):
                if name.startswith("llama-"):
                    llama_url = asset["browser_download_url"]
                elif name.startswith("cudart-llama"):
                    cudart_url = asset["browser_download_url"]
        elif is_mac:
            if "bin-macos" in name or "bin-osx" in name:
                llama_url = asset["browser_download_url"]
                break
        else:
            # Linux
            if ("ubuntu-x64" in name or "linux-x64" in name or "bin-ubuntu" in name) and not name.endswith(".deb"):
                llama_url = asset["browser_download_url"]
                break

    if not llama_url:
        # Fallback search for any matching archive
        for asset in assets:
            name = asset["name"].lower()
            if is_win and "win" in name and (name.endswith(".zip") or name.endswith(".tar.gz")):
                llama_url = asset["browser_download_url"]
                break
            elif not is_win and ("ubuntu" in name or "linux" in name) and (name.endswith(".zip") or name.endswith(".tar.gz")):
                llama_url = asset["browser_download_url"]
                break

    if not llama_url:
        print("[!] Could not find appropriate binary release for platform:", sys.platform)
        return
        
    print(f"[*] Downloading binaries from: {llama_url}")
    archive_path = os.path.join(target_dir, "llama_download" + (".zip" if llama_url.endswith(".zip") else ".tar.gz"))
    urllib.request.urlretrieve(llama_url, archive_path)
    
    if cudart_url:
        cudart_path = os.path.join(target_dir, "cudart.zip")
        print(f"[*] Downloading CUDA dependencies from: {cudart_url}")
        urllib.request.urlretrieve(cudart_url, cudart_path)
        with zipfile.ZipFile(cudart_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(cudart_path)

    print("[*] Extracting binaries...")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(target_dir)

    os.remove(archive_path)

    bin_name = "llama-server.exe" if is_win else "llama-server"
    server_path = os.path.join(target_dir, bin_name)
    
    # Check inside nested subdirectories if extracted into a subfolder
    if not os.path.exists(server_path):
        for root, dirs, files in os.walk(target_dir):
            if bin_name in files:
                server_path = os.path.join(root, bin_name)
                break

    if os.path.exists(server_path):
        if not is_win:
            st = os.stat(server_path)
            os.chmod(server_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"[+] Successfully installed llama-server to {server_path}")
    else:
        print(f"[!] Error: {bin_name} not found after extraction.")

if __name__ == "__main__":
    setup_server()

import os
import importlib
import inspect
import sys
import re
import glob

def get_public_callables(module_name):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        return -1
        
    visited = set()
    callables = set()
    
    def traverse(obj, prefix):
        if id(obj) in visited:
            return
        visited.add(id(obj))
        
        try:
            for name, attr in inspect.getmembers(obj):
                if name.startswith('_'):
                    continue
                full_name = f"{prefix}.{name}"
                if callable(attr):
                    if not inspect.isclass(attr): # only count methods/functions
                        callables.add(full_name)
                    if inspect.isclass(attr) or inspect.ismodule(attr):
                        # Limit depth to avoid massive recursive loops in pandas/numpy
                        if full_name.count('.') < 2:
                            traverse(attr, full_name)
        except Exception:
            pass

    traverse(mod, module_name)
    return len(callables)

def main():
    # Compute harvests dir relative to project root (parent of scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    harvests_dir = os.environ.get("NSTL_HARVESTS_DIR", os.path.join(project_root, "harvests"))
    files = glob.glob(os.path.join(harvests_dir, "*.py"))
    
    print("=" * 60)
    print(" HARVEST COVERAGE REPORT ")
    print("=" * 60)
    print(f"{'LIBRARY':<15} | {'HARVESTED NODES':<15} | {'EST. PUBLIC API':<15} | {'COVERAGE %':<10}")
    print("-" * 60)
    
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith("harvested_"):
            lib_name = basename.replace("harvested_", "").replace(".py", "")
        elif basename.startswith("qwen_"):
            lib_name = basename.replace("qwen_", "").replace(".py", "")
        else:
            continue
            
        # Count harvested
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            # Use regex to count top-level function defs (not matches inside strings/docstrings)
            harvested_count = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
            
        # For python builtins, the module name is 'builtins'
        if lib_name == "python":
            lib_name = "builtins"
            
        est_public = get_public_callables(lib_name)
        
        if est_public > 0:
            coverage = min(100.0, (harvested_count / est_public) * 100)
            cov_str = f"{coverage:.1f}%"
        else:
            est_public = "N/A"
            cov_str = "N/A"
            
        print(f"{lib_name:<15} | {harvested_count:<15} | {est_public:<15} | {cov_str:<10}")

if __name__ == '__main__':
    main()

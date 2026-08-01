import inspect
import importlib
import sys
import json

def extract_skeleton(root_module_name, output_file):
    try:
        root_mod = importlib.import_module(root_module_name)
    except ImportError:
        raise ImportError(f"Could not import module '{root_module_name}'")
        
    visited = set()
    skeletons = {} # Deduplicate by function/method name

    def process_routine(obj, name, parent_name, is_method=False):
        # We use 'name' as the deduplication key to collapse 36 'dropna's into 1
        # But we prepend the root module to keep things scoped (e.g., pandas.dropna)
        dedup_key = f"{root_module_name}.{name}"
        
        # If we already processed this method name, just append the parent to its contexts
        if dedup_key in skeletons:
            skeletons[dedup_key]["contexts"].append(parent_name)
            return

        params = []
        try:
            sig = inspect.signature(obj)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                params.append(param_name)
        except (ValueError, TypeError):
            # C/C++ native fallback
            params = ["*args", "**kwargs"]
            
        doc = inspect.getdoc(obj) or ""
        doc_short = "\n".join(doc.split('\n')[:4]) # Keep it short for the LLM context window

        skeletons[dedup_key] = {
            "name": name,
            "contexts": [parent_name],
            "is_method": is_method,
            "params": params,
            "doc": doc_short
        }

    def walk_module(mod, parent_path):
        if mod in visited:
            return
        visited.add(mod)
        
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
                
            if inspect.ismodule(obj):
                if getattr(obj, '__name__', '').startswith(root_module_name):
                    walk_module(obj, obj.__name__)
                    
            elif inspect.isclass(obj):
                if getattr(obj, '__module__', '').startswith(root_module_name):
                    if obj not in visited:
                        visited.add(obj)
                        for method_name, method_obj in inspect.getmembers(obj):
                            if method_name.startswith("_"):
                                continue
                            if inspect.isfunction(method_obj) or inspect.ismethod(method_obj) or inspect.isroutine(method_obj):
                                process_routine(method_obj, method_name, f"{parent_path}.{name}", is_method=True)
                                
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isroutine(obj):
                process_routine(obj, name, parent_path, is_method=False)

    walk_module(root_mod, root_module_name)
    
    # Convert dict to list
    skeleton_list = list(skeletons.values())
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(skeleton_list, f, indent=2)
        
    print(f"[+] Extracted {len(skeleton_list)} unique skeletons from {root_module_name} into {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_skeleton.py <module_name> <output_file.json>")
        sys.exit(1)
        
    extract_skeleton(sys.argv[1], sys.argv[2])

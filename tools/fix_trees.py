import json
import re
import sys
from copy import deepcopy

def to_camel_case(snake_str):
    components = snake_str.split('_')
    # We capitalize the first letter of each component
    return ''.join(x.title() for x in components)

def fix_union_type(type_name):
    if not type_name:
        return type_name
    if "|" in type_name:
        # Split by | and take the first valid part
        parts = [p.strip() for p in type_name.split('|')]
        # If the first is None, take the second
        for p in parts:
            if p != "None":
                type_name = p
                break
    
    # Strip brackets like ReadCsvBuffer[bytes] -> str for simplicity?
    # Or just keep the first part. FilePath -> str
    if "FilePath" in type_name:
        return "str"
        
    # If it has brackets, just strip them for simple types
    if "[" in type_name:
        type_name = type_name.split("[")[0].strip()
        
    return type_name

VALID_TYPES = {
    'dataframe', 'series', 'ndarray', 'mat', 'image', 'tensor',
    'int', 'float', 'str', 'dict', 'list', 'graph', 'model',
    'bool', 'tuple', 'set', 'bytes', 'object', 'none', 'nonetype',
    'sparsematrix', 'sparse_matrix', 'index', 'multiindex', 'numeric',
    'flask', 'faissindex', 'faiss_index'
}

def is_self_named_type(type_name: str, cell_id: str) -> bool:
    if not type_name:
        return False
    t_lower = type_name.lower().strip()
    if t_lower in VALID_TYPES:
        return False
    clean_id = cell_id.replace('_DEFAULT', '').replace('_CELL', '')
    parts = clean_id.split('_')
    for part in parts:
        if part and len(part) > 2 and part.lower() not in (
            'pandas', 'sklearn', 'scipy', 'numpy', 'cv2', 'default', 'cell',
            'python', 'io', 'libs', 'core', 'arrays'
        ):
            if t_lower == part.lower():
                return True
    if t_lower == clean_id.lower() or t_lower in ('tags', 'resolution', 'scope', 'type', 'dtype', 'slice'):
        return True
    return False

def sanitize_type(type_name: str, cell_id: str, domain_name: str) -> str:
    if not type_name or is_self_named_type(type_name, cell_id):
        domain = (domain_name or '').lower()
        if 'cv2' in domain or 'image' in domain:
            return 'ndarray'
        elif 'pandas' in domain or 'data' in domain:
            t_lower = (type_name or '').lower()
            if 'expanding' in t_lower or 'rolling' in t_lower:
                return 'DataFrame'
            elif 'timestamp' in t_lower or 'datetime' in t_lower or 'timedelta' in t_lower:
                return 'str'
            elif 'index' in t_lower:
                return 'Index'
            else:
                return 'object'
        elif 'sklearn' in domain or 'scipy' in domain or 'numpy' in domain:
            return 'ndarray'
        return 'object'
    return type_name

def fix_node(node, domain_name):
    # Fix union types
    if "inputs" in node and "type_name" in node["inputs"]:
        node["inputs"]["type_name"] = fix_union_type(node["inputs"]["type_name"])
    if "outputs" in node and "type_name" in node["outputs"]:
        node["outputs"]["type_name"] = fix_union_type(node["outputs"]["type_name"])
        
    cell_id = node.get("cell_id", "")
    if "inputs" in node and "type_name" in node["inputs"]:
        node["inputs"]["type_name"] = sanitize_type(node["inputs"]["type_name"], cell_id, domain_name)
    if "outputs" in node and "type_name" in node["outputs"]:
        node["outputs"]["type_name"] = sanitize_type(node["outputs"]["type_name"], cell_id, domain_name)

    parts = cell_id.split('_')
    
    # 1. Infer class from cell_id if input is "any" and it's an instance method
    inferred_class = None
    if len(parts) >= 3:
        # Usually MODULE_CLASS_METHOD
        module_part = parts[0]
        class_part = parts[1]
        
        # Heuristics for common classes
        if class_part == "DATAFRAME":
            inferred_class = "DataFrame"
        elif class_part == "SERIES":
            inferred_class = "Series"
        elif class_part == "NDARRAY":
            inferred_class = "ndarray"
        else:
            inferred_class = to_camel_case(class_part.lower())
            
        # Fix inputs type_name if "any"
        if node.get("inputs", {}).get("type_name") == "any":
            node["inputs"]["type_name"] = inferred_class
            
    # Attempt to correct the casing of inferred_class using keywords
    if inferred_class:
        for kw in node.get("keywords", []):
            if "." in kw:
                cls_part = kw.split(".")[-1] # e.g. "pandas.ArrowDtype" -> "ArrowDtype"
                if cls_part.lower() == inferred_class.lower():
                    inferred_class = cls_part
                    if node.get("inputs", {}).get("type_name", "").lower() == inferred_class.lower():
                        node["inputs"]["type_name"] = inferred_class
                    break

    # 2. Fix state names
    if "inputs" in node and node["inputs"].get("state") == "self":
        if inferred_class:
            node["inputs"]["state"] = f"source_{inferred_class.lower()}"
        else:
            node["inputs"]["state"] = "source_object"
            
    if "outputs" in node and node["outputs"].get("state") == "computed":
        out_type = node["outputs"].get("type_name", "object")
        node["outputs"]["state"] = f"result_{out_type.lower()}"
        
    # 3. Dependencies
    impl = node.get("domain_implementations", {}).get("Python_Core", {})
    code = impl.get("code", "")
    deps = impl.get("dependencies", [])
    if deps is None:
        deps = []
    
    # Always add domain_name if missing, unless it's builtins/python
    base_domain = domain_name.lower().replace("_auto", "").replace("_seed", "").replace("_domain", "")
    if base_domain not in ["python", "builtins"]:
        if base_domain not in deps:
            deps.append(base_domain)
            
    # Check if code uses other modules
    for mod in ["pandas", "numpy", "cv2", "math", "matplotlib", "scipy", "sklearn", "os", "sys"]:
        if f"{mod}." in code and mod not in deps:
            deps.append(mod)
            
    impl["dependencies"] = deps

    # 4. Fix static method calls vs instance method calls
    # If code is `{output_var} = {input_var}.method()` but the input_var is NOT the class itself,
    # it's likely a static/class method that was incorrectly scraped.
    if "{input_var}." in code:
        method_call_match = re.search(r"\{input_var\}\.([a-zA-Z0-9_]+)\(", code)
        if method_call_match:
            method_name = method_call_match.group(1)
            # If the inferred class is set, and the input type is NOT the class
            inp_type = node.get("inputs", {}).get("type_name", "")
            if inferred_class and inp_type.lower() != inferred_class.lower() and inp_type != "any":
                # It's a static method! Rewrite code.
                # Example: pandas.ArrowDtype.construct_from_string({input_var})
                new_code = code.replace(f"{{input_var}}.{method_name}(", f"{base_domain}.{inferred_class}.{method_name}({{input_var}}")
                impl["code"] = new_code
                
    return node

def main():
    if len(sys.argv) < 3:
        print("Usage: python fix_trees.py <input.json> <output.json> [subset_count]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    subset_count = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    domain_name = data.get("domain_name", "unknown")
    cells = data.get("cells", [])
    
    if subset_count:
        cells = cells[:subset_count]
        
    fixed_cells = []
    for cell in cells:
        fixed_cells.append(fix_node(deepcopy(cell), domain_name))
        
    data["cells"] = fixed_cells
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Fixed {len(fixed_cells)} nodes and saved to {output_file}")

if __name__ == "__main__":
    main()

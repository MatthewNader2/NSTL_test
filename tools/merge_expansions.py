import os
import glob

def merge_expansions(harvests_dir):
    expansion_files = glob.glob(os.path.join(harvests_dir, "qwen_EXPANSION_*.py"))
    
    for exp_file in expansion_files:
        basename = os.path.basename(exp_file)
        lib_name = basename.replace("qwen_EXPANSION_", "").replace(".py", "")
        
        # Determine the target main file
        if lib_name == "cv2" or lib_name == "builtins":
            target_file = os.path.join(harvests_dir, f"qwen_{lib_name}.py")
        else:
            target_file = os.path.join(harvests_dir, f"harvested_{lib_name}.py")
            
        if not os.path.exists(target_file):
            # python was harvested as harvested_python.py but expansion is builtins? 
            if lib_name == "python": target_file = os.path.join(harvests_dir, "harvested_python.py")
            print(f"Target not found for {lib_name}, skipping.")
            continue
            
        with open(exp_file, 'r', encoding='utf-8') as f:
            expansion_content = f.read()
            
        with open(target_file, 'a', encoding='utf-8') as f:
            f.write("\n\n" + expansion_content)
            
        print(f"[+] Merged {basename} into {os.path.basename(target_file)}")

if __name__ == "__main__":
    merge_expansions(r"d:\grad_test\nstl_prototype\harvests")

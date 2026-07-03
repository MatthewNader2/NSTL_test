import os

# Directories and files to exclude from the dump
IGNORE_DIRS = {
    '.git', '__pycache__', 'build', 'dist', 
    'frontend_dist', 'model_cache', 'node_modules', '.venv', 'trees', 'frontend', 'android_app'
}
IGNORE_EXTS = {
    '.png', '.ico', '.log', '.spec', '.pdf', '.pyc', '.exe'
}
OUTPUT_FILE = 'nstl_codebase_dump.txt'

def bundle_codebase():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            # Modify the dirs list in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in IGNORE_EXTS or file == OUTPUT_FILE or file == 'bundle_code.py':
                    continue

                filepath = os.path.join(root, file)
                
                # Write a clear header for each file
                outfile.write(f"\n{'='*60}\n")
                outfile.write(f"FILE: {filepath}\n")
                outfile.write(f"{'='*60}\n\n")

                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read() + "\n")
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")

    print(f"Success! Codebase dumped to {OUTPUT_FILE}")

if __name__ == "__main__":
    bundle_codebase()
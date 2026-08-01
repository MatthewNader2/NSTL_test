import os
os.chdir('/content')
print('[*] Extracting skeletons for numpy...')
os.system('python extract_skeleton.py numpy skeleton_numpy.json')
print('[*] Harvesting LLM AST parameters for numpy (This may take a while)...')
# We do not use --limit so it runs on all nodes for the library
os.system('python cloud_harvester.py skeleton_numpy.json qwen_numpy.py')

import sys
import os
import subprocess
os.chdir('/content')
subprocess.run([sys.executable, 'cloud_harvester.py', 'skeleton_cv2.json', 'qwen_cv2_5.py', '--limit', '5'])

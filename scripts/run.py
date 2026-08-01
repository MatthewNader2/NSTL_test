import os
os.chdir('/content')
os.system('python cloud_harvester.py skeleton_cv2.json qwen_cv2_5.py --limit 5 > out.txt 2>&1')

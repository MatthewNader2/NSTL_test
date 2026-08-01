import os
os.chdir('/content')
print('Starting Batch 0')
os.system('python cloud_harvester.py skeleton_cv2.json qwen_cv2_0.py --batch-start 0 --batch-size 50')

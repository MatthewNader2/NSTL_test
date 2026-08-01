import subprocess
print(subprocess.getoutput('ps aux | grep -E "python|cc1|nvcc|gcc|sh"'))

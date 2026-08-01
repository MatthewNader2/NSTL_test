import urllib.request
import os
from PIL import Image

url = "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/So11111111111111111111111111111111111111112/logo.png"
png_path = "app_icon.png"
ico_path = "app_icon.ico"

try:
    print("Downloading icon from:", url)
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(png_path, 'wb') as f:
            f.write(response.read())

    print("Converting to ICO...")
    img = Image.open(png_path)
    img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("Icon download and conversion complete successfully!")
except Exception as e:
    print("Error downloading/converting icon:", str(e))

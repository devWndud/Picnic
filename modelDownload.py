
import os
import gdown

file_id = "1BWQ0_sg862W9EOYakEpo3WMiVQs3sq5w-QfpLM8-_MIv"
url = f"https://drive.google.com/uc?id={file_id}"
output = "Data001"
if not os.path.exists(output):
    print("yolov8x.pt not found, downloading from Google Drive...")
    gdown.download(url, output, quiet=False)
else:
    print("yolov8x.pt already exists.")
    
import kagglehub

# Download latest version
path = kagglehub.dataset_download("liewyousheng/geolocation")

print("Path to dataset files:", path)

import os
# Copy to alignment/dataset
dataset_path = "alignment/dataset"
os.makedirs(dataset_path, exist_ok=True)
for filename in os.listdir(path):
    src_file = os.path.join(path, filename)
    dst_file = os.path.join(dataset_path, filename)
    if os.path.isfile(src_file):
        with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
            fdst.write(fsrc.read())
        print(f"Copied {filename} to {dataset_path}")
    else:
        print(f"Skipping non-file: {filename}")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import msgpack
import glob

from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt

dataset_dir = "data/data_0/"
shard_fnames = [dataset_dir + f"shard_{i}.msg" for i in range(0,1)]



def get_image(record):
        return Image.open(BytesIO(record["image"]))


for shard_fname in shard_fnames[0:]:
    print(f"Processing {shard_fname.split('/')[-1]}")
    with open(shard_fname, "rb") as infile:
        for record in msgpack.Unpacker(infile, raw=False):
            print(f'Image_id={record["id"]}, lat={record["latitude"]}, lon={record["longitude"]}')
            image = get_image(record)
            break
    break
    
# plt.imshow(image)
plt.savefig("results/image.png")


# Nunber of photos to extract 
n = 10000
coords_img_file = []
if not os.path.exists("results/images/"):
    os.makedirs("results/images/")

for shard_fname in shard_fnames[0:1]:
    #print(f"Processing {shard_fname.split('/')[-1]}")
    with open(shard_fname, "rb") as infile:
        for i, record in enumerate(msgpack.Unpacker(infile, raw=False)):
            if i >= n:
                break
            coords_img_file.append((record["latitude"], record["longitude"], f"images/image_{i}.png"))
            image = get_image(record)
            image.save(f"data/data_0/images/image_{i}.png")


# Save CSV file with coordinates and image paths
import pandas as pd
img_file_col = [f"images/image_{i}.png" for i in range(n)]
df = pd.DataFrame(coords_img_file, columns=["LAT", "LON", "IMG_FILE"])
# df["IMG_FILE"] = [f"results/images/image_{i}.png" for i in range(n)]
df.to_csv("data/data_0/shard_0.csv", index=False)

"""
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(x=[y[1] for y in coords], y=[y[0] for y in coords] ,marker=".", alpha=0.1)
plt.savefig("results/coords.png")
"""


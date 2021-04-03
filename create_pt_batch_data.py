from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np 
import random
import os, sys
import pickle
import lmdb
from tqdm import tqdm
import io

'''
This file is to create a .pt data file from a dataset composed of images (like .JPEG, .PNG, etc.).
Usage: 
    python <this_file> <path_to_image_dir>
Here is an example to process ImageNet:
    
'''

# --------------------------------
img_dir = sys.argv[1] # image dir, in which there should be many folders. Each folder includes the .JPEG images of the same class.
lmdb_path = f'{img_dir}/lmdb'
txt_file = open(f'{img_dir}/lmdb_meta_info.txt', 'w')
batch = 1000 # every <batch> images, lmdb commits once.
# --------------------------------

# get the mapping from text name to label
dataset = ImageFolder(img_dir)
class_to_idx = dataset.class_to_idx

# get all image paths
image_paths, labels = [], []
for root, dirs, files in os.walk(img_dir):
    for f in files:
        if f.endswith(".JPEG"):
            f = os.path.join(root, f)
            image_paths.append(f)
            label = class_to_idx[root.split("/")[1]]
            labels.append(label)
print(f'Got {len(image_paths)} images in total')

# get map_size, which is needed by lmdb
img_byte = io.BytesIO()
img_pil = Image.open(image_paths[0]).convert('RGB')
img_pil.save(img_byte, format='PNG')
data_size_per_img = img_byte.tell()
print(f'data_size_per_img: {data_size_per_img} bytes')
data_size = data_size_per_img * len(image_paths)
map_size = data_size * 10

# create lmdb env
env = lmdb.open(lmdb_path, map_size=map_size)

# write data to lmdb
pbar = tqdm(total=len(image_paths))
txn = env.begin(write=True)
for idx, (path, label) in enumerate(zip(image_paths, labels)):
    
    pbar.update(1)
    key = path # lmdb key
    pbar.set_description(f'Write {key}')
    key_byte = key.encode('ascii')
    
    # read image
    img = Image.open(path).convert('RGB')
    h, w = img.size

    # push to lmdb
    value_byte = pickle.dumps([img, label])
    txn.put(key_byte, value_byte)
    if idx % batch == 0:
        txn.commit()
        txn = env.begin(write=True)

    # write meta information
    txt_file.write(f'{idx}: {path} ({h},{w},3) {label}\n')

txn.commit()
env.close()
txt_file.close()
print('\nFinish writing lmdb.')
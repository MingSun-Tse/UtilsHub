import os
import numpy as np, random
import sys
import math


def get_relative_path(x):
    r"""Given an absolute path, get its relative path.
    x example: /home3/wanghuan/Projects/Pruning/data/imagenet/train/n03291819
    """
    cwd = os.path.abspath(os.getcwd()) # E.g., /home3/wanghuan/Projects/Pruning/data/imagenet_subset_200_RandSeed20220521_0/train
    cwd_split = cwd.split(os.sep)
    x_split = x.split(os.sep)
    cnt = min(len(cwd_split), len(x_split))
    for i in range(cnt):
        if cwd_split[i] != x_split[i]:
            break
    prefix = f'..{os.sep}' * len(cwd_split[i:])
    path = os.path.join(prefix, os.sep.join(x_split[i:]))
    return path

"""Usage:
python  prepare_imagenet_subset_200.py  ../data/imagenet 200  ../data/imagenet_subset_200  2  20220521
"""
# ------------------ args
data_dir = sys.argv[1] # Path of the ImageNet dataset folder
n_subset_class = int(sys.argv[2])
out_dir = sys.argv[3]
how_many = int(sys.argv[4]) # Prepare how many sets of subset images
rand_seed = int(sys.argv[5]) # To exact reproduce
# ------------------

# Get all the folders of ImageNet
train_folder, val_folder = 'train', 'val'
data_dir_train = f'{data_dir}/{train_folder}'
data_dir_val = f'{data_dir}/{val_folder}'
folders_train = [os.path.abspath(os.path.join(data_dir_train, x)) for x in os.listdir(data_dir_train)]

# Fix rand seed
random.seed(rand_seed)
np.random.seed(rand_seed)
os.environ['PYTHONHASHSEED'] = str(rand_seed)

# Randomly pick <n_subset_class> out of <n_class>
n_class = len(folders_train)
rand_index = np.random.permutation(n_class)
print(rand_index[:100]) # Check
original_cwd = os.getcwd()

for ix in range(how_many):
    os.chdir(original_cwd)
    # Make folders
    out_dir_ = f'{out_dir}_RandSeed{rand_seed}_{ix}'
    os.makedirs(f'{out_dir_}/{train_folder}', exist_ok=True)
    os.makedirs(f'{out_dir_}/{val_folder}', exist_ok=True)

    begin, end = ix * n_subset_class, ix * n_subset_class + n_subset_class
    picked_folders_train = [folders_train[i] for i in rand_index[begin: end]]

    # Create soft links -- train
    cnt = 0
    os.chdir(f'{out_dir_}/{train_folder}')
    for f in picked_folders_train:
        cnt += 1
        f_name = f.split('/')[-1] # E.g., n04200800
        f_train = get_relative_path(f) # Change to relative path, so that the links can be platform-invariant
        os.symlink(f_train, f_name)
        print('[%d/%d] Creating soft link: %s -> %s' % (cnt, n_subset_class, f_train, f_name))

    # Create soft links -- val
    cnt = 0
    os.chdir(f'../{val_folder}')
    for f in picked_folders_train:
        cnt += 1
        f_train = get_relative_path(f)
        f_name = f.split('/')[-1] # E.g., n04200800
        f_val = f_train.replace(f'/{train_folder}/', f'/{val_folder}/')
        os.symlink(f_val, f_name)
        print('[%d/%d] Creating soft link: %s -> %s' % (cnt, n_subset_class, f_val, f_name))
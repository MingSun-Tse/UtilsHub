import os
import numpy as np 
import sys
import math

'''Usage:
python prepare_imagenet_subset_200.py ../data/imagenet 200 ../data/imagenet_subset_200
'''
data_dir = sys.argv[1] # path of the ImageNet dataset folder
data_dir_train = '%s/train' % data_dir
data_dir_val = '%s/val' % data_dir
n_subset_class = int(sys.argv[2])
out_dir = sys.argv[3]

os.makedirs('%s/train' % out_dir, exist_ok=True)
os.makedirs('%s/val' % out_dir, exist_ok=True)

# get all the folders of ImageNet
folders_train = [os.path.abspath(os.path.join(data_dir_train, x)) for x in os.listdir(data_dir_train)]

# randomly pick <n_subset_class> out of <n_class>
n_class = len(folders_train)
rand_index = np.random.permutation(n_class)
picked_folders_train = [folders_train[i] for i in rand_index[:n_subset_class]]

# create soft links
cnt = 0
for f in picked_folders_train:
    cnt += 1
    f_name = f.split('/')[-1]

    # train
    out_f = '%s/train/%s' % (out_dir, f_name)
    os.symlink(f, out_f)
    print('[%d/%d] creating soft link: %s -> %s' % (cnt, n_subset_class, f, out_f))

    # val
    f_val = f.replace('/train/', '/val/')
    out_f_val = '%s/val/%s' % (out_dir, f_name)
    os.symlink(f_val, out_f_val)
    print('[%d/%d] creating soft link: %s -> %s' % (cnt, n_subset_class, f_val, out_f_val))


    

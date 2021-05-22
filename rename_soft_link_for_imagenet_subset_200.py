import sys
import os
'''Usage:
    python <path_to_this_file>  <path_to_imagenet_subset_200> <path_to_imagenet>

Example:
    
'''
inDir_subset = sys.argv[1] # imagenet subset 200 folder
inDir_full = sys.argv[2] # imagenet folder

train_path = '%s/train' % inDir_subset
val_path = '%s/val' % inDir_subset

for link in os.listdir(train_path):
    os.unlink('%s/%s' % (train_path, link))
    script = 'ln -s %s/train/%s %s/' % (inDir_full, link, train_path)
    os.system(script)

for link in os.listdir(val_path):
    os.unlink('%s/%s' % (val_path, link))
    script = 'ln -s %s/val/%s %s/' % (inDir_full, link, val_path)
    os.system(script)
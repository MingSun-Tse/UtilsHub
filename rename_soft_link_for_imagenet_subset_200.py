import sys
import os
'''Usage:
In the 'tools' dir, run "python <this_file>  ../data/imagenet_subset_200".
Note: the 'data' dir should have 'imagenet' already set up.
'''
inDir = sys.argv[1] # imagenet subset 200 folder
train_path = '%s/train' % inDir
val_path = '%s/val' % inDir

for link in os.listdir(train_path):
    os.unlink('%s/%s' % (train_path, link))
    script = 'ln -s ../../imagenet/train/%s %s/' % (link, train_path)
    os.system(script)

for link in os.listdir(val_path):
    os.unlink('%s/%s' % (val_path, link))
    script = 'ln -s ../../imagenet/val/%s %s/' % (link, val_path)
    os.system(script)
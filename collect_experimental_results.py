import sys
import os
import numpy as np
import glob
pjoin = os.path.join

def _get_value(line, key, type_func=float):
    line_seg = line.split()
    for i in range(len(line_seg)):
        if key in line_seg[i]:
            break
    if i == len(line_seg) - 1:
        return None # did not find the <key> in this line
    value = type_func(line_seg[i + 1]) # example: "Acc: 0.7"
    return value

def _get_exp_id(exp_path):
    '''arg example: kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
    '''
    assert 'SERVER' in exp_path # safety check
    exp_id = exp_path.split('-')[-1]
    assert exp_id.isdigit() # safety check
    return exp_id

def _get_project_name():
    cwd = os.getcwd()
    assert '/Projects/' in cwd
    return cwd.split('/')[-1] 

def _make_acc_str(acc_list, num_digit=2):
    '''Example the output: 75.84, 75.63, 75.45 -- 75.64 (0.16)
    '''
    str_format = '%.{}f'.format(num_digit)
    acc_str = [str_format % x for x in acc_list]
    mean = str_format % np.mean(acc_list)
    std = str_format % np.std(acc_list)
    output = ', '.join(acc_str) + ' -- %s (%s)' % (mean, std)
    return output

'''Usage:
    In the project dir, run:
    python ../UtilsHub/collect_experimental_results.py vid-wrn402wrn162-cifar100-HEVGGAugment_
'''
exp_kw = sys.argv[1]
exp_id = []
acc_last = []
acc_best = []

exps = ['Experiments/%s' % x for x in os.listdir('Experiments') if exp_kw in x]
exps.sort()
for exp in exps:
    log_f = '%s/log/log.txt' % exp
    for line in open(log_f, 'r'):
        if 'Epoch 240 (after update)' in line: # parsing accuracy
            acc_l = _get_value(line, 'Acc1')
            acc_b = _get_value(line, 'Best_Acc1')
            break
    acc_last.append(acc_l)
    acc_best.append(acc_b)
    exp_id.append(_get_exp_id(exp))

# example for the exp_str and acc_last_str
# 138-CRD-174550, 174554, 174558:
# 75.84, 75.63, 75.45 – 75.64 (0.16)
exp_str = '%s-%s-' % (os.environ['SERVER'], _get_project_name()) + ', '.join(exp_id) + ':'
acc_last_str = _make_acc_str(acc_last)
acc_best_str = _make_acc_str(acc_best)
print(exp_str)
print(acc_last_str)
print(acc_best_str)

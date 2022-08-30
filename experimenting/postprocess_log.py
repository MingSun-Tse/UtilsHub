import sys, os, time
import numpy as np
from scipy import stats
import glob, copy
import argparse
from accuracy_analyzer import AccuracyAnalyzer
import matplotlib.pyplot as plt
pjoin = os.path.join


def _get_value(line, key, type_func=float, exact_key=False, raw=False):
    if raw: 
        type_func = str
    if exact_key: # back compatibility
        value = line.split(key)[1].strip().split()[0]
        if value.endswith(')'): # hand-fix case: "Epoch 23)"
            value = value[:-1] 
        if value.endswith('%'):
            value = type_func(value[:-1]) / 100.
        else:
            value = type_func(value)
    else:
        line_seg = line.split()
        for i in range(len(line_seg)):
            if key in line_seg[i]: # example: 'Acc1: 0.7'
                break
        if i == len(line_seg) - 1:
            return None # did not find the <key> in this line
        value = type_func(line_seg[i + 1])
    return value


def _get_exp_name_id(exp_path):
    r"""arg example: Experiments/kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
            or kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
    """
    exp_path = exp_path.strip('/')
    assert 'SERVER' in exp_path # safety check
    exp_id = exp_path.split('-')[-1]
    assert exp_id.isdigit() # safety check
    exp_name = os.path.split(exp_path)[-1].split('_SERVER')[0]
    date = exp_path.split('-')[-2]
    assert date.isdigit() # safety check
    return exp_name, exp_id, date


def _get_project_name():
    cwd = os.getcwd()
    # assert '/Projects/' in cwd
    return cwd.split('/')[-1] 

# acc line example: Acc1 71.1200 Acc5 90.3800 Epoch 840 (after update) lr 5.0000000000000016e-05 (Best_Acc1 71.3500 @ Epoch 817)
# acc line example: Acc1 0.9195 @ Step 46600 (Best = 0.9208 @ Step 38200) lr 0.0001
# acc line example: ==> test acc = 0.7156 @ step 80000 (best = 0.7240 @ step 21300)
def is_acc_line(line, mark=''):
    """This function determines if a line is an accuracy line. Of course the accuracy line should meet some 
    format features which @mst used. So if these format features are changed, this func may not work.
    """
    if mark:
        return mark in line
    else:
        line = line.lower()
        return "acc" in line and "best" in line and '@' in line and 'lr' in line and 'resume' not in line and 'finetune' not in line


def parse_metric(line, metric='Acc1', raw=False):
    r"""Parse out the metric value of interest.
    """
    # Get the last metric
    if f'{metric} =' in line: # previous impl.
        acc_l = _get_value(line, f'{metric} =', exact_key=True, raw=raw)
    elif 'test acc = ' in line: # previous impl.
        acc_l = _get_value(line, 'test acc =', exact_key=True, raw=raw)
    else:
        acc_l = _get_value(line, f'{metric}', exact_key=True, raw=raw)

    # Get the best metric
    if f'Best {metric}' in line: # previous impl.
        acc_b = _get_value(line, f'Best {metric}', exact_key=True, raw=raw)
    elif f'Best_{metric}' in line:
        acc_b = _get_value(line, f'Best_{metric}', exact_key=True, raw=raw)
    else:
        acc_b = -1 # Not found the best metric value (not written in log)
    
    if raw:
        return acc_l, acc_b
    else:
        return acc_l * args.scale, acc_b * args.scale

def print_acc_for_one_exp_group(all_exps, name):
    """In <all_exps>, pick those with <name> in their name for accuracy collection.
    <name> is to locate which experiments
    """
    name = f'{args.exps_folder}/{name}_SERVER'
    for exp in all_exps:
        if name in exp:
            log_f = '%s/log/log.txt' % exp
            lines = open(log_f).readlines()
            all_metric = []
            with open(log_f, 'w') as f:
                for line in lines:
                    if is_acc_line(line, mark=args.accline_mark):
                        metric, _ = parse_metric(line, args.metric, raw=True)
                        n_decimals = len(metric.split('.')[1])
                        all_metric += [float(metric)]
                        avg = np.mean(all_metric[-args.avg:])
                        line = line.strip() + f' AvgLast{args.avg}{args.metric} {avg:.{n_decimals}f}' + '\n'
                    f.write(line)
            print(f'Processed file: {log_f}')

parser = argparse.ArgumentParser()
parser.add_argument('--kw', type=str, required=True, help='keyword for filtering expriment folders')
parser.add_argument('--exact_kw', action='store_true', help='if true, not filter by exp_name but exactly the kw')
parser.add_argument('--accline_mark', type=str, default='')
parser.add_argument('--metric', type=str, default='Acc1')
parser.add_argument('--ignore', type=str, default='')
parser.add_argument('--exps_folder', type=str, default='Experiments')
parser.add_argument('--scale', type=float, default=1.)
parser.add_argument('--avg', type=int, default=5)
args = parser.parse_args()

def main():
    """Usage:
        In the project dir, run:
        python ../UtilsHub/experimenting/collect_experimental_results.py --kw 20200731-18
    """
    # 1st filtering: get all the exps with the keyword
    all_exps_ = glob.glob(f'{args.exps_folder}/*{args.kw}*')
    
    # 2nd filtering: remove all exps in args.ignore
    if args.ignore:
        all_exps_2 = []
        ignores = args.ignore.split(',')
        for exp in all_exps_:
            if not any([x in exp for x in ignores]):
                all_exps_2 += [exp]
        all_exps_ = all_exps_2

    # 3rd filtering: add all the exps with the same name, even it is not included by the 1st filtering by kw
    if args.exact_kw:
        all_exps = all_exps_
    else:
        all_exps = []
        for exp in all_exps_:
            name, *_ = _get_exp_name_id(exp)
            all_exps_with_the_same_name = glob.glob(f'{args.exps_folder}/{name}_SERVER*')
            for x in all_exps_with_the_same_name:
                if x not in all_exps:
                    all_exps.append(x)
    all_exps.sort()
    
    # Get group exps, because each group is made up of multiple times.
    exp_groups = []
    for exp in all_exps:
        name, *_ = _get_exp_name_id(exp)
        if name not in exp_groups:
            exp_groups.append(name)
    
    # Analyze each independent exp (with multi-runs)
    for exp_name in exp_groups:
        print('[%s]' % exp_name)
        print_acc_for_one_exp_group(all_exps, exp_name)
        print('')

if __name__ == '__main__':
    main()
import sys
import os
import numpy as np
import glob
pjoin = os.path.join

def _get_value(line, key, type_func=float, exact_key=False):
    if exact_key: # back compatibility
        value = line.split(key)[1].strip().split()[0]
        value = type_func(value)
    else:
        line_seg = line.split()
        for i in range(len(line_seg)):
            if line_seg[i].startswith(key): # example: 'Acc1: 0.7'
                break
        if i == len(line_seg) - 1:
            return None # did not find the <key> in this line
        value = type_func(line_seg[i + 1])
    return value

def _get_exp_name_id(exp_path):
    '''arg example: Experiments/kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
            or kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
    '''
    exp_path = exp_path.strip('/')
    assert 'SERVER' in exp_path # safety check
    exp_id = exp_path.split('-')[-1]
    assert exp_id.isdigit() # safety check
    exp_name = os.path.split(exp_path)[-1].split('_SERVER')[0]
    return exp_name, exp_id

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

def print_acc_for_one_exp(all_exps, name, epoch=240):
    '''In <all_exps>, pick those with <name> in their name for accuracy collection.
    '''
    exp_id = []
    acc_last = []
    acc_best = []

    for exp in all_exps:
        if name in exp:
            log_f = '%s/log/log.txt' % exp
            acc_l = -1
            acc_b = -1
            for line in open(log_f, 'r'):
                if 'Epoch %d (after update)' % epoch in line: # parsing accuracy
                    if 'Acc1 =' in line: # previous impel
                        acc_l = _get_value(line, 'Acc1 =', exact_key=True)
                    else:
                        acc_l = _get_value(line, 'Acc1')
                    if 'Best Acc1' in line: # previous impel
                        acc_b = _get_value(line, 'Best Acc1', exact_key=True)
                    elif 'Best_Acc1' in line:
                        acc_b = _get_value(line, 'Best_Acc1')
                    else:
                        raise NotImplementedError
                    break
            if acc_b == -1:
                print('This log is broken, so skip it')
                continue
            acc_last.append(acc_l)
            acc_best.append(acc_b)
            _, id = _get_exp_name_id(exp)
            exp_id.append(id)

    # example for the exp_str and acc_last_str:
    # 174550, 174554, 174558 (138-CRD)
    # 75.84, 75.63, 75.45 – 75.64 (0.16)
    exp_str = ', '.join(exp_id) + ' (%s-%s)' % (os.environ['SERVER'], _get_project_name())
    acc_last_str = _make_acc_str(acc_last)
    acc_best_str = _make_acc_str(acc_best)
    print(exp_str)
    print(acc_last_str)
    print(acc_best_str)

def main():
    '''Usage:
        In the project dir, run:
        python ../UtilsHub/collect_experimental_results.py 20200731-18
    '''
    kw = sys.argv[1]
    epoch = int(sys.argv[2])
    all_exps = glob.glob('Experiments/*%s*' % kw)
    all_exps.sort()
    
    # get independent exps, because each independent exp is run for multiple times.
    independent_exps = []
    for exp in all_exps:
        name, _ = _get_exp_name_id(exp)
        if name not in independent_exps:
            independent_exps.append(name)
    
    # analyze each independent exp
    for name in independent_exps:
        print('[%s]' % name)
        print_acc_for_one_exp(all_exps, name + '_SERVER', epoch)
        print('')

if __name__ == '__main__':
    main()

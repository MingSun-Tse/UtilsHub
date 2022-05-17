import sys, os, time
import numpy as np
from scipy import stats
import glob, copy
import argparse
from accuracy_analyzer import AccuracyAnalyzer
import matplotlib.pyplot as plt
pjoin = os.path.join


def _get_value(line, key, type_func=float, exact_key=False):
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


def _make_acc_str(acc_list, num_digit=2, present=False):
    """Example the output: 75.84, 75.63, 75.45 -- 75.64 (0.16)
    """
    str_format = '%.{}f'.format(num_digit)
    acc_str = [str_format % x for x in acc_list]
    mean = str_format % np.mean(acc_list) if len(acc_list) else 0
    std = str_format % np.std(acc_list) if len(acc_list) else 0
    if present:
        output = f'{mean}$_{{\pm{std}}}$' # Only print mean and std: 75.64 (0.16)
    else:
        output = ', '.join(acc_str) + f' -- {mean}$_{{\pm{std}}}$' # Print the result of every experiment: 75.84, 75.63, 75.45 -- 75.64 (0.16)
    return output


def _make_acc_str_one_exp(acc_last, acc_best, num_digit):
    str_format = '%.{}f'.format(num_digit)
    output = '%s/%s' % (str_format % acc_last, str_format % acc_best)
    return output

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


def parse_metric(line, metric='Acc1'):
    r"""Parse out the metric value of interest.
    """
    # Get the last metric
    if f'{metric} =' in line: # previous impl.
        acc_l = _get_value(line, f'{metric} =', exact_key=True)
    elif 'test acc = ' in line: # previous impl.
        acc_l = _get_value(line, 'test acc =', exact_key=True)
    else:
        acc_l = _get_value(line, f'{metric}', exact_key=True)

    # Get the best metric
    if f'Best {metric}' in line: # previous impl.
        acc_b = _get_value(line, f'Best {metric}', exact_key=True)
    elif f'Best_{metric}' in line:
        acc_b = _get_value(line, f'Best_{metric}', exact_key=True)
    else:
        acc_b = -1 # Not found the best metric value (not written in log)
    return acc_l * args.scale, acc_b * args.scale


def parse_time(line): # TODO
    if 'Epoch' in line:
        epoch = line.split('Epoch')[1].split()[0]
        time = int(epoch)
    elif 'Step' in line:
        time = _get_value(line, 'Step', type_func=int, exact_key=True)
    elif 'step' in line:
        time = _get_value(line, 'step', type_func=int, exact_key=True)
    else:
        raise NotImplementedError
    return time


def parse_finish_time(log_f):
    lines = open(log_f, 'r').readlines()
    for k in range(1, min(1000, len(lines))):
        if 'predicted finish time' in lines[-k].lower():
            finish_time = lines[-k].split('time:')[1].split('(')[0].strip() # example: predicted finish time: 2020/10/25-08:21 (speed: 314.98s per timing)
            return finish_time


def remove_outlier(metric, *lists):
    metric = copy.deepcopy(metric)
    mean, std = np.mean(metric), np.std(metric)
    for ix in range(len(metric)-1, -1, -1):
        if abs(metric[ix] - mean) > args.outlier_thresh: # variation larger than std, remove it.
            for l in lists:
                l.pop(ix)


def print_acc_for_one_exp_group(all_exps, name, mark, present_data):
    """In <all_exps>, pick those with <name> in their name for accuracy collection.
    <name> is to locate which experiments; <mark> is to locate the accuracy line in a log.
    """
    exp_id, date = [], []
    acc_last, acc_best, acc_time, finish_time = [], [], [], []
    name = f'{args.exps_folder}/{name}_SERVER'

    # for loss, acc correlation analysis
    acc1_test_just_finished_prune = []
    loss_test_just_finished_prune = []
    acc1_train_just_finished_prune = []
    loss_train_just_finished_prune = []
    acc1_test_after_ft = []
    loss_test_after_ft = []
    acc1_train_after_ft = []
    loss_train_after_ft = []

    for exp in all_exps:
        if name in exp:
            log_f = '%s/log/log.txt' % exp
            acc_l, acc_b = -1, -1 # acc last, acc best
            
            try:
                if mark == 'last': # the last number shown in the log
                    lines = open(log_f, 'r').readlines()
                    for k in range(1, len(lines) + 1):
                        if is_acc_line(lines[-k], mark=args.accline_mark):
                            acc_l, acc_b = parse_metric(lines[-k], args.metric)
                            acc_time_ = parse_time(lines[-k])
                            break
                
                else: # mark is like "Epoch 240 (", which explicitly points out which epoch or step
                    for line in open(log_f, 'r'):
                        if is_acc_line(line, mark=args.accline_mark) and mark in line:
                            acc_time_ = parse_time(line)
                            acc_l, acc_b = parse_metric(line, args.metric)
                            break

                if acc_l == -1 and acc_b == -1:
                    print('Not found mark "%s" in the log "%s", skip it' % (mark, log_f))
                    continue
            
                # parse, for loss, acc correlation analysis
                if args.corr_analysis:
                    for line in open(log_f, 'r'):
                        if 'Just got pruned model' in line and 'Loss_train' in line:
                            acc1_test = _get_value(line, 'Acc1', exact_key=True)
                            loss_test = _get_value(line, 'Loss_test', exact_key=True)
                            acc1_train = _get_value(line, 'Acc1_train', exact_key=True)
                            loss_train = _get_value(line, 'Loss_train', exact_key=True)
                            acc1_test_just_finished_prune.append(acc1_test)
                            loss_test_just_finished_prune.append(loss_test)
                            acc1_train_just_finished_prune.append(acc1_train)
                            loss_train_just_finished_prune.append(loss_train)
                            break
                
                # get statistics for best model
                if args.corr_analysis:
                    for line in open(log_f, 'r'):
                        if is_acc_line(line, mark=args.accline_mark):
                            best_epoch = _get_value(line, 'Best_Acc1_Epoch', exact_key=True, type_func=int)
                            current_epoch = _get_value(line, 'Epoch', exact_key=True, type_func=int)
                            if current_epoch == best_epoch:
                                loss_train = _get_value(line, 'Loss_train', exact_key=True)
                                acc1_train = _get_value(line, 'Acc1_train', exact_key=True)
                                loss_test = _get_value(line, 'Loss_test', exact_key=True)
                    loss_train_after_ft.append(loss_train)
                    acc1_train_after_ft.append(acc1_train)
                    loss_test_after_ft.append(loss_test)
                
                acc_last.append(acc_l)
                acc_best.append(acc_b)
                acc_time.append(acc_time_)
                _, id, d = _get_exp_name_id(exp)
                exp_id.append(id)
                date.append(d)
                finish_t = parse_finish_time(log_f)
                finish_time.append(finish_t)
                acc1_test_after_ft.append(acc_b) # for loss, acc correlation analysis
            
            except:
                print(f'!! There is sth. wrong with "{exp}", please check. Continue first')
    
    # remove outlier
    if args.remove_outlier:
        n_exp_original = len(acc_best)
        remove_outlier(acc_best, acc_best, acc_last, exp_id, finish_time, date, acc_time)
        n_outlier = n_exp_original - len(acc_best)
    
    # print
    current_server_id = os.environ['SERVER'] if 'SERVER' in os.environ else ''
    exp_str = '[%s-%s] ' % (current_server_id, _get_project_name()) + ', '.join(exp_id) # [138-CRD] 174550, 174554, 174558
    
    if len(acc_last) == 1: # only one result
        acc_str = _make_acc_str_one_exp(acc_last[0], acc_best[0], num_digit=args.n_decimals)
        print('[exp date: %s]' % date[-1])
        print(exp_str + ' -- ' + acc_str) # [115-CCL] 225022 -- 0.1926/0.4944 
        print('acc_time: %s' % acc_time[0])
        
    elif len(acc_last) > 1:
        acc_last_str = _make_acc_str(acc_last, num_digit=args.n_decimals, present='last' in present_data) # 75.84, 75.63, 75.45 – 75.64 (0.16)
        acc_best_str = _make_acc_str(acc_best, num_digit=args.n_decimals, present='best' in present_data)
        print(f'[exp date: {", ".join(date)}]')
        print(exp_str)
        print(acc_last_str)
        print(acc_best_str)
        if np.mean(acc_time) == np.max(acc_time):
            print('acc_time: %s' % acc_time)
        else:
            print('acc_time: %s -- Warning: acc times are different!' % acc_time)
    
    print_ft = False
    for ft in finish_time:
        if ft and ('predict' in ft or time.localtime() < time.strptime(ft, '%Y/%m/%d-%H:%M')):
            print_ft = True
            break
    if print_ft:
        print('fin_time: %s' % (finish_time))

    if args.remove_outlier:
        print(f'Note, {n_outlier} outliers have been not included (outlier_thresh = {args.outlier_thresh})')

    # Print acc of the just pruned model
    if len(acc1_test_just_finished_prune):
        print(f'test_acc_just_pruned: {np.mean(acc1_test_just_finished_prune):.4f} ({np.std(acc1_test_just_finished_prune):.4f})')

    # Accuracy analyzer
    if args.acc_analysis:
        mean_metric_first_LR_stage = []
        for exp in all_exps:
            if name in exp:
                _, id, _ = _get_exp_name_id(exp)
                if id in exp_id:
                    log_f = '%s/log/log.txt' % exp
                    analyzer = AccuracyAnalyzer(log_f)
                    mean_metric_first_LR_stage += [analyzer.metric_sketch[0][-1]]
        avg, std = np.mean(mean_metric_first_LR_stage), np.std(mean_metric_first_LR_stage)
        print(f'Avg mean_metric of the 1st LR stage: {avg:.{args.n_decimals}f}$_{{\pm{std:.{args.n_decimals}f}}}$')
    
    # for loss, acc correlation analysis
    if args.corr_analysis and len(loss_train_just_finished_prune):
        # print(len(acc1_test_just_finished_prune), len(loss_test_just_finished_prune), 
        #     len(acc1_train_just_finished_prune), len(loss_train_just_finished_prune), len(acc1_test_after_ft))
        matrix = np.stack([acc1_test_just_finished_prune, loss_test_just_finished_prune, 
            acc1_train_just_finished_prune, loss_train_just_finished_prune, loss_train_after_ft, acc1_test_after_ft])
        matrix = np.stack([loss_train_just_finished_prune, loss_train_after_ft, loss_test_after_ft])
        shape = str(np.shape(matrix))
        if args.corr_stats == 'pearson':
            pass
            # corr = stats.pearsonr(matrix)
        elif args.corr_stats == 'spearman':
            corr, pval = stats.spearmanr(matrix, axis=1) # axis=1, each row is a variable; each column is an observation
        elif args.corr_stats == 'kendall':
            corr01, pval01 = stats.kendalltau(loss_train_just_finished_prune, loss_train_after_ft)
            corr02, pval02 = stats.kendalltau(loss_train_just_finished_prune, loss_test_after_ft)
            corr12, pval12 = stats.kendalltau(loss_train_after_ft, loss_test_after_ft)
            corr, pval = np.ones([3, 3]), np.zeros([3, 3])
            corr[0, 1], corr[0, 2], corr[1, 2] = corr01, corr02, corr12
            pval[0, 1], pval[0, 2], pval[1, 2] = pval01, pval02, pval12

        attr = ['pruned_loss_train', 'final_loss_train', 'final_loss_test'] # what to print is manually set
        print('------------------ matrix shape: %s, correlation matrix: ------------------' % shape)
        print(attr)
        matprint(corr)
        print('------------------ p-value: ------------------')
        matprint(pval)

        # plot a scatter to see correlation
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        ax[0].scatter(loss_train_just_finished_prune, loss_train_after_ft)
        ax[0].set_xlabel('loss_train_just_finished_prune')
        ax[0].set_ylabel('loss_train_after_ft')
        ax[1].scatter(loss_train_just_finished_prune, loss_test_after_ft)
        ax[1].set_xlabel('loss_train_just_finished_prune')
        ax[1].set_ylabel('loss_test_after_ft')
        ax[2].scatter(loss_train_after_ft, loss_test_after_ft)
        ax[2].set_xlabel('loss_train_after_ft')
        ax[2].set_ylabel('loss_test_after_ft')
        fig.tight_layout()
        fig.savefig(args.out_plot_path, dpi=200)


def matprint(mat, fmt="g"):
    try:
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
            print("")
    except:
        print(mat)

parser = argparse.ArgumentParser()
parser.add_argument('--kw', type=str, required=True, help='keyword for filtering expriment folders')
parser.add_argument('--exact_kw', action='store_true', help='if true, not filter by exp_name but exactly the kw')
parser.add_argument('--mark', type=str, default='last') # 'Epoch 240' or 'Step 11200', which is used to pin down the line that prints the best accuracy
parser.add_argument('--accline_mark', type=str, default='')
parser.add_argument('--metric', type=str, default='Acc1')
parser.add_argument('--present_data', type=str, default='', choices=['', 'last', 'best', 'last,best'])
parser.add_argument('--acc_analysis', action='store_true')
parser.add_argument('--corr_analysis', action='store_true')
parser.add_argument('--remove_outlier', action='store_true')
parser.add_argument('--outlier_thresh', type=float, default=0.5, help='if |value - mean| > outlier_thresh, we take this value as an outlier')
parser.add_argument('--corr_stats', type=str, default='spearman', choices=['pearson', 'spearman', 'kendall'])
parser.add_argument('--out_plot_path', type=str, default='plot.jpg')
parser.add_argument('--ignore', type=str, default='')
parser.add_argument('--exps_folder', type=str, default='Experiments')
parser.add_argument('--n_decimals', type=int, default=2)
parser.add_argument('--scale', type=float, default=1.)
args = parser.parse_args()

if 'loss' in args.metric.lower():
    args.n_decimals = 4

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
        print_acc_for_one_exp_group(all_exps, exp_name, args.mark, args.present_data)
        print('')

if __name__ == '__main__':
    main()

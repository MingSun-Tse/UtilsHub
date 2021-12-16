import os
import shutil as sh
import sys
import matplotlib.pyplot as plt
import numpy as np
import math, argparse

'''Usage Example:
    python get_log_and_compare_acc.py 138-DAFL-120845 138-DAFL-120835 5-CRD-070245
'''

# Log path template on different servers
SERVER = {
    '138': 'wanghuan@155.33.198.138:/home/wanghuan/Projects/{}/Experiments/*-{}/log/log.txt',
    '005': 'wanghuan@155.33.199.5:/home3/wanghuan/Projects/{}/Experiments/*-{}/log/log.txt',
    '115': 'yulun@155.33.198.115:/media/yulun/12THD1/Huan_Projects/Projects/{}/Experiments/*-{}/log/log.txt',
    '120': 'wang.huan@129.10.0.120:/home/wang.huan/Projects/{}/Experiments/*-{}/log/log.txt',
    '170': 'huan@155.33.198.170:/home/wanghuan/Projects/{}/Experiments/*-{}/log/log.txt',
    '202': 'huwang@137.203.141.202:/homes/huwang/Projects/{}/Experiments/*-{}/log/log.txt',
}

def smooth(L, window=50):
    if window == 1: return L
    num = len(L)
    out = []
    for i in range(num):
        if i < window:
            out.append(np.mean(L[:i+1]))
        else:
            out.append(np.mean(L[i+1-window:i+1]))
    return np.array(out) if isinstance(L, np.ndarray) else out

def _get_ExpID_from_path():
    for line in open('log.txt'):
        if 'caching various config files' in line:
            ExpID = line.split('SERVER')[1].split('/')[0]
            return "SERVER" + ExpID

def _get_value(line_seg, key, type_func=float):
    for i in range(len(line_seg)):
        if key in line_seg[i]:
            break
    if i == len(line_seg) - 1:
        return None # did not find the <key> in this line
    try:
        value = type_func(line_seg[i + 1]) # example: "Acc: 0.7"
    except:
        value = type_func(line_seg[i + 2]) # example: "Acc = 0.7"
    return value

def _get_value_from_log(log, key, is_acc=True):
    '''Plot some item in a log file
    Example: Acc1 = 0.75 xxx Step xxx
    '''
    value = []
    for line in open(log):
        if key in line:
            line_seg = line.strip().split()
            v = _get_value(line_seg, key, type_func=float)
            if is_acc and v > 1:
                v = v / 100.0 # use accuracy in range [0, 1]
            value.append(v)
    return value
        
def _fetch_log_file(server, project, log_id):
    log_path = SERVER[server].format(project, log_id)
    script = 'scp "%s" .' % log_path
    os.system(script)
    ExpID = _get_ExpID_from_path()
    log_file = 'log_%s.txt' % ExpID
    sh.move('log.txt', log_file)
    return log_file

#################################
parser = argparse.ArgumentParser()
parser.add_argument('--server', '-s', type=str, required=True, help='server name')
parser.add_argument('--project', '-p', type=str, required=True, help='project name') 
parser.add_argument('--log_ids', type=str, required=True, help='example: "000012,000245"')
parser.add_argument('--window', type=int, default=1)
args = parser.parse_args()

acc, loss = [], []
log_ids = args.log_ids.split(',')
for log_id in log_ids:
    # get log file
    local_log_files = [x for x in os.listdir('./') if x.startswith('log_') and x.endswith('.txt')]
    log_file = ''
    for f in local_log_files:
        if log_id in f:
            log_file = f
            break
    if log_file == '':
        log_file = _fetch_log_file(args.server, args.project, log_id)
    
    # parsing from log file ------------------------
    # options = Loss_train, Loss_test, Acc1_train, Acc1 etc. (You may need to modify this manually to your need)
    v_loss = _get_value_from_log(log_file, 'Loss_train', is_acc=False)
    v_acc = _get_value_from_log(log_file, 'Acc1')
    # ----------------------------------------------
    acc.append(v_acc)
    loss.append(v_loss)

min_len = np.min([len(v) for v in acc])
fig, ax = plt.subplots()
ax.set_ylabel('Loss')
ax2 = ax.twinx()
ax2.set_ylabel('Acc1')
for loss_, acc_, log_id in zip(loss, acc, log_ids):
    step = 1 # int(round(len(v) * 1.0  / min_len))

    loss_ = loss_[1:]
    loss_[::step] = smooth(loss_[::step], window=args.window)
    ax.plot(loss_, label=log_id + ' Loss', linestyle='dashed')
    
    # acc_[::step] = smooth(acc_[::step], window=args.window)
    # ax2.plot(acc_, label=log_id + ' Acc1', linestyle='solid')

ax.legend()
ax2.legend(loc=2)

fig.tight_layout()
plt.grid()
plt.show()

import os
import shutil as sh
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

'''Usage Example:
    python get_log_and_compare_acc.py 138-DAFL-120845 138-DAFL-120835 5-CRD-070245
'''

# Log path template on different servers
SERVER = {
    '138': 'wanghuan@155.33.198.138:/home/wanghuan/Projects/{}/Experiments/*-{}/log/log.txt',
    '5'  : 'wanghuan@155.33.199.5:/home3/wanghuan/Projects/{}/Experiments/*-{}/log/log.txt',
    '115': 'yulun@155.33.198.115:/media/yu*/12THD1/Huan_Projects/{}/Experiments/*-{}/log/log.txt',
    'clu': '', # cluster
    '170': 'huan@155.33.198.170:/home/wanghuan/Projects/{}/Experiments/*-{}/log/log.txt',
    '202': 'huwang@137.203.141.202:/homes/huwang/Projects/{}/Experiments/*-{}/log/log.txt',
}

def smooth(L, window=50):
    num = len(L)
    L1 = list(L[:window]) + list(L)
    out = [np.average(L1[i : i + window]) for i in range(num)]
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

def _get_value_from_log(log, key):
    '''Plot some item in a log file
    Example: Acc1 = 0.75 xxx Step xxx
    '''
    value = []
    for line in open(log):
        if key in line:
            line_seg = line.strip().split()
            v = _get_value(line_seg, key, type_func=float)
            if v > 1:
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
value = []
log_ids = []
window = 1
for arg in sys.argv[1:]:
    if "-" not in arg:
        window = int(arg)
    else:
        server, project, log_id = arg.split('-') # arg example: 202-CRD-002234. This means, we want the log on machine 202, project CRD, log_id 002234
        local_log_files = [x for x in os.listdir('./') if x.startswith('log_') and x.endswith('.txt')]
        
        # get log file
        log_file = ''
        for f in local_log_files:
            if log_id in f:
                log_file = f
                break
        if log_file == '':
            log_file = _fetch_log_file(server, project, log_id)
        
        # parsing from log file
        v = _get_value_from_log(log_file, 'Acc1')
        value.append(v)
        log_ids.append(arg)

min_len = np.min([len(v) for v in value])
for v, log_id in zip(value, log_ids):
    step = 1 # int(round(len(v) * 1.0  / min_len))
    v = v[::step]
    v = smooth(v, window=window)
    plt.plot(v, label=log_id)
plt.grid()
plt.legend()
plt.show()

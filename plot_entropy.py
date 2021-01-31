import numpy as np, os, sys
import matplotlib.pyplot as plt
from scipy import stats

inFile = sys.argv[1]
out_plot_path = sys.argv[2]

pruned_train_loss = []
final_train_loss = []
final_test_loss = []
final_test_acc = []

def _get_value(line, key, type_func=float, exact_key=False):
    if exact_key: # back compatibility
        value = line.split(key)[1].strip().split()[0]
        if value.endswith(')'): # hand-fix case: "Epoch 23)"
            value = value[:-1] 
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

step_entropy = []
T_entropy = []
S_entropy = []
TS_kld = []

step_acc = []
test_acc = []
for line in open(inFile):
    if 'T_entropy' in line: # T_entropy 0.3722 S_entropy 0.4619 TS_kld 0.0835 Step 15480
        T_entropy.append(_get_value(line, 'T_entropy'))
        S_entropy.append(_get_value(line, 'S_entropy'))
        TS_kld.append(_get_value(line, 'TS_kld'))
        step_entropy.append(_get_value(line, 'Step'))
    if 'Best_Acc1' in line and '@' in line:
        test_acc.append(_get_value(line, 'Acc1'))
        step_acc.append(_get_value(line,'Step'))

fig, ax = plt.subplots()
ax.plot(step_entropy, T_entropy, label='T_entropy')
ax.plot(step_entropy, S_entropy, label='S_entropy')
ax.plot(step_entropy, TS_kld, label='TS_kld')
ax.plot(step_acc, test_acc, label='test_acc')
ax.grid()
ax.legend()
fig.tight_layout()
fig.savefig(out_plot_path)


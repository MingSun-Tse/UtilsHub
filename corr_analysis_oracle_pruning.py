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

def matprint(mat, fmt="g"):
    try:
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
            print("")
    except:
        print(mat)

for line in open(inFile):
    if 'Start oracle pruning' in line: # Start oracle pruning: 252 pairs of pruned index to ablate
        n_pairs = int(line.split('Start oracle pruning:')[1].split('pairs')[0].strip())
    if not 'n_pairs' in locals():
        continue
    if '/%d]' % n_pairs in line:
        if 'pruned_train_loss' in line:
            pruned_train_loss.append(_get_value(line, 'pruned_train_loss'))
        if 'last5_train_loss' in line and '(mean)' in line:
            final_train_loss.append(_get_value(line, 'last5_train_loss'))
            final_test_loss.append(_get_value(line, 'last5_test_loss'))
            final_test_acc.append(_get_value(line, 'last5_test_acc'))

if len(pruned_train_loss) == len(final_train_loss) + 1:
    pruned_train_loss = pruned_train_loss[:-1]

# corr analysis
corr01, pval01 = stats.kendalltau(pruned_train_loss, final_train_loss)
corr02, pval02 = stats.kendalltau(pruned_train_loss, final_test_loss)
corr12, pval12 = stats.kendalltau(final_train_loss, final_test_loss)
corr, pval = np.ones([3, 3]), np.zeros([3, 3])
corr[0, 1], corr[0, 2], corr[1, 2] = corr01, corr02, corr12
pval[0, 1], pval[0, 2], pval[1, 2] = pval01, pval02, pval12
attr = ['pruned_train_loss', 'final_train_loss', 'final_test_loss'] # what to print is manually set
print('------------------ num sample: %s, correlation matrix: ------------------' % len(pruned_train_loss))
print(attr)
matprint(corr)
print('------------------ p-value: ------------------')
matprint(pval)

# plot a scatter to see correlation
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].scatter(pruned_train_loss, final_train_loss)
ax[0].set_xlabel('pruend_train_loss')
ax[0].set_ylabel('final_train_loss')
ax[1].scatter(pruned_train_loss, final_test_loss)
ax[1].set_xlabel('pruend_train_loss')
ax[1].set_ylabel('final_test_loss')
ax[2].scatter(final_train_loss, final_test_loss)
ax[2].set_xlabel('final_train_loss')
ax[2].set_ylabel('final_test_loss')
fig.tight_layout()
fig.savefig(out_plot_path, dpi=200)




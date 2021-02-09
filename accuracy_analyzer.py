from collections import OrderedDict
import numpy as np
import sys, glob

# parse wanted value from accuracy print log
def parse_acc_log(line, key, type_func=float):
    line_seg = line.strip().lower().split()
    for i in range(len(line_seg)):
        if key in line_seg[i]:
            break
    if i == len(line_seg) - 1:
        return None # did not find the <key> in this line
    try:
        value = type_func(line_seg[i+1])
    except:
        value = type_func(line_seg[i+2])
    return value

class AccuracyAnalyzer():
    def __init__(self, log_path):
        logs = glob.glob(log_path)
        for log in logs: # there may be multiple logs
            print('"%s":' % log)
            self.log = log
            self.lr_state = OrderedDict()
            self.register_from_log()
            self.analyze()
    
    def _register(self, lr, step, acc):
        '''
            <step> is an abstraction, which can be iteration or epoch
        '''
        if any([x == None for x in [lr, step, acc]]):
            return
        lr = str(lr)
        if lr in self.lr_state:
            self.lr_state[lr].append([step, acc])
        else:
            self.lr_state[lr] = [[step, acc]]

    def clear(self):
        self.lr_state = OrderedDict()

    def register_from_log(self):
        '''
            <log> will be like this: 
                [221700 29847 2020/06/13-04:27:58]  Acc1 = 72.6460 Acc5 = 90.9620 Epoch 77 (after update) lr 0.001 (Best Acc1 72.6920 @ Epoch 73)
                [221700 29847 2020/06/13-04:51:17]  Acc1 = 72.5960 Acc5 = 90.9740 Epoch 78 (after update) lr 0.001 (Best Acc1 72.6920 @ Epoch 73)
                [221700 29847 2020/06/13-05:14:40]  Acc1 = 72.6940 Acc5 = 90.9560 Epoch 79 (after update) lr 0.001 (Best Acc1 72.6940 @ Epoch 79)
                [221700 29847 2020/06/13-05:38:03]  Acc1 = 72.6340 Acc5 = 90.9080 Epoch 80 (after update) lr 0.001 (Best Acc1 72.6940 @ Epoch 79)
            Warning: This func depends on the specific log format. Need improvement.
        '''
        for line in open(self.log):
            if 'Acc1' in line and '@' in line:
                lr = parse_acc_log(line, 'lr')
                step = parse_acc_log(line, 'epoch', type_func=int)
                acc = parse_acc_log(line, 'acc1')
                self._register(lr, step, acc)
    
    def analyze(self, print_func=print):
        keys = list(self.lr_state.keys())
        max_len_lr = 0
        for k in keys:
            max_len_lr = max(len(str(k)), max_len_lr) # eg, from 0.1 to 0.0001
        vals = list(self.lr_state.values())
        max_len_step = len(str(vals[-1][-1][0])) # eg, from 0 to 10000
        format_str = 'lr %{}s (%{}d ~ %{}d) -- max_acc %6.4f | min_acc %6.4f | avg_acc %6.4f | max_acc_step %d'.format(max_len_lr, max_len_step, max_len_step)
        for lr in self.lr_state.keys():
            lr_state = np.array(self.lr_state[lr])
            step = lr_state[:, 0]
            acc = lr_state[:, 1]
            max_acc_step = lr_state[np.argmax(lr_state[:, 1])][0]
            print_func(format_str % (lr, step[0], step[-1], acc.max(), acc.min(), acc.mean(), max_acc_step))
    
    def plot(self):
        pass

if __name__ == "__main__":
    AccuracyAnalyzer(sys.argv[1])
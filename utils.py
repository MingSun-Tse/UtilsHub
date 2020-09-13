import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from pprint import pprint
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
import copy
import glob
from PIL import Image
import json
import yaml
import pandas as pd

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

# refer to: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/imagenet/l1-norm-pruning/compute_flops.py
def get_n_params(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    total /= 1e6
    # print('  Number of params: %.4fM' % total)
    return total

def get_n_flops(model=None, input_res=224, multiply_adds=True):
    model = copy.deepcopy(model)

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 0 if self.bias is not None else 0

        # params = output_channels * (kernel_ops + bias_ops) # @mst: commented since not used
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.weight.data != 0).float().sum() # @mst: this should be considering the pruned model
        # could be problematic if some weights happen to be 0.
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(3,input_res,input_res).unsqueeze(0), requires_grad = True)
    out = model(input)


    total_flops = (sum(list_conv) + sum(list_linear)) # + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    total_flops /= 1e9
    # print('  Number of FLOPs: %.2fG' % total_flops)

    return total_flops

# refer to: https://github.com/alecwangcq/EigenDamage-Pytorch/blob/master/utils/common_utils.py
class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = {}
        for k, v in decay_schedule.items(): # a dict, example: {"0":0.001, "30":0.00001, "45":0.000001}
            self.decay_schedule[int(k)] = v
        print('Using a preset learning rate schedule:')
        print(self.decay_schedule)

    def __call__(self, optimizer, e):
        epochs = list(self.decay_schedule.keys())
        epochs = sorted(epochs) # example: [0, 30, 45]
        lr = self.decay_schedule[epochs[-1]]
        for i in range(len(epochs) - 1):
            if epochs[i] <= e < epochs[i+1]:
                lr = self.decay_schedule[epochs[i]]
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        return lr

def plot_weights_heatmap(weights, out_path):
    '''
        weights: [N, C, H, W]. Torch tensor
        averaged in dim H, W so that we get a 2-dim color map of size [N, C]
    '''
    w_abs = weights.abs()
    w_abs = w_abs.data.cpu().numpy()
    
    fig, ax = plt.subplots()
    im = ax.imshow(w_abs, cmap='jet')

    # make a beautiful colorbar        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=0.05, pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel("Channel")
    ax.set_ylabel("Filter")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def strlist_to_list(sstr, ttype):
    '''
        example:
        # self.args.stage_pr = [0, 0.3, 0.3, 0.3, 0, ]
        # self.args.skip_layers = ['1.0', '2.0', '2.3', '3.0', '3.5', ]
        turn these into a list of <ttype> (float or str or int etc.)
    '''
    out = []
    sstr = sstr.split("[")[1].split("]")[0]
    for x in sstr.split(','):
        x = x.strip()
        if x:
            x = ttype(x)
            out.append(x)
    return out

def strdict_to_dict(sstr, ttype):
    '''
        '{"1": 0.04, "2": 0.04, "4": 0.03, "5": 0.02, "7": 0.03, }'
    '''
    out = {}
    sstr = sstr.split("{")[1].split("}")[0]
    for x in sstr.split(','):
        x = x.strip()
        if x:
            k = x.split(':')[0]
            v = ttype(x.split(':')[1].strip())
            out[k] = v
    return out
    
def check_path(x):
    if x:
        complete_path = glob.glob(x)
        assert(len(complete_path) == 1)
        x = complete_path[0]
    return x

def parse_prune_ratio_vgg(sstr):
    # example: [0-4:0.5, 5:0.6, 8-10:0.2]
    out = np.zeros(20) # at most 20 layers, could be problematic but enough for vgg-like cases
    sstr = sstr.split("[")[1].split("]")[0]
    for x in sstr.split(','):
        k = x.split(":")[0].strip()
        v = x.split(":")[1].strip()
        if k.isdigit():
            out[int(k)] = float(v)
        else:
            begin = int(k.split('-')[0].strip())
            end = int(k.split('-')[1].strip())
            out[begin : end+1] = float(v)
    return out


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0),  A.size(1) * B.size(1))
    
    
def np_to_torch(x):
    '''
        np array to pytorch float tensor
    '''
    x = np.array(x)
    x= torch.from_numpy(x).float()
    return x

def kd_loss(student_scores, teacher_scores, temp=1):
    '''Knowledge distillation loss: soft target
    '''
    p = F.log_softmax(student_scores / temp, dim=1)
    q =     F.softmax(teacher_scores / temp, dim=1)
    # l_kl = F.kl_div(p, q, size_average=False) / student_scores.shape[0] # previous working loss
    l_kl = F.kl_div(p, q, reduction='batchmean') # 2020-06-21 @mingsun-tse: Since 'size_average' is deprecated, \
    # use 'reduction' instead. In probation.
    return l_kl

def test(net, test_loader):
    n_example_test = 0
    total_correct = 0
    avg_loss = 0
    net.eval()
    with torch.no_grad():
        pred_total = []
        label_total = []
        for _, (images, labels) in enumerate(test_loader):
            n_example_test += images.size(0)
            images = images.cuda()
            labels = labels.cuda()
            output = net(images)
            avg_loss += nn.CrossEntropyLoss()(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            pred_total.extend(list(pred.data.cpu().numpy()))
            label_total.extend(list(labels.data.cpu().numpy()))
    
    acc = float(total_correct) / n_example_test
    avg_loss /= n_example_test

    # get accuracy per class
    n_class = output.size(1)
    acc_test = [0] * n_class
    cnt_test = [0] * n_class
    for p, l in zip(pred_total, label_total):
        acc_test[l] += int(p == l)
        cnt_test[l] += 1
    acc_per_class = []
    for c in range(n_class):
        acc_test[c] /= float(cnt_test[c])
        acc_per_class.append(acc_test[c])

    return acc, avg_loss.item(), acc_per_class

def get_project_path(ExpID):
    full_path = glob.glob("Experiments/*%s*" % ExpID)
    assert(len(full_path) == 1) # There should be only ONE folder with <ExpID> in its name.
    return full_path[0]

def parse_ExpID(path):
    '''parse out the ExpID from 'path', which can be a file or directory.
    Example: Experiments/AE__ckpt_epoch_240.pth__LR1.5__originallabel__vgg13_SERVER138-20200829-202307/gen_img
    Example: Experiments/AE__ckpt_epoch_240.pth__LR1.5__originallabel__vgg13_SERVER-20200829-202307/gen_img
    '''
    return 'SERVER' + path.split('_SERVER')[1].split('/')[0]

def mkdirs(*paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

class EMA():
    '''
        Exponential Moving Average for pytorch tensor
    '''
    def __init__(self, mu):
        self.mu = mu
        self.history = {}

    def __call__(self, name, x):
        '''
            Note: this func will modify x directly, no return value.
            x is supposed to be a pytorch tensor.
        '''
        if self.mu > 0:
            assert(0 < self.mu < 1)
            if name in self.history.keys():
                new_average = self.mu * self.history[name] + (1.0 - self.mu) * x.clone()
            else:
                new_average = x.clone()
            self.history[name] = new_average.clone()
            return new_average.clone()
        else:
            return x.clone()

# Exponential Moving Average
class EMA2():
  def __init__(self, mu):
    self.mu = mu
    self.shadow = {}
  def register(self, name, value):
    self.shadow[name] = value.clone()
  def __call__(self, name, x):
    assert name in self.shadow
    new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
    self.shadow[name] = new_average.clone()
    return new_average

def register_ema(emas):
    for net, ema in emas:
        for name, param in net.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

def apply_ema(emas):
    for net, ema in emas:
        for name, param in net.named_parameters():
            if param.requires_grad:
                param.data = ema(name, param.data)

colors = ["gray", "blue", "black", "yellow", "green", "yellowgreen", "gold", "royalblue", "peru", "purple"]
def feat_visualize(ax, feat, label):
    '''
        feat:  N x 2 # 2-d feature, N: number of examples
        label: N x 1
    '''
    for ix in range(len(label)):
        x = feat[ix]
        y = label[ix]
        ax.scatter(x[0], x[1], color=colors[y], marker=".")
    return ax

def smart_weights_load(net, w_path, key=None, load_mode='exact'):
    '''
        This func is to load the weights of <w_path> into <net>.
    '''
    common_weights_keys = ['T', 'S', 'G', 'model', 'state_dict']

    ckpt = torch.load(w_path, map_location=lambda storage, location: storage)
    
    # get state_dict
    if isinstance(ckpt, OrderedDict):
        state_dict = ckpt
    else:
        if key:
            state_dict = ckpt[key]
        else:
            intersection = [k for k in ckpt.keys() if k in common_weights_keys]
            if len(intersection) == 1:
                k = intersection[0]
                state_dict = ckpt[k]
            else:
                print('Error: multiple model keys found in ckpt: %s. Please explicitly appoint one' % intersection)
                exit(1)

    if load_mode == 'exact': # net and state_dict have exactly the same architecture (layer names etc. are exactly same)
        try:
            net.load_state_dict(state_dict)
        except:
            ckpt_data_parallel = False
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    ckpt_data_parallel = True # DataParallel was used in the ckpt
                    break
            
            if ckpt_data_parallel:
                # If ckpt used DataParallel, then the reason of the load failure above should be that the <net> does not use 
                # DataParallel. Therefore, remove the surfix 'module.' in ckpt.
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    param_name = k.split("module.")[-1]
                    new_state_dict[param_name] = v
            else:
                # Similarly, if ckpt didn't use DataParallel, here we add the surfix 'module.'.
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    param_name = 'module.' + k
                    new_state_dict[param_name] = v
            net.load_state_dict(new_state_dict)

    else:
    # Here is the case that <net> and ckpt only have part of weights in common. Then load them by module name:
    # for every named module in <net>, if ckpt has a module of the same name, then they are matched and weights are loaded from ckpt to <net>.
        for name, m in net.named_modules():
            module_name = name.split("module.")[-1] # remove 'module.' if any
            # match and load
            matched_param_name = ''
            for k in ckpt.keys():
                if module_name in k:
                    matched_param_name = k
                    break
                if matched_param_name:
                    m.weight.copy_(ckpt[matched_param_name])
                    print("net module name: '%s' <- '%s' (ckpt module name)" % (name, matched_param_name))
                else:
                    print("Error: cannot find matched module in ckpt for module '%s' in net. Please check manually." % name)
                    exit(1)
    
class AccuracyAnalyzer():
    def __init__(self):
        self.lr_state = OrderedDict()
    
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

    def _get_value(self, line_seg, key, type_func=str, shift=1):
        for i in range(len(line_seg)):
            if key in line_seg[i]:
                break
        if i == len(line_seg) - 1:
            return None # did not find the <key> in this line
        value = line_seg[i + shift]
        return type_func(value)
    
    def clear(self):
        self.lr_state = OrderedDict()

    def register_from_log(self, log):
        '''
            <log> will be like this: 
                [221700 29847 2020/06/13-04:27:58]  Acc1 = 72.6460 Acc5 = 90.9620 Epoch 77 (after update) lr 0.001 (Best Acc1 72.6920 @ Epoch 73)
                [221700 29847 2020/06/13-04:51:17]  Acc1 = 72.5960 Acc5 = 90.9740 Epoch 78 (after update) lr 0.001 (Best Acc1 72.6920 @ Epoch 73)
                [221700 29847 2020/06/13-05:14:40]  Acc1 = 72.6940 Acc5 = 90.9560 Epoch 79 (after update) lr 0.001 (Best Acc1 72.6940 @ Epoch 79)
                [221700 29847 2020/06/13-05:38:03]  Acc1 = 72.6340 Acc5 = 90.9080 Epoch 80 (after update) lr 0.001 (Best Acc1 72.6940 @ Epoch 79)
            Warning: This func depends on the specific log format. Need improvement.
        '''
        for line in open(log):
            line_seg = line.strip().lower().split()
            lr = self._get_value(line_seg, 'lr')
            step = self._get_value(line_seg, 'epoch', type_func=int)
            acc = self._get_value(line_seg, 'acc1', type_func=float, shift=2)
            self._register(lr, step, acc)
    
    def analyze(self, print_func=print):
        keys = list(self.lr_state.keys())
        max_len_lr = 0
        for k in keys:
            max_len_lr = max(len(str(k)), max_len_lr) # eg, from 0.1 to 0.0001
        vals = list(self.lr_state.values())
        max_len_step = len(str(vals[-1][-1][0])) # eg, from 0 to 10000
        format_str = 'lr %{}s (%{}d - %{}d): max acc = %.4f, min acc = %.4f, ave acc = %.4f'.format(max_len_lr, max_len_step, max_len_step)
        for lr in self.lr_state.keys():
            lr_state = np.array(self.lr_state[lr])
            step = lr_state[:, 0]
            acc = lr_state[:, 1]
            print_func(format_str % (lr, step[0], step[-1], acc.max(), acc.min(), acc.mean()))
    
    def plot(self):
        pass

def get_layer_by_index(net, index):
    cnt = -1
    for _, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            cnt += 1
            if cnt == index:
                return m
    return None

def get_total_index_by_learnable_index(net, learnable_index):
    '''
        learnable_index: index when only counting learnable layers (conv or fc, no bn);
        total_index: count relu, pooling etc in.
    '''
    layer_type_considered = [nn.Conv2d, nn.ReLU, nn.LeakyReLU, nn.PReLU, 
        nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d, nn.Linear]
    cnt_total = -1
    cnt_learnable = -1
    for _, m in net.named_modules():
        cond = [isinstance(m, x) for x in layer_type_considered]
        if any(cond):
            cnt_total += 1
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                cnt_learnable += 1
                if cnt_learnable == learnable_index:
                    return cnt_total
    return None

def cal_correlation(x, coef=False):
    '''Calculate the correlation matrix for a pytorch tensor.
    Input shape: [n_sample, n_attr]
    Output shape: [n_attr, n_attr]
    Refer to: https://github.com/pytorch/pytorch/issues/1254
    '''
    # calculate covariance matrix
    y = x - x.mean(dim=0)
    c = y.t().mm(y) / (y.size(0) - 1)
    
    if coef:
        # normalize covariance matrix
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())

        # clamp between -1 and 1
        # probably not necessary but numpy does it
        c = torch.clamp(c, -1.0, 1.0)
    return c

def get_class_corr(loader, model):
    model.eval().cuda()
    logits = 0
    n_batch = len(loader)
    with torch.no_grad():
        for ix, data in enumerate(loader):
            input = data[0]
            print('[%d/%d] -- forwarding' % (ix, n_batch))
            input = input.float().cuda()
            if type(logits) == int:
                logits = model(input) # [batch_size, n_class]
            else:
                logits = torch.cat([logits, model(input)], dim=0)
    # Use numpy:
    # logits -= logits.mean(dim=0)
    # logits = logits.data.cpu().numpy()
    # corr = np.corrcoef(logits, rowvar=False)

    # Use pytorch
    corr = cal_correlation(logits, coef=True)
    return corr

def cal_acc(logits, y):
    pred = logits.argmax(dim=1)
    acc = pred.eq(y.data.view_as(pred)).sum().float() / y.size(0)
    return acc

class Timer():
    '''Log down iteration time and predict the left time for the left iterations
    '''
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.time_stamp = []

    def predict_finish_time(self, ave_window=3):
        self.time_stamp.append(time.time()) # update time stamp
        if len(self.time_stamp) == 1:
            return 'only one time stamp, not enough to predict'
        interval = []
        for i in range(len(self.time_stamp) - 1):
            t = self.time_stamp[i + 1] - self.time_stamp[i]
            interval.append(t)
        sec_per_epoch = np.mean(interval[-ave_window:])
        left_t = sec_per_epoch * (self.total_epoch - len(interval))
        finish_t = left_t + time.time()
        finish_t = time.strftime('%Y/%m/%d-%H:%M', time.localtime(finish_t))
        return finish_t + ' (speed: %.2fs per timing)' % sec_per_epoch
    
    def __call__(self):
        return(self.predict_finish_time())

class Dataset_npy_batch(Dataset):
    def __init__(self, npy_dir, transform, f='batch.npy'):
        self.data = np.load(os.path.join(npy_dir, f), allow_pickle=True)
        self.transform = transform
    def __getitem__(self, index):
        img = Image.fromarray(self.data[index][0])
        img = self.transform(img)
        label = self.data[index][1]
        label = torch.LongTensor([label])[0]
        return img.squeeze(0), label
    def __len__(self):
        return len(self.data)

def merge_args(args, params_json):
    '''<args> is from argparser. <params_json> is a json/yaml file.
    merge them, if there is collision, the param in <params_json> has a higher priority.
    '''
    with open(params_json) as f:
        if params_json.endswith('.json'):
            params = json.load(f)
        elif params_json.endswith('.yaml'):
            params = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise NotImplementedError
    for k, v in params.items():
        args.__dict__[k] = v
    return args

class AccuracyManager():
    def __init__(self):
        self.accuracy = pd.DataFrame()
    
    def update(self, time, acc1, acc5=None):
        acc = pd.DataFrame([[time, acc1, acc5]], columns=['time', 'acc1', 'acc5']) # time can be epoch or step
        self.accuracy = self.accuracy.append(acc, ignore_index=True)
    
    def get_best_acc(self, criterion='acc1'):
        assert criterion in ['acc1', 'acc5']
        acc = self.accuracy.sort_values(by=criterion) # ascending sort
        best = acc.iloc[-1] # the last row
        time, acc1, acc5 = best.time, best.acc1, best.acc5
        return time, acc1, acc5
    
    def get_last_acc(self):
        last = self.accuracy.iloc[-1]
        time, acc1, acc5 = last.time, last.acc1, last.acc5
        return time, acc1, acc5


    
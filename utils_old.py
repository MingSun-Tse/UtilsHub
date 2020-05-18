import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil as sh
import time
import sys
pjoin = os.path.join
matplotlib.use("Agg")


class LogPrint():
    def __init__(self, file, ExpID):
        self.file = file
        self.ExpID = ExpID

    def __call__(self, some_str):
        print(self.ExpID[-6:] + "-%s-" % os.getpid() + time.strftime(
            "%Y/%m/%d-%H:%M:%S] ") + str(some_str), file=self.file, flush=True)

    def argprint(self, args):
        logtmp = "{"
        key_map = {}
        for k in args.__dict__:
            key_map[k.lower()] = k
        argdict = [x.lower() for x in args.__dict__]
        for k_ in sorted(argdict):
            k = key_map[k_]
            logtmp += '"%s": %s, ' % (k, args.__dict__[k])
        logtmp = logtmp[:-2] + "}"
        self.__call__(logtmp)


def check_path(x):
    if x:
        complete_path = glob.glob(x)
        assert(len(complete_path) == 1)
        x = complete_path[0]
    return x


colors = ["gray", "blue", "black", "yellow", "green",
          "yellowgreen", "gold", "royalblue", "peru", "purple"]
markers = [".", "x"]


def feat_visualize(ax, feat, label, if_right):
    '''
      feat:  N x 2 # 2-d feature, N: number of examples
      label: N x 1
    '''
    index_1 = np.where(if_right == 1)[0]
    index_0 = np.where(if_right == 0)[0]
    for ix in index_1:
        x = feat[ix]
        y = label[ix]
        ax.scatter(x[0], x[1], color=colors[y], marker=".")
    for ix in index_0:
        x = feat[ix]
        y = label[ix]
        ax.scatter(x[0], x[1], color="red", marker="x")
    return ax


def get_previous_step(e2, resume):
    previous_epoch = previous_step = 0
    if e2 and resume:
        for clip in os.path.basename(e2).split("_"):
            if clip[0] == "E" and "S" in clip:
                num1 = clip.split("E")[1].split("S")[0]
                num2 = clip.split("S")[1]
                if num1.isdigit() and num2.isdigit():
                    previous_epoch = int(num1)
                    previous_step = int(num2)
    return previous_epoch, previous_step


def set_up_dir(project_name, resume, debug):
    TimeID = time.strftime("%Y%m%d-%H%M%S")
    ExpID = "SERVER" + os.environ["SERVER"] + "-" + TimeID
    if not debug:
        assert(project_name != "")  # For a formal exp, name it!
        project_path = pjoin("Experiments", ExpID + "_" + project_name)
        rec_img_path = pjoin(project_path, "reconstructed_images")
        weights_path = pjoin(project_path, "weights")
        os.makedirs(project_path)
        os.makedirs(rec_img_path)
        os.makedirs(weights_path)
        log_path = pjoin(weights_path, "log_" + ExpID + ".txt")
        log = open(log_path, "w+")
    else:
        rec_img_path = pjoin(os.environ["HOME"], "Trash")
        weights_path = pjoin(os.environ["HOME"], "Trash")
        if not os.path.exists(rec_img_path):
            os.makedirs(rec_img_path)
        log = sys.stdout  # print to the screen
    print(" ".join(["CUDA_VISIBLE_DEVICES='0' python", *sys.argv]),
          file=log, flush=True)  # save the script
    return TimeID, ExpID, rec_img_path, weights_path, log

# Exponential Moving Average


class EMA():
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


def register_ema(net, k=0.9):
    ema = EMA(k)
    for name, param in net.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    return ema


def apply_ema(net, ema):
    for name, param in net.named_parameters():
        if param.requires_grad:
            param.data = ema(name, param.data)


def get_CodeID():
    script = "git log --pretty=oneline >> wh_CodeID_file.tmp"
    os.system(script)
    x = open("wh_CodeID_file.tmp").readline()
    os.remove("wh_CodeID_file.tmp")
    return x[:8]


class LogHub(object):
    def __init__(self, momentum=0):
        self.losses = {}
        self.momentum = momentum

    def update(self, name, value):
        if name not in self.losses:
            self.losses[name] = value
        else:
            self.losses[name] = self.losses[name] * \
                self.momentum + value * (1 - self.momentum)

    def format(self, num_digit=3):
        keys = self.losses.keys()
        keys = sorted(keys)
        keys_str = " | ".join(keys)
        values = []
        for k in keys:
            values.append(("%"+".%df" % num_digit) % self.losses[k])
        values_str = " | ".join(values)
        return keys_str, values_str


def is_img(x):
    ext = x.splitext(x)
    return ext.lower() in [".png", ".jpg", ".jpeg", ".bmp"]


def load_param_from_t7(model, in_layer_index, out_layer):
    out_layer.weight.data.copy_(model.get(in_layer_index).weight.data)
    out_layer.bias.data.copy_(model.get(in_layer_index).bias.data)


def smooth(L, window=50):
    num = len(L)
    L1 = list(L[:window]) + list(L)
    out = [np.average(L1[i:i+window]) for i in range(num)]
    return np.array(out) if type(L) == type(np.array([0])) else out

# take model1's params to model2
# for each layer of model2, if model1 has the same layer, then copy the params.


def cut_pth(model1, model2):
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()
    dict_params1 = dict(params1)
    dict_params2 = dict(params2)
    for name2, _ in params2:
        if name2 in dict_params1:
            dict_params2[name2].data.copy_(dict_params1[name2].data)
    model2.load_state_dict(dict_params2)
    torch.save(model2.state_dict(), "model2.pth")


def check_ExpID(exp_id):
    assert(isinstance(exp_id, str))
    server, date, hour = exp_id.split("-")
    assert(server[:6] == "SERVER" and server[6:].isdigit()) # example: SERVER218
    assert(date.isdigit() and date[:2] == "20") # example: 20200128
    assert(hour.isdigit())
    print("==> ExpID checked, it's okay")

def merge_experiment_folder(folderSrc, folderDst):
    '''
      move all the experiments of folderSrc to folderDst.

    '''
    folders_src = []
    for f_ in os.listdir(folderSrc):
        f = pjoin(folderSrc, f_)
        if os.path.isdir(f) and "SERVER" in f_:
            folders_src.append(f)
    print("==> these folders are waiting to be merged:", folders_src)

    for f in folders_src:
        f_ = os.path.basename(f)
        if f_[:6] == "SERVER":
            ExpID = f_.split("_")[0]
        else:
            ExpID = f_.split("_")[-1]
        print("==> processing folder '%s', its ExpID is '%s'." % (f, ExpID))
        check_ExpID(ExpID)

        f_dst = glob.glob(pjoin(folderDst, "*%s*" % ExpID))
        if len(f_dst):
            assert(len(f_dst) == 1)
            print("==> there has been the same experiment in dst folder: %s, so skipped.\n" % f_dst[0])
        else:
            sh.move(f, folderDst)
            print("==> moved.\n")

if __name__ == "__main__":
    func = eval(sys.argv[1])
    func(*sys.argv[2:])
# --------------------------
# last update: 2019-12-23
# --------------------------

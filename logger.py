import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math
import os
import shutil as sh
from distutils.dir_util import copy_tree
import glob
import sys
import copy
try:
    from utils import get_project_path, mkdirs
except:
    from uutils import get_project_path, mkdirs # sometimes, there is a name conflict for 'utils' then we will use 'uutils'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
pjoin = os.path.join

# globals
CONFIDENTIAL_SERVERS = ['202', '008']

class LogPrinter(object):
    def __init__(self, file, ExpID, print_to_screen=False):
        self.file = file
        self.ExpID = ExpID
        self.print_to_screen = print_to_screen

    def __call__(self, *in_str):
        in_str = [str(x) for x in in_str]
        in_str = " ".join(in_str)
        short_exp_id = self.ExpID[-6:]
        pid = os.getpid()
        current_time = time.strftime("%Y/%m/%d-%H:%M:%S")
        out_str = "[%s %s %s] %s" % (short_exp_id, pid, current_time, in_str)
        print(out_str, file=self.file, flush=True) # print to txt
        if self.print_to_screen:
            print(out_str) # print to screen
    
    def accprint(self, *in_str):
        blank = '  ' * int(self.ExpID[-1])
        self.__call__(blank, *in_str)
    
    def netprint(self, *in_str):
        for x in in_str:
            print(x, file=self.file, flush=True)
            if self.print_to_screen:
                print(x)
    
    def print_args(self, args):
        '''
            Example: [('batch_size', 16) ('decoder', models/small16x_ae_base/d5_base.pth)]
        '''
        logtmp = "["
        key_map = {}
        for k in args.__dict__:
            key_map[k.lower()] = k
            argdict = [x.lower() for x in args.__dict__]
        for k_ in sorted(argdict):
            k = key_map[k_]
            logtmp += "('%s', %s) "% (k, args.__dict__[k])
        logtmp = logtmp[:-1] + "]"
        self.__call__(logtmp)

class LogTracker(object):
    def __init__(self, momentum=0.9):
        self.loss = OrderedDict()
        self.momentum = momentum
        self.show = OrderedDict()

    def __call__(self, name, value, step=-1, show=True):
        '''
            Update the loss value of <name>
        '''
        assert(type(step) == int)
        # value = np.array(value)

        if step == -1:
            if name not in self.loss:
                self.loss[name] = value
            else:
                self.loss[name] = self.loss[name] * \
                    self.momentum + value * (1 - self.momentum)
        else:
            if name not in self.loss:
                self.loss[name] = [[step, value]]
            else:
                self.loss[name].append([step, value])
        
        # if the loss item will show in the log printing
        self.show[name] = show

    def avg(self, name):
        nparray = np.array(self.loss[name])
        return np.mean(nparray[:, 1], aixs=0)
    
    def max(self, name):
        nparray = np.array(self.loss[name])
        # TODO: max index
        return np.max(nparray[:, 1], axis=0)

    def format(self):
        '''
            loss example: 
                [[1, xx], [2, yy], ...] ==> [[step, [xx, yy]], ...]
                xx ==> [xx, yy, ...]
        '''
        keys = self.loss.keys()
        k_str, v_str = [], []
        for k in keys:
            if self.show[k] == False:
                continue
            v = self.loss[k]
            if not hasattr(v, "__len__"): # xx
                v = "%.4f" % v
            else:
                if not hasattr(v[0], "__len__"): # [xx, yy, ...]
                    v = " ".join(["%.3f" % x for x in v])
                elif hasattr(v[0][1], "__len__"): # [[step, [xx, yy]], ...]
                    v = " ".join(["%.3f" % x for x in v[-1][1]])
                else: # [[1, xx], [2, yy], ...]
                    v = "%.4f" % v[-1][1]
            
            format_str = "{:<%d}" % (max(len(k), len(v)))
            k_str.append(format_str.format(k))
            v_str.append(format_str.format(v))
        k_str = " | ".join(k_str)
        v_str = " | ".join(v_str)
        return k_str + " |", v_str + " |"

    def plot(self, name, out_path):
        '''
            Plot the loss of <name>, save it to <out_path>.
        '''
        v = self.loss[name]
        if (not hasattr(v, "__len__")) or type(v[0][0]) != int: # do not log the 'step'
            return
        if hasattr(v[0][1], "__len__"):
            # self.plot_heatmap(name, out_path)
            return
        v = np.array(v)
        step, value = v[:, 0], v[:, 1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(step, value)
        ax.set_xlabel("step")
        ax.set_ylabel(name)
        ax.grid()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def plot_heatmap(self, name, out_path, show_ticks=False):
        '''
            A typical case: plot the training process of 10 weights
            x-axis: step
            y-axis: index (10 weights, 0-9)
            value: the weight values
        '''
        v = self.loss[name]
        step, value = [], []
        [(step.append(x[0]), value.append(x[1])) for x in v]
        n_class = len(value[0])
        fig, ax = plt.subplots(figsize=[0.1*len(step), n_class / 5]) # /5 is set manually
        im = ax.imshow(np.transpose(value), cmap='jet')
        
        # make a beautiful colorbar        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.05, pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        
        # set the x and y ticks
        # For now, this can not adjust its range adaptively, so deprecated.
        # ax.set_xticks(range(len(step))); ax.set_xticklabels(step)
        # ax.set_yticks(range(len(value[0]))); ax.set_yticklabels(range(len(value[0])))
        
        interval = step[0] if len(step) == 1 else step[1] - step[0]
        ax.set_xlabel("step (* interval = %d)" % interval)
        ax.set_ylabel("index")
        ax.set_title(name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


class Logger(object):
    '''
        The top logger, which 
            (1) set up all log directories
            (2) maintain the losses and accuracies
    '''

    def __init__(self, args):
        self.args = args

        # set up work folder
        self.ExpID = self.get_ExpID()
        self.set_up_dir()

        self.log_printer = LogPrinter(
            self.logtxt, self.ExpID, self.args.debug or self.args.screen_print)  # for all txt logging
        self.log_tracker = LogTracker()  # for all numerical logging

        # initial print: save args
        self.print_script()
        self.print_note()
        if (not args.debug) and self.SERVER != '':
            # If self.SERVER != '', it shows this is my computer, so call this func, which is just to my need.
            # When others use my code, they probably need not call this func.
            self.__send_to_exp_hub() 
        args.CodeID = self.get_CodeID()
        self.log_printer.print_args(args)
        self.cache_model()
        self.n_log_item = 0 

    def get_CodeID(self):
        if self.args.CodeID:
            return self.args.CodeID
        else:
            script = "git status >> wh_git_status.tmp"
            os.system(script)
            x = open("wh_git_status.tmp").readlines()
            x = "".join(x)
            if "Changes not staged for commit" in x:
                self.log_printer("Warning! Your code is not commited. Cannot be too careful.")
                time.sleep(3)
            os.remove("wh_git_status.tmp")

            script = "git log --pretty=oneline >> wh_CodeID_file.tmp"
            os.system(script)
            x = open("wh_CodeID_file.tmp").readline()
            os.remove("wh_CodeID_file.tmp")
            return x[:8]

    def get_ExpID(self):
        if self.args.resume_ExpID:
            full_path = get_project_path(self.args.resume_ExpID)
            exp_folder_name = os.path.basename(full_path)
            # exp_folder_name is like "run_SERVER5-20191220-212041"
            ExpID = exp_folder_name.split("_")[-1]
            _, date, hr = ExpID.split("-")
            if not (date.isdigit() and hr.isdigit()):
                self.log_printer("ExpID format is wrong! Please check.")
                exit(1)
        else:
            self.SERVER = ''
            TimeID = time.strftime("%Y%m%d-%H%M%S")
            if 'SERVER' in os.environ.keys():
                ExpID = "SERVER" + os.environ["SERVER"] + "-" + TimeID
                self.SERVER = os.environ["SERVER"]
            else:
                ExpID = TimeID
        return ExpID

    def set_up_dir(self):
        project_path = pjoin("Experiments/%s_%s" % (self.args.project_name, self.ExpID))
        if self.args.resume_ExpID:
            project_path = get_project_path(self.args.resume_ExpID)
        if self.args.debug: # debug has the highest priority. If debug, we will make sure all the things will be saved in Debug_dir.
            project_path = "Debug_Dir"

        self.weights_path = pjoin(project_path, "weights")
        self.gen_img_path = pjoin(project_path, "gen_img")
        self.cache_path   = pjoin(project_path, ".caches")
        self.log_path     = pjoin(project_path, "log")
        self.logplt_path  = pjoin(project_path, "log", "plot")
        self.logtxt_path  = pjoin(project_path, "log", "log.txt")
        mkdirs(self.weights_path, self.gen_img_path, self.logplt_path, self.cache_path)
        self.logtxt = open(self.logtxt_path, "a+") # note: append to previous log txt file

    def print_script(self):
        print(" ".join(["CUDA_VISIBLE_DEVICES=xx python", *sys.argv]),
              file=self.logtxt, flush=True)
        print(" ".join(["CUDA_VISIBLE_DEVICES=xx python", *sys.argv]),
              file=sys.stdout, flush=True)

    def print_note(self):
        project = self.get_project_name() # the current project folder name
        exp_id = self.ExpID.split('-')[-1] # SERVER138-20200623-095526
        self.ExpNote = 'ExpNote [%s-%s-%s]: "%s" -- %s' % (self.SERVER, project, exp_id, self.args.note, self.args.project_name)
        print(self.ExpNote, file=self.logtxt, flush=True)
        print(self.ExpNote, file=sys.stdout, flush=True)

    def plot(self, name, out_path):
        self.log_tracker.plot(name, out_path)

    def print(self, step):
        keys, values = self.log_tracker.format()
        k = keys.split("|")[0].strip()
        if k: # only when there is sth to print, print 
            values += " (step = %d)" % step
            if step % (self.args.print_interval * 10) == 0 \
                or len(self.log_tracker.loss.keys()) > self.n_log_item: # when a new loss is added into the loss pool, print
                self.log_printer(keys)
                self.n_log_item = len(self.log_tracker.loss.keys())
            self.log_printer(values)

    def cache_model(self):
        '''
            Save the modle architecture, loss, configs, in case of future check.
        '''
        self.log_printer("==> caching various config files to '%s'" % self.cache_path)
        for root, dirs, files in os.walk("."):
            if "Experiments" in root or "Debug_Dir" in root:
                continue
            for f in files:
                if f.endswith(".py") or f.endswith(".json"):
                    dir_path = pjoin(self.cache_path, root)
                    f_path = pjoin(root, f)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    sh.copy(f_path, dir_path)

    def get_project_name(self):
        '''For example, 'Projects/CRD/logger.py', then return CRD
        '''
        file_path = os.path.abspath(__file__)
        return file_path.split('/')[-2]
    
    def __send_to_exp_hub(self):
        '''For every experiment, it will send <ExpNote> to a hub for the convenience of checking.
        '''
        today_local = time.strftime("%Y%m%d") + "_exps.txt"
        if self.SERVER in CONFIDENTIAL_SERVERS:
            today_remote = 'huwang@137.203.141.202:/homes/huwang/Projects/ExpLogs/%s' % today_local
        else:
            today_remote = 'wanghuan@155.33.198.138:/home/wanghuan/Projects/ExpLogs/%s' % today_local
        try:
            script_pull = 'scp %s .' % today_remote
            os.system(script_pull)
        except:
            pass
        with open(today_local, 'a+') as f:
            f.write(self.ExpNote + '\n')
        script_push = 'scp %s %s' % (today_local, today_remote)
        os.system(script_push)
        os.remove(today_local)

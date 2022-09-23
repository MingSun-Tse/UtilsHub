import sys, os, shutil
import subprocess
import time
import argparse
from fnmatch import fnmatch

#################
r"""Usage
example: python ../UtilsHub/experimenting/auto_rm_experiments.py Experiments/
"""
#################

parser = argparse.ArgumentParser()
parser.add_argument('Experiments', type=str)
parser.add_argument('--ignore', type=str, default=None)
args = parser.parse_args()
ignore = args.ignore.split(',') if args.ignore is not None else []

def remove_exp(exp_path):
    trash_dir = f'{args.Experiments}/Trash'
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)
    script = f'mv {exp_path} {trash_dir}'
    os.system(script)
    return trash_dir


def run_shell_command(cmd, inarg=None):
    r"""Run shell command and return the output (string) in a list
    """
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip().split('\n')

# Get all exps
exps = []
for d in os.listdir(args.Experiments):
    d_path = os.path.join(args.Experiments, d)
    if os.path.isdir(d_path):
        ign = [fnmatch(d, i) for i in ignore]
        if not any(ign):
            exps += [ d_path ]

ERRORS = ['RuntimeError', 'FileNotFoundError', 'AssertionError', 'TypeError', 'NameError', 'KeyError', 
    'OSError', 'bdb.BdbQuit', 'DGLError', 'UnicodeDecodeError', '] *************************************']

for e in exps:
    log_path = os.path.join(e, 'log', 'log.txt')
    if os.path.exists(log_path):
        last_line = run_shell_command(f'tail -n 1 {log_path}')[0]
        # print(last_line)
        cond = [err in last_line for err in ERRORS]

        # CTRL+C
        if last_line.startswith('KeyboardInterrupt'):
            if len(os.listdir(os.path.join(e, 'weights'))) == 0:
                cond += [True]

        remove = False
        if any(cond):
            remove = True
        if remove:
            # shutil.rmtree(e)
            remove_exp(e)
            print(f'==> Rm experiment "{e}", its last line is "{last_line}"')
            continue

        # Too few lines
        lines = open(log_path).readlines()
        thresh = 50
        if len(lines) < thresh:
            remove_exp(e)
            print(f'==> Rm experiment "{e}", too few lines (#lines={len(lines)} < thresh={thresh})')
            continue

        # Too short time (<2min)
        timestr = '-'.join(log_path.split('_SERVER')[1].split('/')[0].split('-')[1:])
        start_time = time.strptime(timestr, '%Y%m%d-%H%M%S')
        start_time = time.mktime(start_time)
        last_modify_time = os.stat(log_path).st_mtime
        duration = last_modify_time - start_time
        thresh = 120
        if duration < thresh:
            remove_exp(e)
            print(f'==> Rm experiment "{e}", too short time (during={duration:.2f}s < thresh={thresh}s)')
            continue
import sys, os, shutil
import subprocess
import time
import argparse
from fnmatch import fnmatch
sys.path.insert(0, '.')
from smilelogging.utils import get_script_from_log

#################
r"""Usage
example: python ../UtilsHub/experimenting/auto_rm_experiments.py Experiments/
"""
#################

parser = argparse.ArgumentParser()
parser.add_argument('Experiments', type=str)
parser.add_argument('--ignore', type=str, default=None)
parser.add_argument('--script_wrong_mark', type=str, default=None)
args = parser.parse_args()
ignore = args.ignore.split(',') if args.ignore is not None else []

def remove_exp(exp_path):
    trash_dir = f'{args.Experiments}/Trash'
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)
    script = f'mv {exp_path} {trash_dir}'
    os.system(script)
    return trash_dir

def legal_exp_full_name(name):
    
    
    

def run_shell_command(cmd, inarg=None):
    r"""Run shell command and return the output (string) in a list
    """
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip().split('\n')


# Rm debug and test experiments
os.system(f'rm -rf {args.Experiments}/Debug*')
os.system(f'rm -rf {args.Experiments}/TEST*')
os.system(f'rm -rf {args.Experiments}/_SERVER*')

# Get all exps
exps = []
for d in os.listdir(args.Experiments):
    if '_SERVER'
    d_path = os.path.join(args.Experiments, d)
    if os.path.isdir(d_path):
        ign = [fnmatch(d, i) for i in ignore]
        if not any(ign):
            exps += [ d_path ]


self_defined_errors = ['bdb.BdbQuit', '] *************************************']
for e in exps:
    log_path = os.path.join(e, 'log', 'log.txt')
    if not os.path.exists(log_path):
        remove_exp(e)
        print(f'==> Rm experiment "{e}", its log txt is missing')
        continue
        
    last_line = run_shell_command(f'tail -n 1 {log_path}')[0]
    err = False
    if 'Error: ' in last_line:
        err = last_line.split(':')[0]
        if fnmatch(err, '*Error'):
            err = True
    if err or any([x in last_line for x in self_defined_errors]):
        # shutil.rmtree(e)
        remove_exp(e)
        print(f'==> Rm experiment "{e}", its last line has error: "{last_line}"')
        continue

    # # CTRL+C
    # if last_line.startswith('KeyboardInterrupt'):
    #     if len(os.listdir(os.path.join(e, 'weights'))) == 0:
    #         conds += [True]

    # Script conditions (some experiments are run by mistake)
    if args.script_wrong_mark is not None:
        wrong_mark = args.script_wrong_mark.split(',')
        script = get_script_from_log(log_path)
        if script is not None:
            if True in [m in script for m in wrong_mark]:
                remove_exp(e)
                print(f'==> Rm experiment "{e}", its script is wrong: "{script}"')
                continue

    # Too few lines
    lines = open(log_path).readlines()
    thresh = 50
    if len(lines) < thresh:
        remove_exp(e)
        print(f'==> Rm experiment "{e}", too few lines (#lines={len(lines)} < thresh={thresh})')
        continue

    # Too short time (<2min)
    try:
        timestr = '-'.join(log_path.split('_SERVER')[1].split('/')[0].split('-')[1:])
    except:
        print(f'Sth wrong, log_path: {log_path}')
        exit(0)
    start_time = time.strptime(timestr, '%Y%m%d-%H%M%S')
    start_time = time.mktime(start_time)
    last_modify_time = os.stat(log_path).st_mtime
    duration = last_modify_time - start_time
    thresh = 120
    if duration < thresh:
        remove_exp(e)
        print(f'==> Rm experiment "{e}", too short time (during={duration:.2f}s < thresh={thresh}s)')
        continue
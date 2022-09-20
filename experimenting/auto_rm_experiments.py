import sys, os, shutil
import subprocess

#################
in_dir = sys.argv[1]
#################

def run_shell_command(cmd, inarg=None):
    r"""Run shell command and return the output (string) in a list
    """
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip().split('\n')

exps = [os.path.join(in_dir, d) for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
errors = ['FileNotFoundError', 'AssertionError', 'TypeError', 'NameError', 'KeyError', 
    'OSError', 'bdb.BdbQuit', 'DGLError']

for e in exps:
    log_path = os.path.join(e, 'log', 'log.txt')
    if os.path.exists(log_path):
        last_line = run_shell_command(f'tail -n 1 {log_path}')[0]
        # print(last_line)
        cond = [err in last_line for err in errors]

        # CTRL+C
        if last_line.startswith('KeyboardInterrupt'):
            if len(os.listdir(os.path.join(e, 'weights'))) == 0:
                cond += [True]

        remove = False
        if any(cond):
            remove = True
        if remove:
            shutil.rmtree(e)
            print(f'==> Rm experiment "{e}", its last line is "{last_line}"')

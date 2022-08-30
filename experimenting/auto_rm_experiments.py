import sys, os, shutil

#################
in_dir = sys.argv[1]
#################

exps = [os.path.join(in_dir, d) for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
errors = ['FileNotFoundError', 'AssertionError']

for e in exps:
    log_path = os.path.join(e, 'log', 'log.txt')
    if os.path.exists(log_path):
        lines = open(log_path).readlines()
        cond = [err in lines[-1] for err in errors]
        remove = False
        if any(cond):
            remove = True
        if remove:
            shutil.rmtree(e)
            print(f'==> Rm experiment "{e}", its last line is "{lines[-1]}"')

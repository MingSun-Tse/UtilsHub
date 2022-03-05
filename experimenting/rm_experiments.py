import sys, os
import shutil
"""Usage: python <this_file> <path_to_exp

Example: python ../UtilsHub/experimenting/rm_experiments.py Test 130832,132945 
"""
exp_dir = sys.argv[1]

# Init
matched = {}
for k in sys.argv[2].split(','):
    matched[k] = []

# Matching
all_dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(f'{exp_dir}/{d}')]
# print(f'All dirs: {all_dirs}')

candidates = []
for d in all_dirs:
    for k in matched:
        if k in d:
            matched[k] += [d]
            candidates += [d]
print(f'Candidates to remove: {candidates}')

# Remove
for k, v in matched.items():
    if len(v) == 1:
        shutil.rmtree(f'{exp_dir}/{v[0]}')
        print(f'==> Rm folder "{exp_dir}/{v[0]}"')
    elif len(v) > 1:
        print(f'==> Keyword "{k}" matches multiple folders: {v}. Please check!')




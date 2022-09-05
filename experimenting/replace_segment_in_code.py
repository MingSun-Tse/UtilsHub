r"""Usage:
python  <this_file>  <code_folder>  AAA:BBB  # In <code_folder>, replace AAA with BBB

Example:
python ../UtilsHub/experimenting/replace_segment_in_code.py
"""


import os, sys, numpy as np, shutil as sh
pjoin = os.path.join
# ----------------- Args -----------------
from smilelogging import argparser as parser
parser.add_argument('--code_dir', type=str, default='./')
parser.add_argument('--replace_dict', type=str, default='mst:none,mingsun-tse:none,MingSun-Tse:none,huan:none,wang:none,cool:none')
parser.add_argument('--total_dir_levels', type=int, default=10)
args = parser.parse_args()
# ----------------------------------------

def strdict_to_dict(sstr, ttype=float):
    r"""Example: '{"1": 0.04, "2": 0.04, "4": 0.03, "5": 0.02, "7": 0.03}'
    """
    if not sstr:
        return sstr
    out = {}
    sstr = sstr.strip()
    if sstr.startswith('{') and sstr.endswith('}'):
        sstr = sstr[1:-1]
    
    sep = ','
    if '/' in sstr:
        sep = '/'
    elif ';' in sstr:
        sep = ';'
    for x in sstr.split(sep):
        x = x.strip()
        if x:
            k = x.split(':')[0] # note: key is always str 
            if k.startswith("'"): k = k.strip("'") # remove ' '
            if k.startswith('"'): k = k.strip('"') # remove " "
            v = ttype(x.split(':')[1].strip())
            out[k] = v
    return out

def edit_one_file(f_path, src_seg, tgt_seg):
    script = f"sed -i 's/{src_seg}/{tgt_seg}/g' {f_path}"
    os.system(script)

# Pre
script = 'rm scripts tools utils.py && cp ../UtilsHub/utils.py .'
os.system(script)

# Run
guide_f = 'replace_guide.txt'
replace_dict = strdict_to_dict(args.replace_dict, ttype=str)
for k, v in replace_dict.items():
    for level in range(args.total_dir_levels):
        script = f"grep -nHR {k} .{'/*' * level}/*.py > {guide_f}"
        os.system(script)
        if not open(guide_f).read():
            os.remove(guide_f)
            break
        processed_files = []
        for line in open(guide_f):
            f, line_number = line.split(':')[0], line.split(':')[1]
            if f not in processed_files:
                ######### Core func #########
                edit_one_file(f, k, v)
                processed_files += [f]
                #############################
                print(f'Finished processing file "{f}": {k}:{v}')
        os.remove(guide_f)


import sys, os, glob


def strdict_to_dict(sstr, ttype):
    '''
        '{"1": 0.04, "2": 0.04, "4": 0.03, "5": 0.02, "7": 0.03, }'
    '''
    out = {}
    if '{' in sstr:
        sstr = sstr.split("{")[1].split("}")[0]
    else:
        sstr = sstr.strip()
    for x in sstr.split(','):
        x = x.strip()
        if x:
            k = x.split(':')[0]
            v = ttype(x.split(':')[1].strip())
            out[k] = v
    return out

def modify_txt(file, old_project_name, new_project_name):
    if not os.path.exists(file):
        return
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_project_name in line:
                line = line.replace(old_project_name, new_project_name)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


'''Usage:
python change_log_project_name.py 20210207 {tinyimagenet:cifar100}
'''
# ----------------------------
# args
kw = sys.argv[1] # to select experiment folders
replace_dict = strdict_to_dict(sys.argv[2], str)
print(replace_dict)
# ----------------------------

exps = glob.glob('Experiments/*%s*' % kw)
cnt, cnt_all = 0, len(exps)
for exp in exps:
    for k, v in replace_dict.items():
        new_exp = exp.replace(k, v) # 'exp' example: Experiments/kd__vgg13vgg8__tinyimagenet__cutmix-pick__save_entropy_log__ceiling0.02_SERVER115-20210208-101647
        
        # chaneg folder name
        os.rename(exp, new_exp)

        # change in the log txt
        logtxt = os.path.join(new_exp, 'log/log.txt')
        old_project_name = exp.split('Experiments/')[1].split('_SERVER')[0] # old_project_name example: kd__vgg13vgg8__tinyimagenet__cutmix-pick__save_entropy_log__ceiling0.02
        new_project_name = new_exp.split('Experiments/')[1].split('_SERVER')[0]
        modify_txt(logtxt, old_project_name, new_project_name)
        
        exp = new_exp
    cnt += 1
    print('[%d/%d] change project name done: "%s" -> "%s"' % (cnt, cnt_all, old_project_name, new_project_name))

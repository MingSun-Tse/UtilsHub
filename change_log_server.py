import sys
import os


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

def modify_txt(file, old_str1, new_str1, old_str2, new_str2):
    if not os.path.exists(file):
        return
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str1 in line:
                line = line.replace(old_str1, new_str1)
            if old_str2 in line:
                line = line.replace(old_str2, new_str2)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


'''Usage:
python change_log_server.py Experiments "{005:008, 5:008, 138:008, 120:008, cluster:008, 115:008}"
'''
# ----------------------------
# args
inDir = sys.argv[1]
server_dict = strdict_to_dict(sys.argv[2], str)
print(server_dict)
# ----------------------------

exps = [os.path.join(inDir, x) for x in os.listdir(inDir) if 'SERVER' in x]
cnt = 0
for exp in exps:
    old_server_id = exp.split('SERVER')[1].split('-')[0]
    if old_server_id in server_dict: 
        new_server_id = server_dict[old_server_id]
        new_exp = exp.replace('SERVER%s-' % old_server_id, 'SERVER%s-' % new_server_id)
        
        # chaneg folder name
        os.rename(exp, new_exp)

        # change server id in the log txt
        logtxt = os.path.join(new_exp, 'log/log.txt')
        old_str1 = 'SERVER%s-' % old_server_id
        new_str1 = 'SERVER%s-' % new_server_id
        old_str2 = 'ExpNote [%s-' % old_server_id
        new_str2 = 'ExpNote [%s-' % new_server_id
        modify_txt(logtxt, old_str1, new_str1, old_str2, new_str2)

        cnt += 1
        print('[%d] change server id done' % (cnt))
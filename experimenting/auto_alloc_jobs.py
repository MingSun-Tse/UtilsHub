import os, sys, time, copy
import argparse
import functools
print = functools.partial(print, flush=True)

EXPS = ['kd__wrn_40_2wrn_16_2__tinyimagenet__flip', 'kd__resnet56ShuffleV2__tinyimagenet__cropflip', 'kd__wrn_40_2wrn_16_2__tinyimagenet__cutout', 'kd__wrn_40_2vgg8__tinyimagenet__cutout', 'kd__wrn_40_2wrn_16_2__tinyimagenet__autoaugment', 'kd__wrn_40_2vgg8__tinyimagenet__autoaugment', 'kd__wrn_40_2vgg8__tinyimagenet__autoaugment', 'kd__resnet56ShuffleV2__tinyimagenet__autoaugment', 'kd__wrn_40_2wrn_16_2__tinyimagenet__mixup', 'kd__wrn_40_2vgg8__tinyimagenet__mixup', 'kd__wrn_40_2vgg8__tinyimagenet__cutmix', 'kd__resnet56ShuffleV2__tinyimagenet__cutmix', 'kd__wrn_40_2wrn_16_2__tinyimagenet__cutmix_pick_Sentropy', 'kd__wrn_40_2wrn_16_2__tinyimagenet__cutmix_pick_Sentropy', 'kd__wrn_40_2vgg8__tinyimagenet__cutmix_pick_Sentropy', 'kd__resnet56ShuffleV2__tinyimagenet__cutmix_pick_Sentropy', 'kd__resnet56ShuffleV2__tinyimagenet__cutmix_pick_Sentropy', 'kd__wrn_40_2wrn_16_2__tinyimagenet__cutmix_pick_Tentropy', 'kd__wrn_40_2vgg8__tinyimagenet__cutmix_pick_Tentropy', 'kd__wrn_40_2vgg8__tinyimagenet__cutmix_pick_Tentropy', 'kd__resnet56ShuffleV2__tinyimagenet__cutmix_pick_Tentropy', 'kd__resnet56ShuffleV2__tinyimagenet__cutmix_pick_Tentropy', 'kd__resnet32x4ShuffleV2__cifar100__cutmix_pick_Sentropy', 'kd__vgg13vgg8__tinyimagenet__cutout', 'kd__vgg13vgg8__tinyimagenet__cutout', 'kd__vgg13vgg8__tinyimagenet__cutout', 'kd__vgg13MobileNetV2__tinyimagenet__cutout', 'kd__vgg13MobileNetV2__tinyimagenet__cutout', 'kd__vgg13MobileNetV2__tinyimagenet__cutout', 'kd__resnet32x4resnet8x4__tinyimagenet__cutout', 'kd__ResNet50vgg8__tinyimagenet__cutout', 'kd__resnet32x4resnet8x4__tinyimagenet__autoaugment', 'kd__ResNet50vgg8__tinyimagenet__autoaugment', 'kd__resnet32x4resnet8x4__tinyimagenet__mixup', 'kd__resnet32x4ShuffleV2__tinyimagenet__mixup', 'kd__ResNet50vgg8__tinyimagenet__mixup', 'kd__resnet32x4resnet8x4__tinyimagenet__cutmix_pick_Sentropy', 'kd__resnet32x4resnet8x4__tinyimagenet__cutmix_pick_Sentropy', 'kd__resnet32x4ShuffleV2__tinyimagenet__cutmix_pick_Sentropy', 'kd__resnet32x4ShuffleV2__tinyimagenet__cutmix_pick_Sentropy', 'kd__ResNet50vgg8__tinyimagenet__cutmix_pick_Sentropy']

def get_exp_name_id(exp_path):
    r"""arg example: Experiments/kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
            or kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
            or Experiments/kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318/weights/ckpt.pth
    """
    seps = exp_path.split(os.sep)
    for s in seps:
        if '_SERVER' in s:
            exp_id = s.split('-')[-1]
            assert exp_id.isdigit()
            ExpID = 'SERVER' + s.split('_SERVER')[1]
            exp_name = s.split('_SERVER')[0]
            date = s.split('-')[-2]
            return ExpID, exp_id, exp_name, date

def replace_var(line, var_dict):
    """This function is to replace the variables in shell script.
    """
    line = line.strip()
    line_copy = copy.deepcopy(line)
    max_len = len(line)
    for i in range(max_len):
        if line[i] == '$': # example: ../save/models/${T}_vanilla/$CKPT
            for start in range(i + 1, max_len):
                if line[start].isalpha():
                    break
            got_end = False
            for end in range(start + 1, max_len):
                if (not line[end].isalpha()) and (not line[end].isdigit()):
                    got_end = True
                    break
            if got_end:     
                var_name = line[start: end]
            else:
                var_name = line[start:] # special case: the $VAR is at the end of the line
            real_var = var_dict[var_name]
            
            if line[i + 1] == '{':
                var_name = '${%s}' % var_name
            else:
                var_name = '$%s' % var_name
            line_copy = line_copy.replace(var_name, real_var)
    return line_copy

def standard_sep(lines):
    return [' '.join(line.split()) for line in lines]

def remove_comments(lines):
    r"""remove # comments in script files
    """
    out = []
    for line in lines:
        if '#' in line:
            index = line.find('#') # get the index of the first #
            line = line[:index]
        out += [line]
    return out

def remove_CUDA(lines):
    r"""Remove CUDA_VISIBLE_DEVICES
    """
    out = []
    for line in lines:
        if line.startswith('CUDA_VISIBLE_DEVICES='):
            line = line.split()[1:]
            line = ' '.join(line)
            out += [line]
    return out

def remove_nohup(lines):
    out = []
    for line in lines:
        if line.startswith('nohup ') and ' > /dev/null' in line:
            line = line.split('nohup ')[1].split(' > /dev/null')[0]
            out += [line]
    return out

def is_ignore(line):
    ignore = False
    for x in args.ignore:
        if len(x) and x in line:
            ignore = True
    for x in args.include:
        if len(x) and x not in line:
            ignore = True
    return ignore

def strftime():
    return time.strftime("%Y/%m/%d-%H:%M:%S")

def query_hub(script, userip='wanghuan@155.33.198.138'):
    exp_name = script.split('--project ')[1].strip().split()[0]
    exp_name = exp_name.split('_SERVER')[0]
    project = os.getcwd().split('/')[-1]
    # os.system(f'echo Y | ssh {userip}')
    script = f'echo Y | sshpass -p 8 ssh {userip} "ls $HOME/Projects/{project}/Experiments | grep {exp_name}_SERVER | wc -l" > tmp.txt'
    os.system(script)
    cnt = open('tmp.txt').readline().strip()
    # print(f'exp_name: {exp_name}, Hub cnt: {int(cnt)}')
    os.remove('tmp.txt')
    return  int(cnt)

class JobManager():
    def __init__(self, script_f):
        jobs, var_dict = [], {}
        lines = [x.strip() for x in open(script_f) if x.strip()]
        
        # Preprocessing
        lines = standard_sep(lines)
        lines = remove_comments(lines)
        lines = remove_CUDA(lines)
        lines = remove_nohup(lines)
        
        for line in lines:
            # if '=' in line and (not line.startswith('python')) :
            #     k, v = line.split('=')
            #     if v[0] == '"' and v[-1] == '"': # example: T="vgg13"
            #         v = v[1:-1]
            #     if '$' in v:
            #         var_dict[k] = replace_var(v, var_dict)
            #     else:
            #         var_dict[k] = v
            
            if ' ==> ' in line:
                line = line.split(' ==> ')[1].strip()

            # Collect jobs
            if 'python ' in line or 'sh ' in line:
                new_line = replace_var(line, var_dict)
                if not is_ignore(new_line):
                    if args.predefined_exps:
                        exp_name = new_line.split('--project ')[1].strip().split()[0]
                        exp_name = exp_name.split('_SERVER')[0]
                        if exp_name in EXPS:
                            how_many = len([x for x in EXPS if x == exp_name])
                            jobs += [new_line] * how_many
                            print(f'[{strftime()}] {len(jobs)} Got a job: "{new_line}". Repeated by {how_many} times')
                    else:
                        jobs.append(new_line)
                        print(f'[{strftime()}] {len(jobs)} Got a job: "{new_line}"')
        
        repeated_jobs = jobs * args.times # repeat
        print(f'[{strftime()}] Jobs will be repeated by {args.times} times. Expected #total_jobs: {len(repeated_jobs)}')

        # Filter by querying the hub
        for j in jobs:
            cnt = query_hub(j)
            left = 0 if cnt >= args.times else args.times - cnt
            for _ in range(args.times - left):
                repeated_jobs.remove(j)
                print(f'[{strftime()}] Remove job "{j}". Left cnt for this job: {left}. Now #total_jobs: {len(repeated_jobs)}')
        jobs = repeated_jobs

        self.jobs_txt = script_f.replace('.sh', '.txt')
        with open(self.jobs_txt, 'a+') as f:
            for ix, j in enumerate(jobs):
                f.write('%s ==> %s\n\n' % (ix, j))
        print(f'[{strftime()}] Save jobs to {self.jobs_txt}')

    def read_jobs(self):
        jobs = []
        for line in open(self.jobs_txt):
            line = line.strip()
            if line and not (line.startswith('[Done') or line.startswith('#')): # TODO: improve end mark
                jobs.append(line)
        return jobs
    
    def update_jobs_txt(self, job, gpu):
        new_txt = ''
        for line in open(self.jobs_txt):
            line = line.strip()
            if line == job:
                line = f'# [Done, GPU={gpu}] ' + line
            new_txt += line + '\n'
        with open(self.jobs_txt, 'w') as f:
            f.write(new_txt)

    def get_free_GPU_once(self):
        free_gpus, busy_gpus = [], []
        get_gpu_successully = False
        while not get_gpu_successully:
            f = '.wh_GPU_status_%s.txt' % time.time()
            os.system('nvidia-smi > %s' % f)
            lines = open(f).readlines()
            os.remove(f)
            get_gpu_successully = True
            for i in range(len(lines)) :
                line = lines[i].strip()
                
                # get free gpus by utility 
                if 'MiB /' in line: # example: | 41%   31C    P8     4W / 260W |      1MiB / 11019MiB |      76%      Default |
                    volatile = line.split('%')[-2].split()[-1]
                    memory = line.split(' | ')[1].split('MiB')[0].strip()
                    if volatile.isdigit() and memory.isdigit():
                        volatile = float(volatile) / 100.
                        memory = float(memory)
                        if volatile < 0.05 and memory < 300: # This condition is empirical. May be improved.
                            gpu_id = lines[i - 1].split()[1] # example: |   1  GeForce RTX 208...  Off  | 00000000:02:00.0 Off |                  N/A |
                            free_gpus.append(gpu_id)
                            print(f'Found free gpu {gpu_id}')
                    else: # the log may be broken, access it again
                        print(line)
                        get_gpu_successully = False
                        print('Trying to get free GPUs: nvidia-smi log may be broken, access it agian')
                        break
                
                # get busy gpus
                if line.endswith('MiB |'): # example: |    0     18134      C   python                                      2939MiB |
                    gpu_id = line.split()[1]
                    program = line.split()[4]
                    if program in ['python', 'python2', 'python3', 'matlab']:
                        busy_gpus.append(gpu_id)
                        print(f'Found busy gpu {gpu_id}')
        return [x for x in free_gpus if x not in busy_gpus]
    
    def get_free_GPU(self):
        """run 3 times to get a stable result
        """
        unavailable_gpus = args.unavailable_gpus.split(',')
        free_gpus_1 = self.get_free_GPU_once()
        free_gpus_2 = self.get_free_GPU_once()
        free_gpus_3 = self.get_free_GPU_once()
        free_gpus = [x for x in free_gpus_1 if x in free_gpus_2 and x in free_gpus_3 and (x not in unavailable_gpus)]
        return free_gpus
    
    def run(self):
        while 1:
            jobs = self.read_jobs()
            n_job = len(jobs)
            if n_job == 0:
                print(f'{strftime()} ==> All jobs have been executed. Congrats!')
                os.remove(self.jobs_txt)
                exit(0)

            # find a gpu to run the job
            job = jobs[0]
            while 1:
                free_gpus = self.get_free_GPU()
                current_time = strftime()
                if len(free_gpus) > 0:
                    gpu = free_gpus[0]
                    core_script = job.split('==>')[1].strip()
                    if args.debug:
                        new_script = 'CUDA_VISIBLE_DEVICES=%s %s --debug' % (gpu, core_script)
                    else:
                        new_script = 'CUDA_VISIBLE_DEVICES=%s nohup %s > /dev/null 2>&1 &' % (gpu, core_script)
                    os.system(new_script)
                    print('[%s] ==> Found free GPUs: %s' % (current_time, ' '.join(free_gpus)))
                    print('[%s] ==> Run the job on GPU %s: [%s] %d jobs left' % (current_time, gpu, new_script, n_job - 1))
                    time.sleep(20) # wait for 20 seconds so that the GPU is fully activated. This is decided by experience, may be improved.
                    break
                else:
                    print('[%s] ==> No free GPUs right now. Wait for another 60 seconds. %d jobs left.' % (current_time, n_job))
                    time.sleep(60)
                
            # after the job is run successfully, update jobs txt
            self.update_jobs_txt(job, gpu)

r"""Usage: python auto_alloc_jobs.py script.sh
"""
parser = argparse.ArgumentParser()
parser.add_argument('--script', type=str, required=True)
parser.add_argument('--times', type=int, default=1, help='each experiment will be run by <times> times')
parser.add_argument('--ignore', type=str, default='', help='ignore scripts that are not expected to run. separated by comma. example: wrn,resnet56')
parser.add_argument('--include', type=str, default='', help='include scripts that are not expected to run. separated by comma. example: wrn,resnet56')
parser.add_argument('--unavailable_gpus', type=str, default=',', help='gpus that are unavailable')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--predefined_exps', action='store_true')
parser.add_argument('--hold', action='store_true')
args = parser.parse_args()
def main():
    args.ignore = args.ignore.split(',')
    args.include = args.include.split(',')
    job_manager = JobManager(args.script)
    job_manager.run()
    if args.hold:
        time.sleep(36000000)

if __name__ == '__main__':
    main()




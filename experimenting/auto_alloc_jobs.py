import os, sys, time, copy
import argparse
import functools
print = functools.partial(print, flush=True)

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

def remove_comments(f):
    """remove # comments in script files
    """
    lines = []
    for line in open(f, 'r'):
        if '#' in line:
            index = line.find('#') # get the index of the first #
            line = line[:index]
        lines.append(line)
    return lines

def is_ignore(line):
    ignore = False
    for x in args.ignore:
        if len(x) and x in line:
            ignore = True
    return ignore

def strftime():
    return time.strftime("%Y/%m/%d-%H:%M:%S")

class JobManager():
    def __init__(self, script_f):
        jobs, var_dict = [], {}
        lines = remove_comments(script_f)
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            if '=' in line and (not line.startswith('python')) :
                k, v = line.split('=')
                if v[0] == '"' and v[-1] == '"': # example: T="vgg13"
                    v = v[1:-1]
                if '$' in v:
                    var_dict[k] = replace_var(v, var_dict)
                else:
                    var_dict[k] = v
            
            # collect jobs
            if line.startswith('python') or line.startswith('sh'):
                new_line = replace_var(line, var_dict)
                if not is_ignore(new_line):
                    jobs.append(new_line)
                    print(f'[{strftime()}] {len(jobs)} Got a job: "{new_line}"')
                    
        
        jobs = jobs * args.times # repeat
        print(f'[{strftime()}] Jobs will be repeated by {args.times} times.')

        self.jobs_txt = '.auto_run_jobs_%s.txt' % time.strftime("%Y%m%d_%H%M%S")
        with open(self.jobs_txt, 'w') as f:
            for ix, j in enumerate(jobs):
                f.write('%s ==> %s\n\n' % (ix, j))
        print(f'[{strftime()}] Save jobs to {self.jobs_txt}')

        # save a summarized script
        summary = script_f.replace('.sh', '_summary.sh')
        with open(summary, 'w') as f:
            for j in jobs:
                script = f'CUDA_VISIBLE_DEVICES=0 nohup {j} > /dev/null 2>&1 &\n\n'
                f.write(script)
        print(f'[{strftime()}] Save jobs to {summary}')

    def read_jobs(self):
        jobs = []
        for line in open(self.jobs_txt):
            line = line.strip()
            if line and (not line.startswith('[Done')):
                jobs.append(line)
        return jobs
    
    def update_jobs_txt(self, job, gpu):
        new_txt = ''
        for line in open(self.jobs_txt):
            line = line.strip()
            if line == job:
                line = f'[Done, GPU #{gpu}] ' + line
            new_txt += line + '\n'
        with open(self.jobs_txt, 'w') as f:
            f.write(new_txt)

    def get_free_GPU_once(self):
        free_gpus, busy_gpus = [], []
        get_gpu_successully = False
        while not get_gpu_successully:
            f = '.wh_GPU_status_%s.txt' % time.time()
            os.system('nvidia-smi >> %s' % f)
            lines = open(f).readlines()
            os.remove(f)
            get_gpu_successully = True
            for i in range(len(lines)) :
                line = lines[i].strip()
                
                # get free gpus by utility 
                if 'MiB /' in line: # example: | 41%   31C    P8     4W / 260W |      1MiB / 11019MiB |      76%      Default |
                    volatile = line.split('%')[-2].split()[-1]
                    memory = line.split('/')[1].split('|')[1].split('MiB')[0].strip()
                    if volatile.isdigit() and memory.isdigit():
                        volatile = float(volatile) / 100.
                        memory = float(memory)
                        if volatile < 0.05 and memory < 100: # this condition is empirical. May be improved.
                            gpu_id = lines[i - 1].split()[1] # example: |   1  GeForce RTX 208...  Off  | 00000000:02:00.0 Off |                  N/A |
                            free_gpus.append(gpu_id)
                            print(f'found free gpu {gpu_id}')
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
                        print(f'got busy gpu {gpu_id}')
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
parser.add_argument('--unavailable_gpus', type=str, default=',', help='gpus that are unavailable')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
def main():
    args.ignore = args.ignore.split(',')
    job_manager = JobManager(args.script)
    job_manager.run()

if __name__ == '__main__':
    main()




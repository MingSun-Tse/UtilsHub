import os
import sys
import time
import copy

def replace_var(line, dict):
    '''This function is to replace the variables in shell script.
    '''
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
            real_var = dict[var_name]
            
            if line[i + 1] == '{':
                var_name = '${%s}' % var_name
            else:
                var_name = '$%s' % var_name
            line_copy = line_copy.replace(var_name, real_var)
    return line_copy

class JobManager():
    def __init__(self, f):
        self.script_f = f

    def read_jobs(self):
        jobs = []
        dict = {}
        for line in open(self.script_f, 'r'):
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            if '=' in line and (not line.startswith('python')) :
                k, v = line.split('=')
                if v[0] == '"' and v[-1] == '"': # example: T="vgg13"
                    v = v[1:-1]
                if '$' in v:
                    dict[k] = replace_var(v, dict)
                else:
                    dict[k] = v
            if line.startswith('python'):
                new_line = replace_var(line, dict)
                jobs.append(new_line)
        return jobs
        
    def get_vacant_GPU(self):
        f = 'wh_GPU_status_%s.tmp' % time.time()
        os.system('nvidia-smi >> %s' % f)
        lines = open(f).readlines()
        os.remove(f)
        vacant_gpus = []
        for i in range(len(lines)) :
            line = lines[i]
            if 'MiB /' in line: # example: | 41%   31C    P8     4W / 260W |      1MiB / 11019MiB |      0%      Default |
                volatile = float(line.split()[12].split('%')[0]) / 100.
                if volatile < 0.05: # now this is the only condition to determine if a GPU is used or not. May be improved.
                    gpu_id = lines[i - 1].split()[1] # example: |   1  GeForce RTX 208...  Off  | 00000000:02:00.0 Off |                  N/A |
                    vacant_gpus.append(gpu_id)
        return vacant_gpus
    
    def run(self):
        jobs = self.read_jobs()
        n_job = len(jobs)
        n_executed = 0
        for job in jobs:
            while 1:
                vacant_gpus = self.get_vacant_GPU()
                current_time = time.strftime("%Y/%m/%d-%H:%M:%S")
                if len(vacant_gpus) > 0:
                    gpu = vacant_gpus[0]
                    new_script = 'CUDA_VISIBLE_DEVICES=%s nohup %s > /dev/null 2>&1 &' % (gpu, job)
                    os.system(new_script)
                    n_executed += 1
                    print('[%s] ==> Found vacant GPUs: %s' % (current_time, ' '.join(vacant_gpus)))
                    print('[%s] ==> Run job on GPU %s: [%s] %d jobs left.\n' % (current_time, gpu, new_script, n_job - n_executed))
                    time.sleep(10) # wait for 10 seconds so that the GPU is fully activated
                    break
                else:
                    print('[%s] ==> Found no vacant GPUs. Wait for another 60 seconds. %d jobs left.' % (current_time, n_job - n_executed))
                    time.sleep(60)
        print('==> All jobs have been executed. Congrats!')

'''Usage: python auto_alloc_jobs.py script.sh
'''
def main():
    f = sys.argv[1]
    job_manager = JobManager(f)
    job_manager.run()

if __name__ == '__main__':
    main()




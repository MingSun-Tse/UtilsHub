import os
import sys
import time

class JobManager():
    def __init__(self, f):
        self.script_f = f

    def read_jobs(self):
        jobs = []
        for line in open(self.script_f, 'r'):
            line = line.strip()
            if line and (not line.startswith('#')):
                jobs.append(line)
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
        for job in jobs:
            while 1:
                vacant_gpus = self.get_vacant_GPU()
                current_time = time.strftime("%Y/%m/%d-%H:%M:%S")
                if len(vacant_gpus) > 0:
                    gpu = vacant_gpus[0]
                    new_script = 'CUDA_VISIBLE_DEVICES=%s nohup %s > /dev/null 2>&1 &' % (gpu, job)
                    print('[%s] ==> Found vacant GPUs: %s. Run job on GPU %s: "%s"' % (current_time, vacant_gpus, gpu, new_script))
                    os.system(new_script)
                    time.sleep(10) # wait for 10 seconds so that the GPU is fully activated
                    break
                else:
                    print('[%s] ==> Found no vacant GPUs. Wait for another 2 minutes' % current_time)
                    time.sleep(120)
        print('==> All jobs have been executed. Congrats :-)')

'''Usage: python auto_alloc_jobs.py script.sh
'''
def main():
    f = sys.argv[1]
    job_manager = JobManager(f)
    job_manager.run()

if __name__ == '__main__':
    main()




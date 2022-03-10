import sys, os, argparse

"""Usage: python <this_file> <src>
Example: python ../UtilsHub/experimenting/sync_exp_folders_across_servers.py -s wanghuan@155.33.199.5:/home/wanghuan/Projects/Efficient-NeRF/Experiments/nerfv3.2__llff*SERVER115*2022* --only_scp_weights
"""

parser = argparse.ArgumentParser()
parser.add_argument('--src', '-s', type=str, required=True)
parser.add_argument('--target', '-t', type=str, default='Experiments')
parser.add_argument('--pw', type=str)
parser.add_argument('--only_scp_weights', action='store_true')
args = parser.parse_args()

host, folder = args.src.split(':')
print(f'Remote SERVER: {host} | target folder: {folder}')

prefix = f'sshpass -p {args.pw} ' if args.pw else ''

if args.only_scp_weights:
    # Get folders of interest
    logfile = 'scp_folders.tmp'
    script1 = prefix + f'ssh {host} ls {folder} > {logfile}'
    os.system(script1)

    # Parse
    fodlers_of_interest = []
    for line in open(logfile, 'r'):
        line = line.strip()
        if line.endswith(':'): 
            line = line[:-1]
        if '_SERVER' in line: # This is the only condition here to check whether a line is a experiment folder or not
            # TODO [@mst,20220310]: To improve this
            fodlers_of_interest.append(line)
    os.remove(logfile)

    # Scp
    for d in fodlers_of_interest:
        os.makedirs(f'{d}/weights', exist_ok=True)
        script = prefix + f'scp -r {host}:{d}/weights/* {d}/weights'
        os.system(script)
        print(f'==> Only scp weights: "{d}/weights"')
else:
    script = prefix + f'scp -r {args.src} {args.target}'
    os.system(script)

print(f'==> Scp done!')
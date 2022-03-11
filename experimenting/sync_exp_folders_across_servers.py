import sys, os, argparse

"""Usage: python <this_file> <src>
Example: python ../UtilsHub/experimenting/sync_exp_folders_across_servers.py -s wanghuan@155.33.199.5:/home/wanghuan/Projects/Efficient-NeRF/Experiments/nerfv3.2__llff*SERVER115*2022* --only_scp_weights
"""

parser = argparse.ArgumentParser()
parser.add_argument('--src', '-s', type=str, required=True)
parser.add_argument('--target_exp', '-t', type=str, default='Experiments')
parser.add_argument('--pw', type=str)
parser.add_argument('--only_scp_weights', action='store_true')
args = parser.parse_args()

host, src_dir = args.src.split(':')
print(f'Remote SERVER: {host} | src directory: {src_dir}')

prefix = f'sshpass -p {args.pw} ' if args.pw else ''
exp_folder_name = os.path.split(args.target_exp)[-1] # to handle two typical cases: 'Experiments' and '../experiments'

if args.only_scp_weights:
    # Get folders of interest
    logfile = 'scp_dir.tmp'
    script1 = prefix + f'ssh {host} ls {src_dir} > {logfile}'
    os.system(script1)

    # Parse
    remote_folders, local_folders = [], []
    for line in open(logfile, 'r'):
        line = line.strip()
        if line.endswith(':'): 
            line = line[:-1]
        if '_SERVER' in line: # This is the only condition here to check whether a line is a experiment folder or not
            # TODO [@mst,20220310]: To improve this
            remote_folders += [line]
            local_folders += [line.split(f'/{exp_folder_name}/')[1]]
    os.remove(logfile)

    # Scp
    for dr, dl in zip(remote_folders, local_folders):
        weights_dir = f'{args.target_exp}/{dl}/weights'
        os.makedirs(weights_dir, exist_ok=True)
        script = prefix + f'scp -r {host}:{dr}/weights/* {weights_dir}'
        os.system(script)
        print(f'==> Only scp weights: "{weights_dir}"')
else:
    script = prefix + f'scp -r {args.src} {args.target_exp}'
    os.system(script)

print(f'==> Scp done!')
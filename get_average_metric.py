from option import args, all_exps, exp_groups, get_value, replace_value
import numpy as np
import os, sys

num_avg = 5
avg_item = f'{args.metric}Last{num_avg}Avg'

for ix, e in enumerate(all_exps):
    logtxt = os.path.join(e, 'log', 'log.txt')
    if not os.path.exists(logtxt):
        print(f'[{ix+1}/{len(all_exps)}] Skip "{e}" -- not found the log txt')
        continue
    
    new_txt = ''
    all_metric_values = []
    for line in open(logtxt):
        line = line.strip()
        if args.metricline_mark in line and args.metric in line:
            all_metric_values += [ get_value(line, key=args.metric) ]
            avg = np.mean(all_metric_values[-num_avg:])
            if f' {avg_item} ' in line:
                line = replace_value(line, avg_item, f'{avg:.6f}')
            else:
                line = line.replace(f' {args.metric} ', f' {args.metric}Last{num_avg}Avg {avg:.6f} {args.metric} ')
        new_txt += line + '\n'
    with open(logtxt, 'w') as f:
        f.write(new_txt)
    print(f'[{ix+1}/{len(all_exps)}] Finish processing "{logtxt}"')
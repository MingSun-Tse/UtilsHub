import configargparse
from ezlogging.utils import check_path, update_args

parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True, 
                    help='config file path')

# routine arguments for ezlogging
parser.add_argument('--project_name', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--screen_print', action="store_true")
parser.add_argument('--cache_ignore', type=str, default='')
parser.add_argument('--note', type=str, default='')

# trial related
parser.add_argument('--trial.ON', action='store_true')

args = parser.parse_args()

# some default args to keep compatibility
# args.xxx = 0.5

# update args to enable the '--xx.ON' feature
args = update_args(args)
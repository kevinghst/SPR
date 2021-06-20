import argparse
import subprocess
import pdb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--command', required=True, type=str)

args = parser.parse_args()


bashCmd = ['sbatch', 'run_47.sbatch']
bashCmd.append(args.command)

games = [
    'breakout',
    'bank_heist',
    'boxing',
    'frostbite',
    'pong',
    'up_n_down',
    'kangaroo',
    'assault',
    'battle_zone',
    'crazy_climber'
]

for game in games:
    curr_bashCmd = bashCmd.copy()
    curr_bashCmd[-1] = curr_bashCmd[-1] + f' --game {game}'
    process = subprocess.Popen(curr_bashCmd, stdout=subprocess.PIPE)
    output, error = process.communicate()


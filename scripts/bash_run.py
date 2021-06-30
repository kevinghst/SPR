import argparse
import subprocess
import pdb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--command', required=True, type=str)
parser.add_argument('--set', required=True, choices=['dev', 'test'])

args = parser.parse_args()


bashCmd = ['sbatch', 'run_47.sbatch']
bashCmd.append(args.command)

dev_games = [
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

test_games = [
    'alien',
    'amidar',
    'asterix',
    'chopper_command',
    'demon_attack',
    'freeway',
    'gopher',
    'hero',
    'jamesbond',
    'krull',
    'kung_fu_master',
    'ms_pacman',
    'private_eye',
    'qbert',
    'road_runner',
    'seaquest',
]

games = dev_games if args.set == 'dev' else test_games

for game in games:
    curr_bashCmd = bashCmd.copy()
    curr_bashCmd[-1] = curr_bashCmd[-1] + f' --game {game}'
    process = subprocess.Popen(curr_bashCmd, stdout=subprocess.PIPE)
    output, error = process.communicate()


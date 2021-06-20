
"""
Parallel sampler version of Atari DQN.  Increasing the number of parallel
environmnets (sampler batch_B) should improve the efficiency of the forward
pass for action sampling on the GPU.  Using a larger batch size in the algorithm
should improve the efficiency of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
import pdb
from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.utils.logging.context import logger_context

import wandb
import torch
import numpy as np
import copy

from src.models import SPRCatDqnModel
from src.rlpyt_utils import OneToOneSerialEvalCollector, SerialSampler, MinibatchRlEvalWandb
from src.algos import SPRCategoricalDQN
from src.agent import SPRAgent
from src.rlpyt_atari_env import AtariEnv
from src.utils import set_config


def build_and_train(game="pong", run_ID=0, cuda_idx=0, args=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = AtariEnv
    config = set_config(args, game)

    sampler = SerialSampler(
        EnvCls=env,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=config["env"],
        eval_env_kwargs=config["eval_env"],
        batch_T=config['sampler']['batch_T'],
        batch_B=config['sampler']['batch_B'],
        max_decorrelation_steps=0,
        eval_CollectorCls=OneToOneSerialEvalCollector,
        eval_n_envs=config["sampler"]["eval_n_envs"],
        eval_max_steps=config['sampler']['eval_max_steps'],
        eval_max_trajectories=config["sampler"]["eval_max_trajectories"],
    )
    args.discount = config["algo"]["discount"]
    algo = SPRCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"])  # Run with defaults.
    agent = SPRAgent(ModelCls=SPRCatDqnModel, model_kwargs=config["model"], **config["agent"])

    wandb.config.update(config)
    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        affinity=dict(cuda_idx=cuda_idx),
        log_interval_steps=args.n_steps//args.num_logs,
        seed=args.seed,
        final_eval_only=args.final_eval_only,
        skip_init_eval=args.skip_init_eval
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "logs"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='ms_pacman')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--grayscale', type=int, default=1)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--imagesize', type=int, default=84)
    parser.add_argument('--n-steps', type=int, default=100000)
    parser.add_argument('--dqn-hidden-size', type=int, default=256)
    parser.add_argument('--target-update-interval', type=int, default=1)
    parser.add_argument('--target-update-tau', type=float, default=1.)
    parser.add_argument('--momentum-tau', type=float, default=0.01)
    parser.add_argument('--batch-b', type=int, default=1)
    parser.add_argument('--batch-t', type=int, default=1)
    parser.add_argument('--beluga', action="store_true")
    parser.add_argument('--jumps', type=int, default=5)
    parser.add_argument('--num-logs', type=int, default=10)
    parser.add_argument('--renormalize', type=int, default=1)
    parser.add_argument('--renormalize_type', type=str, default='minmax', choices=['minmax', 'ln', 'train_ln'])
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--replay-ratio', type=int, default=64)
    parser.add_argument('--dynamics-blocks', type=int, default=0)
    parser.add_argument('--residual-tm', type=int, default=0.)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--tag', type=str, default='', help='Tag for wandb run.')
    parser.add_argument('--wandb-dir', type=str, default='', help='Directory for wandb files.')
    parser.add_argument('--norm-type', type=str, default='bn', choices=["bn", "ln", "in", "none"], help='Normalization')
    parser.add_argument('--aug-prob', type=float, default=1., help='Probability to apply augmentation')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability in convnet.')
    parser.add_argument('--spr', type=int, default=1)
    parser.add_argument('--distributional', type=int, default=1)
    parser.add_argument('--delta-clip', type=float, default=1., help="Huber Delta")
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--momentum-encoder', type=int, default=1)
    parser.add_argument('--shared-encoder', type=int, default=0)
    parser.add_argument('--local-spr', type=int, default=0)
    parser.add_argument('--global-spr', type=int, default=1)
    parser.add_argument('--noisy-nets', type=int, default=1)
    parser.add_argument('--noisy-nets-std', type=float, default=0.1)
    parser.add_argument('--classifier', type=str, default='q_l1', choices=["mlp", "bilinear", "q_l1", "q_l2", "none"], help='Style of NCE classifier')
    parser.add_argument('--final-classifier', type=str, default='linear', choices=["mlp", "linear", "none"], help='Style of NCE classifier')
    parser.add_argument('--augmentation', type=str, default=["shift", "intensity"], nargs="+",
                        choices=["none", "rrc", "affine", "crop", "blur", "shift", "intensity"],
                        help='Style of augmentation')
    parser.add_argument('--q-l1-type', type=str, default=["value", "advantage"], nargs="+",
                        choices=["noisy", "value", "advantage", "relu"],
                        help='Style of q_l1 projection')
    parser.add_argument('--target-augmentation', type=int, default=1, help='Use augmentation on inputs to target networks')
    parser.add_argument('--eval-augmentation', type=int, default=0, help='Use augmentation on inputs at evaluation time')
    parser.add_argument('--reward-loss-weight', type=float, default=0.)
    parser.add_argument('--model-rl-weight', type=float, default=0.)
    parser.add_argument('--model-spr-weight', type=float, default=5.)
    parser.add_argument('--t0-spr-loss-weight', type=float, default=0.)
    parser.add_argument('--eps-steps', type=int, default=2001)
    parser.add_argument('--min-steps-learn', type=int, default=2000)
    parser.add_argument('--eps-init', type=float, default=1.)
    parser.add_argument('--eps-final', type=float, default=0.)
    parser.add_argument('--final-eval-only', type=int, default=1)
    parser.add_argument('--time-offset', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    parser.add_argument('--skip_init_eval', action='store_true', help='no initial evaluation')

    # model
    parser.add_argument('--encoder_type', type=str, default='conv2d', choices=["conv2d", "resnet18"])
    parser.add_argument('--transition_type', type=str, default='conv2d', choices=["conv2d", "gru"])
    parser.add_argument('--proj_hidden_size', type=int, default=0)
    parser.add_argument('--gru_input_size', type=int, default=0)
    parser.add_argument('--gru_proj_size', type=int, default=0)
    parser.add_argument('--gru_in_dropout', type=float, default=0.)
    parser.add_argument('--gru_out_dropout', type=float, default=0.)
    parser.add_argument('--conv_proj_channel', type=int, default=0)
    parser.add_argument('--ln_ratio', type=int, default=1)
    parser.add_argument('--aug_control', action='store_true')
    parser.add_argument('--latent_dists', type=int, default=0)
    parser.add_argument('--latent_dist_size', type=int, default=0)
    parser.add_argument('--latent_proj_size', type=int, default=600)
    parser.add_argument('--activation', type=str, choices=['relu', 'elu'], default='relu')

    # env related
    parser.add_argument('--repeat_action_probability', type=float, default=0.25)


    # spr related
    parser.add_argument('--pred_hidden_ratio', type=float, default=2.)
    parser.add_argument('--pred_decay', type=float, default=0.)

    # wandb
    parser.add_argument('--disable_log', action='store_true', help='no wandb')
    parser.add_argument('--project', type=str, default="mpr")
    parser.add_argument('--entity', type=str, default="kevinghst")
    parser.add_argument('--public', action='store_true', help='If set, uses anonymous wandb logging')
    parser.add_argument('--group', type=str, default="")

    og_args = parser.parse_args()


    games = og_args.game.split(',')
    games = [x.strip() for x in games]

    for game in games:
        for i in range(og_args.seeds):
            args = copy.deepcopy(og_args)

            seed = i + args.seed_start
            args.seed = seed

            exp_name = f'{game}_{seed}_{args.group}'

            if args.disable_log:
                wandb.init(mode="disabled")
            else:
                if args.public:
                    wandb.init(
                        anonymous="allow",
                        group=args.group,
                        name=exp_name,
                        config=args,
                        tags=[args.tag] if args.tag else None, dir=args.wandb_dir
                    )
                else:
                    wandb.init(
                        project=args.project,
                        group=args.group,
                        entity=args.entity,
                        name=exp_name,
                        config=args,
                        tags=[args.tag] if args.tag else None, dir=args.wandb_dir
                    )
            wandb.config.update(vars(args))
            build_and_train(game=game,
                            cuda_idx=args.cuda_idx,
                            args=args)
            wandb.finish()

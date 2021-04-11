# config json for trial run bank heist using follow command:
# python -m scripts.run --public --game bank_heist --augmentation none --target-augmentation 0 --momentum-tau 0.01 --dropout 0.5

{
    "agent": {
        "eps_eval": 0.001,
        "eps_final": 0.0,
        "eps_init": 1.0
    },
    "algo": {
        "batch_size": 32,
        "clip_grad_norm": 10.0,
        "delta_clip": 1.0,
        "discount": 0.99,
        "distributional": 1,
        "double_dqn": true,
        "eps_steps": 2001,
        "learning_rate": 0.0001,
        "min_steps_learn": 2000,
        "model_rl_weight": 0.0,
        "model_spr_weight": 5.0,
        "n_step_return": 10,
        "pri_alpha": 0.5,
        "pri_beta_steps": 100000,
        "prioritized_replay": 1,
        "replay_ratio": 64,
        "replay_size": 1000000,
        "reward_loss_weight": 0.0,
        "t0_spr_loss_weight": 0.0,
        "target_update_interval": 1,
        "target_update_tau": 1.0,
        "time_offset": 0
    },
    "env": {
        "episodic_lives": true,
        "game": "bank_heist",
        "grayscale": 1, # Correct
        "imagesize": 84,
        "num_img_obs": 4,
        "seed": 0
    },
    "eval_env": {
        "episodic_lives": false,
        "game": "bank_heist",
        "grayscale": 1,
        "horizon": 27000,
        "imagesize": 84,
        "num_img_obs": 4,
        "seed": 0
    },
    "model": {
        "aug_prob": 1.0,
        "augmentation": [
            "none"
        ],
        "classifier": "q_l1",
        "distributional": 1,
        "dqn_hidden_size": 256,
        "dropout": 0.5,
        "dueling": true,
        "dynamics_blocks": 0,
        "eval_augmentation": 0,
        "final_classifier": "linear",
        "global_spr": 1,
        "imagesize": 84,
        "jumps": 5,
        "local_spr": 0,
        "model_rl": 0.0,
        "momentum_encoder": 1,
        "momentum_tau": 0.01,
        "n_atoms": 51,
        "noisy_nets": 1,
        "noisy_nets_std": 0.1,
        "norm_type": "bn",
        "q_l1_type": [
            "value",
            "advantage"
        ],
        "renormalize": 1,
        "residual_tm": 0.0,
        "shared_encoder": 0,
        "spr": 1,
        "target_augmentation": 0,
        "time_offset": 0
    },
    "optim": {
        "eps": 0.00015
    },
    "runner": {
        "log_interval_steps": 1000000.0,
        "n_steps": 50000000.0
    },
    "sampler": {
        "batch_B": 1,
        "batch_T": 1,
        "eval_max_steps": 2800000,
        "eval_max_trajectories": 100,
        "eval_n_envs": 100,
        "max_decorrelation_steps": 1000
    }
}

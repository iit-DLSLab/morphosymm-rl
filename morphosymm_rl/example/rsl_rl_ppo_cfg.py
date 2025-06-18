# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from pathlib import Path
from dataclasses import MISSING



@configclass
class MorphologycalSymmetriesCfg:
    """Configuration for the discriminator network."""

    class_name: str = "MorphologycalSymmetries"

    obs_space_names =  None

    action_space_names = None

    joints_order = None

    history_length = None

    robot_name = None


@configclass
class AliengoFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "aliengo_flat_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSymmEquivariantNN", #ActorCritic, ActorCriticRecurrent, ActorCriticSymmEquivariantNN
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="fixed", #fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


    # Symmetry Related Stuff
    morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names = [
            "base_lin_vel:base",
            "base_ang_vel:base",
            "gravity:base",
            "ctrl_commands",
            "default_qpos_js_error",
            "qvel_js",
            "actions",
            "clock_data",
        ],
        
        action_space_names = ["actions"],
        
        joints_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", 
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ],
        
        history_length = 5,
        
        robot_name = "a1",
    )


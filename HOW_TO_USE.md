# How to Use morphosymm-rl

This guide explains what is inside this repository and how to wire it into an IsaacLab/RSL-RL training setup. The short version is:

1. Install this package in the same Python environment you use for IsaacLab.
2. Replace the standard RSL-RL runner with `SymmOnPolicyRunner`.
3. Add a `morphologycal_symmetries_cfg` block to your training config.
4. Choose a `robot_name`, `joints_order`, observation names, and action names that match your IsaacLab environment exactly.

The key idea is that `morphosymm-rl` does not define the robot symmetries itself. It asks the `morpho_symm` package for the robot's symmetry group and its representations, then uses those representations to augment PPO data or build equivariant/invariant neural networks.

## Repository Layout

```text
morphosymm_rl/
  algorithms/
    ppo.py                       # Local PPO fork with RSL-RL-style symmetry hooks.
    ppo_symm_data_augment.py     # PPO variant that augments rollouts with morphological symmetries.
  modules/
    ac_symm.py                   # Equivariant actor and invariant critic.
    ac_moe*.py                   # Mixture-of-experts actor-critic variants.
    normalizer.py                # Equivariant empirical normalization.
  runners/
    symm_on_policy_runner.py     # IsaacLab/RSL-RL runner that constructs the policy and algorithm.
  symm_utils.py                  # Maps observation/action names to group representations.
README.md                       # Project overview and citations.
pyproject.toml                  # Package metadata and dependencies.
```

This repository is a library layer. It does not currently ship a complete IsaacLab task, environment, or standalone `train.py`. Your IsaacLab project still owns the task definition, observation tensors, reward, command manager, terrain, logging setup, and train/play scripts.

## Installation

From the root of this repository:

```bash
pip install -e .
```

Use the same Python environment that already has IsaacLab, Isaac Sim, and RSL-RL available. The package depends on:

- `rsl-rl-lib==3.3.0`
- `morpho-symm[pin]`
- `symm-learning>=0.2.5`
- `torch`, `torchvision`
- `gym-quadruped`

## How the Pieces Fit Together

At runtime the flow is:

```text
IsaacLab VecEnv
  -> SymmOnPolicyRunner
    -> policy module
      -> regular RSL-RL actor-critic, ActorCriticSymm, or ActorCriticMoE
    -> algorithm
      -> PPO or PPOSymmDataAugmented
    -> symm_utils.configure_observation_space_representations(...)
      -> morpho_symm.load_symmetric_system(robot_name=..., joint_space_order=...)
```

The runner expects a training config with these top-level sections:

```python
train_cfg = {
    "policy": {...},
    "algorithm": {...},
    "obs_groups": {...},
    "num_steps_per_env": ...,
    "save_interval": ...,
    "morphologycal_symmetries_cfg": {...},
    ...
}
```

Keep the key name `morphologycal_symmetries_cfg` exactly as written above. The spelling is part of the current code API.

## Choose a Workflow

There are two main ways to use the package.

### Option 1: PPO Data Augmentation

Use this when you want to keep a mostly standard actor-critic network, but train PPO on both original and symmetrically transformed rollout samples.

Set the algorithm class to:

```python
"algorithm": {
    "class_name": "PPOSymmDataAugmented",
    "num_learning_epochs": ...,
    "num_mini_batches": ...,
    "clip_param": ...,
    "gamma": ...,
    "lam": ...,
    "value_loss_coef": ...,
    "entropy_coef": ...,
    "learning_rate": ...,
    "schedule": ...,
    "desired_kl": ...,
    ...
}
```

`PPOSymmDataAugmented` expands the rollout storage by the number of symmetry group elements and transforms:

- policy observations
- critic observations
- actions
- action means and standard deviations
- values, rewards, dones, and returns where appropriate


### Option 2: Equivariant Actor and Invariant Critic

Use this when you want symmetry constraints directly in the network architecture.

Set the policy class to:

```python
"policy": {
    "class_name": "ActorCriticSymm",
    "actor_hidden_dims": ...,
    "critic_hidden_dims": ...,
    "activation": ...,
    ...
}
```

`ActorCriticSymm` builds:

- an equivariant actor using `symm_learning.models.EMLP`
- an invariant critic using `symm_learning.models.IMLP`
- an equivariant multivariate normal action distribution

In this mode you can usually pair the policy with regular `"PPO"`, because the architecture itself carries the symmetry. You can still experiment with the augmented PPO path, but start with one symmetry mechanism first so debugging is easier.

## Use the Symmetric Runner

See [here](https://github.com/iit-DLSLab/basic-locomotion-isaaclab/tree/main/scripts/morphosymm_rl) for an example.

## Configure Morphological Symmetries

The symmetry config is the most important part:

```python
"morphologycal_symmetries_cfg": {
    "robot_name": "go2",
    "joints_order": [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ],
    "obs_space_names_actor": [
        "base_lin_vel",
        "base_ang_vel",
        "gravity",
        "ctrl_commands",
        "qpos_js",
        "qvel_js",
        "actions",
    ],
    "obs_space_names_critic": [
        "base_lin_vel",
        "base_ang_vel",
        "gravity",
        "ctrl_commands",
        "qpos_js",
        "qvel_js",
        "actions",
    ],
    "action_space_names": ["actions"],
}
```

Treat this config as a contract between your IsaacLab environment and the symmetry code:

- `robot_name` selects the robot symmetry description from `morpho_symm`.
- `joints_order` tells `morpho_symm` how your IsaacLab joint vector is ordered.
- `obs_space_names_actor` must describe the actor observation tensor, in order.
- `obs_space_names_critic` must describe the critic observation tensor, in order.
- `action_space_names` must describe the action tensor, in order.

If the listed names do not match the real tensor order, the code may run but train on incorrect symmetry transforms. This is the most important thing to get right.

## Choose Your Robot

Set `robot_name` to a robot configuration known by [morpho_symm](https://github.com/Danfoa/MorphoSymm). 

To inspect the robot names in your own environment:

```bash
python -c "import morpho_symm, pathlib; cfg = pathlib.Path(morpho_symm.__file__).parent / 'cfg' / 'robot'; print('\n'.join(sorted(p.stem for p in cfg.glob('*.yaml') if p.stem != 'base_robot')))"
```

Pick the exact robot when possible. If your IsaacLab robot is a renamed or slightly modified version of a known robot, only reuse an existing `robot_name` when all of these are true:

- the limbs/joints have the same symmetry structure
- the action vector represents the same joint-space quantities
- the joint names can be mapped with `joints_order`
- the observation features have the same physical meaning

If the robot has a different morphology or different symmetry group, add a new robot YAML to `morpho_symm` instead of forcing the closest existing name.

## Set the Joint Order

`joints_order` must match the order used by your IsaacLab action vector and joint observations. For a quadruped, this is usually the order of the actuated joints in the articulation data.

A good workflow is:

1. Print the joint names from your IsaacLab robot/articulation.
2. Print or inspect the action order used by your action manager.
3. Put that exact order in `joints_order`.
4. Make sure `qpos_js`, `qvel_js`, `actions`, and torque-like observations all use the same joint ordering.

The code passes `joints_order` into:

```python
load_symmetric_system(robot_name=robot_name, return_robot=False, joint_space_order=joints_order)
```

If the names or length do not match the robot description, `morpho_symm` will raise an error.

## Supported Observation and Action Names

`symm_utils.py` maps each name to a representation. These are the currently supported names and name patterns.

### Joint-Space Quantities

```text
qpos_js
qvel_js
tau_ctrl_setpoint
actions
position_gains
velocity_gains
friction_static
friction_dynamic
armature
```

Names containing `qpos_js` use the joint position representation. Names containing `qvel_js` use the joint velocity representation. `actions` and `tau_ctrl_setpoint` use the joint velocity/torque-space representation.

### Base and IMU Quantities

```text
base_pos
base_pos_z
base_lin_vel
base_lin_acc
base_ang_vel
base_ori_euler_xyz
base_ori_SO3
gravity
imu_acc
imu_gyro
```

Quaternion observations are intentionally skipped because this package does not define a left-group action for them:

```text
qpos
base_ori_quat_wxyz
```

Use `base_ori_SO3` or Euler-angle features if you need orientation in a symmetry-aware observation.

### Limb and Contact Quantities

```text
feet_pos
feet_vel
contact_state
contact_forces
clock_data
```

Names containing `feet_pos`, `feet_vel`, or `contact_forces` are treated as limb-indexed vector quantities. `contact_state` and `clock_data` are treated as limb permutations.

### Command Quantities

```text
ctrl_commands
```

`ctrl_commands` is represented as planar linear command plus yaw-rate command.

### Heightmaps

Heightmaps must include their flattened grid size in the name:

```text
heightmap:<rows>x<cols>
```

Example:

```text
heightmap:64x32
```

The heightmap transform assumes a signed-permutation action on the horizontal axes. Axis swaps require compatible row and column coordinate sets.

## Match Names to Tensor Dimensions

The observation names are not just labels. Their representation sizes must add up to the actual tensor size.

For example, if the actor observation is:

```text
[base_lin_vel, base_ang_vel, gravity, ctrl_commands, qpos_js, qvel_js, actions]
```

then `obs_space_names_actor` must use the same order:

```python
"obs_space_names_actor": [
    "base_lin_vel",
    "base_ang_vel",
    "gravity",
    "ctrl_commands",
    "qpos_js",
    "qvel_js",
    "actions",
]
```

If your actor and critic observations differ, make `obs_space_names_actor` and `obs_space_names_critic` different. Do not copy the actor list into the critic config unless the critic tensor is actually identical.

## Minimal Config Skeleton


Depending on your RSL-RL/IsaacLab config loader, the regular actor-critic class name may already be expressed differently. The important part for this package is that `SymmOnPolicyRunner` sees:

- `algorithm["class_name"] == "PPOSymmDataAugmented"` for rollout data augmentation
- `policy["class_name"] == "ActorCriticSymm"` for the equivariant actor/invariant critic path

## Checklist for a New Robot Task

1. Confirm the robot exists in `morpho_symm`, or add a new robot description there.
2. Confirm the IsaacLab action order.
3. Set `joints_order` to that exact order.
4. Write down the actor observation tensor as a list of physical blocks.
5. Translate each block to one of the supported observation names.
6. Do the same for critic observations.
7. Set `action_space_names`; for joint actions this is usually `["actions"]`.
8. Run one short training job with a small number of environments.
9. If dimensions fail, compare the summed representation sizes against the actual observation/action tensor sizes.
10. Once the config is correct, scale up training.

## Common Errors

### `ValueError: Invalid observation name`

One of the names in `obs_space_names_actor`, `obs_space_names_critic`, or `action_space_names` is not supported by `symm_utils.py`.

Fix it by either:

- renaming the config entry to a supported name, if it has the same physical meaning
- adding a new mapping in `configure_observation_space_representations`

### Robot Config Not Found

If `morpho_symm` cannot load the robot, the `robot_name` does not match a YAML file in `morpho_symm/cfg/robot`.

Use the robot discovery command above and copy the name exactly, including suffixes such as `-c2` or `-k4`.

### Joint Order Mismatch

If `joints_order` has the wrong length or names, `morpho_symm` will fail while loading the symmetric system.

Fix it by printing the IsaacLab joint order and using those exact names. Do not assume that URDF order, action-manager order, and config-file order are automatically the same.

### Observation Size Mismatch

If the first layer or representation size does not match, the observation list probably does not describe the actual flattened tensor.

Check:

- missing observation blocks
- extra observation blocks
- actor and critic lists accidentally swapped
- command dimensions that do not match `ctrl_commands`
- quaternion observations included even though they are skipped
- joint observations using a different joint order than `joints_order`

### Heightmap Errors

Heightmap observations must be named with dimensions, for example:

```text
heightmap:64x32
```

The transform currently supports signed-permutation horizontal actions. More general geometric transforms need a custom representation.

## Practical Advice

Start with the data-augmentation path first. It keeps the network familiar and mostly changes how rollouts are stored and transformed. Once that is stable, try `ActorCriticSymm` if you want architectural equivariance.

When debugging, use a tiny run:

```text
num_envs: small
num_steps_per_env: small
max_iterations: 1 to 5
```

The first goal is not performance. The first goal is proving that robot name, joint order, observation names, and action names agree with each other.

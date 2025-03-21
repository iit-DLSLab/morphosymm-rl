# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 14/03/25
from typing import Sequence

import numpy as np
from escnn.group import Group, Representation


def configure_observation_space_representations(
        robot_name: str, obs_names: Sequence[str], joints_order: list,
        ) -> [Group, dict[str, Representation]]:
    try:
        import morpho_symm
        from morpho_symm.utils.rep_theory_utils import (
            escnn_representation_form_mapping,
            group_rep_from_gens,
            )
        from morpho_symm.utils.robot_utils import load_symmetric_system
    except ImportError as e:
        raise ImportError('morpho_symm package is required to configure observation group representations') from e

    G = load_symmetric_system(robot_name=robot_name, return_robot=False, joint_space_order=joints_order)
    rep_Q_js = G.representations['Q_js']  # Representation on joint space position coordinates
    rep_TqQ_js = G.representations['TqQ_js']  # Representation on joint space velocity coordinates
    rep_Rd = G.representations['R3']  # Representation on vectors in R^d
    rep_Rd_pseudo = G.representations['R3_pseudo']  # Representation on pseudo vectors in R^d
    rep_euler_xyz = G.representations['euler_xyz']  # Representation on Euler angles
    # TODO: Ensure the limb order in the configuration matches the used order by quadruped gym.
    rep_kin_three = G.representations['kin_chain']  # Permutation of legs
    rep_Rd_on_limbs = rep_kin_three.tensor(rep_Rd)  # Representation on signals R^d on the limbs
    rep_Rd_on_limbs.name = 'Rd_on_limbs'
    rep_Rd_pseudo_on_limbs = rep_kin_three.tensor(rep_Rd_pseudo)  # Representation on pseudo vect R^d on the limbs
    rep_Rd_pseudo_on_limbs.name = 'Rd_pseudo_on_limbs'
    rep_SO3_flat = {}
    for h in G.elements:
        rep_SO3_flat[h] = np.kron(rep_Rd(h), rep_Rd(~h).T)
    rep_SO3_flat = escnn_representation_form_mapping(G, rep_SO3_flat)
    rep_SO3_flat.name = 'SO3_flat'

    # Create a representation for the z dimension alone of the base position
    rep_z = escnn_representation_form_mapping(G, {g: rep_Rd(g)[2:3, 2:3] for g in G.elements}, name='base_z')
    rep_roll = escnn_representation_form_mapping(
        G, {g: rep_Rd_pseudo(g)[0:1, 0:1] for g in G.elements}, name='base_roll'
        )
    rep_pitch = escnn_representation_form_mapping(
        G, {g: rep_Rd_pseudo(g)[1:2, 1:2] for g in G.elements}, name='base_pitch'
        )
    rep_yaw = escnn_representation_form_mapping(G, {g: rep_Rd_pseudo(g)[2:3, 2:3] for g in G.elements}, name='base_yaw')

    rep_ctrl_commands_lin = escnn_representation_form_mapping(
        G, {g: rep_Rd(g)[0:2, 0:2] for g in G.elements}, name='ctrl_commands_lin_xy_dot'
        )
    rep_ctrl_commands_ang = escnn_representation_form_mapping(
        G, {g: rep_Rd_pseudo(g)[2:3, 2:3] for g in G.elements}, name='ctrl_commands_yaw_rate'
        )
    

    obs_reps = {}
    for obs_name in obs_names:
        # Generalized position, velocity, and force (torque) spaces
        if obs_name == 'qpos':
            continue  # Quaternion does not have a left-group action definition.
        elif obs_name == 'qvel':
            obs_reps[obs_name] = rep_Rd + rep_Rd_pseudo + rep_TqQ_js  # lin_vel , ang_vel, joint_vel
        elif obs_name in ['tau_ctrl_setpoint', 'actions']:
            obs_reps[obs_name] = rep_TqQ_js
        elif 'qpos_js' in obs_name:  # Joint space position configuration
            obs_reps[obs_name] = rep_Q_js
        elif 'qvel_js' in obs_name:  # Joint space velocity configuration
            obs_reps[obs_name] = rep_TqQ_js
        elif obs_name == 'base_pos':
            obs_reps[obs_name] = rep_Rd
        elif obs_name == 'base_pos_z':
            obs_reps[obs_name] = rep_z
        elif 'base_lin_vel' in obs_name or 'base_lin_acc' in obs_name:  # base_lin_vel / base_lin_vel:base (base frame)
            obs_reps[obs_name] = rep_Rd
        elif 'base_ang_vel' in obs_name:
            obs_reps[obs_name] = rep_Rd_pseudo
        elif 'base_ori_euler_xyz' in obs_name:
            obs_reps[obs_name] = rep_euler_xyz
        elif obs_name == 'base_ori_quat_wxyz':
            continue  # Quaternion does not have a left-group action definition.
        elif obs_name == 'base_ori_SO3':
            obs_reps[obs_name] = rep_SO3_flat
        elif 'feet_pos' in obs_name or 'feet_vel' in obs_name:  # feet_pos:frame := feet_pos:world or feet_pos:base
            obs_reps[obs_name] = rep_Rd_on_limbs
        elif obs_name in ['contact_state', 'clock_data']:
            obs_reps[obs_name] = rep_kin_three
        elif 'contact_forces' in obs_name:
            obs_reps[obs_name] = rep_Rd_on_limbs
        elif 'gravity' in obs_name or 'imu_acc' in obs_names:
            obs_reps[obs_name] = rep_Rd
        elif 'imu_gyro' in obs_names:  # Same as angular velocity
            obs_reps[obs_name] = rep_Rd_pseudo
        elif 'ctrl_commands' in obs_name:
            obs_reps[obs_name] = rep_ctrl_commands_lin + rep_ctrl_commands_ang
        else:
            raise ValueError(f'Invalid observation name: {obs_name}')

    return G, obs_reps

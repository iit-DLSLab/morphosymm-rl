# Add reference to paper

import escnn
import numpy as np
import torch
import torch.nn as nn
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from rsl_rl.modules.actor_critic import ActorCritic
from torch.distributions import Normal

from morphosymm_rl.symm_utils import configure_observation_space_representations
from symm_learning.nn import EquivMultivariateNormal
from symm_learning.models import EMLP, IMLP

G = None


class ActorCriticSymm(ActorCritic):
    """Symmetric Actor-Critic using an Equivariant Policy and an Invariant Critic."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **morphologycal_symmetries_cfg,
    ):
        # Instead of calling ActorCritic.__init__, call torch.nn.Module.__init__
        torch.nn.Module.__init__(self)
        # Cache init args for export function
        self._ac_kwargs = dict(
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        # MorphoSymm components
        obs_space_names_actor = morphologycal_symmetries_cfg["obs_space_names_actor"]
        obs_space_names_critic = morphologycal_symmetries_cfg["obs_space_names_critic"]
        action_space_names = morphologycal_symmetries_cfg["action_space_names"]
        joints_order = morphologycal_symmetries_cfg["joints_order"]
        robot_name = morphologycal_symmetries_cfg["robot_name"]

        G_actor, obs_reps_actor = configure_observation_space_representations(robot_name, obs_space_names_actor, joints_order)
        G_critic, obs_reps_critic = configure_observation_space_representations(robot_name, obs_space_names_critic, joints_order)

        obs_space_reps_actor = [obs_reps_actor[n] for n in obs_space_names_actor]
        obs_space_reps_critic = [obs_reps_critic[n] for n in obs_space_names_critic]
        act_space_reps = [obs_reps_actor[n] for n in action_space_names]

        self.G = G_actor
        gspace = escnn.gspaces.no_base_space(self.G)
        self.num_replica = len(self.G.elements)
        # G-equivariant actor
        self.actor_in_type = FieldType(gspace, obs_space_reps_actor)
        self.actor_out_type = FieldType(gspace, act_space_reps)
        # G-invariant critic
        self.critic_in_type = FieldType(gspace, obs_space_reps_critic)

        # Policy funciton parameterizes a G-equivariant Multivariate Normal distribution
        self.action_gaussian = EquivMultivariateNormal(y_type=self.actor_out_type)

        self.actor = EMLP(
            in_type=self.actor_in_type,
            out_type=self.action_gaussian.in_type,  # Parameters if the equiv Gaussian
            bias=True,
            hidden_units=actor_hidden_dims,
            activation=activation,
        )

        self.critic = IMLP(
            in_type=self.critic_in_type,
            out_dim=1,  # Invariant scalar Value function
            bias=True,
            hidden_units=critic_hidden_dims,
            activation=activation,
        )

        print(f"Critic MLP: {self.critic}")

        model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Actor [{params / 1e6:.2f}M params]: \n{self.actor}")
        model_parameters = filter(lambda p: p.requires_grad, self.critic.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Critic [{params / 1e6:.2f}M params]: \n{self.critic}")

        # Action distribution (populated in update_distribution)
        self.distribution: torch.distributions.MultivariateNormal = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        """Returns the mean of the action distribution conditioned on last observations."""
        if self.distribution is None:
            raise ValueError("Distribution not updated. Call update_distribution() first.")
        return self.distribution.mean

    @property
    def action_std(self):
        """Returns the standard deviation of the action distribution conditioned on last observations."""
        if self.distribution is None:
            raise ValueError("Distribution not updated. Call update_distribution() first.")
        return self.distribution.stddev

    @property
    def entropy(self):
        """Returns the entropy of the action distribution conditioned on last observations."""
        if self.distribution is None:
            raise ValueError("Distribution not updated. Call update_distribution() first.")
        return self.distribution.entropy()

    def get_actions_log_prob(self, actions: torch.Tensor):
        """Returns the log probability of the given actions under the current action distribution.

        Args:
            actions (torch.Tensor): Present actions (batch_size, action_dim).

        Returns:
            torch.Tensor: Log probabilities of the actions (batch_size, 1).
        """
        probs = self.distribution.log_prob(actions)
        assert probs.shape[0] == actions.shape[0]
        return probs

    def update_distribution(self, observations):
        """Update the action distribution based on the current observations."""
        observations = self.actor_in_type(observations)
        dist_params = self.actor(observations)
        self.distribution = self.action_gaussian.get_distribution(dist_params)

    def act_inference(self, observations):
        """Returns the mean action for the given observations during inference."""
        observations = self.actor_in_type(observations)
        dist_params = self.actor(observations)
        actions_mean = dist_params.tensor[..., : self.actor_out_type.size]
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate the value function for the given critic observations."""
        critic_observations = self.critic_in_type(critic_observations)
        value = self.critic(critic_observations).tensor
        return value

    def export(self):
        """Export the acto-critic model as a torch.module with no Equivariant submodules."""
        torch_ac = ActorCritic(
            num_actor_obs=self.actor_in_type.size,
            num_critic_obs=self.critic_in_type.size,
            num_actions=self.actor_out_type.size,
            **self._ac_kwargs,
        )
        # Replace the actor and critic networks by the learned equivariant/invariant modules.
        torch_ac.actor = self.actor.export()
        torch_ac.critic = self.critic.export()

        return torch_ac

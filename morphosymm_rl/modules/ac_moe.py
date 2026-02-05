# Original implementation by Giuseppe L'erario, https://github.com/ami-iit/amp-rsl-rl

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation


class MLP_net(nn.Sequential):
    def __init__(self, in_dim, hidden_dims, out_dim, act):
        layers = [nn.Linear(in_dim, hidden_dims[0]), act]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
            else:
                layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act])
        super().__init__(*layers)


class ActorMoE(nn.Module):
    """
    Mixture-of-Experts actor:
    ⎡expert_1(x) … expert_K(x)⎤ · softmax(gate(x))
    Optionally uses top-k sparse gating.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims,
        gate_hidden_dims: list[int] | None = None,
        activation="elu", 
        num_experts: int = 4,
        top_k: None = None,
        use_gate_loss: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        self.top_k = top_k
        act = resolve_nn_activation(activation)

        self._last_gate_weights = None
        self.use_gate_loss = use_gate_loss

        # experts
        self.experts = nn.ModuleList(
            [MLP_net(obs_dim, hidden_dims, act_dim, act) for _ in range(num_experts)]
        )

        # gating network
        gate_layers = []
        last_dim = obs_dim
        gate_hidden_dims = gate_hidden_dims or []
        for h in gate_hidden_dims:
            gate_layers += [nn.Linear(last_dim, h), act]
            last_dim = h
        gate_layers.append(nn.Linear(last_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)

        self.softmax = nn.Softmax(dim=-1)  # ONNX-friendly

    def forward(self, x: torch.Tensor, return_gate: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch, obs_dim]
        Returns:
            mean action: [batch, act_dim]
        """
        # [batch, act_dim, K]
        expert_out = torch.stack([e(x) for e in self.experts], dim=-1)

        # [batch, K]
        gate_logits = self.gate(x)

        # ---- gating ----
        if self.top_k is None or self.top_k >= self.num_experts:
            # standard dense MoE
            weights = self.softmax(gate_logits).unsqueeze(1)
        else:
            # top-k sparse MoE
            topk_vals, topk_idx = torch.topk(
                gate_logits, k=self.top_k, dim=-1
            )

            masked_logits = torch.full_like(gate_logits, float("-inf"))
            masked_logits.scatter_(
                dim=-1, index=topk_idx, src=topk_vals
            )

            weights = self.softmax(masked_logits).unsqueeze(1)


        # cache for PPO losses / logging
        self._last_gate_weights = weights


        # weighted sum -> [batch, act_dim]
        return (expert_out * weights).sum(dim=-1)


class ActorCriticMoE(nn.Module):
    """Actor-critic with Mixture-of-Experts policy."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **moe_cfg: dict[str, Any],
    ):

        super().__init__()
        act = resolve_nn_activation(activation)


        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # Actor (Mixture-of-Experts)
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            print("Not supported yet., switching to off")
            self.state_dependent_std = False


        num_experts = moe_cfg["num_experts"]
        top_k = moe_cfg["top_k"]
        use_gate_loss = moe_cfg["use_gate_loss"]

        self.actor = ActorMoE(
            obs_dim=num_actor_obs,
            act_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            gate_hidden_dims=actor_hidden_dims[:-1],  # last layer is output
            activation=activation,
            num_experts=num_experts,
            top_k=top_k,
            use_gate_loss=use_gate_loss
        )

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP_net(num_critic_obs, critic_hidden_dims, 1, act)

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def gate_entropy(self) -> torch.Tensor | None:
        """
        Mean gate entropy from last forward pass (useful for PPO regularization)
        """
        if self.actor._last_gate_weights is None:
            return 0.0
        w = self.actor._last_gate_weights
        return -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        else:
            return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

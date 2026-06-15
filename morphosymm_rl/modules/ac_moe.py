# Original implementation by Giuseppe L'erario, https://github.com/ami-iit/amp-rsl-rl

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, NoReturn
from tensordict import TensorDict
from torch.distributions import Normal
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

from .ac_moe_common import BaseMoENet, DiagonalGaussianMixture, MLP_net, MaskedActionNormal
from .ac_moe_explicit import ExplicitExpertMoENet
from .ac_moe_gated import GatedMoENet


def MoE_net(
    obs_dim: int,
    act_dim: int,
    hidden_dims,
    gate_hidden_dims: list[int] | None = None,
    activation="elu",
    num_experts: int = 4,
    top_k: int = -1,
    use_gate_loss: bool = False,
    use_load_balance_loss: bool = False,
    use_explicit_expert: bool = False,
    use_shared_layers="None",
    expert_output_dims: list[int] | None = None,
) -> BaseMoENet:
    """Compatibility factory for the two MoE implementations."""
    if use_explicit_expert:
        return ExplicitExpertMoENet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            num_experts=num_experts,
            use_gate_loss=use_gate_loss,
            use_load_balance_loss=use_load_balance_loss,
            use_shared_layers=use_shared_layers,
            expert_output_dims=expert_output_dims,
        )

    if expert_output_dims is not None:
        raise ValueError("`expert_output_dims` is supported only with `use_explicit_expert=True`.")

    return GatedMoENet(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=hidden_dims,
        gate_hidden_dims=gate_hidden_dims,
        activation=activation,
        num_experts=num_experts,
        top_k=top_k,
        use_gate_loss=use_gate_loss,
        use_load_balance_loss=use_load_balance_loss,
        use_shared_layers=use_shared_layers,
    )


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

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            print("Not supported yet., switching to off")
            self.state_dependent_std = False

        self.who = moe_cfg["who"]
        num_experts = moe_cfg["num_experts"]
        raw_top_k = moe_cfg.get("top_k", -1)
        top_k = -1 if raw_top_k is None else int(raw_top_k)
        use_gate_loss = moe_cfg["use_gate_loss"]
        use_load_balance_loss = moe_cfg["use_load_balance_loss"]
        use_explicit_expert = moe_cfg["use_explicit_expert"]
        gate_hidden_dims = moe_cfg["gate_hidden_dims"]
        use_shared_layers = moe_cfg["use_shared_layers"]
        expert_output_dims = moe_cfg.get("expert_output_dims", None)
        if expert_output_dims is None:
            expert_output_dims = moe_cfg.get("expert_action_dims", None)
        if expert_output_dims is None:
            expert_output_dims = moe_cfg.get("num_outputs_per_expert", None)
        if expert_output_dims is not None and "actor" not in self.who:
            raise ValueError("`expert_output_dims` can be used only when `who` includes 'actor'.")

        self.use_gate_loss = use_gate_loss
        self.use_load_balance_loss = use_load_balance_loss
        self.use_gaussian_mixture = bool(moe_cfg.get("use_gaussian_mixture", False))

        if "actor" in self.who:
            self.actor = MoE_net(
                obs_dim=num_actor_obs,
                act_dim=num_actions,
                hidden_dims=actor_hidden_dims,
                gate_hidden_dims=gate_hidden_dims,
                activation=activation,
                num_experts=num_experts,
                top_k=top_k,
                use_gate_loss=use_gate_loss,
                use_load_balance_loss=use_load_balance_loss,
                use_explicit_expert=use_explicit_expert,
                use_shared_layers=use_shared_layers,
                expert_output_dims=expert_output_dims,
            )
        else:
            self.actor = MLP_net(num_actor_obs, actor_hidden_dims, num_actions, act)

        if self.use_gaussian_mixture and not isinstance(self.actor, BaseMoENet):
            raise ValueError("`use_gaussian_mixture=True` requires `who` to include 'actor'.")
        self.use_variable_expert_outputs = isinstance(self.actor, BaseMoENet) and self.actor.has_variable_expert_outputs
        self.use_masked_action_kl = self.use_variable_expert_outputs and not self.use_gaussian_mixture
        self.use_log_prob_kl = self.use_gaussian_mixture

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        if "critic" in self.who:
            self.critic = MoE_net(
                obs_dim=num_critic_obs,
                act_dim=1,
                hidden_dims=critic_hidden_dims,
                gate_hidden_dims=gate_hidden_dims,
                activation=activation,
                num_experts=num_experts,
                top_k=top_k,
                use_gate_loss=use_gate_loss,
                use_load_balance_loss=use_load_balance_loss,
                use_explicit_expert=use_explicit_expert,
                use_shared_layers=use_shared_layers,
            )
        else:
            self.critic = MLP_net(num_critic_obs, critic_hidden_dims, 1, act)

        print("actor:", self.actor)
        print("critic:", self.critic)

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

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
                if isinstance(self.actor, BaseMoENet):
                    self.std = nn.Parameter(init_noise_std * torch.ones(self.actor.num_experts, num_actions))
                else:
                    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                if isinstance(self.actor, BaseMoENet):
                    self.log_std = nn.Parameter(
                        torch.log(init_noise_std * torch.ones(self.actor.num_experts, num_actions))
                    )
                else:
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
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
    def action_std_for_logging(self) -> torch.Tensor:
        action_std = self.action_std
        if isinstance(self.distribution, MaskedActionNormal):
            active_mask = self.distribution.action_mask > 0.0
            return action_std[active_mask]
        return action_std

    @property
    def entropy(self) -> torch.Tensor:
        entropy = self.distribution.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)
        return entropy

    def gate_entropy(self) -> torch.Tensor:
        """Mean gate entropy from last forward pass."""
        if "actor" in self.who and "critic" in self.who:
            w = self.actor._last_gate_weights + self.critic._last_gate_weights
        elif "actor" in self.who:
            w = self.actor._last_gate_weights
        elif "critic" in self.who:
            w = self.critic._last_gate_weights
        return -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()

    def load_balance_loss(self) -> torch.Tensor:
        """Aggregate load-balancing loss from MoE modules."""
        if "actor" in self.who and "critic" in self.who:
            return self.actor.load_balance_loss() + self.critic.load_balance_loss()
        elif "actor" in self.who:
            return self.actor.load_balance_loss()
        elif "critic" in self.who:
            return self.critic.load_balance_loss()

    def _expert_action_std(self, batch_size: int) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            expert_std = self.std
        elif self.noise_std_type == "log":
            expert_std = torch.exp(self.log_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        return expert_std.unsqueeze(0).expand(batch_size, -1, -1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        action_mask: torch.Tensor | None = None
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if isinstance(self.actor, BaseMoENet):
                mean = self.actor(obs)
                expert_std = torch.clamp(self._expert_action_std(obs.shape[0]), 1e-3, 2.0)

                if self.use_gaussian_mixture:
                    component_means = self.actor._last_component_outputs.transpose(1, 2)
                    weights = self.actor._last_gate_weights.squeeze(1)
                    component_action_masks = None
                    if self.actor.has_variable_expert_outputs:
                        component_action_masks = self.actor.expert_action_masks.to(
                            device=component_means.device,
                            dtype=component_means.dtype,
                        )
                    self.distribution = DiagonalGaussianMixture(
                        component_means,
                        expert_std,
                        weights,
                        component_action_masks=component_action_masks,
                    )
                    return

                if self.actor.use_explicit_expert:
                    selector_vals = obs[:, -1].round().long().clamp(0, self.actor.num_experts - 1)
                    batch_idx = torch.arange(obs.shape[0], device=obs.device)
                    std = expert_std[batch_idx, selector_vals]
                    if self.actor.has_variable_expert_outputs:
                        action_mask = self.actor.selected_action_mask(obs).to(dtype=mean.dtype, device=mean.device)
                else:
                    weights = self.actor._last_gate_weights.squeeze(1)
                    std = (weights.unsqueeze(-1) * expert_std).sum(dim=1)
            else:
                mean = self.actor(obs)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        std = torch.clamp(std, 1e-3, 2.0)
        if action_mask is None:
            self.distribution = Normal(mean, std)
        else:
            self.distribution = MaskedActionNormal(mean, std, action_mask)

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
        log_prob = self.distribution.log_prob(actions)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model."""
        super().load_state_dict(state_dict, strict=strict)
        return True

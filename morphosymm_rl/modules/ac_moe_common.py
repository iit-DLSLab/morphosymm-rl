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
        self._in_features_override = -1

    @property
    def in_features(self) -> int:
        """Proxy to the first linear layer's input size."""
        if self._in_features_override >= 0:
            return self._in_features_override
        first = self[0]
        return int(first.in_features)

    @in_features.setter
    def in_features(self, value: int) -> None:
        self._in_features_override = int(value)


class DiagonalGaussianMixture:
    """Batch of diagonal Gaussian mixtures over actions."""

    def __init__(
        self,
        component_means: torch.Tensor,
        component_stds: torch.Tensor,
        mixture_weights: torch.Tensor,
        component_action_masks: torch.Tensor | None = None,
    ) -> None:
        if mixture_weights.dim() == 3:
            mixture_weights = mixture_weights.squeeze(1)

        self.component_means = component_means
        self.component_stds = torch.clamp(component_stds, min=1e-6)
        self.mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        self.component_distribution = Normal(self.component_means, self.component_stds)
        if component_action_masks is None:
            self.component_action_masks = torch.ones_like(component_means)
        elif component_action_masks.dim() == 2:
            self.component_action_masks = component_action_masks.unsqueeze(0).expand_as(component_means)
        else:
            self.component_action_masks = component_action_masks.expand_as(component_means)

    @property
    def mean(self) -> torch.Tensor:
        masked_means = self.component_means * self.component_action_masks
        return (self.mixture_weights.unsqueeze(-1) * masked_means).sum(dim=1)

    @property
    def stddev(self) -> torch.Tensor:
        masked_means = self.component_means * self.component_action_masks
        second_moment = (
            self.mixture_weights.unsqueeze(-1)
            * self.component_action_masks
            * (self.component_stds.square() + masked_means.square())
        ).sum(dim=1)
        variance = (second_moment - self.mean.square()).clamp_min(1e-12)
        return torch.sqrt(variance)

    def sample(self) -> torch.Tensor:
        expert_idx = torch.multinomial(self.mixture_weights, num_samples=1)
        gather_idx = expert_idx.unsqueeze(-1).expand(-1, -1, self.component_means.shape[-1])
        means = torch.gather(self.component_means, dim=1, index=gather_idx).squeeze(1)
        stds = torch.gather(self.component_stds, dim=1, index=gather_idx).squeeze(1)
        masks = torch.gather(self.component_action_masks, dim=1, index=gather_idx).squeeze(1)
        return (means + torch.randn_like(means) * stds) * masks

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        component_log_probs = (
            self.component_distribution.log_prob(actions.unsqueeze(1)) * self.component_action_masks
        ).sum(dim=-1)
        log_weights = torch.log(self.mixture_weights.clamp_min(1e-8))
        return torch.logsumexp(log_weights + component_log_probs, dim=-1)

    def entropy(self) -> torch.Tensor:
        weights = self.mixture_weights.clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        mixture_entropy = -(weights * torch.log(weights)).sum(dim=-1)
        component_entropy = (self.component_distribution.entropy() * self.component_action_masks).sum(dim=-1)
        return mixture_entropy + (weights * component_entropy).sum(dim=-1)


class MaskedActionNormal:
    """Diagonal Normal whose inactive action dimensions are deterministic zeros."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, action_mask: torch.Tensor) -> None:
        self.action_mask = action_mask.to(dtype=mean.dtype, device=mean.device)
        self.distribution = Normal(mean, torch.clamp(std, min=1e-6))

    @property
    def mean(self) -> torch.Tensor:
        return self.distribution.mean * self.action_mask

    @property
    def stddev(self) -> torch.Tensor:
        return self.distribution.stddev * self.action_mask

    def sample(self) -> torch.Tensor:
        return self.distribution.sample() * self.action_mask

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions) * self.action_mask

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy() * self.action_mask


class BaseMoENet(nn.Module):
    """Shared expert topology for learned-gate and explicit-expert MoE policies."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims,
        activation: str = "elu",
        num_experts: int = 4,
        use_explicit_expert: bool = False,
        use_shared_layers="None",
        expert_output_dims: list[int] | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        self.use_explicit_expert = use_explicit_expert
        self.use_shared_backbone = use_shared_layers == "backbone"
        self.use_shared_backbone_and_head = use_shared_layers == "backbone+head"

        act = resolve_nn_activation(activation)

        self._last_gate_weights = torch.empty(0)
        self._last_unmasked_gate_weights = torch.empty(0)
        self._last_component_outputs = torch.empty(0)

        if expert_output_dims is None:
            self.expert_output_dims = [act_dim for _ in range(num_experts)]
        else:
            if len(expert_output_dims) != num_experts:
                raise ValueError(
                    f"`expert_output_dims` must contain one value per expert: got {len(expert_output_dims)} "
                    f"values for {num_experts} experts."
                )
            self.expert_output_dims = [int(dim) for dim in expert_output_dims]
            invalid_dims = [dim for dim in self.expert_output_dims if dim <= 0 or dim > act_dim]
            if len(invalid_dims) > 0:
                raise ValueError(
                    f"`expert_output_dims` entries must be in [1, {act_dim}], got {self.expert_output_dims}."
                )
        self.has_variable_expert_outputs = any(dim != act_dim for dim in self.expert_output_dims)
        expert_action_masks = torch.zeros(num_experts, act_dim)
        for expert_idx, output_dim in enumerate(self.expert_output_dims):
            expert_action_masks[expert_idx, :output_dim] = 1.0
        self.register_buffer("expert_action_masks", expert_action_masks)

        self.shared_backbone: nn.Module = nn.Sequential()
        self.shared_head: nn.Module = nn.Linear(1, 1)
        self.gate: nn.Module = nn.Sequential()
        self.softmax: nn.Module = nn.Softmax(dim=-1)

        model_obs_dim = obs_dim - 1 if self.use_explicit_expert else obs_dim
        gate_obs_dim = model_obs_dim

        if self.use_shared_backbone_and_head:
            shared_layers = [nn.Linear(model_obs_dim, hidden_dims[0]), act]
            for i in range(len(hidden_dims) - 2):
                shared_layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act]

            self.shared_backbone = nn.Sequential(*shared_layers)
            last_dim = hidden_dims[-2]
            expert_input_dim = last_dim
            self.experts = nn.ModuleList(
                [MLP_net(expert_input_dim, [hidden_dims[-1]], hidden_dims[-1], act) for _ in range(num_experts)]
            )
            self.shared_head = nn.Linear(hidden_dims[-1], act_dim)
        elif self.use_shared_backbone:
            shared_layers = [nn.Linear(model_obs_dim, hidden_dims[0]), act]
            for i in range(len(hidden_dims) - 2):
                shared_layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act]

            self.shared_backbone = nn.Sequential(*shared_layers)
            last_dim = hidden_dims[-2]
            expert_input_dim = last_dim
            self.experts = nn.ModuleList(
                [
                    MLP_net(expert_input_dim, [last_dim], self.expert_output_dims[i], act)
                    for i in range(num_experts)
                ]
            )
        else:
            expert_input_dim = model_obs_dim
            last_dim = model_obs_dim
            self.experts = nn.ModuleList(
                [
                    MLP_net(expert_input_dim, hidden_dims, self.expert_output_dims[i], act)
                    for i in range(num_experts)
                ]
            )

        self.gate_input_dim = last_dim if (self.use_shared_backbone or self.use_shared_backbone_and_head) else gate_obs_dim

    def __getitem__(self, idx: int):
        module = self.experts[idx]
        module.in_features = self.obs_dim  # type: ignore[attr-defined]
        return module

    @property
    def in_features(self) -> int:
        return self.obs_dim

    def _prepare_observation_input(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :-1] if self.use_explicit_expert else x

    def _prepare_gate_input(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :-1] if self.use_explicit_expert else x

    def _pad_expert_action_output(self, expert_output: torch.Tensor, expert_idx: int) -> torch.Tensor:
        output_dim = self.expert_output_dims[expert_idx]
        if output_dim == self.act_dim:
            return expert_output

        padded_output = expert_output.new_zeros(expert_output.shape[0], self.act_dim)
        padded_output[:, :output_dim] = expert_output
        return padded_output

    def _mask_component_outputs(self, component_out: torch.Tensor) -> torch.Tensor:
        if not self.has_variable_expert_outputs:
            return component_out

        action_masks = self.expert_action_masks.transpose(0, 1).unsqueeze(0).to(dtype=component_out.dtype)
        return component_out * action_masks

    def selected_action_mask(self, x: torch.Tensor) -> torch.Tensor:
        selector_vals = x[:, -1].round().long().clamp(0, self.num_experts - 1)
        return self.expert_action_masks.index_select(0, selector_vals)

    def _experts_separate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_input = self._prepare_observation_input(x)
        expert_out = torch.stack(
            [self._pad_expert_action_output(e(obs_input), i) for i, e in enumerate(self.experts)],
            dim=-1,
        )
        return expert_out, self._prepare_gate_input(x)

    def _experts_shared_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_input = self._prepare_observation_input(x)
        features = self.shared_backbone(obs_input)
        expert_out = torch.stack(
            [self._pad_expert_action_output(e(features), i) for i, e in enumerate(self.experts)],
            dim=-1,
        )
        return expert_out, features

    def _experts_shared_backbone_and_head(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_input = self._prepare_observation_input(x)
        features = self.shared_backbone(obs_input)
        expert_out = torch.stack([e(features) for e in self.experts], dim=-1)
        return expert_out, features

    def _compute_experts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_shared_backbone_and_head:
            return self._experts_shared_backbone_and_head(x)
        if self.use_shared_backbone:
            return self._experts_shared_backbone(x)
        return self._experts_separate(x)

    def _component_outputs(self, expert_out: torch.Tensor) -> torch.Tensor:
        if self.use_shared_backbone_and_head:
            component_out = self.shared_head(expert_out.transpose(1, 2)).transpose(1, 2)
        else:
            component_out = expert_out
        return self._mask_component_outputs(component_out)

    def _combine_direct(self, expert_out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (expert_out * weights).sum(dim=-1)

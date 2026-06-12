# Original implementation by Giuseppe L'erario, https://github.com/ami-iit/amp-rsl-rl

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, NoReturn
from tensordict import TensorDict
from torch.distributions import Normal
from rsl_rl.networks import EmpiricalNormalization
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
        """Proxy to the in_features of the first linear layer so external code can
        query an MLP's input size (e.g. exporter expecting `module.in_features`).
        """
        if self._in_features_override >= 0:
            return self._in_features_override
        first = self[0]
        # first is expected to be nn.Linear
        return int(first.in_features)

    @in_features.setter
    def in_features(self, value: int) -> None:
        self._in_features_override = int(value)


class Identity_net(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DiagonalGaussianMixture:
    """Batch of diagonal Gaussian mixtures over actions.

    Shapes:
        component_means: [batch, num_experts, action_dim]
        component_stds: [batch, num_experts, action_dim]
        mixture_weights: [batch, num_experts]
    """

    def __init__(
        self,
        component_means: torch.Tensor,
        component_stds: torch.Tensor,
        mixture_weights: torch.Tensor,
    ) -> None:
        if mixture_weights.dim() == 3:
            mixture_weights = mixture_weights.squeeze(1)

        self.component_means = component_means
        self.component_stds = torch.clamp(component_stds, min=1e-6)
        self.mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        self.component_distribution = Normal(self.component_means, self.component_stds)

    @property
    def mean(self) -> torch.Tensor:
        return (self.mixture_weights.unsqueeze(-1) * self.component_means).sum(dim=1)

    @property
    def stddev(self) -> torch.Tensor:
        second_moment = (
            self.mixture_weights.unsqueeze(-1) * (self.component_stds.square() + self.component_means.square())
        ).sum(dim=1)
        variance = (second_moment - self.mean.square()).clamp_min(1e-12)
        return torch.sqrt(variance)

    def sample(self) -> torch.Tensor:
        expert_idx = torch.multinomial(self.mixture_weights, num_samples=1)
        gather_idx = expert_idx.unsqueeze(-1).expand(-1, -1, self.component_means.shape[-1])
        means = torch.gather(self.component_means, dim=1, index=gather_idx).squeeze(1)
        stds = torch.gather(self.component_stds, dim=1, index=gather_idx).squeeze(1)
        return means + torch.randn_like(means) * stds

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        component_log_probs = self.component_distribution.log_prob(actions.unsqueeze(1)).sum(dim=-1)
        log_weights = torch.log(self.mixture_weights.clamp_min(1e-8))
        return torch.logsumexp(log_weights + component_log_probs, dim=-1)

    def entropy(self) -> torch.Tensor:
        # Upper-bound approximation H(a, z) = H(z) + E_z[H(a | z)].
        weights = self.mixture_weights.clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        mixture_entropy = -(weights * torch.log(weights)).sum(dim=-1)
        component_entropy = self.component_distribution.entropy().sum(dim=-1)
        return mixture_entropy + (weights * component_entropy).sum(dim=-1)


class MoE_net(nn.Module):
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
        top_k: int = -1,
        use_gate_loss: bool = False,
        use_load_balance_loss: bool = False,
        use_explicit_expert: bool = False,
        use_shared_layers="None",
        use_shared_exteroception: bool = False,
        exteroceptive_start_idx: int = -150,
        exteroceptive_end_idx: int = -2,
        exteroceptive_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        # Use an integer sentinel (-1) for 'no top-k' to avoid Optional/None comparisons
        # which break TorchScript. Store as int.
        self.top_k = -1 if top_k is None else int(top_k)
        self.is_sparse = 0 < self.top_k < self.num_experts
        act = resolve_nn_activation(activation)

        # Store last gate weights as a tensor sentinel (empty tensor) so TorchScript
        # sees a consistent attribute type (Tensor) instead of switching from NoneType
        # to Tensor during execution.
        self._last_gate_weights = torch.empty(0)
        self._last_unmasked_gate_weights = torch.empty(0)
        self._last_component_outputs = torch.empty(0)
        
        # Different modalities
        self.use_gate_loss = use_gate_loss
        self.use_load_balance_loss = use_load_balance_loss
        self.use_explicit_expert = use_explicit_expert
        self.use_shared_exteroception = use_shared_exteroception
        self.use_shared_backbone = use_shared_layers == "backbone"
        self.use_shared_backbone_and_head = use_shared_layers == "backbone+head"
        self.exteroceptive_start_idx = int(exteroceptive_start_idx)
        self.exteroceptive_end_idx = int(exteroceptive_end_idx)


        # TorchScript/ONNX require all attributes to exist regardless of init branch.
        # Provide placeholders; the correct branch below will overwrite them.
        self.shared_backbone: nn.Module = nn.Sequential()
        self.shared_head: nn.Module = nn.Linear(1, 1)
        self.gate: nn.Module = nn.Sequential()
        self.softmax: nn.Module = nn.Softmax(dim=-1)
        self.shared_extero_encoder: nn.Module = Identity_net()
        self.extero_latent_dim = 0


        # If explicit expert is used, the last input variable is not 
        # considered an obs
        if(self.use_explicit_expert):
            obs_dim = obs_dim-1

        model_obs_dim = obs_dim
        gate_obs_dim = obs_dim
        if self.use_shared_exteroception:
            extero_start, extero_end = self._resolve_exteroceptive_slice(obs_dim)
            extero_dim = extero_end - extero_start
            proprio_dim = obs_dim - extero_dim
            if extero_dim <= 0 or proprio_dim <= 0:
                raise ValueError("Invalid exteroceptive slice for MoE_net.")

            exteroceptive_hidden_dims = exteroceptive_hidden_dims or []
            if len(exteroceptive_hidden_dims) == 0:
                self.shared_extero_encoder = Identity_net()
                extero_latent_dim = extero_dim
            else:
                extero_latent_dim = exteroceptive_hidden_dims[-1]
                self.shared_extero_encoder = MLP_net(
                    extero_dim, exteroceptive_hidden_dims[:-1], extero_latent_dim, act
                )
            self.extero_latent_dim = extero_latent_dim
            model_obs_dim = proprio_dim
            gate_obs_dim = proprio_dim


        # We start building the network
        if(self.use_shared_backbone_and_head):
            # Shared trunk + single shared head
            shared_layers = [nn.Linear(model_obs_dim, hidden_dims[0]), act]
            for i in range(len(hidden_dims) - 2):
                shared_layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act]

            self.shared_backbone = nn.Sequential(*shared_layers)
            last_dim = hidden_dims[-2]
            expert_input_dim = last_dim + self.extero_latent_dim

            # Expert-specific nonlinear heads that project shared features into a
            # common latent space before the shared output layer.
            self.experts = nn.ModuleList(
                [MLP_net(expert_input_dim, [hidden_dims[-1]], hidden_dims[-1], act) for _ in range(num_experts)]
            )

            self.shared_head = nn.Linear(hidden_dims[-1], act_dim)
        
        elif(self.use_shared_backbone):
            # Shared trunk + separate expert heads
            shared_layers = [nn.Linear(model_obs_dim, hidden_dims[0]), act]
            for i in range(len(hidden_dims) - 2):
                shared_layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act]

            self.shared_backbone = nn.Sequential(*shared_layers)
            last_dim = hidden_dims[-2]
            expert_input_dim = last_dim + self.extero_latent_dim

            # Expert-specific nonlinear heads on top of the shared backbone.
            self.experts = nn.ModuleList(
                [MLP_net(expert_input_dim, [last_dim], act_dim, act) for _ in range(num_experts)]
            )
        
        else:
            # Separate NN Experts
            expert_input_dim = model_obs_dim + self.extero_latent_dim
            last_dim = model_obs_dim
            self.experts = nn.ModuleList(
                [MLP_net(expert_input_dim, hidden_dims, act_dim, act) for _ in range(num_experts)]
            )

        gate_input_dim = last_dim if (self.use_shared_backbone or self.use_shared_backbone_and_head) else gate_obs_dim


        if(self.use_explicit_expert == False):
            # gating network
            gate_layers = []
            gate_hidden_dims = gate_hidden_dims or []
            last_dim = gate_input_dim
            for h in gate_hidden_dims:
                gate_layers += [nn.Linear(last_dim, h), act]
                last_dim = h
            gate_layers.append(nn.Linear(last_dim, num_experts))
            self.gate = nn.Sequential(*gate_layers)

            self.softmax = nn.Softmax(dim=-1)  # ONNX-friendly

    # ------------------------------------------------------------------ #
    #  Things for jitting/onnx export the net
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int):
        """Allow indexing into the MoE to get the underlying expert module
        (keeps compatibility with code doing `actor[0]`).
        
        Note: For shared backbone/head modes, experts[0] is an inner layer
        (not the input layer). We wrap it so that `actor[0].in_features`
        still reports the correct *network* input dimension (obs_dim).
        """
        module = self.experts[idx]
        if self.use_shared_backbone or self.use_shared_backbone_and_head:
            # The exporter queries actor[0].in_features to size the dummy input.
            # Monkey-patch so it reflects the true obs_dim, not the expert head input.
            module.in_features = self.obs_dim  # type: ignore[attr-defined]
        return module

    @property
    def in_features(self) -> int:
        """Return the observation dimension expected by this MoE network.

        This is used by the policy exporter to create a correctly-sized
        dummy input for JIT tracing / ONNX export.
        """
        return self.obs_dim

    # ------------------------------------------------------------------ #
    #  Expert output computation — one method per topology
    # ------------------------------------------------------------------ #

    def _resolve_exteroceptive_slice(self, feature_dim: int) -> tuple[int, int]:
        start = self.exteroceptive_start_idx if self.exteroceptive_start_idx >= 0 else feature_dim + self.exteroceptive_start_idx
        end = self.exteroceptive_end_idx if self.exteroceptive_end_idx >= 0 else feature_dim + self.exteroceptive_end_idx
        start = max(0, min(start, feature_dim))
        end = max(0, min(end, feature_dim))
        if end <= start:
            raise ValueError(
                f"Resolved exteroceptive slice is empty: [{start}:{end}] from raw [{self.exteroceptive_start_idx}:{self.exteroceptive_end_idx}]."
            )
        return start, end

    def _prepare_observation_input(self, x: torch.Tensor) -> torch.Tensor:
        obs_input = x[:, :-1] if self.use_explicit_expert else x
        if not self.use_shared_exteroception:
            return obs_input

        start, end = self._resolve_exteroceptive_slice(obs_input.shape[-1])
        proprioceptive_input = torch.cat([obs_input[:, :start], obs_input[:, end:]], dim=-1)
        return proprioceptive_input

    def _prepare_gate_input(self, x: torch.Tensor) -> torch.Tensor:
        obs_input = x[:, :-1] if self.use_explicit_expert else x
        if not self.use_shared_exteroception:
            return obs_input

        start, end = self._resolve_exteroceptive_slice(obs_input.shape[-1])
        proprioceptive_input = torch.cat([obs_input[:, :start], obs_input[:, end:]], dim=-1)
        return proprioceptive_input

    def _prepare_exteroceptive_features(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.use_shared_exteroception:
            return None

        obs_input = x[:, :-1] if self.use_explicit_expert else x
        start, end = self._resolve_exteroceptive_slice(obs_input.shape[-1])
        exteroceptive_input = obs_input[:, start:end]
        return self.shared_extero_encoder(exteroceptive_input)

    def _experts_separate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each expert is a full MLP; gate receives the raw observation."""
        obs_input = self._prepare_observation_input(x)
        extero_features = self._prepare_exteroceptive_features(x)
        expert_input = obs_input if extero_features is None else torch.cat([obs_input, extero_features], dim=-1)
        expert_out = torch.stack([e(expert_input) for e in self.experts], dim=-1)  # [B, act_dim, K]
        return expert_out, self._prepare_gate_input(x)

    def _experts_shared_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared backbone → per-expert linear heads; gate receives backbone features."""
        obs_input = self._prepare_observation_input(x)
        features = self.shared_backbone(obs_input)
        extero_features = self._prepare_exteroceptive_features(x)
        expert_input = features if extero_features is None else torch.cat([features, extero_features], dim=-1)
        expert_out = torch.stack([e(expert_input) for e in self.experts], dim=-1)  # [B, act_dim, K]
        return expert_out, features

    def _experts_shared_backbone_and_head(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared backbone → per-expert linear → shared head; gate receives backbone features."""

        obs_input = self._prepare_observation_input(x)
        features = self.shared_backbone(obs_input)
        extero_features = self._prepare_exteroceptive_features(x)
        expert_input = features if extero_features is None else torch.cat([features, extero_features], dim=-1)
        expert_out = torch.stack([e(expert_input) for e in self.experts], dim=-1)  # [B, hidden, K]
        return expert_out, features

    # ------------------------------------------------------------------ #
    #  Gating — one method per routing strategy
    # ------------------------------------------------------------------ #

    def _gate_explicit(self, x: torch.Tensor) -> torch.Tensor:
        """Hard expert selection: the last element of *x* encodes the expert index.

        Returns:
            weights: [B, 1, K] one-hot gating tensor.
        """
        #print("x[:, -1]:", x[:, -1])
        selector_vals = x[:, -1].round().long().clamp(0, self.num_experts - 1)
        #print("selector_vals:", selector_vals)
        weights = torch.zeros(x.shape[0], self.num_experts, device=x.device)
        weights.scatter_(1, selector_vals.unsqueeze(1), 1.0)
        return weights.unsqueeze(1)

    def _gate_dense(self, gate_input: torch.Tensor) -> torch.Tensor:
        """Soft gating over all experts (no sparsity).

        Returns:
            weights: [B, 1, K] soft gating tensor.
        """
        gate_logits = self.gate(gate_input)
        full_weights = self.softmax(gate_logits)
        self._last_unmasked_gate_weights = full_weights
        return full_weights.unsqueeze(1)

    def _gate_sparse(self, gate_input: torch.Tensor) -> torch.Tensor:
        """Top-k sparse gating: only *top_k* experts receive non-zero weight.

        Returns:
            weights: [B, 1, K] sparse gating tensor.
        """
        gate_logits = self.gate(gate_input)
        self._last_unmasked_gate_weights = self.softmax(gate_logits)

        topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        masked_logits = torch.full_like(gate_logits, -1e9)
        masked_logits.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        full_weights = self.softmax(masked_logits)

        self._last_unmasked_gate_weights = full_weights
        return full_weights.unsqueeze(1)

    # ------------------------------------------------------------------ #
    #  Output combination — one method per topology
    # ------------------------------------------------------------------ #

    def _combine_with_shared_head(self, expert_out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted sum of expert outputs followed by a shared linear head."""
        mixed = (expert_out * weights).sum(dim=-1)   # [B, hidden]
        return self.shared_head(mixed)                # [B, act_dim]

    def _combine_direct(self, expert_out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted sum of expert outputs (no extra head)."""
        return (expert_out * weights).sum(dim=-1)     # [B, act_dim]

    # ------------------------------------------------------------------ #
    #  Forward — dispatches to the correct sub-methods
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, return_gate: bool = False) -> torch.Tensor:
        # 1) Expert outputs
        if self.use_shared_backbone_and_head:
            expert_out, gate_input = self._experts_shared_backbone_and_head(x)
        elif self.use_shared_backbone:
            expert_out, gate_input = self._experts_shared_backbone(x)
        else:
            expert_out, gate_input = self._experts_separate(x)

        # 2) Gating weights
        if self.use_explicit_expert:
            weights = self._gate_explicit(x)
        elif self.is_sparse:
            weights = self._gate_sparse(gate_input)
        else:
            weights = self._gate_dense(gate_input)

        self._last_gate_weights = weights
        if self.use_shared_backbone_and_head:
            component_out = self.shared_head(expert_out.transpose(1, 2)).transpose(1, 2)
        else:
            component_out = expert_out
        self._last_component_outputs = component_out

        # 3) Combine
        if self.use_shared_backbone_and_head:
            return self._combine_with_shared_head(expert_out, weights)
        else:
            return self._combine_direct(expert_out, weights)


    def load_balance_loss(self) -> torch.Tensor:
        """Auxiliary load-balancing loss that adapts to the routing mode.

        **Sparse routing** (top_k >= 1 and < num_experts):
            Uses the Switch Transformer formulation (Fedus et al., 2022, Eq. 4-6):
                L = alpha * N * sum_i(f_i * P_i)
            where f_i is the fraction of samples whose argmax expert is i (non-differentiable)
            and P_i is the mean router probability for expert i (differentiable).
            The product f_i * P_i is minimised under a uniform distribution.

        **Dense routing** (all experts used, top_k < 0 or >= num_experts):
            Uses squared deviation from uniform:
                L = sum_k (mean_w_k - 1/K)^2
            Since every expert contributes (weighted by soft probability), soft weights
            accurately reflect utilisation and squared deviation is appropriate.

        Returns:
            Scalar loss tensor. Zero if no gate weights have been cached yet.
        """

        N = self.num_experts

        if self.is_sparse:
            # --- Sparse routing: Switch Transformer loss (Eq. 4-6) ---
            # router_probs: full softmax probabilities [batch, K]
            router_probs = self._last_unmasked_gate_weights.squeeze(1)  # [batch, K]
            #router_probs = self._last_gate_weights.squeeze(1)  # [batch, K]
            
            # f_i: fraction of samples dispatched to expert i (hard assignment, non-differentiable)
            expert_indices = router_probs.argmax(dim=-1)  # [batch]
            f = torch.zeros(N, device=router_probs.device)
            f.scatter_add_(0, expert_indices, torch.ones_like(expert_indices, dtype=router_probs.dtype))
            f = f / router_probs.shape[0]  # [K]
            # P_i: mean router probability for expert i
            P = router_probs.mean(dim=0)  # [K]
            # L = N * sum(f_i * P_i)  (alpha is applied externally in the PPO loss)
            return N * (f * P).sum()
        else:
            # --- Dense routing: squared deviation from uniform ---
            w = self._last_gate_weights.squeeze(1)  # [batch, K]
            mean_w = w.mean(dim=0)  # [K]
            uniform = torch.full_like(mean_w, 1.0 / N)
            return (mean_w * (mean_w / uniform).log()).sum()
            #return ((mean_w - 1.0 / N) ** 2).sum()


    def expert_utilization_stats(self) -> dict[str, torch.Tensor]:
        """Per-expert utilization statistics from the last forward pass.

        Returns a dict with:
            - ``percent_of_expert_<i>_usage``: fraction of batch samples where expert i is utilized
            - ``dead_experts``: number of experts with zero utilization.
            - ``percent_of_most_used_expert``: max utilization across experts.
            - ``percent_of_least_used_expert``: min utilization across experts.
        """
        stats: dict[str, torch.Tensor] = {}

        N = self.num_experts
        batch_size = self._last_gate_weights.shape[0]

        if self.is_sparse:
            topk_idx = self._last_gate_weights.topk(k=self.top_k, dim=-1).indices  # [batch, K]
        else:
            topk_idx = torch.arange(N, device=self._last_gate_weights.device).unsqueeze(0).expand(batch_size, -1)
        
        # Flatten to count occurrences
        flat_idx = topk_idx.flatten()  # [batch * top_k]
        utilization_counts = torch.zeros(N, device=topk_idx.device, dtype=topk_idx.dtype)
        utilization_counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        # Since each sample contributes to top_k experts, divide by batch_size * top_k? No.
        # Fraction of samples where expert is used: utilization_counts / batch_size
        hard_fracs = utilization_counts.float() / batch_size

        # Soft assignment: mean gate weight per expert
        w = self._last_gate_weights.squeeze(1)  # [batch, K]
        soft_fracs = w.mean(dim=0)  # [K]

        # Per-expert stats
        for i in range(N):
            stats[f"percent_of_expert_{i}_usage"] = hard_fracs[i].detach()
            stats[f"mean_weight_for_expert_{i}"] = soft_fracs[i].detach()

        # Summary stats
        stats["dead_experts"] = (hard_fracs == 0).sum().float().detach()
        stats["percent_of_most_used_expert"] = hard_fracs.max().detach()
        stats["percent_of_least_used_expert"] = hard_fracs.min().detach()

        return stats


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

        self.who = moe_cfg["who"]
        num_experts = moe_cfg["num_experts"]
        raw_top_k = moe_cfg.get("top_k", -1)
        top_k = -1 if raw_top_k is None else int(raw_top_k)
        use_gate_loss = moe_cfg["use_gate_loss"]
        use_load_balance_loss = moe_cfg["use_load_balance_loss"]
        use_explicit_expert = moe_cfg["use_explicit_expert"]
        gate_hidden_dims = moe_cfg["gate_hidden_dims"]
        use_shared_layers = moe_cfg["use_shared_layers"]
        use_shared_exteroception = bool(moe_cfg.get("use_shared_exteroception", False))
        exteroceptive_start_idx = int(moe_cfg.get("exteroceptive_start_idx", -150))
        exteroceptive_end_idx = int(moe_cfg.get("exteroceptive_end_idx", -2))
        exteroceptive_hidden_dims = moe_cfg.get("exteroceptive_hidden_dims", None)
        self.use_gate_loss = use_gate_loss
        self.use_load_balance_loss = use_load_balance_loss
        self.use_gaussian_mixture = bool(moe_cfg.get("use_gaussian_mixture", False))
        self.log_expert_stats = moe_cfg["log_expert_stats"]

        if("actor" in self.who):
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
                use_shared_exteroception=use_shared_exteroception,
                exteroceptive_start_idx=exteroceptive_start_idx,
                exteroceptive_end_idx=exteroceptive_end_idx,
                exteroceptive_hidden_dims=exteroceptive_hidden_dims,
            )
        else:
            self.actor = MLP_net(num_actor_obs, actor_hidden_dims, num_actions, act)

        if self.use_gaussian_mixture and not isinstance(self.actor, MoE_net):
            raise ValueError("`use_gaussian_mixture=True` requires `who` to include 'actor'.")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        if("critic" in self.who):
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
                        use_shared_exteroception=use_shared_exteroception,
                        exteroceptive_start_idx=exteroceptive_start_idx,
                        exteroceptive_end_idx=exteroceptive_end_idx,
                        exteroceptive_hidden_dims=exteroceptive_hidden_dims,
            )
        else:
            self.critic = MLP_net(num_critic_obs, critic_hidden_dims, 1, act)

        print("actor:", self.actor)
        print("critic:", self.critic)

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
                if isinstance(self.actor, MoE_net):
                    self.std = nn.Parameter(init_noise_std * torch.ones(self.actor.num_experts, num_actions))
                else:
                    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                #self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                if isinstance(self.actor, MoE_net):
                    self.log_std = nn.Parameter(
                        torch.log(init_noise_std * torch.ones(self.actor.num_experts, num_actions))
                    )
                else:
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                #self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
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
        entropy = self.distribution.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)
        return entropy

    def gate_entropy(self) -> torch.Tensor:
        """
        Mean gate entropy from last forward pass (useful for PPO regularization)
        """
        if("actor" in self.who and "critic" in self.who):
            w = self.actor._last_gate_weights + self.critic._last_gate_weights
        elif("actor" in self.who):
            w = self.actor._last_gate_weights
        elif("critic" in self.who):
            w = self.critic._last_gate_weights
        return -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()

    def load_balance_loss(self) -> torch.Tensor:
        """Aggregate load-balancing loss from the actor MoE."""
        if("actor" in self.who and "critic" in self.who):
            return self.actor.load_balance_loss() + self.critic.load_balance_loss()
        elif("actor" in self.who):
            return self.actor.load_balance_loss()
        elif("critic" in self.who):
            return self.critic.load_balance_loss()

    def get_expert_stats(self) -> dict[str, float]:
        """Return per-expert utilization stats for logging (e.g. to wandb).

        Keys are prefixed with ``MoE/`` so they appear in a dedicated group.
        """
        if isinstance(self.actor, MoE_net):
            raw = self.actor.expert_utilization_stats()
        elif isinstance(self.critic, MoE_net):
            raw = self.critic.expert_utilization_stats()
        else:
            return {}
        return {f"MoE/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

    def _expert_action_std(self, batch_size: int) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            expert_std = self.std
        elif self.noise_std_type == "log":
            expert_std = torch.exp(self.log_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        return expert_std.unsqueeze(0).expand(batch_size, -1, -1)

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
            if isinstance(self.actor, MoE_net):
                mean = self.actor(obs)
                expert_std = torch.clamp(self._expert_action_std(obs.shape[0]), 1e-3, 2.0)

                if self.use_gaussian_mixture:
                    component_means = self.actor._last_component_outputs.transpose(1, 2)
                    weights = self.actor._last_gate_weights.squeeze(1)
                    self.distribution = DiagonalGaussianMixture(component_means, expert_std, weights)
                    return

                if self.actor.use_explicit_expert:
                    selector_vals = obs[:, -1].round().long().clamp(0, self.actor.num_experts - 1)
                    batch_idx = torch.arange(obs.shape[0], device=obs.device)
                    std = expert_std[batch_idx, selector_vals]

                else:
                    # Gating case: collapse expert stds into one diagonal Gaussian.
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

        # -------- safety clamp  --------
        std = torch.clamp(std, 1e-3, 2.0)

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

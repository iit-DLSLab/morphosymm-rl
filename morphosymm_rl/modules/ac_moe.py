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

    @property
    def in_features(self) -> int:
        """Proxy to the in_features of the first linear layer so external code can
        query an MLP's input size (e.g. exporter expecting `module.in_features`).
        """
        first = self[0]
        # first is expected to be nn.Linear
        return int(first.in_features)


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
        explicit_expert_epsilon: float = 0.8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        # Use an integer sentinel (-1) for 'no top-k' to avoid Optional/None comparisons
        # which break TorchScript. Store as int.
        self.top_k = -1 if top_k is None else int(top_k)
        act = resolve_nn_activation(activation)

        # Store last gate weights as a tensor sentinel (empty tensor) so TorchScript
        # sees a consistent attribute type (Tensor) instead of switching from NoneType
        # to Tensor during execution.
        self._last_gate_weights = torch.empty(0)
        self._last_unmasked_gate_weights = torch.empty(0)
        self.use_gate_loss = use_gate_loss
        self.use_load_balance_loss = use_load_balance_loss
        
        self.use_explicit_expert = use_explicit_expert
        self.explicit_expert_epsilon = explicit_expert_epsilon

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

    def __getitem__(self, idx: int):
        """Allow indexing into the MoE to get the underlying expert module
        (keeps compatibility with code doing `actor[0]`).
        """
        return self.experts[idx]

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
        # Use sentinel < 0 to mean 'no top-k' so TorchScript never compares None with ints.
        if self.top_k < 0 or self.top_k >= self.num_experts:
            # standard dense MoE
            weights = self.softmax(gate_logits).unsqueeze(1)
        else:
            self._last_unmasked_gate_weights = self.softmax(gate_logits)  # [batch, K]
            # top-k sparse MoE
            topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)

            masked_logits = torch.full_like(gate_logits, float("-inf"))
            masked_logits.scatter_(dim=-1, index=topk_idx, src=topk_vals)

            weights = self.softmax(masked_logits).unsqueeze(1)

        # cache for PPO losses / logging
        self._last_gate_weights = weights
        
        if(self.use_explicit_expert):
            # Extract expert selectors from last num_experts elements
            expert_selector = x[:, -self.num_experts:]  # [batch, num_experts]
            
            # Use explicit expert selector instead of learned gate
            weights_suggestion = expert_selector.unsqueeze(1)  # [batch, 1, num_experts]
            epsilon = self.explicit_expert_epsilon
        
            # weighted sum -> [batch, act_dim]
            return (expert_out * (epsilon * weights_suggestion + (1-epsilon) * weights)).sum(dim=-1)
        else:
            # weighted sum -> [batch, act_dim]
            return (expert_out * weights).sum(dim=-1)


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

        if self.top_k >= 1 and self.top_k <= self.num_experts:
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
            return ((mean_w - 1.0 / N) ** 2).sum()


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

        if self.top_k >= 1 and self.top_k <= self.num_experts:
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


        num_experts = moe_cfg["num_experts"]
        raw_top_k = moe_cfg.get("top_k", -1)
        top_k = -1 if raw_top_k is None else int(raw_top_k)
        use_gate_loss = moe_cfg["use_gate_loss"]
        use_load_balance_loss = moe_cfg["use_load_balance_loss"]
        use_explicit_expert = moe_cfg["use_explicit_expert"]
        explicit_expert_epsilon = moe_cfg["explicit_expert_epsilon"]
        gate_hidden_dims = moe_cfg["gate_hidden_dims"]
        self.log_expert_stats = moe_cfg["log_expert_stats"]

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
            explicit_expert_epsilon=explicit_expert_epsilon
        )

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        #self.critic = MLP_net(num_critic_obs, critic_hidden_dims, 1, act)
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
            explicit_expert_epsilon=explicit_expert_epsilon
        )

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

    def gate_entropy(self) -> torch.Tensor:
        """
        Mean gate entropy from last forward pass (useful for PPO regularization)
        """
        w = self.actor._last_gate_weights
        return -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()

    def load_balance_loss(self) -> torch.Tensor:
        """Aggregate load-balancing loss from the actor MoE."""
        return self.actor.load_balance_loss()

    def get_expert_stats(self) -> dict[str, float]:
        """Return per-expert utilization stats for logging (e.g. to wandb).

        Keys are prefixed with ``MoE/`` so they appear in a dedicated group.
        """
        raw = self.actor.expert_utilization_stats()
        return {f"MoE/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

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

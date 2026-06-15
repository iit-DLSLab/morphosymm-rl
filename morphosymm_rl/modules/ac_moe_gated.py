from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation

from .ac_moe_common import BaseMoENet


class GatedMoENet(BaseMoENet):
    """Mixture-of-experts actor/critic with learned dense or top-k sparse gating."""

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
        use_shared_layers="None",
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            num_experts=num_experts,
            use_explicit_expert=False,
            use_shared_layers=use_shared_layers,
            expert_output_dims=None,
        )
        self.use_gate_loss = use_gate_loss
        self.use_load_balance_loss = use_load_balance_loss
        self.top_k = -1 if top_k is None else int(top_k)
        self.is_sparse = 0 < self.top_k < self.num_experts

        act = resolve_nn_activation(activation)
        gate_layers = []
        gate_hidden_dims = gate_hidden_dims or []
        last_dim = self.gate_input_dim
        for h in gate_hidden_dims:
            gate_layers += [nn.Linear(last_dim, h), act]
            last_dim = h
        gate_layers.append(nn.Linear(last_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)
        self.softmax = nn.Softmax(dim=-1)

    def _gate_dense(self, gate_input: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(gate_input)
        full_weights = self.softmax(gate_logits)
        self._last_unmasked_gate_weights = full_weights
        return full_weights.unsqueeze(1)

    def _gate_sparse(self, gate_input: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(gate_input)
        self._last_unmasked_gate_weights = self.softmax(gate_logits)

        topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        masked_logits = torch.full_like(gate_logits, -1e9)
        masked_logits.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        full_weights = self.softmax(masked_logits)

        self._last_unmasked_gate_weights = full_weights
        return full_weights.unsqueeze(1)

    def forward(self, x: torch.Tensor, return_gate: bool = False) -> torch.Tensor:
        expert_out, gate_input = self._compute_experts(x)
        if self.is_sparse:
            weights = self._gate_sparse(gate_input)
        else:
            weights = self._gate_dense(gate_input)

        self._last_gate_weights = weights
        component_out = self._component_outputs(expert_out)
        self._last_component_outputs = component_out
        return self._combine_direct(component_out, weights)

    def load_balance_loss(self) -> torch.Tensor:
        n_experts = self.num_experts

        if self.is_sparse:
            router_probs = self._last_unmasked_gate_weights.squeeze(1)
            expert_indices = router_probs.argmax(dim=-1)
            f = torch.zeros(n_experts, device=router_probs.device)
            f.scatter_add_(0, expert_indices, torch.ones_like(expert_indices, dtype=router_probs.dtype))
            f = f / router_probs.shape[0]
            p = router_probs.mean(dim=0)
            return n_experts * (f * p).sum()

        w = self._last_gate_weights.squeeze(1)
        mean_w = w.mean(dim=0)
        uniform = torch.full_like(mean_w, 1.0 / n_experts)
        safe_mean_w = mean_w.clamp_min(1e-8)
        return (mean_w * (safe_mean_w / uniform).log()).sum()

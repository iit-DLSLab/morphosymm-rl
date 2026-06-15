from __future__ import annotations

import torch

from .ac_moe_common import BaseMoENet


class ExplicitExpertMoENet(BaseMoENet):
    """MoE network with hard expert selection encoded in the last observation entry."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims,
        activation="elu",
        num_experts: int = 4,
        use_gate_loss: bool = False,
        use_load_balance_loss: bool = False,
        use_shared_layers="None",
        expert_output_dims: list[int] | None = None,
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            num_experts=num_experts,
            use_explicit_expert=True,
            use_shared_layers=use_shared_layers,
            expert_output_dims=expert_output_dims,
        )
        self.use_gate_loss = use_gate_loss
        # Explicit routing does not optimize gate balancing.
        self.use_load_balance_loss = False
        self.top_k = -1
        self.is_sparse = False

    def _gate_explicit(self, x: torch.Tensor) -> torch.Tensor:
        selector_vals = x[:, -1].round().long().clamp(0, self.num_experts - 1)
        weights = torch.zeros(x.shape[0], self.num_experts, device=x.device)
        weights.scatter_(1, selector_vals.unsqueeze(1), 1.0)
        return weights.unsqueeze(1)

    def forward(self, x: torch.Tensor, return_gate: bool = False) -> torch.Tensor:
        expert_out, _ = self._compute_experts(x)
        weights = self._gate_explicit(x)

        self._last_gate_weights = weights
        component_out = self._component_outputs(expert_out)
        self._last_component_outputs = component_out
        return self._combine_direct(component_out, weights)

    def load_balance_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self._last_gate_weights.device)

from __future__ import annotations

import symm_learning.stats as symm_stats
import torch
from escnn.group import Representation
from rsl_rl.networks import EmpiricalNormalization


class EquivEmpiricalNormalization(EmpiricalNormalization):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, rep: Representation, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            rep (Representation): Representation of the input space.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.rep = rep
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(self.rep.size).unsqueeze(0))
        self.register_buffer("_var", torch.ones(self.rep.size).unsqueeze(0))
        self.register_buffer("_std", torch.ones(self.rep.size).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (torch.Tensor): Input values of shape (n_samples, self.rep.size)

        Returns:
            torch.Tensor: Normalized output values
        """
        assert x.shape[1] == self.rep.size, f"Input shape {x.shape} does not match representation size {self.rep.size}"
        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x, mean_x = symm_stats.var_mean(x, self.rep)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        """Inverse normalization of the values."""
        return y * (self._std + self.eps) + self._mean

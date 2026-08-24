"""The continuum-PINN network.

A plain MLP mapping normalized coordinates to the nondimensional displacement
field. SiLU activation (Balmer, Kaufmann & Kraus 2024) is smooth, so the
second derivatives the equilibrium residual needs are well defined.

Fourier-feature encoding was tried and removed: the equilibrium residual is
built from second derivatives of u, and high-frequency features are amplified
by frequency-squared in that residual, which made it worse.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class DisplacementPINN(nn.Module):
    """(x_n, y_n) -> (u_x, u_y) nondimensional displacement field."""

    def __init__(self, width: int = 96, depth: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, xy: Tensor) -> Tensor:
        return self.net(xy)

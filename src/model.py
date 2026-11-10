"""MLP that maps encoded launch parameters to a full trajectory.

A shared trunk feeds three output heads:

    trajectory:  (B, N_TIME, 3)  positions [x, y, z] on the normalized time grid
    impact:      (B, 2)          ground impact [x, z]
    flight_time: (B,)            seconds

All inputs and outputs are in normalized units (see dataset.normalize).
"""

import torch
from torch import nn

from . import config


class TrajectoryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int] | None = None,
                 n_time: int = config.N_TIME):
        super().__init__()
        hidden_dims = list(hidden_dims or config.HIDDEN_DIMS)
        self.n_time = n_time

        layers: list[nn.Module] = []
        width = in_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.trunk = nn.Sequential(*layers)

        self.trajectory_head = nn.Linear(width, n_time * 3)
        self.impact_head = nn.Linear(width, 2)
        self.flight_time_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> dict:
        hidden = self.trunk(x)
        return {
            "trajectory": self.trajectory_head(hidden).view(-1, self.n_time, 3),
            "impact": self.impact_head(hidden),
            "flight_time": self.flight_time_head(hidden).squeeze(1),
        }

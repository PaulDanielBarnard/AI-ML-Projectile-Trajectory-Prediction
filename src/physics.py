"""GPU-accelerated 3D ballistic trajectory simulator with quadratic air drag.

State vector: [x, y, z, vx, vy, vz], with y = altitude and x/z = ground plane.

Equations of motion (per sample, drag parameter k = 0.5 * Cd * rho * A / m):

    dx/dt = vx                dvx/dt = -k * |v| * vx
    dy/dt = vy                dvy/dt = -g - k * |v| * vy
    dz/dt = vz                dvz/dt = -k * |v| * vz

All operations are vectorized over the batch dimension and run on the GPU
(RK4 with a fixed time step). Simulation happens in two phases:

    Phase A: integrate until ground impact (y <= 0), recording impact time and
             impact position via linear interpolation between integration steps.
    Phase B: re-integrate and resample each trajectory onto a per-sample
             normalized time grid (num_points samples in [0, 1]).

Input features layout (B, 7): [v0, theta_deg, psi_deg, mass, area, cd, rho].
"""

import torch

from . import config


def drag_coefficient_k(features: torch.Tensor) -> torch.Tensor:
    """Drag parameter k = 0.5 * Cd * rho * A / m, shape (B, 1)."""
    mass = features[:, 3].clamp(min=1e-6)
    area = features[:, 4]
    cd = features[:, 5]
    rho = features[:, 6]
    return (0.5 * cd * rho * area / mass).unsqueeze(1)


def initial_states(features: torch.Tensor) -> torch.Tensor:
    """Launch from the origin with speed v0, elevation theta, azimuth psi."""
    v0 = features[:, 0]
    theta = torch.deg2rad(features[:, 1])
    psi = torch.deg2rad(features[:, 2])

    state = torch.zeros(features.shape[0], 6, device=features.device,
                        dtype=features.dtype)
    state[:, 3] = v0 * torch.cos(theta) * torch.cos(psi)
    state[:, 4] = v0 * torch.sin(theta)
    state[:, 5] = v0 * torch.cos(theta) * torch.sin(psi)
    return state


def _derivative(state: torch.Tensor, k: torch.Tensor, g: float) -> torch.Tensor:
    """Time derivative of the state (B, 6) -> (B, 6)."""
    v = state[:, 3:6]
    speed = torch.linalg.vector_norm(v, dim=1, keepdim=True)
    drag = k * speed  # (B, 1), drag acceleration magnitude factor

    ds = torch.empty_like(state)
    ds[:, 0:3] = v
    ds[:, 3] = -drag[:, 0] * v[:, 0]
    ds[:, 4] = -g - drag[:, 0] * v[:, 1]
    ds[:, 5] = -drag[:, 0] * v[:, 2]
    return ds


def _rk4_step(state: torch.Tensor, k: torch.Tensor, dt: float, g: float) -> torch.Tensor:
    """One classical RK4 step over the whole batch."""
    k1 = _derivative(state, k, g)
    k2 = _derivative(state + 0.5 * dt * k1, k, g)
    k3 = _derivative(state + 0.5 * dt * k2, k, g)
    k4 = _derivative(state + dt * k3, k, g)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _find_impact(features: torch.Tensor, k: torch.Tensor,
                 dt: float, g: float, t_max: float):
    """Phase A: integrate until every projectile lands.

    Returns (impact_time, impact_pos) where impact_pos is (B, 3) with y ~ 0.
    """
    device = features.device
    batch = features.shape[0]

    state = initial_states(features)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    impact_time = torch.full((batch,), t_max, device=device, dtype=torch.float32)
    impact_pos = torch.zeros(batch, 3, device=device, dtype=torch.float32)

    t = 0.0
    while active.any() and t < t_max:
        prev = state
        state = _rk4_step(state, k, dt, g)
        t += dt

        above = state[:, 1] > 0.0
        just_landed = active & ~above
        if just_landed.any():
            y_prev = prev[just_landed, 1]
            y_new = state[just_landed, 1]
            fraction = y_prev / (y_prev - y_new + 1e-12)  # t within this step
            impact_time[just_landed] = (t - dt) + fraction * dt
            impact_pos[just_landed] = (
                prev[just_landed][:, 0:3]
                + fraction.unsqueeze(1)
                * (state[just_landed][:, 0:3] - prev[just_landed][:, 0:3])
            )
        active = above

    return impact_time, impact_pos


def _resample_trajectories(features: torch.Tensor, k: torch.Tensor,
                           impact_time: torch.Tensor, num_points: int,
                           dt: float, g: float, t_max: float) -> torch.Tensor:
    """Phase B: re-integrate and store each trajectory on a normalized time grid.

    Returns (B, num_points, 3) of positions [x, y, z] at t = j/(num_points-1) * T_i.
    """
    device = features.device
    batch = features.shape[0]

    state = initial_states(features)
    # Absolute target times per sample: tgrid * T_i
    tgrid = torch.linspace(0.0, 1.0, num_points, device=device, dtype=torch.float32)
    t_abs = tgrid.unsqueeze(0).expand(batch, -1) * impact_time.unsqueeze(1)

    trajectory = torch.zeros(batch, num_points, 3, device=device, dtype=torch.float32)
    idx = torch.zeros(batch, dtype=torch.long, device=device)

    t = 0.0
    while idx.min() < num_points and t < t_max:
        prev = state
        state = _rk4_step(state, k, dt, g)
        t += dt
        t_prev = t - dt

        # Fill every target time that falls inside this integration interval.
        while True:
            remaining = idx < num_points
            if not remaining.any():
                break
            rows = remaining.nonzero(as_tuple=False).squeeze(1)
            nxt = t_abs[rows, idx[rows]]
            can_fill = nxt <= t
            if not can_fill.any():
                break
            sel = rows[can_fill]
            alpha = ((t_abs[sel, idx[sel]] - t_prev) / dt).clamp(0.0, 1.0)
            traj_from = prev[sel][:, 0:3]
            traj_to = state[sel][:, 0:3]
            trajectory[sel, idx[sel]] = traj_from + alpha.unsqueeze(1) * (traj_to - traj_from)
            idx[sel] += 1

    if idx.min() < num_points:
        raise RuntimeError(
            f"t_max={t_max}s reached before {num_points} points were sampled for all "
            f"trajectories ({int(idx.min())}/{num_points} filled)."
        )
    return trajectory


def simulate_trajectories(features: torch.Tensor, num_points: int = config.N_TIME,
                          dt: float = config.DT, t_max: float = config.T_MAX,
                          g: float = config.G, device: str | None = None) -> dict:
    """Simulate a batch of ballistic flights with quadratic drag.

    Args:
        features: (B, 7) tensor with columns [v0, theta_deg, psi_deg, mass, area, cd, rho].
        num_points: number of samples per trajectory (normalized time grid).
        dt: RK4 time step (s).
        t_max: safety bound on flight time (s).
        g: gravitational acceleration (m/s^2).
        device: torch device; defaults to CUDA if available.

    Returns:
        dict with:
            trajectory:  (B, num_points, 3) positions [x, y, z]
            impact:      (B, 2) ground impact position [x, z]
            flight_time: (B,) time of flight (s)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    features = features.to(device=device, dtype=torch.float32)

    k = drag_coefficient_k(features)
    impact_time, impact_pos = _find_impact(features, k, dt, g, t_max)
    trajectory = _resample_trajectories(features, k, impact_time,
                                        num_points, dt, g, t_max)

    return {
        "trajectory": trajectory,     # (B, N, 3)
        "impact": impact_pos[:, [0, 2]],  # (B, 2)  [x, z]
        "flight_time": impact_time,       # (B,)
    }

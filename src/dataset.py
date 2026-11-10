"""Dataset generation: sample launch parameters, simulate, split, normalize.

Pipeline:

    sample_parameters  -> (n, 7) raw launch parameters (config.PARAM_ORDER layout)
    generate_dataset   -> simulate every launch with physics.simulate_trajectories
    split_indices      -> seeded train / val / test permutation
    compute_norm_stats -> mean / std from the training split only
    get_dataloaders    -> normalized train / val / test DataLoaders

Files written under config.DATA_DIR:

    dataset.pt     raw tensors + split indices
    norm_stats.pt  normalization statistics (recomputed by get_dataloaders)
    summary.csv    one row per launch, for inspection (optional, --csv)

Generate from the command line:

    python -m src.dataset --samples 50000 --csv
"""

import argparse
import csv

import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config, physics

DATASET_PATH = config.DATA_DIR / "dataset.pt"
NORM_STATS_PATH = config.DATA_DIR / "norm_stats.pt"
SUMMARY_CSV_PATH = config.DATA_DIR / "summary.csv"

TARGET_KEYS = ("trajectory", "impact", "flight_time")


# ---------------------------------------------------------------------------
# Sampling and simulation
# ---------------------------------------------------------------------------
def sample_parameters(n: int, seed: int = config.SEED) -> torch.Tensor:
    """Draw n launch configurations uniformly from config.PARAM_RANGES.

    Returns (n, 7) float32 with columns in config.PARAM_ORDER.
    """
    generator = torch.Generator().manual_seed(seed)
    low = torch.tensor([config.PARAM_RANGES[p][0] for p in config.PARAM_ORDER])
    high = torch.tensor([config.PARAM_RANGES[p][1] for p in config.PARAM_ORDER])
    unit = torch.rand(n, len(config.PARAM_ORDER), generator=generator)
    return low + unit * (high - low)


def generate_dataset(n: int = config.DATASET_SIZE, device: str | None = None,
                     seed: int = config.SEED, chunk_size: int = 10_000) -> dict:
    """Sample n launches and simulate them in chunks.

    Returns a dict of CPU tensors:
        features:    (n, 7)          raw launch parameters
        trajectory:  (n, N_TIME, 3)  positions [x, y, z] on normalized time
        impact:      (n, 2)          ground impact [x, z]
        flight_time: (n,)            seconds
    """
    features = sample_parameters(n, seed)
    chunks = {key: [] for key in TARGET_KEYS}
    for start in range(0, n, chunk_size):
        out = physics.simulate_trajectories(features[start:start + chunk_size],
                                            device=device)
        for key in TARGET_KEYS:
            chunks[key].append(out[key].cpu())

    data = {"features": features}
    data.update({key: torch.cat(parts) for key, parts in chunks.items()})
    return data


def split_indices(n: int, seed: int = config.SEED) -> dict:
    """Seeded random split into train / val / test index tensors."""
    generator = torch.Generator().manual_seed(seed + 1)
    perm = torch.randperm(n, generator=generator)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
# Feature encoding and normalization
# ---------------------------------------------------------------------------
def encode_features(features: torch.Tensor) -> torch.Tensor:
    """Turn raw launch parameters into the model's input features.

    Args:
        features: (B, 7) raw parameters, columns
                  [v0, theta_deg, psi_deg, mass, area, cd, rho].

    Returns:
        (B, 6) float tensor with columns
        [v0, sin(theta), cos(theta), sin(psi), cos(psi), log(k)].

    sin/cos of the angles remove the azimuth wrap-around and give the model
    the launch velocity direction directly. mass, area, cd and rho only affect
    the flight through k = 0.5 * cd * rho * area / mass, so they collapse to
    one input; the log spreads its three orders of magnitude evenly.
    Normalization is applied afterwards.
    """
    v0 = features[:, 0:1]
    theta = torch.deg2rad(features[:, 1:2])
    psi = torch.deg2rad(features[:, 2:3])

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    sin_psi = torch.sin(psi)
    cos_psi = torch.cos(psi)

    k = physics.drag_coefficient_k(features)
    log_k = torch.log(k.clamp(min=1e-9))  # clamp keeps log finite when cd = 0
    return torch.cat([v0, sin_theta, cos_theta, sin_psi, cos_psi, log_k], dim=1)

def _mean_std(x: torch.Tensor, dims) -> tuple[torch.Tensor, torch.Tensor]:
    return x.mean(dim=dims), x.std(dim=dims).clamp(min=1e-8)


def compute_norm_stats(data: dict, train_idx: torch.Tensor) -> dict:
    """Mean / std of inputs and targets, from the training split only.

    Trajectory statistics are one value per coordinate (x, y, z), shared over
    all time steps, so normalization rescales curves without reshaping them.
    """
    stats = {}
    inputs = encode_features(data["features"][train_idx])
    stats["features_mean"], stats["features_std"] = _mean_std(inputs, 0)
    stats["trajectory_mean"], stats["trajectory_std"] = _mean_std(
        data["trajectory"][train_idx], (0, 1))
    stats["impact_mean"], stats["impact_std"] = _mean_std(
        data["impact"][train_idx], 0)
    stats["flight_time_mean"], stats["flight_time_std"] = _mean_std(
        data["flight_time"][train_idx], 0)
    return stats


def normalize(x: torch.Tensor, stats: dict, key: str) -> torch.Tensor:
    return (x - stats[f"{key}_mean"].to(x.device)) / stats[f"{key}_std"].to(x.device)


def denormalize(x: torch.Tensor, stats: dict, key: str) -> torch.Tensor:
    return x * stats[f"{key}_std"].to(x.device) + stats[f"{key}_mean"].to(x.device)


# ---------------------------------------------------------------------------
# Persistence and loaders
# ---------------------------------------------------------------------------
def save_dataset(data: dict, splits: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({**data, "splits": splits}, DATASET_PATH)


def load_dataset() -> tuple[dict, dict]:
    """Returns (data, splits) as saved by save_dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Generate it with: python -m src.dataset")
    payload = torch.load(DATASET_PATH, weights_only=True)
    splits = payload.pop("splits")
    return payload, splits


def get_dataloaders(batch_size: int = config.BATCH_SIZE) -> tuple[dict, dict]:
    """Normalized DataLoaders for the saved dataset.

    Returns (loaders, stats). Each batch is
    (inputs, trajectory, impact, flight_time), all normalized. Statistics are
    recomputed here so they always match the current encode_features().
    """
    data, splits = load_dataset()
    stats = compute_norm_stats(data, splits["train"])
    torch.save(stats, NORM_STATS_PATH)

    inputs = normalize(encode_features(data["features"]), stats, "features")
    targets = [normalize(data[key], stats, key) for key in TARGET_KEYS]

    loaders = {}
    for name, idx in splits.items():
        subset = TensorDataset(inputs[idx], *(t[idx] for t in targets))
        loaders[name] = DataLoader(subset, batch_size=batch_size,
                                   shuffle=(name == "train"))
    return loaders, stats


# ---------------------------------------------------------------------------
# Checks, export, CLI
# ---------------------------------------------------------------------------
def validate_dataset(data: dict, tolerance: float = 1.0) -> None:
    """Raise if the simulated data is not physically sensible."""
    n = data["features"].shape[0]
    traj = data["trajectory"]
    assert traj.shape == (n, config.N_TIME, 3), traj.shape
    assert data["impact"].shape == (n, 2), data["impact"].shape
    assert data["flight_time"].shape == (n,), data["flight_time"].shape
    for key, value in data.items():
        assert torch.isfinite(value).all(), f"non-finite values in {key}"
    assert traj[:, :, 1].min() >= -tolerance, "trajectory dips below ground"
    assert traj[:, -1, 1].abs().max() <= tolerance, "last point is not on the ground"
    end_error = (traj[:, -1][:, [0, 2]] - data["impact"]).abs().max()
    assert end_error <= tolerance, f"trajectory end != impact ({end_error:.3f} m)"
    assert (data["flight_time"] > 0).all(), "non-positive flight time"
    assert (data["flight_time"] < config.T_MAX).all(), "flight hit the T_MAX bound"


def export_summary_csv(data: dict, splits: dict) -> None:
    """One row per launch: parameters, impact point, range, flight time, split."""
    split_of = [""] * data["features"].shape[0]
    for name, idx in splits.items():
        for i in idx.tolist():
            split_of[i] = name

    ground_range = torch.linalg.vector_norm(data["impact"], dim=1)
    max_altitude = data["trajectory"][:, :, 1].max(dim=1).values
    with open(SUMMARY_CSV_PATH, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*config.PARAM_ORDER, "impact_x", "impact_z",
                         "range", "max_altitude", "flight_time", "split"])
        for i in range(len(split_of)):
            writer.writerow([
                *(f"{v:.6g}" for v in data["features"][i].tolist()),
                f"{data['impact'][i, 0]:.3f}", f"{data['impact'][i, 1]:.3f}",
                f"{ground_range[i]:.3f}", f"{max_altitude[i]:.3f}",
                f"{data['flight_time'][i]:.4f}", split_of[i],
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the trajectory dataset.")
    parser.add_argument("--samples", type=int, default=config.DATASET_SIZE)
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--csv", action="store_true",
                        help=f"also write {SUMMARY_CSV_PATH.name}")
    args = parser.parse_args()

    data = generate_dataset(args.samples, device=args.device, seed=args.seed)
    validate_dataset(data)
    splits = split_indices(args.samples, seed=args.seed)
    save_dataset(data, splits)
    torch.save(compute_norm_stats(data, splits["train"]), NORM_STATS_PATH)
    if args.csv:
        export_summary_csv(data, splits)

    ground_range = torch.linalg.vector_norm(data["impact"], dim=1)
    print(f"Saved {args.samples} trajectories to {DATASET_PATH}")
    for key, value in data.items():
        print(f"  {key:<12}{tuple(value.shape)}")
    print("  splits      " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"  range       {ground_range.min():.0f} .. {ground_range.max():.0f} m")
    print(f"  flight time {data['flight_time'].min():.1f} .. "
          f"{data['flight_time'].max():.1f} s")


if __name__ == "__main__":
    main()

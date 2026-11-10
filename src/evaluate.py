"""Test-set metrics, baselines and plots.

The trained MLP is compared with two baselines on the held-out test split:

    vacuum:  closed-form ballistic flight with drag ignored
    linear:  least-squares linear regression on the same encoded inputs

All metrics are reported in physical units (metres, seconds).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import config, dataset
from .model import TrajectoryMLP

TARGET_KEYS = dataset.TARGET_KEYS


# ---------------------------------------------------------------------------
# Predictors (each returns physical-unit trajectory / impact / flight_time)
# ---------------------------------------------------------------------------
def load_model(device: str) -> tuple[TrajectoryMLP, dict]:
    checkpoint = torch.load(config.MODEL_PATH, weights_only=True, map_location=device)
    model = TrajectoryMLP(checkpoint["in_dim"], checkpoint["hidden_dims"],
                          checkpoint["n_time"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["stats"]


@torch.no_grad()
def predict_mlp(model: TrajectoryMLP, stats: dict, features: torch.Tensor,
                device: str) -> dict:
    inputs = dataset.normalize(dataset.encode_features(features), stats, "features")
    out = model(inputs.to(device))
    return {key: dataset.denormalize(out[key].cpu(), stats, key)
            for key in TARGET_KEYS}


def predict_vacuum(features: torch.Tensor) -> dict:
    """Closed-form flight with no air drag, on the same normalized time grid."""
    v0 = features[:, 0:1]
    theta = torch.deg2rad(features[:, 1:2])
    psi = torch.deg2rad(features[:, 2:3])
    flight_time = 2.0 * v0 * torch.sin(theta) / config.G           # (B, 1)
    t = torch.linspace(0.0, 1.0, config.N_TIME).unsqueeze(0) * flight_time
    horizontal = v0 * torch.cos(theta) * t
    trajectory = torch.stack([
        horizontal * torch.cos(psi),
        v0 * torch.sin(theta) * t - 0.5 * config.G * t ** 2,
        horizontal * torch.sin(psi),
    ], dim=2)
    return {"trajectory": trajectory,
            "impact": trajectory[:, -1][:, [0, 2]],
            "flight_time": flight_time.squeeze(1)}


def predict_linear(data: dict, splits: dict, stats: dict) -> dict:
    """Linear regression fitted on the training split, applied to the test split."""
    def design(idx):
        x = dataset.normalize(dataset.encode_features(data["features"][idx]),
                              stats, "features")
        return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)

    def flat_targets(idx):
        return torch.cat([
            dataset.normalize(data["trajectory"][idx], stats, "trajectory").flatten(1),
            dataset.normalize(data["impact"][idx], stats, "impact"),
            dataset.normalize(data["flight_time"][idx], stats, "flight_time").unsqueeze(1),
        ], dim=1)

    weights = torch.linalg.lstsq(design(splits["train"]),
                                 flat_targets(splits["train"])).solution
    pred = design(splits["test"]) @ weights
    n_traj = config.N_TIME * 3
    return {
        "trajectory": dataset.denormalize(
            pred[:, :n_traj].reshape(-1, config.N_TIME, 3), stats, "trajectory"),
        "impact": dataset.denormalize(pred[:, n_traj:n_traj + 2], stats, "impact"),
        "flight_time": dataset.denormalize(pred[:, -1], stats, "flight_time"),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred: dict, truth: dict) -> dict:
    point_error = torch.linalg.vector_norm(
        pred["trajectory"] - truth["trajectory"], dim=2)             # (B, N)
    impact_error = torch.linalg.vector_norm(pred["impact"] - truth["impact"], dim=1)
    true_range = torch.linalg.vector_norm(truth["impact"], dim=1)
    time_error = (pred["flight_time"] - truth["flight_time"]).abs()
    return {
        "traj_mean_m": point_error.mean().item(),
        "impact_mean_m": impact_error.mean().item(),
        "impact_median_m": impact_error.median().item(),
        "impact_rel_median_pct": (100.0 * impact_error / true_range).median().item(),
        "time_mae_s": time_error.mean().item(),
        "impact_error": impact_error,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _downrange(trajectory: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(trajectory[..., [0, 2]], dim=-1)


def plot_loss_curves(history: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="train")
    ax.plot(epochs, history["val"], label="validation")
    ax.axvline(history["best_epoch"], color="grey", linestyle=":",
               label=f"best epoch ({history['best_epoch']})")
    ax.set(xlabel="epoch", ylabel="loss (normalized units)", yscale="log",
           title="Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_sample_trajectories(truth: dict, predictions: dict, count: int = 4) -> None:
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 3.6))
    picks = torch.linspace(0, truth["trajectory"].shape[0] - 1, count).long()
    styles = {"MLP": "--", "linear": ":"}
    for ax, i in zip(axes, picks):
        true_traj = truth["trajectory"][i]
        ax.plot(_downrange(true_traj) / 1000, true_traj[:, 1] / 1000,
                color="black", label="simulator")
        for name, style in styles.items():
            traj = predictions[name]["trajectory"][i]
            ax.plot(_downrange(traj) / 1000, traj[:, 1] / 1000, style, label=name)
        ax.set(xlabel="downrange (km)", ylabel="altitude (km)")
    axes[0].legend()
    fig.suptitle("Test trajectories: simulator vs predictions")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "sample_trajectories.png", dpi=150)
    plt.close(fig)


def plot_impact_errors(metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, result in metrics.items():
        errors = result["impact_error"].clamp(min=1e-2)
        bins = torch.logspace(-1, 5, 60).tolist()
        ax.hist(errors.tolist(), bins=bins, alpha=0.6, label=name)
    ax.set(xscale="log", xlabel="impact point error (m)", ylabel="test launches",
           title="Impact error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "impact_error.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def evaluate(device: str | None = None, history: dict | None = None) -> dict:
    """Score the saved model and the baselines on the test split, save plots."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data, splits = dataset.load_dataset()
    test = splits["test"]
    truth = {key: data[key][test] for key in TARGET_KEYS}
    features = data["features"][test]

    model, stats = load_model(device)
    predictions = {
        "MLP": predict_mlp(model, stats, features, device),
        "linear": predict_linear(data, splits, stats),
        "vacuum": predict_vacuum(features),
    }
    metrics = {name: compute_metrics(pred, truth)
               for name, pred in predictions.items()}

    print(f"  test launches: {len(test)}")
    print(f"  {'model':<8}{'traj mean':>12}{'impact mean':>13}{'impact median':>15}"
          f"{'impact / range':>16}{'time MAE':>10}")
    for name, m in metrics.items():
        print(f"  {name:<8}{m['traj_mean_m']:>10.1f} m{m['impact_mean_m']:>11.1f} m"
              f"{m['impact_median_m']:>13.1f} m{m['impact_rel_median_pct']:>14.2f} %"
              f"{m['time_mae_s']:>8.2f} s")

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if history is not None:
        plot_loss_curves(history)
    plot_sample_trajectories(truth, predictions)
    plot_impact_errors(metrics)
    print(f"  plots saved to {config.PLOTS_DIR}")
    return metrics

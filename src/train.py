"""Training loop with learning-rate decay, early stopping and checkpointing."""

import torch
from torch.nn import functional as F

from . import config, dataset
from .model import TrajectoryMLP

BATCH_KEYS = ("trajectory", "impact", "flight_time")


def combined_loss(pred: dict, target: dict,
                  weights: dict = config.LOSS_WEIGHTS) -> torch.Tensor:
    """Weighted sum of the per-output mean squared errors (normalized units).

    Every output is standardized, so equal weights treat a one-standard-
    deviation error the same for the trajectory, the impact point and the
    flight time. Raise a weight in config.LOSS_WEIGHTS to make the model care
    more about that output.
    """
    return sum(weights[key] * F.mse_loss(pred[key], target[key])
               for key in BATCH_KEYS)


def _run_epoch(model, loader, device, optimizer=None) -> float:
    """One pass over a loader; trains when an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for inputs, *targets in loader:
            inputs = inputs.to(device)
            target = {key: t.to(device) for key, t in zip(BATCH_KEYS, targets)}
            loss = combined_loss(model(inputs), target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * inputs.shape[0]
            count += inputs.shape[0]
    return total / count


def train(epochs: int = config.EPOCHS, device: str | None = None,
          seed: int = config.SEED) -> dict:
    """Train on the saved dataset and checkpoint the best validation model.

    Returns the loss history: {"train": [...], "val": [...], "best_epoch": int}.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    loaders, stats = dataset.get_dataloaders()
    in_dim = loaders["train"].dataset.tensors[0].shape[1]
    model = TrajectoryMLP(in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5)

    history = {"train": [], "val": [], "best_epoch": 0}
    best_val, stale = float("inf"), 0
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, loaders["train"], device, optimizer)
        val_loss = _run_epoch(model, loaders["val"], device)
        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val, stale = val_loss, 0
            history["best_epoch"] = epoch
            torch.save({
                "model_state": model.state_dict(),
                "in_dim": in_dim,
                "hidden_dims": list(config.HIDDEN_DIMS),
                "n_time": config.N_TIME,
                "stats": stats,
            }, config.MODEL_PATH)
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:>3}  train {train_loss:.5f}  "
                  f"val {val_loss:.5f}  lr {lr:.1e}")
        if stale >= config.PATIENCE:
            print(f"  early stop at epoch {epoch} "
                  f"(no improvement for {config.PATIENCE} epochs)")
            break

    print(f"  best val loss {best_val:.5f} at epoch {history['best_epoch']}, "
          f"saved to {config.MODEL_PATH}")
    return history

"""Export the trained model for the static website in docs/.

Writes:
    docs/model.bin   every layer's weights then biases, as little-endian float32
    docs/model.json  layer shapes, normalization statistics, parameter ranges,
                     test-set metrics and a few reference predictions the page
                     uses to check its JavaScript against this Python code

Run after training:

    python -m src.export_web
"""

import json

import torch
from torch import nn

from . import config, dataset, evaluate, physics

DOCS_DIR = config.ROOT / "docs"
METRIC_KEYS = ("traj_mean_m", "impact_mean_m", "impact_median_m",
               "impact_rel_median_pct", "time_mae_s")


def export() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    model, stats = evaluate.load_model("cpu")

    named_layers = [("trunk", m) for m in model.trunk if isinstance(m, nn.Linear)]
    named_layers += [("trajectory", model.trajectory_head),
                     ("impact", model.impact_head),
                     ("flight_time", model.flight_time_head)]

    layers, blobs = [], []
    for role, layer in named_layers:
        layers.append({"role": role, "in": layer.in_features,
                       "out": layer.out_features})
        blobs.append(layer.weight.detach().flatten())   # row-major (out, in)
        blobs.append(layer.bias.detach())
    weights = torch.cat(blobs).to(torch.float32)
    (DOCS_DIR / "model.bin").write_bytes(weights.numpy().astype("<f4").tobytes())

    # Reference launches: the page recomputes these in JavaScript and compares.
    features = dataset.sample_parameters(5, seed=123)
    mlp = evaluate.predict_mlp(model, stats, features, "cpu")
    sim = physics.simulate_trajectories(features, device="cpu")
    checks = [{
        "features": features[i].tolist(),
        "mlp_impact": mlp["impact"][i].tolist(),
        "mlp_flight_time": mlp["flight_time"][i].item(),
        "sim_impact": sim["impact"][i].tolist(),
        "sim_flight_time": sim["flight_time"][i].item(),
    } for i in range(features.shape[0])]

    metrics = evaluate.evaluate()
    _, splits = dataset.load_dataset()

    meta = {
        "param_order": config.PARAM_ORDER,
        "param_ranges": {k: list(v) for k, v in config.PARAM_RANGES.items()},
        "g": config.G,
        "dt": config.DT,
        "t_max": config.T_MAX,
        "n_time": config.N_TIME,
        "layers": layers,
        "stats": {key: value.flatten().tolist() for key, value in stats.items()},
        "train_launches": len(splits["train"]),
        "test_launches": len(splits["test"]),
        "metrics": {name: {key: m[key] for key in METRIC_KEYS}
                    for name, m in metrics.items()},
        "checks": checks,
    }
    (DOCS_DIR / "model.json").write_text(json.dumps(meta, indent=1))
    print(f"  exported {weights.numel()} weights to {DOCS_DIR}")


if __name__ == "__main__":
    export()

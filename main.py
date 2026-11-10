"""Single entry point: generate the dataset -> train the model -> evaluate it."""

import argparse

import torch

from src import config, dataset, evaluate, train


def ensure_dataset(samples: int, device: str, seed: int, regenerate: bool) -> None:
    """Reuse the saved dataset when it has the requested size, else generate it."""
    if not regenerate and dataset.DATASET_PATH.exists():
        data, _ = dataset.load_dataset()
        if data["features"].shape[0] == samples:
            print(f"  reusing {dataset.DATASET_PATH} ({samples} launches)")
            return
    print(f"  simulating {samples} launches on {device} ...")
    data = dataset.generate_dataset(samples, device=device, seed=seed)
    dataset.validate_dataset(data)
    dataset.save_dataset(data, dataset.split_indices(samples, seed=seed))
    print(f"  saved {dataset.DATASET_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=config.DATASET_SIZE,
                        help="number of random launch configurations")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--regenerate", action="store_true",
                        help="rebuild the dataset even if one is already saved")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/3] Dataset")
    ensure_dataset(args.samples, device, args.seed, args.regenerate)
    print("[2/3] Training")
    history = train.train(epochs=args.epochs, device=device, seed=args.seed)
    print("[3/3] Evaluation")
    evaluate.evaluate(device=device, history=history)


if __name__ == "__main__":
    main()

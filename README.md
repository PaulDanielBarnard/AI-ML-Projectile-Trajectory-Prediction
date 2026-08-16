# AI/ML Projectile Trajectory Prediction

Simulates 3D ballistic missile trajectories with quadratic air drag, trains a PyTorch
neural network to predict full trajectories and impact points from launch parameters,
and evaluates the model against analytic and linear-regression baselines.

## Project structure

```
├── README.md
├── requirements.txt
├── main.py                  # single entry point: generate -> train -> evaluate
├── src/
│   ├── __init__.py
│   ├── config.py            # physics ranges, hyperparameters, file paths
│   ├── physics.py           # 3D drag + gravity RK4 simulator
│   ├── dataset.py           # sampling, batch simulation, normalization
│   ├── model.py             # PyTorch MLP
│   ├── train.py             # training loop, early stopping, checkpointing
│   └── evaluate.py          # metrics, baselines, plots
├── data/                    # generated dataset + normalization stats (runtime)
├── models/                  # saved model checkpoint (runtime)
└── plots/                   # output figures (runtime)
```

## Requirements

- Python 3.14+
- NVIDIA GPU with CUDA 12.8 support (e.g. RTX 50-series), or CPU-only mode

## Installation

Create a virtual environment (one time):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

`torch` from PyPI installs the CUDA build. For the exact GPU wheel (CUDA 12.8,
required for RTX 50-series / Blackwell), use:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify the GPU is detected:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True NVIDIA GeForce RTX 5070 Laptop GPU`.

## Usage

With the venv active:

```powershell
python main.py
```

Or without activating the venv:

```powershell
.venv\Scripts\python main.py
```

Optional flags (see `python main.py --help`):

```
--samples  N      number of random launch configurations (default 50000)
--epochs   N      training epochs (default 100)
--device   str    'cuda' or 'cpu' (default auto-detect)
--seed     N      random seed (default 42)
```

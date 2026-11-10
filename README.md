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
│   ├── evaluate.py          # metrics, baselines, plots
│   └── export_web.py        # writes the trained model into docs/ for the website
├── docs/                    # static website (GitHub Pages): live simulator vs model
├── data/                    # generated dataset + normalization stats (runtime)
├── models/                  # saved model checkpoint (runtime)
└── plots/                   # output figures (runtime)
```

## Requirements

- Python 3.14+
- NVIDIA GPU (the requirements install the CUDA 13.2 build of PyTorch), or CPU-only mode

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

On Windows, `torch` from PyPI is CPU-only, so `requirements.txt` adds the PyTorch
`cu132` index to get the CUDA build. Verify the GPU is detected:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` followed by your GPU's name.

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
--regenerate      rebuild the dataset even if one is already saved
```

## Website

`docs/` is a static page that runs the simulator and the trained network side by
side in the browser. After training, refresh the model it uses:

```powershell
python -m src.export_web
```

Preview it locally (the page must be served, not opened from disk):

```powershell
python -m http.server 8765 --directory docs
```

To publish, push to GitHub and set Settings → Pages → "Deploy from a branch" to
`master` and `/docs`.

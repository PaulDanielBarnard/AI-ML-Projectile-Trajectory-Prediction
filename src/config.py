"""Central configuration: paths, physics constants, parameter ranges, hyperparameters."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
G = 9.81            # gravitational acceleration (m/s^2)
DT = 0.02           # RK4 integration step (s)
T_MAX = 250.0       # safety bound on flight time (s); max no-drag flight ~203 s

# ---------------------------------------------------------------------------
# Trajectory output resolution
# ---------------------------------------------------------------------------
N_TIME = 100        # points per trajectory on a normalized time grid

# ---------------------------------------------------------------------------
# Launch parameter ranges [low, high]
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "v0":    (100.0, 1000.0),   # launch speed (m/s)
    "theta": (15.0, 85.0),      # elevation angle (deg)
    "psi":   (0.0, 360.0),      # azimuth angle (deg)
    "mass":  (50.0, 500.0),     # mass (kg)
    "area":  (0.1, 0.5),        # cross-sectional area (m^2)
    "cd":    (0.05, 0.4),       # drag coefficient
    "rho":   (0.5, 1.5),        # air density (kg/m^3)
}
PARAM_ORDER = ["v0", "theta", "psi", "mass", "area", "cd", "rho"]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_SIZE = 50_000
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
HIDDEN_DIMS = [128, 256, 128]
EPOCHS = 100
BATCH_SIZE = 512
LR = 1e-3

__version__ = "0.1.0"

from pathlib import Path

# Directory containing this __init__.py
PACKAGE_DIR = Path(__file__).resolve().parent

# Paths to files
path_to_balmers = PACKAGE_DIR / "balmers" / "balmers.dat"
path_to_models = PACKAGE_DIR / "ssp_models" / "smiles_bimodal.npz"

if not path_to_balmers.is_file():
    raise FileNotFoundError(path_to_balmers)

if not path_to_models.is_file():
    raise FileNotFoundError(path_to_models)

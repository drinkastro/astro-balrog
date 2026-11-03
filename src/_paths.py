"""Locations of the data files that ship with the package."""

from pathlib import Path

# Directory containing this file (i.e. the installed package directory)
PACKAGE_DIR = Path(__file__).resolve().parent

# Paths to files
path_to_balmers = PACKAGE_DIR / "balmers" / "balmers.dat"
path_to_models = PACKAGE_DIR / "ssp_models" / "smiles_bimodal.npz"

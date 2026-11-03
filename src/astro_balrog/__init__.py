import warnings
from importlib.metadata import PackageNotFoundError, version

from ._paths import path_to_balmers, path_to_models
from .fit import fit_BR
from .measure import measure_BR

try:
    __version__ = version("astro-balrog")
except PackageNotFoundError:          # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

__all__ = ["measure_BR", "fit_BR", "plot_BR", "cornerplot",
           "path_to_balmers", "path_to_models", "__version__"]

# The data files are looked up when they are needed; warn early (but do not fail on import).
for _p in (path_to_balmers, path_to_models):
    if not _p.is_file():
        warnings.warn(f"Data file not found: {_p}. Functions that need it will fail "
                      f"until it is added (see the README).", RuntimeWarning, stacklevel=2)

def __getattr__(name):
    if name == "plotting":
        import importlib
        return importlib.import_module(".plotting", __name__)
    if name in ("plot_BR", "cornerplot"):
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

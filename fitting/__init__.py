"""Superconductor V-I (Data Fitting) package.

Everything the Data Fitting tab needs is in this folder:

* ``service.py``    — pure-Python fitting math (numpy / scipy only).
* ``extras.py``     — graph-settings dialog, export dialog, presets, widgets.
* ``tab.py``        — the Qt tab layout + action functions.
* ``standalone.py`` — minimal QMainWindow launcher.
* ``__main__.py``   — ``python -m src.fitting`` entry point.

Standalone use:
  From the project root: ``python -m src.fitting``
  As its own project: copy this folder, then ``python -m fitting``
  (Only PyQt5, pyqtgraph, numpy, scipy, nptdms are required.)

Embedded use:
  Call :func:`setup_data_fitting_tab_layout` on a host app whose
  ``ui_state.data_fitting_tab`` is a ``QWidget`` to populate. The host must
  expose the ``data_fitting_*`` method stubs used by the tab's signal
  wiring — see ``standalone.DataFittingWindow`` for a minimal example.

The math layer (``service``) has no Qt dependency and can be imported in
headless scripts: ``from src.fitting.service import run_full_fit``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .service import (
    FitResult,
    FitSettings,
    robust_view_range,
    run_full_fit,
)

BASE_VERSION = "v1.0rc1"
# Keep this fallback in sync with repo commit count when shipping artifacts
# outside a git checkout (e.g., PyInstaller one-file EXE).
FALLBACK_BUILD_NUMBER = 29


def _git_build_number() -> int | None:
    """Return `git rev-list --count HEAD` when the repo metadata exists."""
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").exists():
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(out)
    except Exception:
        return None


def get_app_version_label() -> str:
    build = _git_build_number() or FALLBACK_BUILD_NUMBER
    return f"{BASE_VERSION} (build {build})"


__version__ = get_app_version_label()

# Tab-layer symbols require PyQt5 / pyqtgraph — exposed lazily via __getattr__
# so that plain ``from src.fitting.service import ...`` works without Qt.
_LAZY_TAB = {
    "setup_data_fitting_tab_layout",
    "open_file_dialog",
    "refresh_current_recording",
    "refresh_preview",
    "run_fit",
    "load_metadata_from_tdms",
    "robust_view",
    "reset_view",
    "toggle_zoom",
    "region_mode_changed",
    "sync_region_to_inputs",
}


def __getattr__(name):
    if name in _LAZY_TAB:
        from . import tab as _tab
        return getattr(_tab, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BASE_VERSION",
    "FitResult",
    "FitSettings",
    "FALLBACK_BUILD_NUMBER",
    "get_app_version_label",
    "robust_view_range",
    "run_full_fit",
    "__version__",
    *sorted(_LAZY_TAB),
]

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
    DEFAULT_RESAMPLE_TARGET_SPS,
    FitResult,
    FitSettings,
    compute_resample_avg_window,
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


def _ensure_tab_patches_applied() -> None:
    """Apply the runtime ``tab.py`` patches exactly once.

    * ``_pct_anchor_patch`` anchors Step 3/4/5 fit windows to the untrimmed
      sweep and persists ``linear_fit_window`` in TDMS metadata.
    * ``_resample_patch`` removes the legacy "Resample everything to 100 S/s"
      checkbox and forces ``_apply_resample_to_100sps`` to always run, so
      every load converges on ~100 S/s without an opt-out.

    Wrapping is done lazily so headless ``service`` imports don't pull in
    Qt or pyqtgraph. Patch failures are swallowed — the worst case is the
    original (unpatched) behaviour, never a crashed tab import.
    """
    for module_name in ("_pct_anchor_patch", "_resample_patch"):
        try:
            module = __import__(
                f"{__name__}.{module_name}", fromlist=[module_name]
            )
        except Exception:
            continue
        try:
            module.apply_patches()
        except Exception:
            pass


def __getattr__(name):
    if name in _LAZY_TAB:
        from . import tab as _tab
        _ensure_tab_patches_applied()
        return getattr(_tab, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BASE_VERSION",
    "DEFAULT_RESAMPLE_TARGET_SPS",
    "FitResult",
    "FitSettings",
    "FALLBACK_BUILD_NUMBER",
    "compute_resample_avg_window",
    "get_app_version_label",
    "robust_view_range",
    "run_full_fit",
    "__version__",
    *sorted(_LAZY_TAB),
]

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
import sys
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


def _ensure_tab_patches_applied() -> None:
    """Apply runtime wrappers to ``tab`` exactly once.

    ``_pct_anchor_patch`` anchors Step 3/4/5 fit windows to the untrimmed
    sweep and persists ``linear_fit_window`` in TDMS metadata.
    ``_preset_state_patch`` extends ``_settings_to_preset`` /
    ``_apply_preset`` so the JSON preset captures every Step 1-5, Config
    and Settings widget. ``_auto_resample_patch`` auto-fills the AVG
    textbox so high-rate DAQUniversal recordings (kS/s) drop down to
    ~100 S/s — the regime the IEC log-log decade fit is calibrated for —
    and keeps the Plot summary preview row's Avg/Effective rate cells
    consistent with the textbox. All modules are imported lazily so
    headless ``service`` imports don't pull in Qt or pyqtgraph.
    """
    for module_name in (
        "_pct_anchor_patch",
        "_preset_state_patch",
        "_auto_resample_patch",
    ):
        try:
            module = __import__(f"{__name__}.{module_name}", fromlist=["apply_patches"])
        except Exception:
            continue
        try:
            module.apply_patches()
        except Exception:
            # Patch failures must not break tab import — fall back to the
            # original (unpatched) behaviour rather than crashing the app.
            pass


def __getattr__(name):
    if name in _LAZY_TAB:
        from . import tab as _tab
        _ensure_tab_patches_applied()
        return getattr(_tab, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ----------------------------------------------------------------------------
# Reliable patch activation for ``fitting.tab``
# ----------------------------------------------------------------------------
#
# ``__getattr__`` above only fires for ``fitting.<name>`` lookups. Both the
# standalone window and DAQUniversal's tab shim import names directly with
# ``from fitting.tab import …``, which Python resolves as a submodule lookup
# and bypasses ``__getattr__``. Without the hook below, ``_ensure_tab_patches_applied``
# would never fire in those flows and the runtime fixes (Step 2 trim editors,
# 0-d array guards, …) would silently no-op.
#
# A ``sys.meta_path`` finder watches for the ``fitting.tab`` import. It lets
# the normal import machinery resolve the spec and load the module, then
# applies the patches once execution finishes. The finder is idempotent and
# avoids re-entering itself by skipping its own entry on the path lookup.

_TAB_MODULE_NAME = f"{__name__}.tab"


class _TabPatchFinder:
    """``sys.meta_path`` finder that applies patches after ``fitting.tab`` loads."""

    _resolving = False

    def find_spec(self, fullname, path, target=None):  # noqa: D401 - stdlib API
        if fullname != _TAB_MODULE_NAME:
            return None
        if _TabPatchFinder._resolving:
            # Avoid recursing into ourselves while delegating to the next
            # finder; the original loader handles its own work.
            return None
        _TabPatchFinder._resolving = True
        try:
            for finder in list(sys.meta_path):
                if isinstance(finder, _TabPatchFinder):
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except Exception:
                    spec = None
                if spec is None or spec.loader is None:
                    continue
                spec.loader = _TabPatchLoader(spec.loader)
                return spec
        finally:
            _TabPatchFinder._resolving = False
        return None


class _TabPatchLoader:
    """Wrap ``fitting.tab``'s loader so patches run once execution finishes."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        creator = getattr(self._wrapped, "create_module", None)
        if creator is None:
            return None
        return creator(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        try:
            _ensure_tab_patches_applied()
        except Exception:
            # Never let a patch failure abort tab import.
            pass


if not any(isinstance(f, _TabPatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _TabPatchFinder())


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

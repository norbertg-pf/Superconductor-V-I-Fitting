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

from .service import (
    FitResult,
    FitSettings,
    robust_view_range,
    run_full_fit,
)

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
    "FitResult",
    "FitSettings",
    "robust_view_range",
    "run_full_fit",
    *sorted(_LAZY_TAB),
]

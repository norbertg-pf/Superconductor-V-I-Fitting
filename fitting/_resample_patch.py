"""Runtime patch: remove the 'Resample everything to 100 S/s' checkbox.

Resampling to ~100 S/s is now mandatory (not user-toggleable), so the
matching UI element is hidden after the tab layout runs and
``_apply_resample_to_100sps`` is replaced with a checkbox-ignoring
version that always sets the Plot summary Avg from the source rate.

Applying this at import time keeps the change minimal relative to
``tab.py`` itself — it ships without an in-place edit to the 312 KB
module — and stays consistent with the existing
``_pct_anchor_patch.py`` runtime-wrapping pattern.
"""

from __future__ import annotations

import functools

import numpy as np


_PATCHED = False


def _apply_resample_to_100sps_no_cb(app) -> None:
    """Set Avg so the effective sample rate is ~100 S/s.

    Mirrors the original ``tab._apply_resample_to_100sps`` minus the
    ``data_fit_resample_100sps_cb`` gate, so the textbox always reflects
    the source recording's effective rate after a load.
    """
    avg_widget = getattr(app, "data_fit_avg_input", None)
    if avg_widget is None:
        return
    controller = getattr(app, "data_fit_controller", None)
    if controller is None:
        return
    t_raw = getattr(controller, "time_array", None)
    if t_raw is None or np.asarray(t_raw).size < 2:
        return
    t_arr = np.asarray(t_raw, dtype=float)
    dt = float(np.mean(np.diff(t_arr)))
    if not np.isfinite(dt) or dt <= 0:
        return
    fs_orig = 1.0 / dt
    if fs_orig <= 100.0:
        avg_widget.setText("1")
        return
    avg = max(1, int(round(fs_orig / 100.0)))
    avg_widget.setText(str(avg))


def _hide_checkbox_widget(app) -> None:
    """Remove the legacy Resample-to-100Sps checkbox from the layout.

    Calling ``setParent(None)`` detaches the widget from any layout it was
    placed into so the Settings dialog does not allocate a row for it; the
    follow-up ``deleteLater`` schedules teardown without touching unrelated
    widgets. Guarded so a missing/already-detached widget is a no-op.
    """
    cb = getattr(app, "data_fit_resample_100sps_cb", None)
    if cb is None:
        return
    try:
        cb.setVisible(False)
    except Exception:
        pass
    try:
        cb.setParent(None)
    except Exception:
        pass
    try:
        cb.deleteLater()
    except Exception:
        pass
    try:
        delattr(app, "data_fit_resample_100sps_cb")
    except Exception:
        try:
            app.data_fit_resample_100sps_cb = None
        except Exception:
            pass


def apply_patches() -> None:
    """Wrap the tab functions exactly once. Idempotent on repeat calls."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        from . import tab as _tab
    except Exception:
        return

    if hasattr(_tab, "_apply_resample_to_100sps"):
        _tab._apply_resample_to_100sps = _apply_resample_to_100sps_no_cb

    setup_fn = getattr(_tab, "setup_data_fitting_tab_layout", None)
    if setup_fn is not None:
        @functools.wraps(setup_fn)
        def _setup_data_fitting_tab_layout_no_resample_cb(app):
            result = setup_fn(app)
            _hide_checkbox_widget(app)
            return result

        _tab.setup_data_fitting_tab_layout = (
            _setup_data_fitting_tab_layout_no_resample_cb
        )

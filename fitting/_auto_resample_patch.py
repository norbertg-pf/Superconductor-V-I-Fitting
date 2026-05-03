"""Runtime patch: auto-resample high-rate recordings on load.

Why this lives in a separate module
-----------------------------------
``tab.py`` is large enough that re-uploading it through the GitHub Contents
API is impractical (the file exceeds the per-call payload budget). The same
behavioural change is applied by wrapping a small set of functions in
``tab`` at import time.

What it does
------------
DAQUniversal acquisitions can run at kS/s, but the IEC 61788 log-log decade
fit is calibrated around ~100 S/s. After the Data Fitting tab finishes
loading a recording (via ``Use current measurement`` or the post-acquisition
auto-load), the patch:

1. Pre-fills the main AVG textbox (``data_fit_avg_input``) with the smallest
   block-average factor that brings the effective rate down to
   ``TARGET_EFFECTIVE_RATE_HZ``.
2. Stamps the same factor onto every *replayed source curve* added by
   ``_replay_saved_fits_into_plot`` (the curves DAQUniversal hands to the tab
   after a fit-enabled acquisition) and block-averages their (t, x, y) so
   plotted samples match Run Fit.
3. Fixes the Plot summary dialog's preview row so its Avg column reflects
   the active AVG factor instead of the hardcoded ``1`` literal that the
   underlying ``_open_plot_summary`` function builds inline.

How
---
1. Wrap ``_reset_data_fitting_defaults`` to clear the per-recording marker
   so re-loading the same TDMS triggers a fresh auto-apply (the reset
   helper runs before every ``open_file_dialog`` /
   ``refresh_current_recording`` flow).
2. Wrap ``refresh_preview`` so the first call after a reset (the one inside
   ``_post_load_setup`` once the controller has the recording loaded)
   pre-fills the AVG textbox.
3. Wrap ``_add_replayed_source_curve`` so each replayed curve picks up the
   active AVG factor and its (t, x, y) arrays are block-averaged.
4. Wrap ``_open_plot_summary`` so a one-shot ``QTimer`` callback patches
   the dialog's preview row to display ``_active_avg_window(app)`` instead
   of ``1`` once the table has been built. The QTimer fires on Qt's event
   loop while ``dialog.exec_()`` is blocking, so it runs while the dialog
   is visible but before the user can interact with the row.

Idempotent: safe to call ``apply_patches`` multiple times.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Target effective sample rate (Hz). High-rate DAQ recordings are resampled
# down to this rate via block averaging so the IEC log-log fit operates on
# the regime it was designed for.
TARGET_EFFECTIVE_RATE_HZ = 100.0

_AUTO_APPLIED_ATTR = "_data_fit_auto_resample_applied_path"

_PATCHED = False


def _sample_rate_from_time(t) -> Optional[float]:
    """Median sample rate (Hz) from a time array, or ``None``.

    Uses the median of positive finite diffs so a few duplicated or out-of-
    order timestamps in a TDMS recording do not skew the result.
    """
    if t is None:
        return None
    arr = np.asarray(t, dtype=float)
    if arr.size < 2:
        return None
    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def _avg_window_for_target_rate(
    rate_hz: Optional[float],
    target_hz: float = TARGET_EFFECTIVE_RATE_HZ,
) -> int:
    """Block-average factor that brings ``rate_hz`` down to ~``target_hz``.

    Returns 1 (no averaging) when the rate is unknown, the target is
    invalid, or the recording already runs at or below the target rate.
    """
    if rate_hz is None or target_hz <= 0 or rate_hz <= target_hz:
        return 1
    return max(1, int(round(rate_hz / target_hz)))


def _auto_apply_target_rate_resampling(
    app,
    _tab,
    *,
    target_hz: float = TARGET_EFFECTIVE_RATE_HZ,
) -> None:
    """Pre-fill the AVG textbox so a high-rate recording resamples to ~target_hz.

    Looks at the loaded controller's time array to derive the recording's
    sample rate, then sets ``app.data_fit_avg_input`` to the smallest N
    that brings the effective rate to or below ``target_hz``. No-op when:

    * the controller hasn't loaded any data yet,
    * the recording is already at or below the target rate,
    * the time array is too short or non-monotonic to derive a rate from.
    """
    avg_input = getattr(app, "data_fit_avg_input", None)
    if avg_input is None:
        return
    controller = getattr(app, "data_fit_controller", None)
    if controller is None:
        return
    rate_hz = _sample_rate_from_time(getattr(controller, "time_array", None))
    n = _avg_window_for_target_rate(rate_hz, target_hz)
    if n <= 1:
        return
    set_silently = getattr(_tab, "_set_silently", None)
    if set_silently is not None:
        set_silently(avg_input, str(int(n)))
    else:
        try:
            avg_input.setText(str(int(n)))
        except Exception:
            pass


def _controller_path(app) -> str:
    controller = getattr(app, "data_fit_controller", None)
    if controller is None:
        return ""
    return str(getattr(controller, "tdms_path", "") or "")


def _block_average_safe(arr, window: int):
    """Block-average ``arr`` with safety against ``None`` / zero-size inputs."""
    if window <= 1 or arr is None:
        return arr
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    n_bins = a.size // window
    if n_bins == 0:
        return a
    return a[: n_bins * window].reshape(n_bins, window).mean(axis=1)


def apply_patches() -> None:
    """Apply runtime patches to ``fitting.tab`` exactly once."""
    global _PATCHED
    if _PATCHED:
        return
    from . import tab as _tab
    _PATCHED = True

    # 1) Clear the per-recording marker on every reset so re-loading the
    # *same* file again re-triggers auto-apply.
    _orig_reset = _tab._reset_data_fitting_defaults

    def _patched_reset(app):
        try:
            if hasattr(app, _AUTO_APPLIED_ATTR):
                delattr(app, _AUTO_APPLIED_ATTR)
        except Exception:
            pass
        return _orig_reset(app)

    _tab._reset_data_fitting_defaults = _patched_reset

    # 2) Wrap ``refresh_preview`` so the first call after a reset pre-fills
    # the AVG textbox before the underlying preview is rebuilt.
    _orig_refresh_preview = _tab.refresh_preview

    def _patched_refresh_preview(app):
        path = _controller_path(app)
        if path and getattr(app, _AUTO_APPLIED_ATTR, None) != path:
            try:
                _auto_apply_target_rate_resampling(app, _tab)
            finally:
                try:
                    setattr(app, _AUTO_APPLIED_ATTR, path)
                except Exception:
                    pass
        return _orig_refresh_preview(app)

    _tab.refresh_preview = _patched_refresh_preview

    # 3) Wrap ``_add_replayed_source_curve`` so replayed curves pick up
    # the active AVG factor and have their plotted samples block-averaged.
    _orig_add_replayed = getattr(_tab, "_add_replayed_source_curve", None)
    _active_avg_window = getattr(_tab, "_active_avg_window", None)
    if _orig_add_replayed is not None and _active_avg_window is not None:

        def _patched_add_replayed(app, name, x, y, t, meta, *, visible,
                                  x_name="", x_meta=None):
            try:
                avg = max(1, int(_active_avg_window(app)))
            except Exception:
                avg = 1
            if avg > 1:
                x = _block_average_safe(x, avg)
                y = _block_average_safe(y, avg)
                t = _block_average_safe(t, avg)
            entry = _orig_add_replayed(
                app, name, x, y, t, meta,
                visible=visible, x_name=x_name, x_meta=x_meta,
            )
            try:
                if entry is not None and avg > 1:
                    entry["avg_window"] = avg
            except Exception:
                pass
            return entry

        _tab._add_replayed_source_curve = _patched_add_replayed

    # 4) Wrap ``_open_plot_summary`` so the dialog's preview row Avg column
    # shows the active textbox value instead of the hardcoded ``1`` from
    # the inline dict literal in ``_open_plot_summary``.
    #
    # We schedule a one-shot ``QTimer`` callback before invoking the
    # original. The original calls ``dialog.exec_()`` which blocks but
    # spins Qt's event loop — so the timer fires while the modal dialog
    # is up and lets us locate the preview row's QLineEdit and rewrite
    # its text to match ``_active_avg_window(app)``.
    _orig_open_plot_summary = getattr(_tab, "_open_plot_summary", None)
    if _orig_open_plot_summary is not None and _active_avg_window is not None:

        def _patched_open_plot_summary(app):
            try:
                from PyQt5.QtCore import QTimer
                from PyQt5.QtWidgets import (
                    QApplication, QDialog, QLineEdit, QTableWidget,
                )

                def _fixup_preview_avg():
                    try:
                        # Only the active modal dialog is the one we just
                        # opened; bail otherwise.
                        dialog = QApplication.activeModalWidget()
                        if not isinstance(dialog, QDialog):
                            return
                        if dialog.windowTitle() != "Plot summary":
                            return
                        if not getattr(app, "data_fit_preview_visible", True):
                            return
                        table = dialog.findChild(QTableWidget)
                        if table is None or table.rowCount() == 0:
                            return
                        # The preview row is row 0 when the preview is
                        # visible. The Avg column is index 3 (after Color,
                        # Label, Skip pts).
                        widget = table.cellWidget(0, 3)
                        if not isinstance(widget, QLineEdit):
                            return
                        try:
                            current_text = widget.text().strip()
                        except Exception:
                            return
                        try:
                            current_value = int(float(current_text))
                        except (ValueError, TypeError):
                            return
                        # Only override if the dialog is still showing the
                        # hardcoded ``1`` placeholder — leaving any user-
                        # edited values alone.
                        if current_value != 1:
                            return
                        try:
                            active = max(1, int(_active_avg_window(app)))
                        except Exception:
                            return
                        if active > 1:
                            widget.setText(str(active))
                    except Exception:
                        pass

                QTimer.singleShot(0, _fixup_preview_avg)
            except Exception:
                pass
            return _orig_open_plot_summary(app)

        _tab._open_plot_summary = _patched_open_plot_summary

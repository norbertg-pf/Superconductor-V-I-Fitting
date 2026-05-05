"""Runtime patch for ``fitting.tab`` that auto-resamples high-rate
DAQUniversal recordings to ~100 S/s by filling ``data_fit_avg_input``.

When a TDMS file produced by DAQUniversal is loaded into the Data
Fitting tab and its acquisition rate exceeds the IEC log-log decade
fit's calibrated target of 100 S/s, the Avg window is automatically
set to ``round(rate / 100)`` so the effective sample rate matches the
target. The same factor is stamped onto every existing curve in
``app.data_fit_curves`` and the preview is refreshed so the Plot
summary dialog and live plot both reflect the new value.

Detection
---------
DAQUniversal stamps the acquisition rate onto every TDMS root as the
``Sample_Rate_Hz`` property (see
``DAQUniversal/src/workers/tdms_writer_worker.py:_collect_root_properties``).
Files produced elsewhere — third-party TDMS, hand-crafted recordings —
typically have no such property, so they are left untouched and the
user keeps full manual control of the AVG factor.

What this patches
-----------------
1. ``_post_load_setup`` — runs the resample helper after every
   ``open_file_dialog`` and ``refresh_current_recording`` call,
   covering both the "Load File…" button and the post-acquisition
   auto-load path.
2. ``_open_plot_summary`` — pre-fills the preview row's Avg cell from
   ``data_fit_avg_input`` (column 3) so the Plot summary dialog opens
   showing the active AVG and the Effective rate column matches.
   This is the same belt-and-braces fix DAQUniversal's
   ``data_fitting_tab.py`` shim applies; the marker attribute is
   shared so the DAQ-side fallback won't double-wrap.

Idempotent: safe to call ``apply_patches`` multiple times.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_PATCHED = False

_TARGET_EFFECTIVE_RATE_HZ = 100.0


def _read_sample_rate_from_tdms(path: str) -> Optional[float]:
    """Return the TDMS root ``Sample_Rate_Hz`` property, else ``None``.

    Used as the DAQUniversal marker: only files with this property are
    treated as candidates for auto-resampling. Reads via ``nptdms`` in
    a context manager so the file handle is released immediately.
    """
    if not path:
        return None
    try:
        from nptdms import TdmsFile
    except Exception:
        return None
    try:
        with TdmsFile.open(path) as tdms_file:
            props = dict(getattr(tdms_file, "properties", {}) or {})
    except Exception:
        return None
    rate = props.get("Sample_Rate_Hz")
    try:
        rate_value = float(rate)
    except (TypeError, ValueError):
        return None
    if rate_value <= 0:
        return None
    return rate_value


def _apply_target_rate_resample(app,
                                target_hz: float = _TARGET_EFFECTIVE_RATE_HZ) -> None:
    """Set ``data_fit_avg_input`` so the effective rate matches ``target_hz``.

    Mirrors DAQUniversal's runtime helper of the same name but runs from
    the SVIF side so it covers manual ``Load File…`` opens as well as
    post-acquisition auto-loads.
    """
    avg_input = getattr(app, "data_fit_avg_input", None)
    if avg_input is None:
        return
    controller = getattr(app, "data_fit_controller", None)
    if controller is None:
        return

    rate_hz = _read_sample_rate_from_tdms(getattr(controller, "tdms_path", "") or "")
    if rate_hz is None or rate_hz <= target_hz:
        return
    n = max(1, int(round(rate_hz / target_hz)))
    if n <= 1:
        return

    # ``editingFinished`` is suppressed because the curve-profile
    # autosave handler reads the textbox and would persist the auto-
    # filled AVG over the freshly reset profile if it ran here.
    try:
        avg_input.blockSignals(True)
        avg_input.setText(str(int(n)))
    finally:
        try:
            avg_input.blockSignals(False)
        except Exception:
            pass

    try:
        from . import tab as _tab
    except Exception:
        return
    recompute = getattr(_tab, "_recompute_curve_from_source", None)
    refresh_item = getattr(_tab, "_refresh_curve_item", None)
    curves = getattr(app, "data_fit_curves", None) or []
    if recompute is not None:
        for entry in curves:
            try:
                # Fit-result overlays come from saved parameters, not
                # raw samples — re-averaging would corrupt them.
                if entry.get("is_fit_result"):
                    continue
                if not entry.get("source"):
                    continue
                entry["avg_window"] = int(n)
                recompute(app, entry)
                if refresh_item is not None:
                    refresh_item(entry)
            except Exception:
                pass

    refresh_preview = getattr(_tab, "refresh_preview", None)
    if refresh_preview is not None:
        try:
            refresh_preview(app)
        except Exception:
            pass


def _patch_post_load_setup(_tab) -> None:
    orig = getattr(_tab, "_post_load_setup", None)
    if orig is None or getattr(orig, "_auto_resample_patched", False):
        return

    def _patched_post_load_setup(app, *, auto_plot_fits: bool = True) -> None:
        orig(app, auto_plot_fits=auto_plot_fits)
        try:
            _apply_target_rate_resample(app)
        except Exception:
            # Never let the resample helper abort a successful load.
            pass

    try:
        _patched_post_load_setup._auto_resample_patched = True
    except Exception:
        pass
    _tab._post_load_setup = _patched_post_load_setup


def _patch_open_plot_summary(_tab) -> None:
    """Wrap ``_open_plot_summary`` so the preview row's Avg cell is
    pre-filled from ``data_fit_avg_input`` when the dialog opens.

    The preview entry is built inline in ``_open_plot_summary`` with a
    hardcoded ``"avg_window": 1`` literal — even when ``data_fit_avg_input``
    already holds the auto-filled rate-based factor. We post a one-shot
    fixup onto the event loop so the dialog renders first, then we read
    column 3's QLineEdit and update its text without touching the rest
    of the table.

    Shares the ``_daq_avg_resample_patched`` marker so DAQUniversal's
    fallback patch in ``data_fitting_tab.py`` recognises this wrapper
    and skips re-wrapping.
    """
    orig = getattr(_tab, "_open_plot_summary", None)
    active_avg = getattr(_tab, "_active_avg_window", None)
    if orig is None or active_avg is None:
        return
    if getattr(orig, "_daq_avg_resample_patched", False):
        return

    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import (
            QApplication, QDialog, QLineEdit, QTableWidget,
        )
    except Exception:
        return

    def _patched_open_plot_summary(app):
        def _fixup_preview_avg():
            try:
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
                # Preview row is row 0 (when visible); Avg column is 3
                # (after Color, Label, Skip pts).
                widget = table.cellWidget(0, 3)
                if not isinstance(widget, QLineEdit):
                    return
                try:
                    current = int(float(widget.text().strip()))
                except (ValueError, TypeError):
                    return
                if current != 1:
                    return  # User-edited or already auto-filled.
                try:
                    target = max(1, int(active_avg(app)))
                except Exception:
                    return
                if target > 1:
                    widget.setText(str(target))
                    # Trigger the avg.editingFinished handler so the
                    # Effective rate cell (column 6) re-computes from
                    # the same averaged time array as the preview plot.
                    try:
                        widget.editingFinished.emit()
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            QTimer.singleShot(0, _fixup_preview_avg)
        except Exception:
            pass
        return orig(app)

    try:
        _patched_open_plot_summary._daq_avg_resample_patched = True
        _patched_open_plot_summary._auto_resample_patched = True
    except Exception:
        pass
    _tab._open_plot_summary = _patched_open_plot_summary


def apply_patches() -> None:
    """Apply runtime patches to ``fitting.tab`` exactly once."""
    global _PATCHED
    if _PATCHED:
        return
    from . import tab as _tab
    _PATCHED = True
    _patch_post_load_setup(_tab)
    _patch_open_plot_summary(_tab)

"""Runtime patch: auto-resample high-rate recordings on load.

Why this lives in a separate module
-----------------------------------
``tab.py`` is large enough that re-uploading it through the GitHub Contents
API is impractical (the file exceeds the per-call payload budget). The same
behavioural change is applied by wrapping a small set of functions in
``tab`` at import time. The wrappers honour the contract described in the
surrounding commit message.

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
   after a fit-enabled acquisition). The Plot summary dialog reads each
   curve's ``avg_window`` field for its Avg column, so without this stamp
   the column shows the hardcoded ``1`` even though the textbox is set
   correctly.
3. Block-averages the (t, x, y) arrays of those replayed curves so the
   plotted samples match what Run Fit would see at the new factor.

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
    # ``_set_silently`` blocks ``editingFinished`` so the curve-profile
    # autosave doesn't fire mid-load and overwrite the freshly-reset
    # profile with the auto-filled AVG.
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

    # 1) Clear the per-recording marker on every reset so that opening the
    # *same* file again re-triggers auto-apply. ``_reset_data_fitting_defaults``
    # also resets the AVG textbox back to "1", so by the time ``refresh_preview``
    # runs inside ``_post_load_setup`` we are guaranteed a clean slate.
    _orig_reset = _tab._reset_data_fitting_defaults

    def _patched_reset(app):
        try:
            if hasattr(app, _AUTO_APPLIED_ATTR):
                delattr(app, _AUTO_APPLIED_ATTR)
        except Exception:
            pass
        return _orig_reset(app)

    _tab._reset_data_fitting_defaults = _patched_reset

    # 2) Wrap ``refresh_preview`` so the *first* call after a reset (the one
    # invoked from inside ``_post_load_setup`` once the controller has the
    # recording loaded) pre-fills the AVG textbox. Subsequent calls for the
    # same recording are left alone so user-edited values are not clobbered
    # whenever a downstream interaction triggers another preview refresh.
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

    # 3) Wrap ``_add_replayed_source_curve`` so the replayed curves DAQUniversal
    # hands to the tab after a fit-enabled acquisition pick up the active AVG
    # factor (and have their plotted samples block-averaged to match). Without
    # this, the Plot summary dialog's Avg column reads the hardcoded ``1`` from
    # the entry even though the main AVG textbox now reads the auto-filled N.
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

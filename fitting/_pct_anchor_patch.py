"""Runtime patches for ``fitting.tab``.

Anchors Step 3 / Step 4 / non-linear Step 5 fit windows to the **untrimmed**
current sweep so the displayed Low(X)/High(X) editors match the absolute
amperes the fit actually uses, and persists ``linear_fit_window`` in TDMS
metadata so reloading an auto-fit recording shows the real Step 4 window
instead of falling back to ``(0, 0)``.

Why this lives in a separate module
-----------------------------------
``tab.py`` is large enough that re-uploading it through the GitHub Contents
API is impractical (the file exceeds the per-call payload budget). The same
behavioural fix is applied by wrapping a small set of public functions in
``tab`` at import time. The wrappers honour the same contract as the inline
edits described in the surrounding commit message.

What it patches
---------------
1. ``_settings_for_entry`` — fills ``pct_x_min`` / ``pct_x_max`` on the
   returned ``FitSettings`` from the entry's untrimmed snapshot so
   ``run_full_fit`` (already updated in ``fitting.service``) anchors the
   ``didt_*_frac`` / ``linear_*_frac`` / ``power_low_frac`` fields to the
   same range as the UI editors.
2. ``_settings_from_inputs`` — same anchor for the preview path that has
   no stored profile yet (uses ``_apply_transforms(apply_trim=False)``).
3. ``_compute_step123_result`` — converts the percent fields into the
   equivalent fractions on the *trimmed* array when the override is
   present, then delegates to the original implementation. This keeps the
   helper-curve / "Add corrected curve" paths consistent with Run Fit.
4. ``_fit_result_properties`` — adds ``linear_window_I_lo_A`` /
   ``linear_window_I_hi_A`` to the saved property dict.
5. ``_fit_result_from_props`` — hydrates ``linear_fit_window`` from those
   keys instead of hardcoding ``(0.0, 0.0)``.

Idempotent: safe to call ``apply_patches`` multiple times.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Optional

import numpy as np

_PATCHED = False


def _entry_untrimmed_x(entry: dict) -> np.ndarray:
    """Return the entry's untrimmed x array, falling back to the live x."""
    src = entry.get("x_orig")
    if src is None:
        src = entry.get("x", [])
    return np.asarray(src, dtype=float)


def _bounds_from_array(arr: np.ndarray) -> Optional[tuple[float, float]]:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)] if arr.size else arr
    if finite.size == 0:
        return None
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return None
    return lo, hi


def _untrimmed_bounds_for_entry(entry: Optional[dict]) -> Optional[tuple[float, float]]:
    if not entry:
        return None
    return _bounds_from_array(_entry_untrimmed_x(entry))


def _untrimmed_bounds_for_app(app, _tab) -> Optional[tuple[float, float]]:
    """Best-effort untrimmed bounds for the live preview path."""
    try:
        transformed = _tab._apply_transforms(app, apply_trim=False)
    except Exception:
        return None
    x = transformed.get("x_original")
    if x is None or getattr(x, "size", 0) == 0:
        x = transformed.get("x", [])
    return _bounds_from_array(np.asarray(x, dtype=float))


def apply_patches() -> None:
    """Apply runtime patches to ``fitting.tab`` exactly once."""
    global _PATCHED
    if _PATCHED:
        return
    from . import tab as _tab
    _PATCHED = True

    # 1) _settings_for_entry → seed pct_x_min/pct_x_max from entry snapshot
    _orig_settings_for_entry = _tab._settings_for_entry

    def _patched_settings_for_entry(app, entry):
        settings = _orig_settings_for_entry(app, entry)
        bounds = _untrimmed_bounds_for_entry(entry)
        if bounds is None:
            return settings
        try:
            return _dc_replace(settings, pct_x_min=bounds[0], pct_x_max=bounds[1])
        except Exception:
            return settings

    _tab._settings_for_entry = _patched_settings_for_entry

    # 2) _settings_from_inputs → seed from the live preview / first curve
    _orig_settings_from_inputs = _tab._settings_from_inputs

    def _patched_settings_from_inputs(app):
        settings = _orig_settings_from_inputs(app)
        try:
            bounds = None
            for c in getattr(app, "data_fit_curves", []) or []:
                if bool(c.get("is_fit_result", False)):
                    continue
                bounds = _untrimmed_bounds_for_entry(c)
                if bounds is not None:
                    break
            if bounds is None:
                bounds = _untrimmed_bounds_for_app(app, _tab)
            if bounds is not None:
                return _dc_replace(settings, pct_x_min=bounds[0], pct_x_max=bounds[1])
        except Exception:
            pass
        return settings

    _tab._settings_from_inputs = _patched_settings_from_inputs

    # 3) _compute_step123_result → translate fractions before delegating
    _orig_step123 = _tab._compute_step123_result

    def _patched_step123(t, x, y, settings):
        pct_min = getattr(settings, "pct_x_min", None)
        pct_max = getattr(settings, "pct_x_max", None)
        if pct_min is None or pct_max is None:
            return _orig_step123(t, x, y, settings)
        bounds = _bounds_from_array(np.asarray(x, dtype=float).ravel())
        if bounds is None:
            return _orig_step123(t, x, y, settings)
        trim_min, trim_max = bounds
        pct_min_f = float(pct_min)
        pct_max_f = float(pct_max)
        pct_span = pct_max_f - pct_min_f
        trim_span = trim_max - trim_min
        if pct_span <= 0 or trim_span <= 0:
            return _orig_step123(t, x, y, settings)

        def _xtrans(frac: float) -> float:
            # Map a fraction expressed against [pct_min, pct_max] to the
            # equivalent fraction on the (trimmed) input array's range, so
            # the unpatched body of ``_compute_step123_result`` reproduces
            # the absolute X values the UI displays.
            x_abs = pct_min_f + float(frac) * pct_span
            return (x_abs - trim_min) / trim_span

        try:
            new_settings = _dc_replace(
                settings,
                didt_low_frac=_xtrans(settings.didt_low_frac),
                didt_high_frac=_xtrans(settings.didt_high_frac),
                linear_low_frac=_xtrans(settings.linear_low_frac),
                linear_high_frac=_xtrans(settings.linear_high_frac),
                power_low_frac=_xtrans(settings.power_low_frac),
                pct_x_min=None,
                pct_x_max=None,
            )
        except Exception:
            return _orig_step123(t, x, y, settings)
        return _orig_step123(t, x, y, new_settings)

    _tab._compute_step123_result = _patched_step123

    # 4) _fit_result_properties → write Step 4 window to TDMS metadata
    _orig_props = _tab._fit_result_properties

    def _patched_props(result) -> dict:
        props = _orig_props(result)
        try:
            window = getattr(result, "linear_fit_window", (0.0, 0.0)) or (0.0, 0.0)
            props["linear_window_I_lo_A"] = float(window[0])
            props["linear_window_I_hi_A"] = float(window[1])
        except Exception:
            pass
        return props

    _tab._fit_result_properties = _patched_props

    # 5) _fit_result_from_props → hydrate linear_fit_window on reload
    _orig_from_props = _tab._fit_result_from_props
    _coerce_float = _tab._coerce_float
    _prop_lookup = _tab._prop_lookup

    def _patched_from_props(props):
        result = _orig_from_props(props)
        try:
            lo = _coerce_float(_prop_lookup(props, "linear_window_I_lo_A"), 0.0)
            hi = _coerce_float(_prop_lookup(props, "linear_window_I_hi_A"), 0.0)
            if lo or hi:
                result.linear_fit_window = (float(lo), float(hi))
        except Exception:
            pass
        return result

    _tab._fit_result_from_props = _patched_from_props

"""Runtime patches for ``fitting.tab`` preset persistence.

Extends ``_settings_to_preset`` and ``_apply_preset`` so the JSON preset
captures every textbox / checkbox / radio button / combo box the user can
edit in the Data Fitting tab — Step 1 to Step 5, the Config window
(Ic iteration & criterion) and the Settings window. The base
implementations only persisted a subset of those widgets, so widgets like
the Step 1 thermal-offset toggle, the Step 2 trim inputs, the Show/edit
band checkboxes, the Step 3/4 estimator combos, the Step 5 weighting combo
and the Channels group scale/offset editors were silently dropped on every
save.

Why this lives in a separate module
-----------------------------------
``tab.py`` is large enough that re-uploading it through the GitHub Contents
API is impractical. Instead we wrap the existing public-ish entry points
at import time, mirroring the approach in ``_pct_anchor_patch``.

What it patches
---------------
1. ``_settings_to_preset`` — calls the original to get a base ``FitPreset``,
   then overrides every newly-tracked field by reading the corresponding
   widget directly. This keeps the contract of returning a fully populated
   ``FitPreset`` while ensuring nothing is dropped.
2. ``_apply_preset`` — calls the original first so existing fields and the
   method-mode UI are restored exactly as before, then writes the new
   widgets from the preset. Old preset JSON files without the new keys
   keep working because every read uses ``getattr(preset, key, default)``.
3. ``_resolve_reference_curve_data`` — fixes an ``IndexError`` that fires
   when the user toggles a Show/edit band before any data is loaded.
   ``_apply_transforms`` returns ``None`` for unselected channels and
   ``np.asarray(None, dtype=float)`` produces a 0-d array of size 1, which
   the original function then tried to slice with ``x[:n]``. We coerce to a
   1-D float array (empty when the source is ``None``/0-d) before the size
   check so empty inputs return ``None`` cleanly.

Idempotent: ``apply_patches`` checks an internal flag so wrapping happens
at most once per process.
"""

from __future__ import annotations

import numpy as np


_PATCHED = False


def _safe_checkbox_get(app, attr_name: str, default: bool) -> bool:
    cb = getattr(app, attr_name, None)
    if cb is None:
        return default
    try:
        return bool(cb.isChecked())
    except RuntimeError:
        return default


def _safe_checkbox_set(app, attr_name: str, value: bool) -> None:
    cb = getattr(app, attr_name, None)
    if cb is None:
        return
    try:
        cb.blockSignals(True)
        try:
            cb.setChecked(bool(value))
        finally:
            cb.blockSignals(False)
    except RuntimeError:
        pass


def _safe_text_get(app, attr_name: str, fallback: str = "") -> str:
    widget = getattr(app, attr_name, None)
    if widget is None:
        return fallback
    try:
        return str(widget.text())
    except (RuntimeError, AttributeError):
        return fallback


def _safe_text_set(app, attr_name: str, text: str) -> None:
    widget = getattr(app, attr_name, None)
    if widget is None:
        return
    try:
        widget.blockSignals(True)
        try:
            widget.setText(str(text))
        finally:
            widget.blockSignals(False)
    except (RuntimeError, AttributeError):
        pass


def _safe_float_get(app, attr_name: str, fallback: float) -> float:
    widget = getattr(app, attr_name, None)
    if widget is None:
        return fallback
    try:
        return float(widget.text())
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return fallback


def _safe_combo_get(app, attr_name: str, fallback: str) -> str:
    combo = getattr(app, attr_name, None)
    if combo is None:
        return fallback
    try:
        data = combo.currentData()
    except (RuntimeError, AttributeError):
        return fallback
    return fallback if data is None else str(data)


def _safe_combo_set(app, attr_name: str, value: str) -> None:
    combo = getattr(app, attr_name, None)
    if combo is None:
        return
    try:
        idx = combo.findData(value)
    except (RuntimeError, AttributeError):
        return
    if idx < 0:
        idx = 0
    try:
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
    except RuntimeError:
        pass


# Widgets covered by this patch, grouped by tab section. The lists here are
# the single source of truth — both the snapshot and apply routines walk the
# same tuples so adding a new widget only requires one edit.

_SCALE_OFFSET_WIDGETS = (
    ("data_fit_time_scale", "time_scale", 1.0),
    ("data_fit_time_offset", "time_offset", 0.0),
    ("data_fit_x_scale", "x_scale", 1.0),
    ("data_fit_x_offset", "x_offset", 0.0),
    ("data_fit_y_scale", "y_scale", 1.0),
    ("data_fit_y_offset", "y_offset", 0.0),
)

# (attr, preset_field, default) for plain checkboxes.
_CHECKBOX_WIDGETS = (
    ("data_fit_subtract_vofs_cb", "subtract_vofs", True),
    ("data_fit_trim_quench_cb", "trim_quench", True),
    ("data_fit_show_didt", "show_didt", False),
    ("data_fit_show_linear", "show_linear", False),
    ("data_fit_show_power", "show_power", False),
)

# Plain numeric line edits whose value lives in a percent-of-Imax convention
# or absolute units; we store the raw number so the JSON matches the UI text.
_FLOAT_WIDGETS = (
    ("data_fit_zero_i_frac", "zero_i_frac_pct", 2.0),
    ("data_fit_trim_start_abs", "trim_start_abs", 30.0),
    ("data_fit_trim_start_pct", "trim_start_pct", 5.0),
)

# Step 3/4/5 estimator/weighting combos. The combo data string ("ols",
# "huber", "theil_sen", "equal", "weighted", "robust") is what
# fitting.service expects, so storing that keeps everything aligned.
_COMBO_WIDGETS = (
    ("data_fit_didt_mode_cb", "didt_mode", "huber"),
    ("data_fit_baseline_mode_cb", "baseline_mode", "huber"),
    ("data_fit_weight_mode_cb", "weight_mode", "equal"),
)

# Low(X)/High(X) editors below the Step 3/4/5 percentage rows. Empty
# strings are valid — they show the "—" placeholder until a fit runs.
_X_VALUE_WIDGETS = (
    ("data_fit_didt_low_x", "didt_low_x"),
    ("data_fit_didt_high_x", "didt_high_x"),
    ("data_fit_linear_low_x", "linear_low_x"),
    ("data_fit_linear_high_x", "linear_high_x"),
    ("data_fit_power_low_x", "power_low_x"),
    ("data_fit_power_high_x", "power_high_x"),
)


def _augment_preset_from_app(app, preset) -> None:
    """Overwrite the new FitPreset fields from current widget state."""
    for attr, field_name, fallback in _SCALE_OFFSET_WIDGETS:
        setattr(preset, field_name, _safe_float_get(app, attr, fallback))
    for attr, field_name, fallback in _CHECKBOX_WIDGETS:
        setattr(preset, field_name, _safe_checkbox_get(app, attr, fallback))
    for attr, field_name, fallback in _FLOAT_WIDGETS:
        setattr(preset, field_name, _safe_float_get(app, attr, fallback))
    for attr, field_name, fallback in _COMBO_WIDGETS:
        setattr(preset, field_name, _safe_combo_get(app, attr, fallback))
    for attr, field_name in _X_VALUE_WIDGETS:
        setattr(preset, field_name, _safe_text_get(app, attr, ""))


def _restore_preset_into_app(app, preset) -> None:
    """Apply every new FitPreset field to the corresponding widget."""
    for attr, field_name, fallback in _SCALE_OFFSET_WIDGETS:
        try:
            value = float(getattr(preset, field_name, fallback))
        except (TypeError, ValueError):
            value = fallback
        _safe_text_set(app, attr, f"{value:g}")
    for attr, field_name, fallback in _CHECKBOX_WIDGETS:
        _safe_checkbox_set(app, attr, bool(getattr(preset, field_name, fallback)))
    for attr, field_name, fallback in _FLOAT_WIDGETS:
        try:
            value = float(getattr(preset, field_name, fallback))
        except (TypeError, ValueError):
            value = fallback
        _safe_text_set(app, attr, f"{value:g}")
    for attr, field_name, fallback in _COMBO_WIDGETS:
        _safe_combo_set(app, attr, str(getattr(preset, field_name, fallback)))
    for attr, field_name in _X_VALUE_WIDGETS:
        _safe_text_set(app, attr, str(getattr(preset, field_name, "") or ""))
    # Refresh band visibility/draggability now that the show-edit checkboxes
    # may have flipped. Tolerate the case where the helper isn't available
    # (e.g. an older tab.py without _update_band_states).
    update_band_states = getattr(_tab_module(), "_update_band_states", None)
    if callable(update_band_states):
        try:
            update_band_states(app)
        except Exception:
            pass


def _tab_module():
    from . import tab as _tab
    return _tab


def _to_clean_1d_array(value) -> np.ndarray:
    """Return a 1-D float array, treating ``None`` and 0-d inputs as empty.

    ``np.asarray(None, dtype=float)`` produces a 0-d ``array(nan)`` with
    ``size == 1``; the caller's ``size <= 0`` guard then misses it and
    ``arr[:n]`` raises ``IndexError: too many indices for array``. Coercing
    here keeps the rest of the code path on its happy path.
    """
    if value is None:
        return np.empty(0, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.empty(0, dtype=float)
    return arr


def _patch_resolve_reference_curve_data(tab) -> None:
    """Patch ``_resolve_reference_curve_data`` to handle empty/0-d inputs.

    The original function calls ``np.asarray(transformed.get("x", []), …)``
    on the preview branch. When no channel is selected, ``transformed["x"]``
    is ``None`` and the resulting 0-d array crashes the subsequent slice.
    We re-implement the same logic but route every array through
    :func:`_to_clean_1d_array` first.
    """
    original = getattr(tab, "_resolve_reference_curve_data", None)
    if original is None:
        return

    # Pull the helpers we need by name from tab so we don't duplicate them.
    curve_profile_key_from_ui = getattr(tab, "_curve_profile_key_from_ui", None)
    ensure_entry_origin_snapshot = getattr(tab, "_ensure_entry_origin_snapshot", None)
    entry_untrimmed_xyt = getattr(tab, "_entry_untrimmed_xyt", None)
    trim_xyz_with_step15 = getattr(tab, "_trim_xyz_with_step15", None)
    apply_transforms = getattr(tab, "_apply_transforms", None)
    if any(
        helper is None
        for helper in (
            curve_profile_key_from_ui,
            ensure_entry_origin_snapshot,
            entry_untrimmed_xyt,
            trim_xyz_with_step15,
            apply_transforms,
        )
    ):
        # tab.py is shaped differently than expected; leave the original in
        # place rather than installing a patch that could mask real errors.
        return

    def _resolve_reference_curve_data_patched(app):
        active_key = curve_profile_key_from_ui(app)
        for entry in getattr(app, "data_fit_curves", []):
            if bool(entry.get("is_fit_result", False)):
                continue
            if str(entry.get("signature")) != str(active_key):
                continue
            ensure_entry_origin_snapshot(entry)
            x, y, t = entry_untrimmed_xyt(entry)
            x = _to_clean_1d_array(x)
            y = _to_clean_1d_array(y)
            t = _to_clean_1d_array(t)
            n = int(min(x.size, y.size))
            if n <= 0:
                continue
            x_t, y_t, t_t = trim_xyz_with_step15(
                app,
                x[:n],
                y[:n],
                t[:n] if t.size else np.empty(0, dtype=float),
            )
            if x_t.size <= 0 or y_t.size <= 0:
                continue
            return (
                entry,
                entry.get("signature", entry.get("label", "curve")),
                entry.get("label", "Curve"),
                x_t,
                y_t,
                (t_t if t_t is not None else np.empty(0, dtype=float)),
            )

        transformed = apply_transforms(app, apply_trim=True)
        x = _to_clean_1d_array(transformed.get("x") if transformed else None)
        y = _to_clean_1d_array(transformed.get("y") if transformed else None)
        t = _to_clean_1d_array(transformed.get("time") if transformed else None)
        n = int(min(x.size, y.size))
        if n <= 0:
            return None
        y_label = ""
        try:
            y_label = app.data_fit_y_cb.currentText()
        except (AttributeError, RuntimeError):
            y_label = ""
        return (
            None,
            ("__preview__", y_label),
            y_label or "Preview",
            x[:n],
            y[:n],
            (t[:n] if t.size else np.empty(0, dtype=float)),
        )

    tab._resolve_reference_curve_data = _resolve_reference_curve_data_patched


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    tab = _tab_module()

    original_settings_to_preset = getattr(tab, "_settings_to_preset", None)
    original_apply_preset = getattr(tab, "_apply_preset", None)
    if original_settings_to_preset is None or original_apply_preset is None:
        return

    def _settings_to_preset_patched(app):
        preset = original_settings_to_preset(app)
        try:
            _augment_preset_from_app(app, preset)
        except Exception:
            pass
        return preset

    def _apply_preset_patched(app, preset):
        original_apply_preset(app, preset)
        try:
            _restore_preset_into_app(app, preset)
        except Exception:
            pass

    tab._settings_to_preset = _settings_to_preset_patched
    tab._apply_preset = _apply_preset_patched
    _patch_resolve_reference_curve_data(tab)
    _PATCHED = True

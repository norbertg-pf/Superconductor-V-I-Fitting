"""Builds and wires the Data Fitting tab.

Functions/Classes:
- setup_data_fitting_tab_layout: creates the Data Fitting tab used to extract
  R (or Rho), Ic and n-value from a recorded V-I curve via the procedure in
  src.services.data_fitting_service.
"""

from __future__ import annotations

import os
import traceback
from functools import partial
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pyqtgraph as pg
from nptdms import ChannelObject, GroupObject, TdmsFile, TdmsWriter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .service import (
    adaptive_smooth_for_ec_window,
    BASELINE_MODE_HUBER,
    BASELINE_MODE_OLS,
    BASELINE_MODE_THEIL_SEN,
    DEFAULT_CHI_SQR_TOL,
    DEFAULT_BASELINE_MODE,
    DEFAULT_DIDT_HIGH_FRAC,
    DEFAULT_DIDT_LOW_FRAC,
    DEFAULT_EC_V_PER_CM,
    DEFAULT_EC1_V_PER_CM,
    DEFAULT_EC2_V_PER_CM,
    DEFAULT_EC_WINDOW_GUARD_FRAC,
    DEFAULT_FIT_METHOD,
    DEFAULT_IC_TOLERANCE,
    DEFAULT_LINEAR_HIGH_FRAC,
    DEFAULT_LINEAR_LOW_FRAC,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TRIM_START_A,
    DEFAULT_TRIM_START_FRAC,
    DEFAULT_POWER_LOW_FRAC,
    DEFAULT_POWER_V_FRAC,
    DEFAULT_VC_VOLTS,
    DEFAULT_ZERO_I_FRAC,
    estimate_di_dt,
    estimate_thermal_offset,
    FIT_METHOD_LOG_LOG,
    FIT_METHOD_NONLINEAR,
    fit_linear_baseline,
    WEIGHT_MODE_EQUAL,
    WEIGHT_MODE_ROBUST,
    WEIGHT_MODE_WEIGHTED,
    FitSettings,
    MIN_N_WINDOW_POINTS,
    RAMP_INDUCTIVE_WARN_RATIO,
    robust_view_range,
    run_full_fit,
    trim_vi_curve,
    pick_loglog_i_window_from_thresholds,
)
from .extras import (
    ExportPlotDialog,
    FitPreset,
    GraphSettings,
    GraphSettingsDialog,
    apply_graph_settings,
    load_preset_from_file,
    save_preset_to_file,
)
from . import get_app_version_label


class _FitParamTable(pg.TextItem):
    """Draggable, editable text overlay that shows the engineering-formatted
    fit parameters. Drag with the mouse to position; double-click to edit."""

    def __init__(self):
        super().__init__(text="", anchor=(0, 0), color=(30, 30, 30))
        self.setHtml(self._empty_html())
        self.setZValue(50)
        self._dragging = False
        self._drag_offset = None
        self.setFlag(self.ItemIsMovable, True) if hasattr(self, "setFlag") else None

    @staticmethod
    def _empty_html() -> str:
        return "<i style='color:#777'>Run a fit to see parameters here.</i>"

    def set_parameters(self, result) -> None:
        self.set_parameters_for_curves([("Fit", result)])

    def set_parameters_for_curves(self, results: list[tuple[str, object]]) -> None:
        if not results:
            self.clear_parameters()
            return
        param_names = ("Criterion", "Ic", "n", "R", "V0")
        header_cells = "".join(
            f"<th style='padding:2px 8px; border-bottom:1px solid #bbb;'>{name}</th>"
            for name, _ in results
        )
        body_rows = []
        for p in param_names:
            row_cells = []
            for _, result in results:
                r_name = "Rho" if result.uses_sample_length else "R"
                r_unit = "Ω/cm" if result.uses_sample_length else "Ω"
                v_name = "Ec" if result.uses_sample_length else "Vc"
                v_unit = "V/cm" if result.uses_sample_length else "V"
                value_map = {
                    "n": f"{result.n_value:.2f}",
                    "Ic": f"{result.Ic:.6g} A",
                    "R": _format_engineering(result.R, r_unit, 2),
                    "Criterion": _format_engineering(result.criterion, v_unit, 2),
                    "V0": _format_engineering(result.V0, v_unit, 2),
                }
                value = value_map[p]
                row_cells.append(
                    f"<td style='padding:1px 8px; text-align:right; font-family:monospace;'>{value}</td>"
                )
            row_name = "R/Rho" if p == "R" else ("Vc/Ec" if p == "Criterion" else p)
            body_rows.append(
                f"<tr><td style='padding-right:10px;'><b>{row_name}</b></td>{''.join(row_cells)}</tr>"
            )
        self.setHtml(
            "<div style='background:rgba(255,255,255,200); border:1px solid #999; padding:5px 8px;'>"
            "<table cellspacing='0'>"
            f"<tr><th></th>{header_cells}</tr>"
            + "".join(body_rows)
            + "</table></div>"
        )

    def clear_parameters(self) -> None:
        self.setHtml(self._empty_html())

    # Mouse-drag support ---------------------------------------------------
    def hoverEvent(self, ev):  # noqa: N802 pyqtgraph API
        if hasattr(ev, "acceptDrags"):
            ev.acceptDrags(Qt.LeftButton)

    def mouseDragEvent(self, ev):  # noqa: N802 pyqtgraph API
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            self._drag_offset = self.pos() - ev.buttonDownPos(Qt.LeftButton)
        new_pos = ev.pos() + (self._drag_offset or 0)
        self.setPos(new_pos)

    def mouseDoubleClickEvent(self, ev):  # noqa: N802 pyqtgraph API
        """Allow the user to edit the HTML content directly."""
        from PyQt5.QtWidgets import QInputDialog
        widget = self.getViewWidget()
        current = self.toHtml() if hasattr(self, "toHtml") else ""
        text, ok = QInputDialog.getMultiLineText(
            widget, "Edit parameter table (HTML)", "Content:", current,
        )
        if ok:
            self.setHtml(text)


def _percent_edit(value_fraction: float) -> QLineEdit:
    w = QLineEdit(f"{value_fraction * 100:.2f}")
    w.setMaximumWidth(80)
    return w


def _make_scale_offset_inputs() -> tuple[QLineEdit, QLineEdit]:
    scale = QLineEdit("1.0")
    scale.setMaximumWidth(70)
    scale.setToolTip("Multiplier applied to the raw channel before plotting/fitting.")
    offset = QLineEdit("0.0")
    offset.setMaximumWidth(70)
    offset.setToolTip("Offset subtracted after scaling: displayed = raw * scale - offset.")
    return scale, offset


def _xvalue_edit() -> QLineEdit:
    w = QLineEdit("")
    w.setMaximumWidth(100)
    w.setPlaceholderText("—")
    w.setToolTip("Current (X-axis) value corresponding to the percentage. Editable; changes sync back.")
    return w


def _fill_window_grid(layout, show_cb, *, low_label, low_pct, low_x,
                      high_label, high_pct, high_x, base_row: int = 0,
                      extra_widget=None):
    """Show checkbox on base_row, percents on base_row+1, X values on base_row+2."""
    if extra_widget is None:
        layout.addWidget(show_cb, base_row, 0, 1, 4)
    else:
        layout.addWidget(show_cb, base_row, 0, 1, 2)
        layout.addWidget(extra_widget, base_row, 2, 1, 2)
    layout.addWidget(QLabel(low_label), base_row + 1, 0)
    layout.addWidget(low_pct, base_row + 1, 1)
    layout.addWidget(QLabel(high_label), base_row + 1, 2)
    layout.addWidget(high_pct, base_row + 1, 3)
    layout.addWidget(QLabel("Low (X)"), base_row + 2, 0)
    layout.addWidget(low_x, base_row + 2, 1)
    layout.addWidget(QLabel("High (X)"), base_row + 2, 2)
    layout.addWidget(high_x, base_row + 2, 3)


def _set_silently(widget: QLineEdit, text: str) -> None:
    widget.blockSignals(True)
    try:
        widget.setText(text)
    finally:
        widget.blockSignals(False)


def _capture_fit_window_profile(app, prior: Optional[dict] = None) -> dict:
    """Snapshot the Active-fitting widgets into a per-curve profile dict.

    The Step-4 power_low / power_vfrac widgets reuse the same QLineEdits for
    both fit methods, but the values mean very different things (Ec1/Ec2 in
    µV/cm for log-log; fractions of Imax/Vmax for non-linear). We keep
    independent slots — ``loglog_low/high`` and ``nonlinear_low/high`` — so
    switching methods on a curve restores that method's last-edited values
    instead of clobbering them with defaults.
    """
    prior = prior or {}
    method = _active_fit_method(app)
    snapshot = {
        "fit_method": method,
        "didt_low": app.data_fit_didt_low.text(),
        "didt_high": app.data_fit_didt_high.text(),
        "linear_low": app.data_fit_linear_low.text(),
        "linear_high": app.data_fit_linear_high.text(),
        "power_low": app.data_fit_power_low.text(),
        "power_vfrac": app.data_fit_power_vfrac.text(),
        "max_iter": app.data_fit_max_iter.text(),
        "ic_tol": app.data_fit_ic_tol.text(),
        "chi_tol": app.data_fit_chi_tol.text(),
        "vc": app.data_fit_vc_input.text(),
        "weight_mode": (
            app.data_fit_weight_mode_cb.currentData()
            if getattr(app, "data_fit_weight_mode_cb", None) is not None
            else WEIGHT_MODE_EQUAL
        ),
        "baseline_mode": (
            app.data_fit_baseline_mode_cb.currentData()
            if getattr(app, "data_fit_baseline_mode_cb", None) is not None
            else DEFAULT_BASELINE_MODE
        ),
        "subtract_vofs": (
            app.data_fit_subtract_vofs_cb.isChecked()
            if getattr(app, "data_fit_subtract_vofs_cb", None) is not None
            else True
        ),
        "zero_i_frac": (
            app.data_fit_zero_i_frac.text()
            if getattr(app, "data_fit_zero_i_frac", None) is not None
            else ""
        ),
        "trim_start_abs_a": getattr(app, "data_fit_trim_start_abs_a", QLineEdit("")).text(),
        "trim_start_frac": getattr(app, "data_fit_trim_start_frac", QLineEdit("")).text(),
        "trim_use_percent": (
            app.data_fit_trim_use_percent_cb.isChecked()
            if getattr(app, "data_fit_trim_use_percent_cb", None) is not None
            else False
        ),
        "trim_rampdown_tail": (
            app.data_fit_trim_rampdown_cb.isChecked()
            if getattr(app, "data_fit_trim_rampdown_cb", None) is not None
            else True
        ),
        "show_didt": (
            app.data_fit_show_didt.isChecked()
            if getattr(app, "data_fit_show_didt", None) is not None
            else False
        ),
        "show_linear": (
            app.data_fit_show_linear.isChecked()
            if getattr(app, "data_fit_show_linear", None) is not None
            else False
        ),
        "show_power": (
            app.data_fit_show_power.isChecked()
            if getattr(app, "data_fit_show_power", None) is not None
            else False
        ),
        # Step-4 I-window fields (Low/High X) are persisted per curve so
        # the Active fitting settings selector can restore the latest
        # recalculated window for each plotted curve.
        "power_low_x": app.data_fit_power_low_x.text(),
        "power_high_x": app.data_fit_power_high_x.text(),
    }
    # Method-specific Step-4 slots: active widget values go into the active
    # method's slot; the inactive method's slot is preserved from the prior
    # profile, falling back to the IEC/non-linear defaults.
    default_loglog_low = f"{DEFAULT_EC1_V_PER_CM * 1.0e6:g}"
    default_loglog_high = f"{DEFAULT_EC2_V_PER_CM * 1.0e6:g}"
    default_nonlin_low = f"{DEFAULT_POWER_LOW_FRAC * 100:.2f}"
    default_nonlin_high = f"{DEFAULT_POWER_V_FRAC * 100:.2f}"
    if method == FIT_METHOD_LOG_LOG:
        snapshot["loglog_low"] = app.data_fit_power_low.text()
        snapshot["loglog_high"] = app.data_fit_power_vfrac.text()
        snapshot["nonlinear_low"] = prior.get("nonlinear_low", default_nonlin_low)
        snapshot["nonlinear_high"] = prior.get("nonlinear_high", default_nonlin_high)
    else:
        snapshot["nonlinear_low"] = app.data_fit_power_low.text()
        snapshot["nonlinear_high"] = app.data_fit_power_vfrac.text()
        snapshot["loglog_low"] = prior.get("loglog_low", default_loglog_low)
        snapshot["loglog_high"] = prior.get("loglog_high", default_loglog_high)
    return snapshot


def _apply_fit_window_profile(app, profile: dict) -> None:
    if not profile:
        return
    method = profile.get("fit_method", DEFAULT_FIT_METHOD)
    rb_loglog = getattr(app, "data_fit_method_loglog_rb", None)
    rb_nonlinear = getattr(app, "data_fit_method_nonlinear_rb", None)
    if rb_loglog is not None and rb_nonlinear is not None:
        rb_loglog.blockSignals(True)
        rb_nonlinear.blockSignals(True)
        try:
            if method == FIT_METHOD_LOG_LOG:
                rb_loglog.setChecked(True)
            else:
                rb_nonlinear.setChecked(True)
        finally:
            rb_loglog.blockSignals(False)
            rb_nonlinear.blockSignals(False)
    for widget, key in (
        (app.data_fit_didt_low, "didt_low"),
        (app.data_fit_didt_high, "didt_high"),
        (app.data_fit_linear_low, "linear_low"),
        (app.data_fit_linear_high, "linear_high"),
        (app.data_fit_max_iter, "max_iter"),
        (app.data_fit_ic_tol, "ic_tol"),
        (app.data_fit_chi_tol, "chi_tol"),
        (app.data_fit_vc_input, "vc"),
    ):
        if key in profile:
            _set_silently(widget, str(profile[key]))
    if "weight_mode" in profile and getattr(app, "data_fit_weight_mode_cb", None) is not None:
        idx = app.data_fit_weight_mode_cb.findData(profile["weight_mode"])
        if idx < 0:
            idx = 0
        app.data_fit_weight_mode_cb.blockSignals(True)
        try:
            app.data_fit_weight_mode_cb.setCurrentIndex(idx)
        finally:
            app.data_fit_weight_mode_cb.blockSignals(False)
    if "baseline_mode" in profile and getattr(app, "data_fit_baseline_mode_cb", None) is not None:
        idx = app.data_fit_baseline_mode_cb.findData(profile["baseline_mode"])
        if idx < 0:
            idx = 0
        app.data_fit_baseline_mode_cb.blockSignals(True)
        try:
            app.data_fit_baseline_mode_cb.setCurrentIndex(idx)
        finally:
            app.data_fit_baseline_mode_cb.blockSignals(False)
    if method == FIT_METHOD_LOG_LOG:
        low_key, high_key = "loglog_low", "loglog_high"
    else:
        low_key, high_key = "nonlinear_low", "nonlinear_high"
    low_val = profile.get(low_key, profile.get("power_low"))
    high_val = profile.get(high_key, profile.get("power_vfrac"))
    if low_val is not None:
        _set_silently(app.data_fit_power_low, str(low_val))
    if high_val is not None:
        _set_silently(app.data_fit_power_vfrac, str(high_val))
    if "power_low_x" in profile:
        _set_silently(app.data_fit_power_low_x, str(profile["power_low_x"]))
    if "power_high_x" in profile:
        _set_silently(app.data_fit_power_high_x, str(profile["power_high_x"]))
    if "zero_i_frac" in profile and getattr(app, "data_fit_zero_i_frac", None) is not None:
        _set_silently(app.data_fit_zero_i_frac, str(profile["zero_i_frac"]))
    if "trim_start_abs_a" in profile and getattr(app, "data_fit_trim_start_abs_a", None) is not None:
        _set_silently(app.data_fit_trim_start_abs_a, str(profile["trim_start_abs_a"]))
    if "trim_start_frac" in profile and getattr(app, "data_fit_trim_start_frac", None) is not None:
        _set_silently(app.data_fit_trim_start_frac, str(profile["trim_start_frac"]))
    if "subtract_vofs" in profile and getattr(app, "data_fit_subtract_vofs_cb", None) is not None:
        cb = app.data_fit_subtract_vofs_cb
        cb.blockSignals(True)
        try:
            cb.setChecked(bool(profile["subtract_vofs"]))
        finally:
            cb.blockSignals(False)
    if "trim_use_percent" in profile and getattr(app, "data_fit_trim_use_percent_cb", None) is not None:
        app.data_fit_trim_use_percent_cb.setChecked(bool(profile["trim_use_percent"]))
    if "trim_rampdown_tail" in profile and getattr(app, "data_fit_trim_rampdown_cb", None) is not None:
        app.data_fit_trim_rampdown_cb.setChecked(bool(profile["trim_rampdown_tail"]))
    for attr, key in (
        ("data_fit_show_didt", "show_didt"),
        ("data_fit_show_linear", "show_linear"),
        ("data_fit_show_power", "show_power"),
    ):
        widget = getattr(app, attr, None)
        if widget is not None and key in profile:
            widget.blockSignals(True)
            try:
                widget.setChecked(bool(profile[key]))
            finally:
                widget.blockSignals(False)
    _update_method_mode_ui(app)
    _update_band_states(app)
    _update_equation_label(app)
    sync_region_to_inputs(app)


def _float_from(widget: QLineEdit, fallback: float, as_fraction: bool = False) -> float:
    if widget is None:
        return fallback / 100.0 if as_fraction else fallback
    try:
        v = float(widget.text())
    except (TypeError, ValueError):
        return fallback
    if as_fraction:
        return v / 100.0
    return v


def _adaptive_smooth_visual(y: np.ndarray, ec1: float, ec2: float) -> np.ndarray:
    """UI wrapper around the shared Ec-window smoothing implementation."""
    return adaptive_smooth_for_ec_window(y, ec1, ec2)


def _read_time_channel(tdms_file):
    if "RawData" in tdms_file and "Time" in tdms_file["RawData"]:
        return np.asarray(tdms_file["RawData"]["Time"][:], dtype=float)
    if "Time" in tdms_file and "Time" in tdms_file["Time"]:
        return np.asarray(tdms_file["Time"]["Time"][:], dtype=float)
    return None


_VTAP_KEYS = ("VTap_Distance_cm", "Voltage_Tap_Distance_cm",
              "Voltage_Tap_Distance", "Voltage_Tab_Distance")


def _read_channel_metadata(channel) -> dict:
    """Extract Scale_Factor / Offset / VTap distance metadata.

    The voltage-tap separation is written by the Tape and Cable presets of
    DAQUniversal and by its QD configuration (see TDMS metadata spec). When
    present, the Data Fitting tab auto-populates the voltage-tap separation
    and enables the E-field path so Ic is computed per IEC 61788 with a
    1 µV/cm criterion. Multiple key spellings are accepted to stay
    compatible with older recordings.
    """
    props = getattr(channel, "properties", {}) or {}
    try:
        scale = float(props.get("Scale_Factor", 1.0))
    except (TypeError, ValueError):
        scale = 1.0
    try:
        offset = float(props.get("Offset", 0.0))
    except (TypeError, ValueError):
        offset = 0.0
    v_tap: Optional[float] = None
    for key in _VTAP_KEYS:
        v_tap_raw = props.get(key, "")
        if v_tap_raw in ("", None):
            continue
        try:
            v_tap = float(v_tap_raw)
        except (TypeError, ValueError):
            v_tap = None
            continue
        if v_tap <= 0:
            v_tap = None
            continue
        break
    return {"scale": scale, "offset": offset, "voltage_tap_cm": v_tap}


def _read_signal_channel(tdms_file, signal_name: str):
    if "RawData" in tdms_file and signal_name in tdms_file["RawData"]:
        return np.asarray(tdms_file["RawData"][signal_name][:], dtype=float)
    for group in tdms_file.groups():
        for channel in group.channels():
            if getattr(channel, "name", "") == signal_name:
                return np.asarray(channel[:], dtype=float)
    return None




class DataFittingController:
    """Keeps the Data Fitting tab state and computation in one place."""

    def __init__(self, app):
        self.app = app
        self.tdms_path: str = ""
        self.time_array = None
        self.channel_cache: dict[str, np.ndarray] = {}
        self.channel_names: list[str] = []
        self.channel_metadata: dict[str, dict] = {}
        self.last_result = None
        # Most recent (label, FitResult) pairs from a multi-curve run_fit.
        # Used by the manual "Save metadata" button to persist results that
        # weren't tied to a stored ``data_fit_curves`` entry — e.g. fits run
        # against the preview curve.
        self.last_fit_results: list[tuple[str, object]] = []
        # Saved-fit metadata keyed by channel label, populated from either a
        # FitResults group inside the TDMS or fit_* properties attached to the
        # source channel itself. Used to redraw fitted curves on file load.
        self.saved_fit_results: dict[str, dict] = {}

    # --- data source -----------------------------------------------------
    def load_recording(self, path: str) -> tuple[bool, str]:
        self.channel_cache.clear()
        self.channel_metadata.clear()
        self.channel_names = []
        self.time_array = None
        self.saved_fit_results = {}
        if not path or not os.path.exists(path):
            return False, "No recording found. Click 'Load File…' to choose a TDMS."
        try:
            with TdmsFile.read(path) as tdms_file:
                self.time_array = _read_time_channel(tdms_file)
                names: list[str] = []
                for group in tdms_file.groups():
                    is_fit_results_group = (group.name == "FitResults")
                    for channel in group.channels():
                        name = getattr(channel, "name", "")
                        if not name or name.lower() == "time":
                            continue
                        props = dict(getattr(channel, "properties", {}) or {})
                        if is_fit_results_group:
                            # FitResults group → property dict per fitted label.
                            self.saved_fit_results[name] = props
                            continue
                        if name in self.channel_cache:
                            continue
                        self.channel_cache[name] = np.asarray(channel[:], dtype=float)
                        self.channel_metadata[name] = _read_channel_metadata(channel)
                        names.append(name)
                        # Same-group mode: fit_* properties live on the source
                        # channel itself. Strip the prefix so consumers see the
                        # same dict shape as a FitResults entry.
                        same_group_fit = {
                            key[len(_FIT_PROPERTY_PREFIX):]: value
                            for key, value in props.items()
                            if key.startswith(_FIT_PROPERTY_PREFIX)
                        }
                        if same_group_fit and name not in self.saved_fit_results:
                            self.saved_fit_results[name] = same_group_fit
                self.channel_names = names
        except Exception as exc:
            return False, f"Could not read TDMS: {exc}"
        if self.time_array is None:
            return False, "Recording has no 'Time' channel."
        self.tdms_path = path
        return True, f"Loaded {os.path.basename(path)} with {len(self.channel_names)} channels."

    def get_channel(self, name: str):
        if not name:
            return None
        return self.channel_cache.get(name)

    def get_metadata(self, name: str) -> dict:
        return self.channel_metadata.get(
            name, {"scale": 1.0, "offset": 0.0, "voltage_tap_cm": None}
        )

    @staticmethod
    def apply_transform(values, scale: float, offset: float):
        """Apply the project-standard transform: out = (raw * scale) - offset."""
        if values is None:
            return None
        return np.asarray(values, dtype=float) * float(scale) - float(offset)

    # --- fit orchestration ----------------------------------------------
    def compute_fit(self, *, time_sig: str, x_sig: str, y_sig: str,
                    x_scale: float, x_offset: float,
                    y_scale: float, y_offset: float,
                    t_scale: float, t_offset: float,
                    settings: FitSettings, avg_window: int = 1):
        x_raw = self.get_channel(x_sig)
        y_raw = self.get_channel(y_sig)
        t_raw = self.get_channel(time_sig) if time_sig and time_sig != "Time" else self.time_array
        if t_raw is None or x_raw is None or y_raw is None:
            return None, "Please choose Time, X and Y channels that exist in the recording."
        x = self.apply_transform(x_raw, x_scale, x_offset)
        y = self.apply_transform(y_raw, y_scale, y_offset)
        t = self.apply_transform(t_raw, t_scale, t_offset)
        if avg_window and avg_window > 1:
            win = int(avg_window)
            t = _block_average(t, win)
            x = _block_average(x, win)
            y = _block_average(y, win)
        if settings.sample_length_cm and settings.sample_length_cm > 0:
            y = y / float(settings.sample_length_cm)
        result = run_full_fit(t, x, y, settings)
        self.last_result = result
        return result, result.message


def _connect_data_fitting_actions(app):
    app.data_fit_clear_btn.clicked.connect(lambda: _reset_data_fitting_defaults(app))
    app.data_fit_load_btn.clicked.connect(app.data_fitting_open_file)
    app.data_fit_refresh_btn.clicked.connect(app.data_fitting_refresh_current)
    app.data_fit_run_btn.clicked.connect(app.data_fitting_run)
    app.data_fit_load_metadata_btn.clicked.connect(app.data_fitting_load_metadata)
    app.data_fit_robust_view_btn.clicked.connect(app.data_fitting_robust_view)
    app.data_fit_reset_view_btn.clicked.connect(app.data_fitting_reset_view)
    app.data_fit_zoom_mode_btn.toggled.connect(app.data_fitting_toggle_zoom)
    for (window, which), (pct_widget, x_widget, _axis) in app.data_fit_window_inputs.items():
        pct_widget.editingFinished.connect(partial(_handle_window_edit, app, window, which, "pct"))
        x_widget.editingFinished.connect(partial(_handle_window_edit, app, window, which, "x"))
    app.data_fit_use_length_cb.stateChanged.connect(lambda _: _on_use_length_changed(app))
    for widget in (
        app.data_fit_time_scale, app.data_fit_time_offset,
        app.data_fit_x_scale, app.data_fit_x_offset,
        app.data_fit_y_scale, app.data_fit_y_offset,
    ):
        if hasattr(widget, "editingFinished"):
            widget.editingFinished.connect(lambda: _on_transform_inputs_changed(app))
    for axis in ("time", "x", "y"):
        combo = getattr(app, f"data_fit_{axis}_cb")
        combo.currentIndexChanged.connect(
            lambda _, a=axis: _on_channel_selection_changed(app, a)
        )
    for w in (app.data_fit_max_iter, app.data_fit_ic_tol, app.data_fit_chi_tol, app.data_fit_vc_input):
        w.editingFinished.connect(lambda: _save_active_curve_profile(app))
    if getattr(app, "data_fit_weight_mode_cb", None) is not None:
        app.data_fit_weight_mode_cb.currentIndexChanged.connect(lambda _: _save_active_curve_profile(app))
    if getattr(app, "data_fit_baseline_mode_cb", None) is not None:
        app.data_fit_baseline_mode_cb.currentIndexChanged.connect(lambda _: _save_active_curve_profile(app))
    app.data_fit_graph_btn.clicked.connect(lambda: _open_graph_settings(app))
    app.data_fit_save_preset_btn.clicked.connect(lambda: _save_preset(app))
    app.data_fit_load_preset_btn.clicked.connect(lambda: _load_preset(app))
    app.data_fit_help_btn.clicked.connect(lambda: _open_help_dialog(app))
    app.data_fit_show_didt.toggled.connect(
        lambda _: (_update_band_states(app), _save_active_curve_profile(app))
    )
    app.data_fit_show_linear.toggled.connect(
        lambda _: (_update_band_states(app), _save_active_curve_profile(app))
    )
    app.data_fit_show_power.toggled.connect(
        lambda checked: (_on_show_power_toggled(app, checked), _save_active_curve_profile(app))
    )
    if getattr(app, "data_fit_subtract_vofs_cb", None) is not None:
        app.data_fit_subtract_vofs_cb.toggled.connect(
            lambda _: _save_active_curve_profile(app)
        )
    if getattr(app, "data_fit_zero_i_frac", None) is not None:
        app.data_fit_zero_i_frac.editingFinished.connect(
            lambda: _save_active_curve_profile(app)
        )
    for attr in ("data_fit_trim_start_abs_a", "data_fit_trim_start_frac"):
        w = getattr(app, attr, None)
        if w is not None:
            w.editingFinished.connect(lambda: _save_active_curve_profile(app))
    for attr in ("data_fit_trim_use_percent_cb", "data_fit_trim_rampdown_cb"):
        w = getattr(app, attr, None)
        if w is not None:
            w.toggled.connect(lambda _: _save_active_curve_profile(app))
    app.data_fit_add_smoothed_btn.clicked.connect(lambda: (_add_smoothed_curve_from_current(app), robust_view(app)))
    app.data_fit_export_btn.clicked.connect(lambda: _open_export_dialog(app))
    app.data_fit_settings_btn.clicked.connect(lambda: _open_settings_dialog(app))
    app.data_fit_save_metadata_btn.clicked.connect(lambda: _save_metadata_clicked(app))
    # Autosave gates the two layout checkboxes; save-separate gates same-group.
    # Initialise the enabled state once and re-evaluate on every toggle.
    app.data_fit_autosave_cb.toggled.connect(lambda _: _refresh_save_settings_enabled(app))
    app.data_fit_save_separate_cb.toggled.connect(lambda _: _refresh_save_settings_enabled(app))
    _refresh_save_settings_enabled(app)
    app.data_fit_add_plot_btn.clicked.connect(lambda: (_add_plot_from_current(app), robust_view(app)))
    app.data_fit_add_corrected_btn.clicked.connect(lambda: (_add_corrected_curve_from_last_fit(app), robust_view(app)))
    app.data_fit_plot_summary_btn.clicked.connect(lambda: _open_plot_summary(app))
    app.data_fit_curve_profile_cb.currentIndexChanged.connect(lambda _: _on_curve_profile_changed(app))
    # Mutually exclusive radios fire toggled() twice per click (deselect +
    # select). Filter to the "checked" edge so the per-curve profile only
    # snapshots once and we don't overwrite the just-loaded other-method slot.
    app.data_fit_method_loglog_rb.toggled.connect(
        lambda checked: _on_fit_method_changed(app) if checked else None
    )
    app.data_fit_method_nonlinear_rb.toggled.connect(
        lambda checked: _on_fit_method_changed(app) if checked else None
    )
    app.data_fit_plot_scale_btn.clicked.connect(lambda: _toggle_plot_scale(app))


def _on_transform_inputs_changed(app) -> None:
    app.data_fit_plot_dirty = True
    _update_avg_rate_label(app)
    _refresh_all_x_values(app)
    ctx = _data_ctx(app)
    if ctx is not None:
        _, _, x_arr, y_arr, _ = ctx
        _update_fit_bands(app, x_arr, y_arr)


def _reset_data_fitting_defaults(app) -> None:
    app.data_fit_controller.tdms_path = ""
    app.data_fit_controller.time_array = None
    app.data_fit_controller.channel_cache = {}
    app.data_fit_controller.channel_names = []
    app.data_fit_controller.channel_metadata = {}
    app.data_fit_path_label.setText("No file loaded.")
    app.data_fit_path_label.setStyleSheet("color: gray;")
    for cb in (app.data_fit_time_cb, app.data_fit_x_cb, app.data_fit_y_cb):
        cb.blockSignals(True)
        cb.clear()
        cb.blockSignals(False)
    app.data_fit_time_scale.setText("1.0")
    app.data_fit_time_offset.setText("0.0")
    app.data_fit_x_scale.setText("1.0")
    app.data_fit_x_offset.setText("0.0")
    app.data_fit_y_scale.setText("1.0")
    app.data_fit_y_offset.setText("0.0")
    app.data_fit_avg_input.setText("1")
    # IEC 61788 defaults: enable voltage-tap path with Ec = 1 µV/cm. The box
    # auto-unchecks if the next loaded TDMS lacks Voltage_Tab_Distance.
    app.data_fit_use_length_cb.setChecked(True)
    app.data_fit_length_input.setText("1.0")
    app.data_fit_vc_input.setText(f"{DEFAULT_EC_V_PER_CM * 1.0e6:.6g}")
    if getattr(app, "data_fit_weight_mode_cb", None) is not None:
        app.data_fit_weight_mode_cb.setCurrentIndex(0)
    if getattr(app, "data_fit_baseline_mode_cb", None) is not None:
        idx_default_baseline = app.data_fit_baseline_mode_cb.findData(DEFAULT_BASELINE_MODE)
        app.data_fit_baseline_mode_cb.setCurrentIndex(max(0, idx_default_baseline))
    if getattr(app, "data_fit_subtract_vofs_cb", None) is not None:
        app.data_fit_subtract_vofs_cb.setChecked(True)
    if getattr(app, "data_fit_zero_i_frac", None) is not None:
        app.data_fit_zero_i_frac.setText(f"{DEFAULT_ZERO_I_FRAC * 100:.2f}")
    if getattr(app, "data_fit_trim_start_abs_a", None) is not None:
        app.data_fit_trim_start_abs_a.setText(f"{DEFAULT_TRIM_START_A:.6g}")
    if getattr(app, "data_fit_trim_start_frac", None) is not None:
        app.data_fit_trim_start_frac.setText(f"{DEFAULT_TRIM_START_FRAC * 100:.2f}")
    if getattr(app, "data_fit_trim_use_percent_cb", None) is not None:
        app.data_fit_trim_use_percent_cb.setChecked(False)
    if getattr(app, "data_fit_trim_rampdown_cb", None) is not None:
        app.data_fit_trim_rampdown_cb.setChecked(True)
    app.data_fit_didt_low.setText(f"{DEFAULT_DIDT_LOW_FRAC * 100:.2f}")
    app.data_fit_didt_high.setText(f"{DEFAULT_DIDT_HIGH_FRAC * 100:.2f}")
    app.data_fit_linear_low.setText(f"{DEFAULT_LINEAR_LOW_FRAC * 100:.2f}")
    app.data_fit_linear_high.setText(f"{DEFAULT_LINEAR_HIGH_FRAC * 100:.2f}")
    # Block radio signals during reset so _on_fit_method_changed doesn't try
    # to snapshot the half-reset profile.
    app.data_fit_method_loglog_rb.blockSignals(True)
    app.data_fit_method_nonlinear_rb.blockSignals(True)
    try:
        if DEFAULT_FIT_METHOD == FIT_METHOD_LOG_LOG:
            app.data_fit_method_loglog_rb.setChecked(True)
            app.data_fit_power_low.setText(f"{DEFAULT_EC1_V_PER_CM * 1.0e6:g}")
            app.data_fit_power_vfrac.setText(f"{DEFAULT_EC2_V_PER_CM * 1.0e6:g}")
        else:
            app.data_fit_method_nonlinear_rb.setChecked(True)
            app.data_fit_power_low.setText(f"{DEFAULT_POWER_LOW_FRAC * 100:.2f}")
            app.data_fit_power_vfrac.setText(f"{DEFAULT_POWER_V_FRAC * 100:.2f}")
    finally:
        app.data_fit_method_loglog_rb.blockSignals(False)
        app.data_fit_method_nonlinear_rb.blockSignals(False)
    for cb in (app.data_fit_show_didt, app.data_fit_show_linear, app.data_fit_show_power):
        cb.blockSignals(True)
        try:
            cb.setChecked(False)
        finally:
            cb.blockSignals(False)
    for entry in list(getattr(app, "data_fit_curves", [])):
        item = entry.get("plot_item")
        if item is not None:
            app.data_fit_plot.removeItem(item)
        fit_item = entry.get("fit_plot_item")
        if fit_item is not None:
            app.data_fit_plot.removeItem(fit_item)
    app.data_fit_curves = []
    app.data_fit_power_ref_curve = None
    app.data_fit_power_window_manual = False
    app.data_fit_preview_visible = True
    app.data_fit_preview_include_in_fit = True
    app.data_fit_preview_color = "#1f77b4"
    app.data_fit_preview_alpha_pct = 100
    app.data_fit_preview_style = {"draw_mode": "Auto", "line_width": 1.5, "point_size": 4}
    app.data_fit_curve_profiles = {"__preview__": _capture_fit_window_profile(app)}
    app.data_fit_plot_dirty = True
    app.data_fit_raw_curve.setData([], [])
    app.data_fit_model_curve.setData([], [])
    app.data_fit_result_text.clear()
    app.data_fit_xrange_label.setText("X window: full")
    _hide_fit_overlays(app)
    _clear_warning(app)
    _update_avg_rate_label(app)
    _update_method_mode_ui(app)
    _refresh_curve_profile_selector(app)
    # Reset the Settings checkboxes to their defaults so a Clear gives the
    # user a clean slate for the next session.
    if hasattr(app, "data_fit_auto_load_cb"):
        app.data_fit_auto_load_cb.setChecked(True)
    if hasattr(app, "data_fit_autosave_cb"):
        app.data_fit_autosave_cb.setChecked(True)
    if hasattr(app, "data_fit_save_separate_cb"):
        app.data_fit_save_separate_cb.setChecked(False)
    if hasattr(app, "data_fit_same_group_cb"):
        app.data_fit_same_group_cb.setChecked(True)
    _refresh_save_settings_enabled(app)
    if hasattr(app, "data_fit_save_metadata_btn"):
        app.data_fit_save_metadata_btn.setEnabled(False)


def _curve_profile_key_from_ui(app) -> str:
    key = app.data_fit_curve_profile_cb.currentData()
    return str(key) if key else "__preview__"


def _save_active_curve_profile(app) -> None:
    # Suppressed while we're applying a profile to the widgets — otherwise
    # downstream signal handlers would snapshot half-loaded state into the
    # newly selected curve's slot and undo what we're about to load.
    if getattr(app, "_data_fit_suspend_profile_save", False):
        return
    key = _curve_profile_key_from_ui(app)
    profiles = getattr(app, "data_fit_curve_profiles", {}) or {}
    prior = profiles.get(key, {}) if isinstance(profiles, dict) else {}
    profiles[key] = _capture_fit_window_profile(app, prior=prior)
    app.data_fit_curve_profiles = profiles


def _on_curve_profile_changed(app) -> None:
    key = _curve_profile_key_from_ui(app)
    profiles = getattr(app, "data_fit_curve_profiles", {})
    if key not in profiles:
        profiles[key] = _capture_fit_window_profile(app)
        app.data_fit_curve_profiles = profiles
    # Snapshot the target profile up front so it survives the in-flight
    # widget mutations driven by length-sync and apply.
    target = dict(profiles.get(key, {}))
    app._data_fit_suspend_profile_save = True
    try:
        _sync_active_length_settings_from_profile_key(app, key)
        _apply_fit_window_profile(app, target)
    finally:
        app._data_fit_suspend_profile_save = False
    # Re-save once at the end so the active profile reflects exactly what
    # the widgets now show (idempotent for a clean apply).
    _save_active_curve_profile(app)


def _refresh_curve_profile_selector(app) -> None:
    combo = app.data_fit_curve_profile_cb
    current_key = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    if getattr(app, "data_fit_preview_visible", True):
        combo.addItem("Preview", "__preview__")
    for entry in getattr(app, "data_fit_curves", []):
        if bool(entry.get("is_fit_result", False)):
            continue
        label = entry.get("label", "Curve")
        sig = str(entry.get("signature", label))
        combo.addItem(label, sig)
    if combo.count() == 0:
        combo.addItem("No plotted curve", "__none__")
    idx = combo.findData(current_key)
    combo.setCurrentIndex(idx if idx >= 0 else 0)
    combo.blockSignals(False)


def _find_curve_for_profile_key(app, key: str) -> Optional[dict]:
    key_str = str(key or "")
    for entry in getattr(app, "data_fit_curves", []):
        if bool(entry.get("is_fit_result", False)):
            continue
        signature = str(entry.get("signature", entry.get("label", "Curve")))
        if signature == key_str:
            return entry
    return None


def _sync_active_length_settings(app, *, use_length: bool, length_cm: float) -> None:
    app.data_fit_use_length_cb.blockSignals(True)
    app.data_fit_use_length_cb.setChecked(bool(use_length))
    app.data_fit_use_length_cb.blockSignals(False)
    if length_cm > 0:
        _set_silently(app.data_fit_length_input, f"{float(length_cm):g}")
    _on_use_length_changed(app)


def _sync_active_length_settings_from_profile_key(app, key: str) -> None:
    if str(key) == "__preview__":
        return
    entry = _find_curve_for_profile_key(app, key)
    if entry is None:
        return
    src = entry.get("source") or {}
    use_length = bool(src.get("use_length", False))
    length_cm = float(src.get("length_cm", 1.0) or 1.0)
    _sync_active_length_settings(app, use_length=use_length, length_cm=length_cm)


def setup_data_fitting_tab_layout(app):
    root = QHBoxLayout(app.ui_state.data_fitting_tab)

    left_widget = QWidget()
    left = QVBoxLayout(left_widget)
    left_widget.setMaximumWidth(504)

    file_row = QHBoxLayout()
    app.data_fit_path_label = QLabel("No file loaded.")
    app.data_fit_path_label.setStyleSheet("color: gray;")
    app.data_fit_clear_btn = QPushButton("Clear")
    app.data_fit_clear_btn.setToolTip("Reset Data Fitting to defaults and clear loaded curves and preview state.")
    app.data_fit_load_btn = QPushButton("Load file…")
    app.data_fit_load_btn.setToolTip("Open a TDMS recording from disk for fitting.")
    app.data_fit_refresh_btn = QPushButton("Use current")
    app.data_fit_refresh_btn.setToolTip("Reload the most recent recording from the active acquisition into this tab.")
    app.data_fit_export_btn = QPushButton("Export…")
    app.data_fit_export_btn.setToolTip("Export the plot in a chosen image or data format (PNG, SVG, CSV, …).")
    app.data_fit_settings_btn = QPushButton("Settings…")
    app.data_fit_settings_btn.setToolTip(
        "Open the Data Fitting settings dialog: auto-load behaviour, automatic\n"
        "metadata saving, and where fit parameters are written in the TDMS file."
    )
    # Compact buttons fit four actions plus the Settings button on one row.
    for compact_btn in (
        app.data_fit_clear_btn,
        app.data_fit_load_btn,
        app.data_fit_refresh_btn,
        app.data_fit_export_btn,
        app.data_fit_settings_btn,
    ):
        compact_btn.setStyleSheet("padding: 4px 8px; font-size: 11px;")
    file_row.addWidget(app.data_fit_clear_btn)
    file_row.addWidget(app.data_fit_load_btn)
    file_row.addWidget(app.data_fit_refresh_btn)
    file_row.addWidget(app.data_fit_export_btn)
    file_row.addWidget(app.data_fit_settings_btn)
    left.addLayout(file_row)
    left.addWidget(app.data_fit_path_label)

    # Settings checkboxes live in the Settings dialog (constructed lazily when
    # the user clicks the button) but the QCheckBox instances themselves are
    # kept on ``app`` so the rest of the tab can read their state directly.
    # All four default to "checked" — i.e. metadata is saved automatically,
    # in the same group/channel, and the tab auto-loads the fitted recording.
    app.data_fit_auto_load_cb = QCheckBox(
        "Auto-load fitted recording into plot after acquisition or on file load"
    )
    app.data_fit_auto_load_cb.setChecked(True)
    app.data_fit_auto_load_cb.setToolTip(
        "Checked (default): after Stop Read, the just-finished TDMS is loaded\n"
        "into the Data Fitting tab and every fit-enabled voltage channel is\n"
        "plotted with its fitted curve. Loading a previously fitted TDMS also\n"
        "redraws the saved fit overlays.\n"
        "Unchecked: post-acquisition fits are still written to the TDMS as\n"
        "channel metadata, but the Data Fitting plot is not refreshed."
    )
    app.data_fit_autosave_cb = QCheckBox(
        "Automatically save fitting parameters as metadata"
    )
    app.data_fit_autosave_cb.setChecked(True)
    app.data_fit_autosave_cb.setToolTip(
        "Checked (default): every fit attempt — successful or failed — is "
        "written into the loaded TDMS file as fit metadata, using the layout\n"
        "selected by the two options below.\n"
        "Unchecked: the fit runs and shows the parameters in the result panel "
        "but nothing is written to disk. Use the manual 'Save metadata' button\n"
        "next to Run Fit when you do want to persist a particular result."
    )
    app.data_fit_save_separate_cb = QCheckBox(
        "Save fit results to a separate TDMS file"
    )
    app.data_fit_save_separate_cb.setChecked(False)
    app.data_fit_save_separate_cb.setToolTip(
        "Checked: write fit results to <source>_fit_report.tdms next to the source file.\n"
        "Unchecked (default): write fit metadata into the source TDMS itself "
        "(the layout depends on 'Use the same group/channel for fit metadata' below)."
    )
    app.data_fit_same_group_cb = QCheckBox(
        "Use the same group/channel for fit metadata"
    )
    app.data_fit_same_group_cb.setChecked(True)
    app.data_fit_same_group_cb.setToolTip(
        "Checked (default): write fit parameters as TDMS properties on the "
        "fitted voltage channel itself, in its original group. No extra\n"
        "FitResults group is created and the file layout stays familiar.\n"
        "Unchecked: write fit parameters into a dedicated 'FitResults' group, "
        "one channel per fitted curve. Useful when several fits with different\n"
        "settings should coexist without overwriting each other.\n"
        "Greyed out when 'Save fit results to a separate TDMS file' is checked."
    )

    app.data_fit_channels_group = QGroupBox("Channels (displayed = raw * scale - offset)")
    ch_grid = QGridLayout(app.data_fit_channels_group)
    ch_grid.setContentsMargins(9, 6, 9, 9)
    ch_grid.setVerticalSpacing(2)
    ch_grid.setAlignment(Qt.AlignTop)
    app.data_fit_time_cb = QComboBox()
    app.data_fit_x_cb = QComboBox()
    app.data_fit_y_cb = QComboBox()
    (app.data_fit_time_scale, app.data_fit_time_offset) = _make_scale_offset_inputs()
    (app.data_fit_x_scale, app.data_fit_x_offset) = _make_scale_offset_inputs()
    (app.data_fit_y_scale, app.data_fit_y_offset) = _make_scale_offset_inputs()
    ch_grid.addWidget(QLabel("Channel"), 0, 0)
    ch_grid.addWidget(QLabel("Scale"), 0, 2)
    ch_grid.addWidget(QLabel("Offset"), 0, 3)
    ch_grid.addWidget(QLabel("Time axis:"), 1, 0)
    ch_grid.addWidget(app.data_fit_time_cb, 1, 1)
    ch_grid.addWidget(app.data_fit_time_scale, 1, 2)
    ch_grid.addWidget(app.data_fit_time_offset, 1, 3)
    ch_grid.addWidget(QLabel("X axis (current):"), 2, 0)
    ch_grid.addWidget(app.data_fit_x_cb, 2, 1)
    ch_grid.addWidget(app.data_fit_x_scale, 2, 2)
    ch_grid.addWidget(app.data_fit_x_offset, 2, 3)
    ch_grid.addWidget(QLabel("Y axis (voltage):"), 3, 0)
    ch_grid.addWidget(app.data_fit_y_cb, 3, 1)
    ch_grid.addWidget(app.data_fit_y_scale, 3, 2)
    ch_grid.addWidget(app.data_fit_y_offset, 3, 3)
    app.data_fit_avg_input = QLineEdit("1")
    app.data_fit_avg_input.setMaximumWidth(80)
    app.data_fit_avg_input.setToolTip(
        "Block-average factor (samples per bin) applied to the X/Y/time signals "
        "before plotting and fitting. 1 = no averaging; N > 1 resamples to "
        "original_rate / N."
    )
    app.data_fit_avg_rate_label = QLabel("Effective rate: —")
    app.data_fit_avg_rate_label.setStyleSheet("color: gray;")
    app.data_fit_load_metadata_btn = QPushButton("Load metadata from TDMS")
    app.data_fit_load_metadata_btn.setToolTip(
        "Populate scale and offset fields with the Scale_Factor and Offset properties "
        "saved alongside each channel in the TDMS file."
    )
    app.data_fit_add_plot_btn = QPushButton("Add to plot")
    app.data_fit_add_plot_btn.setToolTip(
        "Add current selection to plot when channel/scale/offset/length changed."
    )
    app.data_fit_plot_summary_btn = QPushButton("Plot summary…")
    app.data_fit_plot_summary_btn.setToolTip(
        "Show all plotted curves: color, label, skip points, include-in-fit, remove."
    )
    app.data_fit_graph_btn = QPushButton("Graph settings…")
    app.data_fit_graph_btn.setToolTip(
        "Open graph settings for scale, tick labels, title, grids, line and ticks."
    )
    app.data_fit_zoom_mode_btn = QPushButton("Zoom Mode")
    app.data_fit_zoom_mode_btn.setCheckable(True)
    app.data_fit_zoom_mode_btn.setToolTip(
        "When on: left-drag on the plot draws a rectangle to zoom in.\n"
        "When off: left-drag pans the view (pyqtgraph default)."
    )
    app.data_fit_robust_view_btn = QPushButton("Robust Auto-Range")
    app.data_fit_robust_view_btn.setToolTip(
        "Set the view to the 1st-99th percentile of the data with a 10% margin, "
        "ignoring extreme outliers (common at the edges of V-I curves)."
    )
    app.data_fit_reset_view_btn = QPushButton("Full View")
    app.data_fit_reset_view_btn.setToolTip("Show the complete data range (includes outliers).")
    for compact_btn in (
        app.data_fit_load_metadata_btn,
        app.data_fit_add_plot_btn,
        app.data_fit_plot_summary_btn,
        app.data_fit_graph_btn,
        app.data_fit_robust_view_btn,
        app.data_fit_zoom_mode_btn,
        app.data_fit_reset_view_btn,
    ):
        compact_btn.setMinimumHeight(24)
        compact_btn.setStyleSheet("padding:2px 8px; font-size:11px;")
    ch_grid.addWidget(app.data_fit_load_metadata_btn, 5, 0)
    ch_grid.addWidget(app.data_fit_add_plot_btn, 5, 1)
    ch_grid.addWidget(app.data_fit_plot_summary_btn, 5, 2, 1, 2)
    ch_grid.addWidget(app.data_fit_graph_btn, 6, 0)
    ch_grid.addWidget(app.data_fit_robust_view_btn, 6, 1)
    ch_grid.addWidget(app.data_fit_zoom_mode_btn, 6, 2)
    ch_grid.addWidget(app.data_fit_reset_view_btn, 6, 3)

    # Voltage-tap separation + criterion widgets.
    # When checked, Y is divided by the tap distance so the fit is in V/cm
    # (electric field) and the Ic criterion is an E-field per IEC 61788-3/-21.
    app.data_fit_use_length_cb = QCheckBox("Use voltage-tap separation (Y → E in V/cm)")
    app.data_fit_use_length_cb.setToolTip(
        "Enable IEC 61788 electric-field path: Y is divided by the voltage-tap\n"
        "separation L_v so the Ic criterion Ec has units of V/cm (defaults to\n"
        "1 µV/cm for HTS at 77 K). Auto-enabled when the TDMS metadata\n"
        "provides a per-channel Voltage_Tab_Distance."
    )
    app.data_fit_length_input = QLineEdit("1.0")
    app.data_fit_length_input.setMaximumWidth(80)
    app.data_fit_length_label = QLabel("Tap distance L_v (cm):")
    app.data_fit_length_input.setToolTip(
        "Distance between the two voltage taps, in cm. Written by DAQUniversal\n"
        "to the TDMS file as per-channel property 'Voltage_Tab_Distance'."
    )
    app.data_fit_vc_input = QLineEdit(f"{DEFAULT_EC_V_PER_CM * 1.0e6:.6g}")
    app.data_fit_vc_input.setMaximumWidth(80)
    app.data_fit_vc_label = QLabel("Ec (µV/cm):")
    ch_grid.addWidget(app.data_fit_use_length_cb, 4, 0, 1, 2)
    ch_grid.addWidget(app.data_fit_length_label, 4, 2)
    ch_grid.addWidget(app.data_fit_length_input, 4, 3)
    app.data_fit_avg_rate_label.setVisible(False)

    profile_group = QGroupBox("Active fitting settings")
    profile_layout = QHBoxLayout(profile_group)
    profile_layout.addWidget(QLabel("Curve label:"))
    app.data_fit_curve_profile_cb = QComboBox()
    app.data_fit_curve_profile_cb.setToolTip("Select curve label to edit/load fit-window settings for that curve.")
    profile_layout.addWidget(app.data_fit_curve_profile_cb, stretch=1)
    left.addWidget(profile_group)

    # --- Step 1: Thermal offset (V_ofs) subtraction from the I = 0 segment ---
    offset_group = QGroupBox("Step 1: Subtract thermal offset from I = 0 segment")
    offset_layout = QGridLayout(offset_group)
    app.data_fit_subtract_vofs_cb = QCheckBox(
        "Subtract Vₒƒₛ estimated from |I| ≤ threshold·|I|ₘₐₓ"
    )
    app.data_fit_subtract_vofs_cb.setChecked(True)
    app.data_fit_subtract_vofs_cb.setToolTip(
        "Step 1: median of Y on the quiescent I = 0 segment is used as the "
        "thermal offset V_ofs and subtracted before Step 2 (dI/dt) and Step 3 "
        "(baseline fit V0, R). This separates V_ofs, L·dI/dt and R·I in the "
        "result."
    )
    offset_layout.addWidget(app.data_fit_subtract_vofs_cb, 1, 0, 1, 4)
    offset_layout.addWidget(QLabel("Zero-I threshold (% of Imax):"), 2, 0)
    app.data_fit_zero_i_frac = _percent_edit(DEFAULT_ZERO_I_FRAC)
    app.data_fit_zero_i_frac.setToolTip(
        "Samples with |I| below this fraction of max|I| are treated as the "
        "I = 0 segment. Typical value: 2% (0.02)."
    )
    offset_layout.addWidget(app.data_fit_zero_i_frac, 2, 1)
    offset_layout.addWidget(QLabel("Trim curve start (A):"), 3, 0)
    app.data_fit_trim_start_abs_a = QLineEdit(f"{DEFAULT_TRIM_START_A:.6g}")
    app.data_fit_trim_start_abs_a.setMaximumWidth(90)
    offset_layout.addWidget(app.data_fit_trim_start_abs_a, 3, 1)
    offset_layout.addWidget(QLabel("or start trim (% of Imax):"), 3, 2)
    app.data_fit_trim_start_frac = _percent_edit(DEFAULT_TRIM_START_FRAC)
    offset_layout.addWidget(app.data_fit_trim_start_frac, 3, 3)
    app.data_fit_trim_use_percent_cb = QCheckBox("Use % instead of A")
    app.data_fit_trim_use_percent_cb.setChecked(False)
    offset_layout.addWidget(app.data_fit_trim_use_percent_cb, 4, 0, 1, 2)
    app.data_fit_trim_rampdown_cb = QCheckBox("Trim ramp-down tail automatically")
    app.data_fit_trim_rampdown_cb.setChecked(True)
    app.data_fit_trim_rampdown_cb.setToolTip(
        "Automatically stop before the first strong current drop and keep a 1% guard."
    )
    offset_layout.addWidget(app.data_fit_trim_rampdown_cb, 4, 2, 1, 2)
    left.addWidget(offset_group)

    didt_group = QGroupBox("Step 2: di/dt window (fraction of Imax)")
    didt_layout = QGridLayout(didt_group)
    app.data_fit_didt_low = _percent_edit(DEFAULT_DIDT_LOW_FRAC)
    app.data_fit_didt_high = _percent_edit(DEFAULT_DIDT_HIGH_FRAC)
    app.data_fit_didt_low_x = _xvalue_edit()
    app.data_fit_didt_high_x = _xvalue_edit()
    app.data_fit_show_didt = QCheckBox("Show / edit")
    app.data_fit_show_didt.setChecked(False)
    app.data_fit_show_didt.setToolTip("Show/hide the blue band for this window on the plot.")
    _fill_window_grid(
        didt_layout, app.data_fit_show_didt,
        low_label="Low (%)", low_pct=app.data_fit_didt_low, low_x=app.data_fit_didt_low_x,
        high_label="High (%)", high_pct=app.data_fit_didt_high, high_x=app.data_fit_didt_high_x,
    )
    left.addWidget(didt_group)

    linear_group = QGroupBox("Step 3: Linear baseline window (fraction of Imax)")
    linear_layout = QGridLayout(linear_group)
    app.data_fit_linear_low = _percent_edit(DEFAULT_LINEAR_LOW_FRAC)
    app.data_fit_linear_high = _percent_edit(DEFAULT_LINEAR_HIGH_FRAC)
    app.data_fit_linear_low_x = _xvalue_edit()
    app.data_fit_linear_high_x = _xvalue_edit()
    app.data_fit_show_linear = QCheckBox("Show / edit")
    app.data_fit_show_linear.setChecked(False)
    app.data_fit_show_linear.setToolTip("Show/hide the green band for this window on the plot.")
    app.data_fit_baseline_mode_cb = QComboBox()
    app.data_fit_baseline_mode_cb.addItem("OLS (legacy)", BASELINE_MODE_OLS)
    app.data_fit_baseline_mode_cb.addItem("Huber robust", BASELINE_MODE_HUBER)
    app.data_fit_baseline_mode_cb.addItem("Theil-Sen robust", BASELINE_MODE_THEIL_SEN)
    idx_default_baseline = app.data_fit_baseline_mode_cb.findData(DEFAULT_BASELINE_MODE)
    app.data_fit_baseline_mode_cb.setCurrentIndex(max(0, idx_default_baseline))
    # Keep compact so it does not collide with Low(X)/High(X) widgets.
    app.data_fit_baseline_mode_cb.setMaximumWidth(145)
    app.data_fit_baseline_mode_cb.setToolTip(
        "Step-3 baseline estimator.\n"
        "OLS: standard least-squares (legacy behavior).\n"
        "Huber robust: reduces outlier influence.\n"
        "Theil-Sen robust: median-slope fit, very stable on noisy ramps."
    )
    baseline_mode_wrap = QWidget()
    baseline_mode_row = QHBoxLayout(baseline_mode_wrap)
    baseline_mode_row.setContentsMargins(0, 0, 0, 0)
    baseline_mode_row.setSpacing(6)
    baseline_mode_row.addWidget(QLabel("Baseline mode:"))
    baseline_mode_row.addWidget(app.data_fit_baseline_mode_cb)
    baseline_mode_row.addStretch(1)
    _fill_window_grid(
        linear_layout, app.data_fit_show_linear,
        low_label="Low (%)", low_pct=app.data_fit_linear_low, low_x=app.data_fit_linear_low_x,
        high_label="High (%)", high_pct=app.data_fit_linear_high, high_x=app.data_fit_linear_high_x,
        extra_widget=baseline_mode_wrap,
    )
    left.addWidget(linear_group)

    power_group = QGroupBox("Step 4: Ic and n-value")
    power_layout = QGridLayout(power_group)

    # --- method selector (row 0) ---
    app.data_fit_method_group = QButtonGroup(power_group)
    app.data_fit_method_loglog_rb = QRadioButton(
        "Log E vs log I linear fit (IEC 61788)"
    )
    app.data_fit_method_loglog_rb.setToolTip(
        "IEC 61788-3 decade method: baseline-subtract (E_sc = E − V0 − R·I),\n"
        "select points with E_sc ∈ [Ec1, Ec2], then fit\n"
        "log10(E_sc) = log10(Ec2) + n · log10(I / Ic).\n"
        "Ic is reported at E = Ec2 (typically 1 µV/cm for HTS at 77 K)."
    )
    app.data_fit_method_nonlinear_rb = QRadioButton(
        "Non-linear V-I fit (V = V0 + R·I + Vc·(I/Ic)^n)"
    )
    app.data_fit_method_nonlinear_rb.setToolTip(
        "Legacy coupled non-linear fit. Uses the full voltage model with\n"
        "V0, R frozen from Step 3 and Ic, n as free parameters. Not IEC\n"
        "standardised — kept for backwards compatibility and cross-checks."
    )
    app.data_fit_method_group.addButton(app.data_fit_method_loglog_rb, 0)
    app.data_fit_method_group.addButton(app.data_fit_method_nonlinear_rb, 1)
    if DEFAULT_FIT_METHOD == FIT_METHOD_LOG_LOG:
        app.data_fit_method_loglog_rb.setChecked(True)
    else:
        app.data_fit_method_nonlinear_rb.setChecked(True)
    power_layout.addWidget(app.data_fit_method_loglog_rb, 0, 0, 1, 3)
    power_layout.addWidget(app.data_fit_method_nonlinear_rb, 1, 0, 1, 3)
    app.data_fit_weight_mode_cb = QComboBox()
    app.data_fit_weight_mode_cb.addItem("Equal", WEIGHT_MODE_EQUAL)
    app.data_fit_weight_mode_cb.addItem("Weighted", WEIGHT_MODE_WEIGHTED)
    app.data_fit_weight_mode_cb.addItem("Robust", WEIGHT_MODE_ROBUST)
    app.data_fit_weight_mode_cb.setCurrentIndex(0)
    app.data_fit_weight_mode_cb.setMaximumWidth(145)
    app.data_fit_weight_mode_cb.setToolTip(
        "Step-4 point weighting mode.\n"
        "Equal: all points same weight (legacy behavior).\n"
        "Weighted: auto-estimate per-point noise and weight by 1/sigma².\n"
        "Robust: weighted + Huber reweighting to suppress outliers."
    )
    power_layout.addWidget(QLabel("Point weighting:"), 2, 0, 1, 1)
    power_layout.addWidget(app.data_fit_weight_mode_cb, 2, 1, 1, 2)

    # Toggle between linear/linear and log/log axes. Text tracks current mode.
    app.data_fit_plot_scale_btn = QPushButton("Switch to log-log plot")
    app.data_fit_plot_scale_btn.setToolTip(
        "Toggle the V-I plot between linear axes and log-log axes.\n"
        "Log-log also updates the graph-settings dialog (Scale → Log10\n"
        "on both axes) so the setting persists when you re-open it."
    )
    power_layout.addWidget(app.data_fit_plot_scale_btn, 0, 3, 2, 1)
    app.data_fit_add_corrected_btn = QPushButton("Add corrected curve")
    app.data_fit_add_corrected_btn.setToolTip(
        "Add the baseline-corrected curve from the last successful fit:\n"
        "Y_corrected = Y - (V0 + R·I). Useful for checking the log-log Step 4 window."
    )
    app.data_fit_add_smoothed_btn = QPushButton("Add smoothed and corrected curve")
    app.data_fit_add_smoothed_btn.setToolTip(
        "Build a Step-4 guide curve from the last successful fit:\n"
        "first baseline-corrected (Y - (V0 + R·I)), then Ec-aware smoothed.\n"
        "The Show/edit Step-4 window uses this corrected+smoothed curve,\n"
        "even if it is hidden from the graph."
    )
    power_layout.addWidget(app.data_fit_add_smoothed_btn, 6, 0, 1, 2)
    power_layout.addWidget(app.data_fit_add_corrected_btn, 6, 2, 1, 2)

    # --- window editors (rows 2-4) ---
    app.data_fit_power_low = _percent_edit(DEFAULT_POWER_LOW_FRAC)
    app.data_fit_power_vfrac = _percent_edit(DEFAULT_POWER_V_FRAC)
    app.data_fit_power_low_x = _xvalue_edit()
    app.data_fit_power_high_x = _xvalue_edit()
    app.data_fit_show_power = QCheckBox("Show / edit")
    app.data_fit_show_power.setChecked(False)
    app.data_fit_show_power.setToolTip("Show/hide the orange band for this window on the plot.")
    # Pre-fill the low/high editors with the IEC Ec1/Ec2 defaults so they
    # are correct the first time the user sees them with the log-log
    # method (the default). The labels switch via _update_method_mode_ui.
    app.data_fit_power_low.setText(f"{DEFAULT_EC1_V_PER_CM * 1.0e6:g}")
    app.data_fit_power_vfrac.setText(f"{DEFAULT_EC2_V_PER_CM * 1.0e6:g}")
    # Offset the grid so the method radios/button stay visible above.
    _fill_window_grid(
        power_layout, app.data_fit_show_power,
        low_label="Ec1 (µV/cm)", low_pct=app.data_fit_power_low, low_x=app.data_fit_power_low_x,
        high_label="Ec2 (µV/cm)", high_pct=app.data_fit_power_vfrac, high_x=app.data_fit_power_high_x,
        base_row=3,
    )
    # Keep references to the text labels so the method-mode handler can
    # swap them when the user switches between IEC and non-linear modes.
    # (row 3 holds the Low/High value labels; row 4 holds the X-value row.)
    app.data_fit_power_low_label = power_layout.itemAtPosition(4, 0).widget()
    app.data_fit_power_high_label = power_layout.itemAtPosition(4, 2).widget()
    app.data_fit_power_low_x_label = power_layout.itemAtPosition(5, 0).widget()
    app.data_fit_power_high_x_label = power_layout.itemAtPosition(5, 2).widget()
    left.addWidget(power_group)

    app.data_fit_window_inputs = {
        ("didt", "low"): (app.data_fit_didt_low, app.data_fit_didt_low_x, "Imax"),
        ("didt", "high"): (app.data_fit_didt_high, app.data_fit_didt_high_x, "Imax"),
        ("linear", "low"): (app.data_fit_linear_low, app.data_fit_linear_low_x, "Imax"),
        ("linear", "high"): (app.data_fit_linear_high, app.data_fit_linear_high_x, "Imax"),
        ("power", "low"): (app.data_fit_power_low, app.data_fit_power_low_x, "Imax"),
        ("power", "high"): (app.data_fit_power_vfrac, app.data_fit_power_high_x, "Vmax"),
    }

    iter_group = QGroupBox("Ic iteration && criterion")
    iter_layout = QGridLayout(iter_group)
    app.data_fit_max_iter = QLineEdit(str(DEFAULT_MAX_ITERATIONS))
    app.data_fit_max_iter.setMaximumWidth(80)
    app.data_fit_ic_tol = QLineEdit(f"{DEFAULT_IC_TOLERANCE * 100:g}")
    app.data_fit_ic_tol.setMaximumWidth(80)
    app.data_fit_chi_tol = QLineEdit(f"{DEFAULT_CHI_SQR_TOL:g}")
    app.data_fit_chi_tol.setMaximumWidth(120)
    app.data_fit_chi_tol.setToolTip(
        "Stopping tolerance passed to the power-law fitter (scipy curve_fit ftol/xtol/gtol).\n"
        "It is a RELATIVE threshold on the change of the cost function between fitter\n"
        "iterations, not an absolute chi-squared target. With real data the final\n"
        "chi-squared is bounded below by the noise, so a fit can be fully converged\n"
        "while chi-squared remains much larger than this tolerance.\n"
        "OriginLab's default is 1e-9."
    )
    # Two-column layout: iteration knobs on the left, criterion on the right.
    iter_layout.addWidget(QLabel("Max iterations"), 0, 0)
    iter_layout.addWidget(app.data_fit_max_iter, 0, 1)
    iter_layout.addWidget(QLabel("Ic stop tol (%)"), 1, 0)
    iter_layout.addWidget(app.data_fit_ic_tol, 1, 1)
    iter_layout.addWidget(QLabel("Chi-sqr tol"), 0, 2)
    iter_layout.addWidget(app.data_fit_chi_tol, 0, 3)
    iter_layout.addWidget(app.data_fit_vc_label, 1, 2)
    iter_layout.addWidget(app.data_fit_vc_input, 1, 3)
    left.addWidget(iter_group)

    run_row = QHBoxLayout()
    app.data_fit_run_btn = QPushButton("Run Fit")
    app.data_fit_run_btn.setStyleSheet("font-weight: bold; background-color: #0078D7; color: white; padding: 8px;")
    app.data_fit_save_metadata_btn = QPushButton("Save metadata")
    app.data_fit_save_metadata_btn.setToolTip(
        "Manually save the most recent fit parameters into the loaded TDMS file.\n"
        "Useful when 'Automatically save fitting parameters as metadata' is\n"
        "unchecked in Settings, or to re-save after toggling the layout\n"
        "(separate file vs. same group/channel).\n"
        "Disabled until at least one fit has run."
    )
    app.data_fit_save_metadata_btn.setStyleSheet(
        "font-weight: bold; background-color: #f1f4f8; color: #003a75; padding: 8px;"
    )
    app.data_fit_save_metadata_btn.setEnabled(False)
    # Stretch ratios make Run Fit three times as wide as Save metadata.
    run_row.addWidget(app.data_fit_run_btn, stretch=3)
    run_row.addWidget(app.data_fit_save_metadata_btn, stretch=1)
    left.addLayout(run_row)

    preset_row = QHBoxLayout()
    app.data_fit_save_preset_btn = QPushButton("Save preset…")
    app.data_fit_save_preset_btn.setToolTip("Save the current fit-window preset to a JSON file.")
    app.data_fit_load_preset_btn = QPushButton("Load preset…")
    app.data_fit_load_preset_btn.setToolTip("Load a fit-window preset from a JSON file.")
    app.data_fit_help_btn = QPushButton("?  Help")
    app.data_fit_help_btn.setToolTip("Open the help window with a full overview of the fitting workflow.")
    app.data_fit_help_btn.setStyleSheet("font-weight: bold; background-color: #e6f2ff; color: #003a75; padding: 6px;")
    preset_row.addWidget(app.data_fit_save_preset_btn)
    preset_row.addWidget(app.data_fit_load_preset_btn)
    preset_row.addWidget(app.data_fit_help_btn)
    left.addLayout(preset_row)

    # Save-result checkboxes were moved to the Settings dialog; the four
    # QCheckBox instances are constructed earlier in this layout so other
    # parts of the code can still read their state directly.

    app.data_fit_warning_label = QLabel("")
    app.data_fit_warning_label.setWordWrap(True)
    app.data_fit_warning_label.setStyleSheet(
        "background-color: #fff7c2; color: #665200; border: 1px solid #e6cc00; padding: 6px;"
    )
    app.data_fit_warning_label.setVisible(False)
    left.addWidget(app.data_fit_warning_label)

    left.addStretch()
    root.addWidget(left_widget)

    right_widget = QWidget()
    right = QVBoxLayout(right_widget)

    # ---- Header row: equation + channels ----
    header = QHBoxLayout()
    left_header = QVBoxLayout()
    app.data_fit_equation_label = QLabel()
    app.data_fit_equation_label.setTextFormat(Qt.RichText)
    app.data_fit_equation_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    _update_equation_label(app)
    left_header.addWidget(app.data_fit_equation_label)

    app.data_fit_graph_settings = GraphSettings()

    app.data_fit_result_text = QTextEdit()
    app.data_fit_result_text.setReadOnly(True)
    app.data_fit_result_text.setPlaceholderText("Fit results will appear here.")
    app.data_fit_result_text.setMinimumHeight(110)
    app.data_fit_result_text.setMaximumHeight(140)
    left_header.addWidget(app.data_fit_result_text)
    header.addLayout(left_header, stretch=1)

    # Channels (displayed = raw * scale - offset) — moved from left panel.
    header.addWidget(app.data_fit_channels_group, stretch=2)

    right.addLayout(header, stretch=0)

    app.data_fit_plot = pg.PlotWidget(
        title="V-I preview",
        axisItems={
            "bottom": EngineeringAxisItem(orientation="bottom"),
            "left": EngineeringAxisItem(orientation="left"),
        },
    )
    app.data_fit_plot.setLabel("bottom", "Current (A)")
    app.data_fit_plot.setLabel("left", "Voltage (V)")
    app.data_fit_plot.showGrid(x=True, y=True)
    app.data_fit_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.PanMode)
    app.data_fit_raw_curve = app.data_fit_plot.plot(pen=pg.mkPen("b", width=1.5), name="Raw")
    app.data_fit_model_curve = app.data_fit_plot.plot(pen=pg.mkPen("r", width=2), name="Fit")

    # One band per step. It is BOTH the colored tint and the draggable editor
    # for that step (when "Edit visually" is selected). No separate gray region.
    app.data_fit_band_didt = pg.LinearRegionItem(
        brush=pg.mkBrush(0, 150, 255, 45), pen=pg.mkPen(0, 150, 255, 180), movable=False,
    )
    app.data_fit_band_linear = pg.LinearRegionItem(
        brush=pg.mkBrush(0, 200, 120, 45), pen=pg.mkPen(0, 200, 120, 180), movable=False,
    )
    app.data_fit_band_power = pg.LinearRegionItem(
        brush=pg.mkBrush(230, 120, 0, 45), pen=pg.mkPen(230, 120, 0, 180), movable=False,
    )
    for band in (app.data_fit_band_didt, app.data_fit_band_linear, app.data_fit_band_power):
        band.setZValue(5)
        app.data_fit_plot.addItem(band, ignoreBounds=True)
    app.data_fit_band_didt.sigRegionChanged.connect(
        lambda *_: _on_band_dragged(app, "didt")
    )
    app.data_fit_band_linear.sigRegionChanged.connect(
        lambda *_: _on_band_dragged(app, "linear")
    )
    app.data_fit_band_power.sigRegionChanged.connect(
        lambda *_: _on_band_dragged(app, "power")
    )

    # Ic marker + criterion line shown after a successful fit.
    app.data_fit_ic_line = pg.InfiniteLine(
        angle=90, pen=pg.mkPen("m", style=Qt.DashLine, width=2),
        label="Ic", labelOpts={"position": 0.9, "color": "m", "movable": True},
    )
    app.data_fit_ic_line.setVisible(False)
    app.data_fit_plot.addItem(app.data_fit_ic_line, ignoreBounds=True)
    app.data_fit_criterion_line = pg.InfiniteLine(
        angle=0, pen=pg.mkPen("m", style=Qt.DashDotLine, width=1.5),
        label="Vc", labelOpts={"position": 0.05, "color": "m"},
    )
    app.data_fit_criterion_line.setVisible(False)
    app.data_fit_plot.addItem(app.data_fit_criterion_line, ignoreBounds=True)

    # IEC 61788 decade lines (Ec1, Ec2) — visible only in the log-log method.
    app.data_fit_ec1_line = pg.InfiniteLine(
        angle=0, pen=pg.mkPen(230, 120, 0, 200, style=Qt.DashLine, width=1.2),
        label="Ec1", labelOpts={"position": 0.02, "color": (230, 120, 0)},
    )
    app.data_fit_ec1_line.setVisible(False)
    app.data_fit_plot.addItem(app.data_fit_ec1_line, ignoreBounds=True)
    app.data_fit_ec2_line = pg.InfiniteLine(
        angle=0, pen=pg.mkPen(230, 120, 0, 200, style=Qt.DashLine, width=1.2),
        label="Ec2", labelOpts={"position": 0.02, "color": (230, 120, 0)},
    )
    app.data_fit_ec2_line.setVisible(False)
    app.data_fit_plot.addItem(app.data_fit_ec2_line, ignoreBounds=True)

    # Draggable parameter-table overlay (visible after a successful fit).
    app.data_fit_param_table = _FitParamTable()
    app.data_fit_param_table.setVisible(False)
    app.data_fit_plot.addItem(app.data_fit_param_table, ignoreBounds=True)

    right.addWidget(app.data_fit_plot, stretch=3)

    # Residuals sub-plot, shown after a fit.
    app.data_fit_resid_plot = pg.PlotWidget(
        title="Residuals (data − model)",
        axisItems={
            "bottom": EngineeringAxisItem(orientation="bottom"),
            "left": EngineeringAxisItem(orientation="left"),
        },
    )
    app.data_fit_resid_plot.setLabel("bottom", "Current (A)")
    app.data_fit_resid_plot.setLabel("left", "Residual")
    app.data_fit_resid_plot.showGrid(x=True, y=True)
    app.data_fit_resid_plot.setXLink(app.data_fit_plot)
    app.data_fit_resid_curve = app.data_fit_resid_plot.plot(
        pen=None, symbol="o", symbolSize=3,
        symbolBrush=pg.mkBrush(120, 50, 180, 200), symbolPen=None,
    )
    zero_line = pg.InfiniteLine(angle=0, pos=0, pen=pg.mkPen("k", style=Qt.DashLine, width=1))
    app.data_fit_resid_plot.addItem(zero_line)
    app.data_fit_resid_plot.setVisible(False)
    right.addWidget(app.data_fit_resid_plot, stretch=1)

    app.data_fit_xrange_label = QLabel("X window: full")
    app.data_fit_xrange_label.setStyleSheet("color: gray;")
    right.addWidget(app.data_fit_xrange_label)

    app.data_fit_version_label = QLabel(get_app_version_label())
    app.data_fit_version_label.setStyleSheet("color: #8a8a8a; font-size: 11px;")
    app.data_fit_version_label.setAlignment(Qt.AlignRight)
    right.addWidget(app.data_fit_version_label)

    root.addWidget(right_widget, stretch=1)

    app.data_fit_controller = DataFittingController(app)
    app.data_fit_preview_visible = True
    app.data_fit_preview_include_in_fit = True
    app.data_fit_preview_color = "#1f77b4"
    app.data_fit_preview_alpha_pct = 100
    app.data_fit_preview_style = {"draw_mode": "Auto", "line_width": 1.5, "point_size": 4}
    app.data_fit_curve_profiles = {"__preview__": _capture_fit_window_profile(app)}
    _on_use_length_changed(app)
    _update_method_mode_ui(app)
    _update_plot_scale_button_text(app)
    _refresh_curve_profile_selector(app)
    _connect_data_fitting_actions(app)


# ---------------------------------------------------------------------------
# Actions (implemented as free functions and attached to app through the facade)
# ---------------------------------------------------------------------------

def _scale_offset_from_inputs(app, axis: str) -> tuple[float, float]:
    scale_widget = getattr(app, f"data_fit_{axis}_scale")
    offset_widget = getattr(app, f"data_fit_{axis}_offset")
    return (
        _float_from(scale_widget, 1.0),
        _float_from(offset_widget, 0.0),
    )


def _active_sample_length(app) -> Optional[float]:
    if not app.data_fit_use_length_cb.isChecked():
        return None
    try:
        value = float(app.data_fit_length_input.text())
    except ValueError:
        return None
    return value if value > 0 else None


def _active_avg_window(app) -> int:
    try:
        value = int(float(app.data_fit_avg_input.text()))
    except ValueError:
        return 1
    return max(1, value)


def _block_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr is None or arr.size == 0:
        return arr
    n_bins = arr.size // window
    if n_bins == 0:
        return arr
    return arr[: n_bins * window].reshape(n_bins, window).mean(axis=1)


def _apply_transforms(app):
    controller = app.data_fit_controller
    t_scale, t_offset = _scale_offset_from_inputs(app, "time")
    x_scale, x_offset = _scale_offset_from_inputs(app, "x")
    y_scale, y_offset = _scale_offset_from_inputs(app, "y")
    time_name = app.data_fit_time_cb.currentText()
    x_name = app.data_fit_x_cb.currentText()
    y_name = app.data_fit_y_cb.currentText()
    t_raw = controller.get_channel(time_name) if time_name and time_name != "Time" else controller.time_array
    x_raw = controller.get_channel(x_name)
    y_raw = controller.get_channel(y_name)
    t = controller.apply_transform(t_raw, t_scale, t_offset)
    x = controller.apply_transform(x_raw, x_scale, x_offset)
    y = controller.apply_transform(y_raw, y_scale, y_offset)
    avg_window = _active_avg_window(app)
    if avg_window > 1:
        t = _block_average(t, avg_window)
        x = _block_average(x, avg_window)
        y = _block_average(y, avg_window)
    sample_length = _active_sample_length(app)
    if sample_length is not None and y is not None:
        y = y / sample_length
    return {
        "time": t,
        "x": x,
        "y": y,
        "scales": {
            "time": (t_scale, t_offset),
            "x": (x_scale, x_offset),
            "y": (y_scale, y_offset),
        },
    }


def _settings_from_inputs(app) -> FitSettings:
    sample_length = None
    if app.data_fit_use_length_cb.isChecked():
        try:
            sample_length = float(app.data_fit_length_input.text())
        except ValueError:
            sample_length = None
    if sample_length is not None:
        # Input is Ec in µV/cm → convert to V/cm for the service.
        ec_uv_per_cm = _float_from(app.data_fit_vc_input, 1.0)
        criterion_value = ec_uv_per_cm * 1.0e-6
    else:
        # Input is Vc in mV → convert to V for the service.
        vc_mv = _float_from(app.data_fit_vc_input, DEFAULT_VC_VOLTS * 1000.0)
        criterion_value = vc_mv * 1.0e-3

    method = _active_fit_method(app)
    # Ec1/Ec2 share the Step 4 Low/High editors when log-log is active.
    # Absolute units: V/cm when Y has been divided by L_v, V otherwise.
    to_si = 1.0e-6 if sample_length is not None else 1.0e-3
    ec1 = _float_from(app.data_fit_power_low, DEFAULT_EC1_V_PER_CM * 1.0e6) * to_si
    ec2 = _float_from(app.data_fit_power_vfrac, DEFAULT_EC2_V_PER_CM * 1.0e6) * to_si
    # In log-log mode the user-entered Ec/Vc is the criterion at which Ic is
    # reported. Ec1/Ec2 only define the fit window for the n-value slope.

    settings = FitSettings(
        didt_low_frac=_float_from(app.data_fit_didt_low, DEFAULT_DIDT_LOW_FRAC * 100, as_fraction=True),
        didt_high_frac=_float_from(app.data_fit_didt_high, DEFAULT_DIDT_HIGH_FRAC * 100, as_fraction=True),
        linear_low_frac=_float_from(app.data_fit_linear_low, DEFAULT_LINEAR_LOW_FRAC * 100, as_fraction=True),
        linear_high_frac=_float_from(app.data_fit_linear_high, DEFAULT_LINEAR_HIGH_FRAC * 100, as_fraction=True),
        power_low_frac=_float_from(app.data_fit_power_low, DEFAULT_POWER_LOW_FRAC * 100, as_fraction=True),
        power_v_frac=_float_from(app.data_fit_power_vfrac, DEFAULT_POWER_V_FRAC * 100, as_fraction=True),
        max_iterations=int(_float_from(app.data_fit_max_iter, DEFAULT_MAX_ITERATIONS)),
        ic_tolerance=_float_from(app.data_fit_ic_tol, DEFAULT_IC_TOLERANCE * 100, as_fraction=True),
        chi_sqr_tolerance=_float_from(app.data_fit_chi_tol, DEFAULT_CHI_SQR_TOL),
        criterion_voltage=criterion_value,
        sample_length_cm=sample_length,
        fit_method=method,
        ec1=ec1,
        ec2=ec2,
        subtract_thermal_offset=bool(
            getattr(app, "data_fit_subtract_vofs_cb", None) is None
            or app.data_fit_subtract_vofs_cb.isChecked()
        ),
        zero_i_frac=_float_from(
            getattr(app, "data_fit_zero_i_frac", None),
            DEFAULT_ZERO_I_FRAC * 100,
            as_fraction=True,
        ),
        weight_mode=(
            app.data_fit_weight_mode_cb.currentData()
            if getattr(app, "data_fit_weight_mode_cb", None) is not None
            else WEIGHT_MODE_EQUAL
        ),
        baseline_mode=(
            app.data_fit_baseline_mode_cb.currentData()
            if getattr(app, "data_fit_baseline_mode_cb", None) is not None
            else DEFAULT_BASELINE_MODE
        ),
        trim_start_abs_a=_float_from(getattr(app, "data_fit_trim_start_abs_a", None), DEFAULT_TRIM_START_A),
        trim_start_frac=_float_from(getattr(app, "data_fit_trim_start_frac", None), DEFAULT_TRIM_START_FRAC * 100, as_fraction=True),
        trim_use_percent=bool(getattr(app, "data_fit_trim_use_percent_cb", None) is not None and app.data_fit_trim_use_percent_cb.isChecked()),
        trim_rampdown_tail=bool(getattr(app, "data_fit_trim_rampdown_cb", None) is None or app.data_fit_trim_rampdown_cb.isChecked()),
    )
    return settings


def _profile_text_float(profile: dict, key: str, fallback: float, as_fraction: bool = False) -> float:
    raw = profile.get(key)
    if raw is None or raw == "":
        return fallback
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    return v / 100.0 if as_fraction else v


def _entry_length_settings(app, entry: dict) -> tuple[bool, float]:
    """Return (use_length, length_cm) for a fit entry.

    Stored curves carry their own length context in ``entry["source"]``;
    fall back to the live UI for the preview / single-curve case.
    """
    src = entry.get("source") or {}
    if "use_length" in src:
        try:
            length_cm = float(src.get("length_cm", 1.0) or 1.0)
        except (TypeError, ValueError):
            length_cm = 1.0
        return bool(src.get("use_length", False)), length_cm
    use_length = bool(app.data_fit_use_length_cb.isChecked())
    try:
        length_cm = float(app.data_fit_length_input.text())
    except (TypeError, ValueError):
        length_cm = 1.0
    return use_length, (length_cm if length_cm > 0 else 1.0)


def _profile_key_for_entry(entry: dict) -> str:
    sig = entry.get("signature")
    if sig is None or sig == "__preview__":
        return "__preview__"
    return str(sig)


def _settings_from_profile(profile: dict, *, use_length: bool, length_cm: float) -> FitSettings:
    """Build a FitSettings from a per-curve profile dict.

    The profile stores raw QLineEdit text plus the curve's chosen fit method,
    so the same dict is sufficient to reproduce the fit on its own — no live
    UI access needed. ``use_length`` / ``length_cm`` come from the entry's
    captured source context (or the current UI for the preview).
    """
    sample_length = float(length_cm) if use_length and length_cm and length_cm > 0 else None
    method = profile.get("fit_method", DEFAULT_FIT_METHOD)
    weight_mode = profile.get("weight_mode", WEIGHT_MODE_EQUAL)
    baseline_mode = profile.get("baseline_mode", DEFAULT_BASELINE_MODE)

    if sample_length is not None:
        # vc input is Ec in µV/cm → convert to V/cm.
        ec_uv_per_cm = _profile_text_float(profile, "vc", DEFAULT_EC_V_PER_CM * 1.0e6)
        criterion_value = ec_uv_per_cm * 1.0e-6
    else:
        # vc input is Vc in mV → convert to V.
        vc_mv = _profile_text_float(profile, "vc", DEFAULT_VC_VOLTS * 1000.0)
        criterion_value = vc_mv * 1.0e-3

    if method == FIT_METHOD_LOG_LOG:
        to_si = 1.0e-6 if sample_length is not None else 1.0e-3
        ec1_uv = _profile_text_float(
            profile, "loglog_low",
            _profile_text_float(profile, "power_low", DEFAULT_EC1_V_PER_CM * 1.0e6),
        )
        ec2_uv = _profile_text_float(
            profile, "loglog_high",
            _profile_text_float(profile, "power_vfrac", DEFAULT_EC2_V_PER_CM * 1.0e6),
        )
        ec1 = ec1_uv * to_si
        ec2 = ec2_uv * to_si
        # In log-log mode the non-linear power_low_frac/power_v_frac are unused;
        # leave them at defaults so the dataclass stays well-formed.
        power_low_frac = DEFAULT_POWER_LOW_FRAC
        power_v_frac = DEFAULT_POWER_V_FRAC
    else:
        ec1 = DEFAULT_EC1_V_PER_CM
        ec2 = DEFAULT_EC2_V_PER_CM
        power_low_frac = _profile_text_float(
            profile, "nonlinear_low",
            _profile_text_float(profile, "power_low", DEFAULT_POWER_LOW_FRAC * 100),
            as_fraction=True,
        )
        power_v_frac = _profile_text_float(
            profile, "nonlinear_high",
            _profile_text_float(profile, "power_vfrac", DEFAULT_POWER_V_FRAC * 100),
            as_fraction=True,
        )

    return FitSettings(
        didt_low_frac=_profile_text_float(profile, "didt_low", DEFAULT_DIDT_LOW_FRAC * 100, as_fraction=True),
        didt_high_frac=_profile_text_float(profile, "didt_high", DEFAULT_DIDT_HIGH_FRAC * 100, as_fraction=True),
        linear_low_frac=_profile_text_float(profile, "linear_low", DEFAULT_LINEAR_LOW_FRAC * 100, as_fraction=True),
        linear_high_frac=_profile_text_float(profile, "linear_high", DEFAULT_LINEAR_HIGH_FRAC * 100, as_fraction=True),
        power_low_frac=power_low_frac,
        power_v_frac=power_v_frac,
        max_iterations=int(_profile_text_float(profile, "max_iter", DEFAULT_MAX_ITERATIONS)),
        ic_tolerance=_profile_text_float(profile, "ic_tol", DEFAULT_IC_TOLERANCE * 100, as_fraction=True),
        chi_sqr_tolerance=_profile_text_float(profile, "chi_tol", DEFAULT_CHI_SQR_TOL),
        criterion_voltage=criterion_value,
        sample_length_cm=sample_length,
        fit_method=method,
        ec1=ec1,
        ec2=ec2,
        subtract_thermal_offset=bool(profile.get("subtract_vofs", True)),
        zero_i_frac=_profile_text_float(profile, "zero_i_frac", DEFAULT_ZERO_I_FRAC * 100, as_fraction=True),
        weight_mode=weight_mode,
        baseline_mode=baseline_mode,
        trim_start_abs_a=_profile_text_float(profile, "trim_start_abs_a", DEFAULT_TRIM_START_A),
        trim_start_frac=_profile_text_float(profile, "trim_start_frac", DEFAULT_TRIM_START_FRAC * 100, as_fraction=True),
        trim_use_percent=bool(profile.get("trim_use_percent", False)),
        trim_rampdown_tail=bool(profile.get("trim_rampdown_tail", True)),
    )


def _settings_for_entry(app, entry: dict) -> FitSettings:
    """Pick the right FitSettings for a single fit entry.

    Looks up the curve's stored profile (or the preview profile, or finally
    the live UI) so each curve can carry its own fit method and windows.
    """
    profiles = getattr(app, "data_fit_curve_profiles", {}) or {}
    key = _profile_key_for_entry(entry)
    profile = profiles.get(key) if isinstance(profiles, dict) else None
    if not profile:
        return _settings_from_inputs(app)
    use_length, length_cm = _entry_length_settings(app, entry)
    return _settings_from_profile(profile, use_length=use_length, length_cm=length_cm)


def _populate_channel_combos(app):
    names = list(app.data_fit_controller.channel_names)
    time_options = ["Time"] + names
    _refill_combo(app.data_fit_time_cb, time_options)
    _refill_combo(app.data_fit_x_cb, names)
    _refill_combo(app.data_fit_y_cb, names)
    _try_select(app.data_fit_x_cb, ("AI0", "Current", "I", "current"))
    _try_select(app.data_fit_y_cb, ("AI1", "Voltage", "V", "voltage"))


def _refill_combo(combo: QComboBox, items):
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    for item in items:
        combo.addItem(item)
    restored = combo.findText(current) if current else -1
    if restored >= 0:
        combo.setCurrentIndex(restored)
    combo.blockSignals(False)


def _try_select(combo: QComboBox, preferred_substrings):
    if combo.count() == 0:
        return
    for i in range(combo.count()):
        text = combo.itemText(i)
        for needle in preferred_substrings:
            if needle.lower() in text.lower():
                combo.setCurrentIndex(i)
                return


def _clear_plot_state_for_new_recording(app) -> None:
    """Wipe plotted curves, fit overlays, and result text so the next load
    starts from a clean slate.

    Called before every new TDMS load (manual or auto). Without this, every
    Stop Read of a new acquisition piles new replayed source curves and fit
    overlays on top of the previous run's plot, and the result panel
    accumulates stale summaries.

    Preserves user-configurable settings (Settings checkboxes, fit-window
    inputs, scale/offset inputs) — only the per-recording plot state is
    reset. Use ``_reset_data_fitting_defaults`` (the Clear button) when the
    user wants a full reset.
    """
    # Remove every curve item that was added to the plot.
    for entry in list(getattr(app, "data_fit_curves", []) or []):
        item = entry.get("plot_item")
        if item is not None:
            try:
                app.data_fit_plot.removeItem(item)
            except Exception:
                pass
        fit_item = entry.get("fit_plot_item")
        if fit_item is not None:
            try:
                app.data_fit_plot.removeItem(fit_item)
            except Exception:
                pass
    app.data_fit_curves = []
    app.data_fit_power_ref_curve = None
    app.data_fit_power_window_manual = False
    # Reset preview state and clear the static raw/model curve items.
    app.data_fit_preview_visible = True
    app.data_fit_preview_include_in_fit = True
    if hasattr(app, "data_fit_raw_curve"):
        app.data_fit_raw_curve.setData([], [])
    if hasattr(app, "data_fit_model_curve"):
        app.data_fit_model_curve.setData([], [])
    # Clear the result panel and any stale warning so the next fit's
    # output isn't appended to the previous run's text.
    if hasattr(app, "data_fit_result_text"):
        app.data_fit_result_text.clear()
    _hide_fit_overlays(app)
    _clear_warning(app)
    # Reset cached fit results so the manual Save metadata button doesn't
    # carry over stale (label, FitResult) tuples from the previous file.
    controller = getattr(app, "data_fit_controller", None)
    if controller is not None:
        controller.last_result = None
        controller.last_fit_results = []
    if hasattr(app, "data_fit_save_metadata_btn"):
        app.data_fit_save_metadata_btn.setEnabled(False)


def _post_load_setup(app, *, auto_plot_fits: bool = True) -> None:
    """Shared post-load wiring for ``open_file_dialog`` and
    ``refresh_current_recording``: rebuild combos, populate metadata, refresh
    the preview, and (when configured) replay saved fits onto the plot.
    """
    app.data_fit_preview_visible = True
    app.data_fit_preview_include_in_fit = True
    app.data_fit_preview_color = _next_pastel_color(app)
    app.data_fit_preview_alpha_pct = 100
    app.data_fit_preview_style = {"draw_mode": "Auto", "line_width": 1.5, "point_size": 4}
    app.data_fit_curve_profiles = {"__preview__": _capture_fit_window_profile(app)}
    app.data_fit_plot_dirty = True
    _populate_channel_combos(app)
    load_metadata_from_tdms(app)
    _refresh_curve_profile_selector(app)
    refresh_preview(app)
    if auto_plot_fits:
        _replay_saved_fits_into_plot(app)


def _safe_checkbox_checked(app, attr_name: str, *, default: bool) -> bool:
    """Return checkbox state without crashing when a stale Qt wrapper exists.

    In some UI flows (e.g. settings widgets recreated), app attributes can
    still point to already-deleted C++ QCheckBox objects. Calling methods on
    those wrappers raises RuntimeError; treat that as missing and use default.
    """
    checkbox = getattr(app, attr_name, None)
    if checkbox is None:
        return bool(default)
    try:
        return bool(checkbox.isChecked())
    except RuntimeError:
        return bool(default)


def open_file_dialog(app):
    runtime_state = getattr(app, "runtime_state", None)
    start_dir = getattr(runtime_state, "output_folder", "") or ""
    path, _ = QFileDialog.getOpenFileName(app, "Select TDMS recording", start_dir, "TDMS Files (*.tdms);;All Files (*)")
    if not path:
        return
    # Clear stale curves and result text so the new file starts on an
    # empty plot rather than piling on top of the previous load.
    _clear_plot_state_for_new_recording(app)
    ok, msg = app.data_fit_controller.load_recording(path)
    app.data_fit_path_label.setText(msg)
    app.data_fit_path_label.setStyleSheet("color: black;" if ok else "color: #b35a00;")
    if ok:
        # Auto-load checkbox controls whether we also redraw any saved fit
        # overlays into the plot — separate from whether the preview itself
        # is shown, which always happens on a successful load.
        auto_load = _safe_checkbox_checked(app, "data_fit_auto_load_cb", default=True)
        _post_load_setup(app, auto_plot_fits=auto_load)


def refresh_current_recording(app, path: Optional[str] = None):
    """Reload the active recording into the Data Fitting tab.

    When ``path`` is given, it is loaded directly — used by the post-
    acquisition auto-load path so the lookup does not depend on
    ``current_tdms_filepath`` still being set when the GUI thread
    finally services the request. Otherwise: prefer the currently
    running file, fall back to the most recently completed one so
    "Use Current Recording" still works after Stop Read.
    """
    if not path:
        path = getattr(app, "current_tdms_filepath", "") or ""
    if not path:
        runtime_state = getattr(app, "runtime_state", None)
        path = getattr(runtime_state, "last_tdms_filepath", "") or ""
    # Clear stale curves and result text so a new acquisition starts on an
    # empty plot rather than piling on top of the previous run's results.
    _clear_plot_state_for_new_recording(app)
    ok, msg = app.data_fit_controller.load_recording(path)
    app.data_fit_path_label.setText(msg)
    app.data_fit_path_label.setStyleSheet("color: black;" if ok else "color: #b35a00;")
    if ok:
        auto_load = _safe_checkbox_checked(app, "data_fit_auto_load_cb", default=True)
        _post_load_setup(app, auto_plot_fits=auto_load)


def _y_signature_from_meta(name: str, meta: dict) -> tuple:
    """Synthesise a curve signature for a replayed channel that matches what
    ``_curve_signature`` would produce if the user had clicked Add to plot
    on the same channel themselves. Lets the dedup in ``_add_plot_from_current``
    swap the replay for a fresh plot without leaving an orphan entry behind.
    """
    return (
        "Time",
        "",
        name,
        float(meta.get("scale", 1.0) or 1.0),
        float(meta.get("offset", 0.0) or 0.0),
        1.0,
        0.0,
        1.0,
        0.0,
        1,
        meta.get("voltage_tap_cm"),
    )


def _add_replayed_source_curve(app, name: str, x: np.ndarray, y: np.ndarray,
                               t: np.ndarray, meta: dict, *, visible: bool):
    """Add a source curve (the raw V-I data) that was loaded from TDMS."""
    color = _next_pastel_color(app)
    label = name
    item = app.data_fit_plot.plot(
        [], [], pen=None, symbol="o", symbolSize=4,
        symbolBrush=pg.mkColor(color), symbolPen=pg.mkColor(color), name=label,
    )
    sig = ("__replayed__", name)
    entry = {
        "signature": sig,
        "label": label,
        "color": color,
        "alpha_pct": 100,
        "skip_points": 1,
        "include_in_fit": True,
        "visible": bool(visible),
        "x": x,
        "y": y,
        "t": t,
        "plot_item": item,
        "fit_result": None,
        "curve_style": {"draw_mode": "Auto", "line_width": 1.0, "point_size": 4},
        "avg_window": 1,
        "show_criterion": False,
        "show_ic": False,
        "source": {
            "time_sig": "Time",
            "x_sig": "",
            "y_sig": name,
            "t_scale": 1.0,
            "t_offset": 0.0,
            "x_scale": 1.0,
            "x_offset": 0.0,
            "y_scale": float(meta.get("scale", 1.0) or 1.0),
            "y_offset": float(meta.get("offset", 0.0) or 0.0),
            "use_length": bool(meta.get("voltage_tap_cm")),
            "length_cm": float(meta.get("voltage_tap_cm") or 1.0),
        },
        "is_replayed": True,
    }
    app.data_fit_curves.append(entry)
    _refresh_curve_item(entry)
    _apply_curve_visibility(entry)
    return entry


# Suffix appended to a fitted-channel label so it can be told apart from the
# source channel in the channel combo box. Also serves as the marker used by
# ``DataFittingController`` to remember which entries are reconstructed.
_FITTED_CHANNEL_SUFFIX = " - fitted"


def _sample_fit_curve_at(x_source: np.ndarray, fit_x: np.ndarray,
                         fit_y: np.ndarray) -> Optional[np.ndarray]:
    """Sample the smooth (fit_x, fit_y) model onto the source X positions.

    The reconstructed fit-channel needs to be the same length as the
    source data so it can be plotted as a regular Y channel. Uses linear
    interpolation; values outside the fit's X range get NaN so they
    don't pollute the plot range.
    """
    if x_source is None or fit_x is None or fit_y is None:
        return None
    if not (np.isfinite(fit_x).all() and np.isfinite(fit_y).all()):
        return None
    order = np.argsort(fit_x)
    fx = fit_x[order]
    fy = fit_y[order]
    sampled = np.interp(x_source, fx, fy, left=float("nan"), right=float("nan"))
    return sampled


def _replay_saved_fits_into_plot(app) -> None:
    """Replay every saved fit overlay we found in the just-loaded TDMS.

    For each labelled entry in ``controller.saved_fit_results``:
    1. If a matching source channel exists, plot it as a replayed source
       curve (so the fit has something to overlay).
    2. Reconstruct a ``FitResult`` from the saved properties and call
       ``_upsert_fit_curve_entry`` to attach the model curve.
    3. Register the reconstructed fit curve in the controller's channel
       cache as ``"<name> - fitted"`` so it appears in the Y-axis combo
       and can be selected like any other channel.
    The full per-channel ``_format_result`` block is shown in the result
    panel so users see the same parameter detail they would after running
    the fit fresh — Ic ± σ, n ± σ, R, V_ofs, R², chi², ramp ratio, …
    Only the first source curve is shown by default — the rest are added
    hidden so the plot stays uncluttered. The user can toggle visibility
    from the Plot summary dialog.
    """
    controller = getattr(app, "data_fit_controller", None)
    saved = getattr(controller, "saved_fit_results", {}) if controller is not None else {}
    if not saved:
        return
    if not hasattr(app, "data_fit_curves"):
        app.data_fit_curves = []
    # Hide the preview when we have replayed curves so the plot doesn't
    # contain both the raw preview and the replayed source curve for the
    # same channel.
    if any(name in (controller.channel_names or []) for name in saved):
        app.data_fit_preview_visible = False
        app.data_fit_raw_curve.setData([], [])

    summary_blocks: list[str] = []
    plotted = 0
    failed = 0
    first = True
    new_fitted_channels: list[str] = []
    for name, props in saved.items():
        if not name:
            continue
        meta = controller.get_metadata(name)
        x_raw = controller.get_channel("")  # placeholder, set below
        # X (current) defaults to the recording's first non-time channel
        # whose name contains "current"/"I"; otherwise the first numeric
        # channel that's not the Y itself. Falls back to ``time_array`` if
        # nothing matches — no overlay is drawn in that case.
        x_name = _guess_current_channel(controller, exclude=name)
        x_raw = controller.get_channel(x_name) if x_name else None
        y_raw = controller.get_channel(name)
        t_raw = controller.time_array
        if y_raw is None or x_raw is None or t_raw is None:
            failed += 1
            continue
        x_arr = controller.apply_transform(
            x_raw,
            float(controller.get_metadata(x_name).get("scale", 1.0) or 1.0),
            float(controller.get_metadata(x_name).get("offset", 0.0) or 0.0),
        )
        y_arr = controller.apply_transform(
            y_raw,
            float(meta.get("scale", 1.0) or 1.0),
            float(meta.get("offset", 0.0) or 0.0),
        )
        t_arr = np.asarray(t_raw, dtype=float)
        # Apply per-unit-length scaling iff the saved fit was in V/cm AND
        # the channel has a recorded tap distance.
        v_tap = meta.get("voltage_tap_cm")
        if v_tap and float(v_tap) > 0 and _coerce_bool(_prop_lookup(props, "uses_sample_length")):
            y_arr = y_arr / float(v_tap)
        result = _fit_result_from_props(props)
        fit_x, fit_y = _build_fit_curve(result, x_arr)
        result.fit_x = fit_x
        result.fit_y = fit_y
        source_entry = _add_replayed_source_curve(
            app, name, x_arr, y_arr, t_arr, meta, visible=first,
        )
        if result.ok and fit_x is not None and fit_y is not None:
            _upsert_fit_curve_entry(app, source_entry, result)
            for c in app.data_fit_curves:
                if (
                    c.get("is_fit_result")
                    and c.get("fit_parent_signature") == source_entry.get("signature")
                ):
                    c["visible"] = bool(first)
                    _apply_curve_visibility(c)
            # Register the reconstructed fit curve as a regular channel so
            # the user can pick it from the Y-axis combo. Sample the smooth
            # fit_y at the source X positions so the array length matches
            # the source channel — required for the channel combo flow.
            #
            # Storage units matter: ``_apply_transforms`` divides Y by the
            # active sample length when the voltage-tap checkbox is on. The
            # fit was already in V/cm when ``uses_sample_length`` is True,
            # so storing it as-is would make the display pipeline divide
            # by the tap distance a second time and flatten the transition
            # off-screen. Multiply by the tap distance here so the cached
            # values are in raw V, mirroring how the source channel is
            # stored, and the standard transform reproduces the V/cm scale.
            fitted_name = f"{name}{_FITTED_CHANNEL_SUFFIX}"
            sampled = _sample_fit_curve_at(x_arr, np.asarray(fit_x), np.asarray(fit_y))
            if sampled is not None and sampled.size:
                if (
                    v_tap and float(v_tap) > 0
                    and _coerce_bool(_prop_lookup(props, "uses_sample_length"))
                ):
                    sampled = sampled * float(v_tap)
                controller.channel_cache[fitted_name] = sampled
                controller.channel_metadata[fitted_name] = {
                    "scale": 1.0,
                    "offset": 0.0,
                    "voltage_tap_cm": meta.get("voltage_tap_cm"),
                }
                if fitted_name not in controller.channel_names:
                    controller.channel_names.append(fitted_name)
                    new_fitted_channels.append(fitted_name)
            plotted += 1
            first = False
            summary_blocks.append(f"[{name}] saved fit replayed:\n" + _format_result(result))
        else:
            failed += 1
            msg = _prop_lookup(props, "fit_message", "message") or "no parameters available"
            summary_blocks.append(f"[{name}] saved fit FAILED: {msg}")
            if first:
                first = False

    # Refresh the channel combos so the new fitted entries are selectable.
    if new_fitted_channels:
        _populate_channel_combos(app)
    _refresh_curve_profile_selector(app)
    if summary_blocks:
        intro = (
            f"Replayed {plotted} saved fit{'s' if plotted != 1 else ''}"
            + (f", {failed} failed" if failed else "")
            + ".\n\n"
        )
        app.data_fit_result_text.setPlainText(intro + "\n\n".join(summary_blocks))
    try:
        robust_view(app)
    except Exception:
        traceback.print_exc()


def _guess_current_channel(controller, *, exclude: str) -> str:
    """Pick the most likely current (X-axis) channel from a loaded recording.

    Used during fit replay when the user hasn't picked an X channel yet.
    Heuristic: prefer names containing 'I' or 'current' that aren't the
    excluded Y label; fall back to the first non-excluded channel.

    Skips ``" - fitted"`` reconstructions added by the replay so a later
    iteration doesn't pick a synthesised model curve as the current axis.
    """
    if not controller:
        return ""
    candidates = [
        n for n in (controller.channel_names or [])
        if n and n != exclude and not n.endswith(_FITTED_CHANNEL_SUFFIX)
    ]
    for needle in ("Current", "current", "_I", "AI0", "Ic"):
        for name in candidates:
            if needle in name:
                return name
    return candidates[0] if candidates else ""


def _apply_curve_visibility(entry: dict) -> None:
    """Honour ``entry['visible']`` on the underlying pyqtgraph plot item."""
    item = entry.get("plot_item")
    if item is None:
        return
    try:
        item.setVisible(bool(entry.get("visible", True)))
    except Exception:
        pass


_Y_TITLE_VOLTAGE = "Voltage (V)"
_Y_TITLE_E_FIELD = "Electric field (V/cm)"


_AUTO_LEFT_TITLES = {
    "",
    _Y_TITLE_VOLTAGE,
    _Y_TITLE_E_FIELD,
    f"{_Y_TITLE_VOLTAGE}  [log scale]",
    f"{_Y_TITLE_E_FIELD}  [log scale]",
}


def _update_y_axis_label(app):
    """Swap the Y-axis title in the graph-settings model so apply_graph_settings
    re-renders it (preserving the user's color/size/font choices).

    If the user has customised the title to something other than one of the
    auto values, we leave it alone.
    """
    desired = _Y_TITLE_E_FIELD if app.data_fit_use_length_cb.isChecked() else _Y_TITLE_VOLTAGE
    if _current_plot_scale(app) == _PLOT_SCALE_LOGLOG:
        desired = f"{desired}  [log scale]"
    settings = getattr(app, "data_fit_graph_settings", None)
    if settings is not None and settings.title_left.text in _AUTO_LEFT_TITLES:
        settings.title_left.text = desired


def _update_equation_label(app):
    has_length = (
        getattr(app, "data_fit_use_length_cb", None)
        and app.data_fit_use_length_cb.isChecked()
    )
    method = _active_fit_method(app) if hasattr(app, "data_fit_method_loglog_rb") else DEFAULT_FIT_METHOD
    if method == FIT_METHOD_LOG_LOG:
        if has_length:
            eq = (
                'Step 4 (IEC 61788, log E vs log I):<br>'
                '<span style="font-size: 14pt;">'
                '<b>log</b> E<sub>sc</sub> = <b>log</b> E<sub>c2</sub> + '
                '<b>n · log</b> (I / I<sub>c</sub>)'
                '</span><br>'
                '<span style="color:#555">where '
                'E<sub>sc</sub> = E − V<sub>0</sub> − ρ·I, '
                'E<sub>c1</sub> ≤ E<sub>sc</sub> ≤ E<sub>c2</sub></span>'
            )
        else:
            eq = (
                'Step 4 (IEC 61788, log V vs log I):<br>'
                '<span style="font-size: 14pt;">'
                '<b>log</b> V<sub>sc</sub> = <b>log</b> V<sub>c2</sub> + '
                '<b>n · log</b> (I / I<sub>c</sub>)'
                '</span>'
            )
    elif has_length:
        eq = (
            'Fitting model (IEC 61788, per unit length):<br>'
            '<span style="font-size: 14pt;">'
            '<b>E</b> = <b>(L/L<sub>s</sub>)·dI/dt</b> + <b>ρ·I</b> + '
            '<b>E<sub>c</sub>·(I/I<sub>c</sub>)<sup>n</sup></b>'
            '</span>'
        )
    else:
        eq = (
            'Fitting model (IEC 61788 power-law criterion):<br>'
            '<span style="font-size: 14pt;">'
            '<b>V</b> = <b>L·dI/dt</b> + <b>R·I</b> + '
            '<b>V<sub>c</sub>·(I/I<sub>c</sub>)<sup>n</sup></b>'
            '</span>'
        )
    app.data_fit_equation_label.setText(eq)


def _format_rate(rate_hz: float) -> str:
    if rate_hz >= 1.0e6:
        return f"{rate_hz / 1.0e6:.3g} MHz"
    if rate_hz >= 1.0e3:
        return f"{rate_hz / 1.0e3:.3g} kHz"
    return f"{rate_hz:.3g} Hz"


def _update_avg_rate_label(app):
    label = getattr(app, "data_fit_avg_rate_label", None)
    if label is None:
        return
    controller = getattr(app, "data_fit_controller", None)
    t_raw = None
    if controller is not None:
        time_name = app.data_fit_time_cb.currentText()
        t_raw = controller.get_channel(time_name) if time_name and time_name != "Time" else controller.time_array
    window = _active_avg_window(app)
    if t_raw is None or np.asarray(t_raw).size < 2:
        label.setText(f"Effective rate: — (avg window = {window})")
        return
    t_arr = np.asarray(t_raw, dtype=float)
    dt = float(np.mean(np.diff(t_arr)))
    if dt <= 0:
        label.setText(f"Effective rate: — (avg window = {window})")
        return
    fs_orig = 1.0 / dt
    fs_eff = fs_orig / max(1, window)
    if window <= 1:
        label.setText(f"Sample rate: {_format_rate(fs_orig)} (no averaging)")
    else:
        label.setText(
            f"Resampled to {_format_rate(fs_eff)} "
            f"(from {_format_rate(fs_orig)} / {window})"
        )


def _active_fit_method(app) -> str:
    """Return FIT_METHOD_LOG_LOG or FIT_METHOD_NONLINEAR from the radio buttons."""
    rb = getattr(app, "data_fit_method_loglog_rb", None)
    if rb is None:
        return DEFAULT_FIT_METHOD
    return FIT_METHOD_LOG_LOG if rb.isChecked() else FIT_METHOD_NONLINEAR


def _update_method_mode_ui(app) -> None:
    """Relabel Step 4 editors and gray out widgets irrelevant to the IEC mode.

    Log-log (IEC) mode: Low/High become Ec1/Ec2 in µV/cm; the Ic iteration
    knobs (max iterations, Ic stop tol, chi-sqr tol, Vc) are disabled — the
    decade method is a single closed-form linear fit.

    Non-linear mode: Low/High become fractions of Imax / Vmax as before, and
    the iteration knobs are re-enabled.
    """
    method = _active_fit_method(app)
    is_loglog = method == FIT_METHOD_LOG_LOG
    has_length = app.data_fit_use_length_cb.isChecked()
    units = "µV/cm" if has_length else "µV"
    if is_loglog:
        if getattr(app, "data_fit_power_low_label", None) is not None:
            app.data_fit_power_low_label.setText(f"Ec1 ({units})")
        if getattr(app, "data_fit_power_high_label", None) is not None:
            app.data_fit_power_high_label.setText(f"Ec2 ({units})")
    else:
        if getattr(app, "data_fit_power_low_label", None) is not None:
            app.data_fit_power_low_label.setText("Low (% of Imax)")
        if getattr(app, "data_fit_power_high_label", None) is not None:
            app.data_fit_power_high_label.setText("High (% of Vmax)")
    # Gray out the non-linear-only inputs in log-log mode.
    for widget in (
        app.data_fit_max_iter,
        app.data_fit_ic_tol,
        app.data_fit_chi_tol,
    ):
        widget.setEnabled(not is_loglog)
    # Vc input stays enabled in log-log mode so the user can override the
    # IEC default (Ec2) and report Ic at any criterion value of their choice.
    app.data_fit_vc_input.setEnabled(True)
    app.data_fit_vc_label.setEnabled(True)
    # The X-value editors in Step 4 map a percentage to a current, which is
    # meaningless when the editors hold Ec1/Ec2. Disable them in log-log mode.
    for widget in (
        app.data_fit_power_low_x,
        app.data_fit_power_high_x,
        getattr(app, "data_fit_power_low_x_label", None),
        getattr(app, "data_fit_power_high_x_label", None),
    ):
        if widget is not None:
            widget.setEnabled(not is_loglog)
    # In log-log mode the Show checkbox controls the Step 4 shaded region and
    # dragging it updates Ec1/Ec2 directly.
    app.data_fit_show_power.setEnabled(True)
    _save_active_curve_profile(app)


def _on_fit_method_changed(app) -> None:
    """Switch Step-4 editors to the mode-specific values for the active curve.

    Each curve's profile remembers per-method Ec1/Ec2 (log-log) and
    Imax/Vmax fractions (non-linear) so toggling between methods restores
    the user's last-edited values for the new method instead of clobbering
    them with defaults.
    """
    new_method = _active_fit_method(app)
    old_method = (
        FIT_METHOD_NONLINEAR if new_method == FIT_METHOD_LOG_LOG else FIT_METHOD_LOG_LOG
    )
    key = _curve_profile_key_from_ui(app)
    profiles = getattr(app, "data_fit_curve_profiles", {}) or {}
    profile = dict(profiles.get(key, {}))
    # Snapshot the still-old widget contents into the OLD method's slot so
    # they survive a later switch back.
    if old_method == FIT_METHOD_LOG_LOG:
        profile["loglog_low"] = app.data_fit_power_low.text()
        profile["loglog_high"] = app.data_fit_power_vfrac.text()
    else:
        profile["nonlinear_low"] = app.data_fit_power_low.text()
        profile["nonlinear_high"] = app.data_fit_power_vfrac.text()
    if new_method == FIT_METHOD_LOG_LOG:
        new_low = profile.get("loglog_low") or f"{DEFAULT_EC1_V_PER_CM * 1.0e6:g}"
        new_high = profile.get("loglog_high") or f"{DEFAULT_EC2_V_PER_CM * 1.0e6:g}"
        app.data_fit_power_window_manual = False
    else:
        new_low = profile.get("nonlinear_low") or f"{DEFAULT_POWER_LOW_FRAC * 100:.2f}"
        new_high = profile.get("nonlinear_high") or f"{DEFAULT_POWER_V_FRAC * 100:.2f}"
    _set_silently(app.data_fit_power_low, str(new_low))
    _set_silently(app.data_fit_power_vfrac, str(new_high))
    profile["fit_method"] = new_method
    profile["power_low"] = str(new_low)
    profile["power_vfrac"] = str(new_high)
    profiles[key] = profile
    app.data_fit_curve_profiles = profiles
    _update_method_mode_ui(app)
    _refresh_all_x_values(app)
    ctx = _data_ctx(app)
    if ctx is not None:
        _, _, x_arr, y_arr, _ = ctx
        _update_fit_bands(app, x_arr, y_arr)
    _update_band_states(app)
    _update_equation_label(app)


_PLOT_SCALE_LINEAR = "linear"
_PLOT_SCALE_LOGLOG = "loglog"


def _current_plot_scale(app) -> str:
    settings = getattr(app, "data_fit_graph_settings", None)
    if settings is None:
        return _PLOT_SCALE_LINEAR
    if (settings.scale_h.scale_type == "Log10"
            and settings.scale_v.scale_type == "Log10"):
        return _PLOT_SCALE_LOGLOG
    return _PLOT_SCALE_LINEAR


def _update_plot_scale_button_text(app) -> None:
    btn = getattr(app, "data_fit_plot_scale_btn", None)
    if btn is None:
        return
    if _current_plot_scale(app) == _PLOT_SCALE_LOGLOG:
        btn.setText("Switch to linear-linear plot")
    else:
        btn.setText("Switch to log-log plot")


def _toggle_plot_scale(app) -> None:
    """Switch the V-I plot between linear/linear and log/log axes.

    Writes the choice into ``data_fit_graph_settings`` so the graph-settings
    dialog reflects it, then applies it directly to the plot and lets
    pyqtgraph auto-range in log space. (Going through refresh_preview would
    re-impose a linear-space robust-view range on top of the log transform,
    which pushes the data off screen.)
    """
    settings = getattr(app, "data_fit_graph_settings", None)
    if settings is None:
        return
    to_loglog = _current_plot_scale(app) != _PLOT_SCALE_LOGLOG
    if to_loglog:
        settings.scale_h.scale_type = "Log10"
        settings.scale_v.scale_type = "Log10"
        settings.scale_h.auto_range = True
        settings.scale_v.auto_range = True
    else:
        settings.scale_h.scale_type = "Linear"
        settings.scale_v.scale_type = "Linear"
    _update_plot_scale_button_text(app)
    plot_widget = getattr(app, "data_fit_plot", None)
    if plot_widget is None:
        return
    plot_item = plot_widget.getPlotItem()
    plot_item.setLogMode(x=to_loglog, y=to_loglog)
    # Keep the linked residuals plot on a matching log x-axis so tick labels
    # match up with the main plot. Y stays linear — residuals are signed.
    resid_plot = getattr(app, "data_fit_resid_plot", None)
    if resid_plot is not None:
        resid_plot.getPlotItem().setLogMode(x=to_loglog, y=False)
    _apply_axis_labels_for_scale(app, to_loglog)
    vb = plot_item.getViewBox()
    vb.enableAutoRange(axis="x")
    vb.enableAutoRange(axis="y")
    # Re-render the step bands / Ec1-Ec2 lines in the new coordinate system.
    try:
        refresh_preview(app)
    except Exception:
        traceback.print_exc()
    if not to_loglog:
        # Back to linear: restore the robust, outlier-trimmed view.
        try:
            robust_view(app)
        except Exception:
            traceback.print_exc()


_AUTO_BOTTOM_TITLES = {
    "",
    "Current (A)",
    "Current (A)  [log scale]",
}


def _apply_axis_labels_for_scale(app, to_loglog: bool) -> None:
    """Rewrite axis titles in the graph-settings model so the log transform is
    explicit. apply_graph_settings re-renders the labels with the user's
    color/size/font choices preserved.

    Tick labels are handled by EngineeringAxisItem (which shows SI-prefixed
    values in both linear and log mode), so here we only adjust the title.
    """
    settings = getattr(app, "data_fit_graph_settings", None)
    if settings is None:
        return
    use_length = (
        getattr(app, "data_fit_use_length_cb", None)
        and app.data_fit_use_length_cb.isChecked()
    )
    y_base = _Y_TITLE_E_FIELD if use_length else _Y_TITLE_VOLTAGE
    bottom_desired = "Current (A)  [log scale]" if to_loglog else "Current (A)"
    left_desired = f"{y_base}  [log scale]" if to_loglog else y_base
    if settings.title_bottom.text in _AUTO_BOTTOM_TITLES:
        settings.title_bottom.text = bottom_desired
    if settings.title_left.text in _AUTO_LEFT_TITLES:
        settings.title_left.text = left_desired


def _on_use_length_changed(app):
    app.data_fit_plot_dirty = True
    checked = app.data_fit_use_length_cb.isChecked()
    app.data_fit_length_input.setEnabled(checked)
    app.data_fit_length_label.setEnabled(checked)
    if checked:
        app.data_fit_vc_label.setText("Ec (µV/cm):")
        app.data_fit_vc_input.setText(f"{DEFAULT_EC_V_PER_CM * 1.0e6:.6g}")
    else:
        app.data_fit_vc_label.setText("Vc (mV):")
        app.data_fit_vc_input.setText(f"{DEFAULT_VC_VOLTS * 1000:.6g}")
    _update_equation_label(app)
    _update_method_mode_ui(app)
    if hasattr(app, "data_fit_curve_profile_cb"):
        _save_active_curve_profile(app)


def refresh_preview(app):
    _update_y_axis_label(app)
    _update_avg_rate_label(app)
    transformed = _apply_transforms(app)
    x = transformed["x"]
    y = transformed["y"]
    app.data_fit_model_curve.setData([], [])
    _hide_fit_overlays(app)
    if x is None or y is None or x.size == 0 or y.size == 0:
        app.data_fit_raw_curve.setData([], [])
        # Apply styling-only so the Graph-settings dialog still affects the
        # plot before the user has loaded data.
        apply_graph_settings(
            app.data_fit_plot, None, None, None, app.data_fit_graph_settings,
        )
        app.data_fit_xrange_label.setText("X window: full")
        return
    n = min(len(x), len(y))
    step = max(1, n // 5000)
    if getattr(app, "data_fit_preview_visible", True):
        apply_graph_settings(
            app.data_fit_plot, app.data_fit_raw_curve,
            x[:n:step], y[:n:step], app.data_fit_graph_settings,
        )
        preview_entry = {
            "x": x[:n:step],
            "y": y[:n:step],
            "color": getattr(app, "data_fit_preview_color", "#1f77b4"),
            "alpha_pct": int(getattr(app, "data_fit_preview_alpha_pct", 100)),
            "label": "Preview",
            "plot_item": app.data_fit_raw_curve,
            "skip_points": 1,
            "curve_style": dict(getattr(app, "data_fit_preview_style", {})),
        }
        _refresh_curve_item(preview_entry)
    else:
        app.data_fit_raw_curve.setData([], [])
        # Preview hidden — still apply the dialog's styling.
        apply_graph_settings(
            app.data_fit_plot, None, None, None, app.data_fit_graph_settings,
        )
    x_min_full = float(np.min(x[:n]))
    x_max_full = float(np.max(x[:n]))
    _apply_robust_view(app, x[:n], y[:n])
    _refresh_all_x_values(app)
    _update_fit_bands(app, x[:n], y[:n])
    _update_band_states(app)
    app.data_fit_xrange_label.setText(f"X window: [{x_min_full:.6g}, {x_max_full:.6g}]")


def _plot_is_loglog(app) -> bool:
    return _current_plot_scale(app) == _PLOT_SCALE_LOGLOG


def _xform_for_view(val: float, is_log: bool) -> float:
    """Linear data value → plot view coordinate (log10 when in log mode)."""
    if not is_log:
        return float(val)
    try:
        v = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return float(np.log10(v)) if v > 0 else float("nan")


def _xform_from_view(val: float, is_log: bool) -> float:
    """Plot view coordinate → linear data value."""
    if not is_log:
        return float(val)
    return float(10.0 ** float(val))


def _update_fit_bands(app, x: np.ndarray, y: np.ndarray) -> None:
    """Update the three semi-transparent bands that show the configured windows.

    LinearRegionItem values are in plot view coordinates, which in log-log
    mode are log10(data). We transform once here so the bands line up with
    the underlying curve regardless of axis scale.
    """
    if x is None or x.size == 0:
        return
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = x_max - x_min
    if span <= 0:
        return
    def from_pct(pct_widget, fallback):
        return x_min + _float_from(pct_widget, fallback * 100, as_fraction=True) * span

    didt_lo = from_pct(app.data_fit_didt_low, DEFAULT_DIDT_LOW_FRAC)
    didt_hi = from_pct(app.data_fit_didt_high, DEFAULT_DIDT_HIGH_FRAC)
    lin_lo = from_pct(app.data_fit_linear_low, DEFAULT_LINEAR_LOW_FRAC)
    lin_hi = from_pct(app.data_fit_linear_high, DEFAULT_LINEAR_HIGH_FRAC)

    band_pairs = [
        (app.data_fit_band_didt, (didt_lo, didt_hi)),
        (app.data_fit_band_linear, (lin_lo, lin_hi)),
    ]

    is_loglog = _plot_is_loglog(app)

    def _window_from_saved_fit() -> Optional[tuple[float, float]]:
        """Best-effort current window [I(Ec1), I(Ec2)] for the active Y channel.

        Do not restore Low/High (X) from TDMS metadata saved by past runs.
        This keeps the I-window fields clean on file load and only reuses
        windows computed in the current app session.

        Priority:
        1) Fresh run-fit cache keyed by curve label.
        2) Last single-result fallback (only when clearly valid).
        """
        if _active_fit_method(app) != FIT_METHOD_LOG_LOG:
            return None
        y_name = app.data_fit_y_cb.currentText().strip() if hasattr(app, "data_fit_y_cb") else ""
        if y_name.endswith(_FITTED_CHANNEL_SUFFIX):
            y_name = y_name[: -len(_FITTED_CHANNEL_SUFFIX)]
        controller = getattr(app, "data_fit_controller", None)
        if controller is None:
            return None

        def _coerce_window(raw) -> Optional[tuple[float, float]]:
            if not raw or len(raw) != 2:
                return None
            try:
                lo = float(raw[0])
                hi = float(raw[1])
            except (TypeError, ValueError):
                return None
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                return None
            return lo, hi

        if y_name:
            for label, result in reversed(list(getattr(controller, "last_fit_results", []) or [])):
                if str(label).strip() != y_name:
                    continue
                if getattr(result, "fit_method", "") != FIT_METHOD_LOG_LOG:
                    continue
                got = _coerce_window(getattr(result, "n_window_I", None))
                if got is not None:
                    return got

        last_result = getattr(controller, "last_result", None)
        if getattr(last_result, "fit_method", "") == FIT_METHOD_LOG_LOG:
            return _coerce_window(getattr(last_result, "n_window_I", None))
        return None

    ec1 = ec2 = None
    if _active_fit_method(app) == FIT_METHOD_LOG_LOG:
        ref = getattr(app, "data_fit_power_ref_curve", None) or {}
        ref_x = np.asarray(ref.get("x", []), dtype=float)
        ref_y = np.asarray(ref.get("y", []), dtype=float)
        if ref_x.size and ref_y.size:
            n_ref = int(min(ref_x.size, ref_y.size))
            x_for_power = ref_x[:n_ref]
            y_for_power = ref_y[:n_ref]
        else:
            x_for_power = x
            y_for_power = y
        has_length = app.data_fit_use_length_cb.isChecked()
        to_si = 1.0e-6 if has_length else 1.0e-3
        ec1 = _float_from(app.data_fit_power_low, DEFAULT_EC1_V_PER_CM * 1.0e6) * to_si
        ec2 = _float_from(app.data_fit_power_vfrac, DEFAULT_EC2_V_PER_CM * 1.0e6) * to_si
        try:
            ui_lo = float((app.data_fit_power_low_x.text() or "").strip())
            ui_hi = float((app.data_fit_power_high_x.text() or "").strip())
            ui_window = (ui_lo, ui_hi) if (np.isfinite(ui_lo) and np.isfinite(ui_hi) and ui_hi > ui_lo) else None
        except (TypeError, ValueError):
            ui_window = None
        exact_window = None if bool(getattr(app, "data_fit_power_window_manual", False)) else _window_from_saved_fit()
        if ui_window is not None:
            band_pairs.append((app.data_fit_band_power, ui_window))
        elif exact_window is not None:
            band_pairs.append((app.data_fit_band_power, exact_window))
            _set_silently(app.data_fit_power_low_x, f"{exact_window[0]:.6g}")
            _set_silently(app.data_fit_power_high_x, f"{exact_window[1]:.6g}")
        else:
            # Pre-fit fallback: use the same High/Low rule as the fit service.
            if y_for_power is not None and y_for_power.size:
                valid = np.isfinite(x_for_power) & np.isfinite(y_for_power) & (x_for_power > 0)
                if np.any(valid):
                    x_pick = np.asarray(x_for_power[valid], dtype=float)
                    y_pick = np.asarray(y_for_power[valid], dtype=float)
                    order = np.argsort(x_pick)
                    x_pick = x_pick[order]
                    y_pick = y_pick[order]
                    pow_lo, pow_hi = pick_loglog_i_window_from_thresholds(
                        x_pick, y_pick, ec1=ec1, ec2=ec2, guard_fraction=DEFAULT_EC_WINDOW_GUARD_FRAC,
                    )
                    if np.isfinite(pow_lo) and np.isfinite(pow_hi) and pow_hi > pow_lo:
                        band_pairs.append((app.data_fit_band_power, (pow_lo, pow_hi)))
    else:
        pow_lo = from_pct(app.data_fit_power_low, DEFAULT_POWER_LOW_FRAC)
        v_f = _float_from(
            app.data_fit_power_vfrac, DEFAULT_POWER_V_FRAC * 100, as_fraction=True
        )
        y_max = float(np.max(y)) if (y is not None and y.size) else 0.0
        threshold = v_f * y_max
        above = np.where(y >= threshold)[0] if y is not None else np.array([], dtype=int)
        pow_hi = float(x[above[0]]) if above.size else x_max
        band_pairs.append((app.data_fit_band_power, (pow_lo, pow_hi)))

    for band, pair in band_pairs:
        lo, hi = pair
        view_pair = (_xform_for_view(lo, is_loglog), _xform_for_view(hi, is_loglog))
        if not (np.isfinite(view_pair[0]) and np.isfinite(view_pair[1])):
            continue
        band.blockSignals(True)
        try:
            band.setRegion(view_pair)
        finally:
            band.blockSignals(False)

    # Keep Ec1/Ec2 guide lines hidden to avoid extra dashed overlays/text.
    ec1_line = getattr(app, "data_fit_ec1_line", None)
    ec2_line = getattr(app, "data_fit_ec2_line", None)
    if ec1_line is not None and ec2_line is not None:
        ec1_line.setVisible(False)
        ec2_line.setVisible(False)


def _robust_log_view_range(values, low_pct: float = 1.0, high_pct: float = 99.0,
                           margin: float = 0.1):
    """Robust percentile bounds in log10 space, for axes in log mode.

    Negative / zero / non-finite values are dropped (log10 is undefined). The
    percentiles are taken on the log10-transformed data so the trim is
    decade-aware, and the margin is also in log space so the padding is a
    fixed fraction of decades rather than blowing up to many decades.
    Returns ``(None, None)`` if there's no usable data.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None, None
    log_arr = np.log10(arr)
    lo = float(np.percentile(log_arr, low_pct))
    hi = float(np.percentile(log_arr, high_pct))
    if hi <= lo:
        lo = float(np.min(log_arr))
        hi = float(np.max(log_arr))
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def _apply_robust_view(app, x: np.ndarray, y: np.ndarray) -> None:
    plot_item = app.data_fit_plot.getPlotItem()
    view_box = plot_item.getViewBox()
    settings = getattr(app, "data_fit_graph_settings", None)
    is_log_x = settings is not None and settings.scale_h.scale_type == "Log10"
    is_log_y = settings is not None and settings.scale_v.scale_type == "Log10"
    auto_x = settings is None or settings.scale_h.auto_range
    auto_y = settings is None or settings.scale_v.auto_range

    # Honour user-specified ranges from the Graph-settings dialog: when
    # auto-range is off for an axis, leave whatever apply_graph_settings set.
    if auto_x:
        if is_log_x:
            x_lo, x_hi = _robust_log_view_range(x)
        else:
            x_lo, x_hi = robust_view_range(x)
        if x_lo is not None and x_hi is not None:
            view_box.setXRange(x_lo, x_hi, padding=0.0)
        else:
            view_box.enableAutoRange(axis="x")
    if auto_y:
        if is_log_y:
            y_lo, y_hi = _robust_log_view_range(y)
        else:
            y_lo, y_hi = robust_view_range(y)
        if y_lo is not None and y_hi is not None:
            view_box.setYRange(y_lo, y_hi, padding=0.0)
        else:
            view_box.enableAutoRange(axis="y")


def robust_view(app):
    x_arrays = []
    y_arrays = []
    if getattr(app, "data_fit_preview_visible", True):
        transformed = _apply_transforms(app)
        x = transformed["x"]
        y = transformed["y"]
        if x is not None and y is not None and x.size and y.size:
            x_arrays.append(np.asarray(x))
            y_arrays.append(np.asarray(y))
    for entry in getattr(app, "data_fit_curves", []):
        x = entry.get("x")
        y = entry.get("y")
        if x is None or y is None:
            continue
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if x_arr.size and y_arr.size:
            x_arrays.append(x_arr)
            y_arrays.append(y_arr)
    if not x_arrays or not y_arrays:
        return
    all_x = np.concatenate(x_arrays)
    all_y = np.concatenate(y_arrays)
    _apply_robust_view(app, all_x, all_y)


def reset_view(app):
    app.data_fit_plot.getPlotItem().getViewBox().autoRange(padding=0.05)


def toggle_zoom(app, checked: bool):
    view_box = app.data_fit_plot.getPlotItem().getViewBox()
    view_box.setMouseMode(pg.ViewBox.RectMode if checked else pg.ViewBox.PanMode)
    app.data_fit_zoom_mode_btn.setText("Zoom Mode (on)" if checked else "Zoom Mode")


def _update_band_states(app) -> None:
    """Show and allow dragging for every window whose Show checkbox is enabled.

    In log-log mode the power band remains draggable and updates Ec1/Ec2.
    """
    for window, band, show_cb in (
        ("didt", app.data_fit_band_didt, app.data_fit_show_didt),
        ("linear", app.data_fit_band_linear, app.data_fit_show_linear),
        ("power", app.data_fit_band_power, app.data_fit_show_power),
    ):
        checked = bool(show_cb.isChecked())
        band.setMovable(checked)
        band.setVisible(checked)


def _on_show_power_toggled(app, checked: bool) -> None:
    """When Step-4 Show/Edit is enabled, ensure corrected+smoothed reference exists."""
    if checked and _active_fit_method(app) == FIT_METHOD_LOG_LOG:
        _ensure_step4_reference_curve(app, create_plot_entry=False, auto_run_fit=True)
        _update_loglog_power_x_from_ec(app, auto_run_fit=True)
    _update_band_states(app)
    refresh_preview(app)


def _on_band_dragged(app, window: str) -> None:
    """A band was dragged — write its new region back to that step's textboxes."""
    band = {
        "didt": app.data_fit_band_didt,
        "linear": app.data_fit_band_linear,
        "power": app.data_fit_band_power,
    }[window]
    if not bool(getattr(band, "movable", False)):
        return
    lo, hi = band.getRegion()
    # Region values are in view coordinates — in log mode that means log10(I).
    is_loglog = _plot_is_loglog(app)
    lo = _xform_from_view(lo, is_loglog)
    hi = _xform_from_view(hi, is_loglog)
    ctx = _data_ctx(app)
    if ctx is None:
        return
    x_min, x_max, x, y, y_max = ctx
    if window == "power" and _active_fit_method(app) == FIT_METHOD_LOG_LOG:
        lo, hi = sorted((lo, hi))
        ref = getattr(app, "data_fit_power_ref_curve", None) or {}
        ref_x = np.asarray(ref.get("x", []), dtype=float)
        ref_y = np.asarray(ref.get("y", []), dtype=float)
        if ref_x.size and ref_y.size:
            n_ref = int(min(ref_x.size, ref_y.size))
            x_arr = ref_x[:n_ref]
            y_arr = ref_y[:n_ref]
        else:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        if y_arr.size == 0:
            return
        has_length = app.data_fit_use_length_cb.isChecked()
        to_si = 1.0e-6 if has_length else 1.0e-3
        ec1_guess = _float_from(app.data_fit_power_low, DEFAULT_EC1_V_PER_CM * 1.0e6) * to_si
        ec2_guess = _float_from(app.data_fit_power_vfrac, DEFAULT_EC2_V_PER_CM * 1.0e6) * to_si
        y_ref = y_arr if (ref_x.size and ref_y.size) else _adaptive_smooth_visual(y_arr, ec1_guess, ec2_guess)
        idx_lo = int(np.argmin(np.abs(x_arr - lo)))
        idx_hi = int(np.argmin(np.abs(x_arr - hi)))
        ec1 = max(float(y_ref[idx_lo]), 1.0e-30)
        ec2 = max(float(y_ref[idx_hi]), ec1 * 1.000001)
        from_si = 1.0e6 if has_length else 1.0e3
        _set_silently(app.data_fit_power_low, f"{ec1 * from_si:.6g}")
        _set_silently(app.data_fit_power_vfrac, f"{ec2 * from_si:.6g}")
        _set_silently(app.data_fit_power_low_x, f"{lo:.6g}")
        _set_silently(app.data_fit_power_high_x, f"{hi:.6g}")
        app.data_fit_power_window_manual = True
        app.data_fit_xrange_label.setText(f"power (Ec): [{ec1 * from_si:.6g}, {ec2 * from_si:.6g}]")
        _save_active_curve_profile(app)
        return
    low_pct_widget, low_x_widget, _ = app.data_fit_window_inputs[(window, "low")]
    high_pct_widget, high_x_widget, high_axis = app.data_fit_window_inputs[(window, "high")]
    _set_silently(low_pct_widget, f"{_x_to_pct(lo, x_min, x_max):.4f}")
    _set_silently(low_x_widget, f"{lo:.6g}")
    if high_axis == "Vmax":
        _set_silently(high_pct_widget, f"{_x_to_vpct(hi, x, y, y_max):.4f}")
    else:
        _set_silently(high_pct_widget, f"{_x_to_pct(hi, x_min, x_max):.4f}")
    _set_silently(high_x_widget, f"{hi:.6g}")
    app.data_fit_xrange_label.setText(f"{window}: [{lo:.6g}, {hi:.6g}]")
    _save_active_curve_profile(app)


def _data_ctx(app):
    transformed = _apply_transforms(app)
    x = transformed["x"]
    y = transformed["y"]
    if _active_fit_method(app) == FIT_METHOD_LOG_LOG:
        ref = getattr(app, "data_fit_power_ref_curve", None) or {}
        ref_x = np.asarray(ref.get("x", []), dtype=float)
        ref_y = np.asarray(ref.get("y", []), dtype=float)
        if ref_x.size and ref_y.size:
            n_ref = int(min(ref_x.size, ref_y.size))
            x = ref_x[:n_ref]
            y = ref_y[:n_ref]
    if x is None or x.size == 0:
        return None
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return None
    y_max = float(np.max(y)) if (y is not None and y.size) else 0.0
    return x_min, x_max, x, y, y_max


def _pct_to_x(pct: float, x_min: float, x_max: float) -> float:
    return x_min + (pct / 100.0) * (x_max - x_min)


def _x_to_pct(x_val: float, x_min: float, x_max: float) -> float:
    span = x_max - x_min
    return 0.0 if span <= 0 else (x_val - x_min) / span * 100.0


def _vpct_to_x(vpct: float, x: np.ndarray, y: np.ndarray, y_max: float, x_max: float) -> float:
    if y_max == 0 or x is None or x.size == 0:
        return x_max
    threshold = (vpct / 100.0) * y_max
    above = np.where(y >= threshold)[0]
    return float(x[above[0]]) if above.size else x_max


def _x_to_vpct(x_val: float, x: np.ndarray, y: np.ndarray, y_max: float) -> float:
    if y_max == 0 or x is None or x.size == 0:
        return 0.0
    idx = int(np.argmin(np.abs(x - x_val)))
    return float(y[idx]) / y_max * 100.0


def _update_loglog_power_x_from_ec(app, *, auto_run_fit: bool = True) -> bool:
    """Update Step-4 low/high X from Ec1/Ec2 using corrected+smoothed reference.

    Mapping used by the UI in log-log mode:
      0) Call ``_ensure_step4_reference_curve(...)``. That helper applies
         baseline correction ``y - (V0 + R·I)`` and adaptive smoothing, then
         stores the result in ``app.data_fit_power_ref_curve``.
      1) Convert Ec1/Ec2 from displayed units (µV/cm or µV) to SI (V/cm or V).
      2) On the Step-4 reference trace ``(x_arr, y_arr)``, find High(X)
         at the first ``y_arr >= Ec2`` after the guard region.
      3) Starting from High(X), walk backwards toward lower current until
         ``y_arr >= Ec1`` -> Low(X).
      4) If a threshold is not reached, clamp to the trace max current.
      5) Enforce ``High(X) > Low(X)`` with a tiny guard band.
    """
    if _active_fit_method(app) != FIT_METHOD_LOG_LOG:
        return False
    ok = _ensure_step4_reference_curve(app, create_plot_entry=False, auto_run_fit=auto_run_fit)
    if not ok:
        return False
    ref = getattr(app, "data_fit_power_ref_curve", None) or {}
    # ``ref`` is the corrected+smoothed Step-4 reference from
    # ``_ensure_step4_reference_curve``.
    x_arr = np.asarray(ref.get("x", []), dtype=float)
    y_arr = np.asarray(ref.get("y", []), dtype=float)
    n = int(min(x_arr.size, y_arr.size))
    if n == 0:
        return False
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    # Align UI threshold picking with service.fit_n_value_log_log:
    # use positive-current points and sort by current before searching Ec hits.
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0)
    if not np.any(valid):
        return False
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    has_length = app.data_fit_use_length_cb.isChecked()
    to_si = 1.0e-6 if has_length else 1.0e-3
    ec1 = max(_float_from(app.data_fit_power_low, DEFAULT_EC1_V_PER_CM * 1.0e6) * to_si, 1.0e-30)
    ec2 = max(_float_from(app.data_fit_power_vfrac, DEFAULT_EC2_V_PER_CM * 1.0e6) * to_si, ec1 * 1.000001)
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    x_lo, x_hi = pick_loglog_i_window_from_thresholds(
        x_arr, y_arr, ec1=ec1, ec2=ec2, guard_fraction=DEFAULT_EC_WINDOW_GUARD_FRAC,
    )
    if not np.isfinite(x_lo):
        x_lo = x_min
    if not np.isfinite(x_hi):
        x_hi = x_max
    _set_silently(app.data_fit_power_low_x, f"{x_lo:.6g}")
    _set_silently(app.data_fit_power_high_x, f"{x_hi:.6g}")
    app.data_fit_power_window_manual = True
    return True


def _refresh_x_from_pct(app, window: str, which: str) -> None:
    # In log-log mode the Step 4 editors hold Ec1/Ec2 (µV/cm), not a
    # percentage of Imax — no meaningful X mapping until a fit has run.
    if window == "power" and _active_fit_method(app) == FIT_METHOD_LOG_LOG:
        return
    ctx = _data_ctx(app)
    if ctx is None:
        return
    x_min, x_max, x, y, y_max = ctx
    pct_widget, x_widget, axis = app.data_fit_window_inputs[(window, which)]
    try:
        pct = float(pct_widget.text())
    except ValueError:
        return
    x_val = _vpct_to_x(pct, x, y, y_max, x_max) if axis == "Vmax" else _pct_to_x(pct, x_min, x_max)
    _set_silently(x_widget, f"{x_val:.6g}")


def _refresh_pct_from_x(app, window: str, which: str) -> None:
    if window == "power" and _active_fit_method(app) == FIT_METHOD_LOG_LOG:
        return
    ctx = _data_ctx(app)
    if ctx is None:
        return
    x_min, x_max, x, y, y_max = ctx
    pct_widget, x_widget, axis = app.data_fit_window_inputs[(window, which)]
    try:
        x_val = float(x_widget.text())
    except ValueError:
        return
    pct = _x_to_vpct(x_val, x, y, y_max) if axis == "Vmax" else _x_to_pct(x_val, x_min, x_max)
    _set_silently(pct_widget, f"{pct:.4f}")


def _handle_window_edit(app, window: str, which: str, source: str) -> None:
    pct_widget, x_widget, _axis = app.data_fit_window_inputs[(window, which)]
    widget = pct_widget if source == "pct" else x_widget
    # editingFinished also fires on focus loss without edits; skip those so
    # round-trip pct <-> X discretisation does not accumulate drift.
    if not widget.isModified():
        return
    widget.setModified(False)
    if window == "power" and _active_fit_method(app) == FIT_METHOD_LOG_LOG:
        if source == "pct":
            _update_loglog_power_x_from_ec(app)
        else:
            _refresh_pct_from_x(app, window, which)
        ctx = _data_ctx(app)
        if ctx is not None:
            _, _, x_arr, y_arr, _ = ctx
            _update_fit_bands(app, x_arr, y_arr)
        _update_band_states(app)
        _save_active_curve_profile(app)
        return
    if source == "pct":
        _refresh_x_from_pct(app, window, which)
    else:
        _refresh_pct_from_x(app, window, which)
    ctx = _data_ctx(app)
    if ctx is not None:
        _, _, x_arr, y_arr, _ = ctx
        _update_fit_bands(app, x_arr, y_arr)
    _update_band_states(app)
    _save_active_curve_profile(app)


def _refresh_all_x_values(app) -> None:
    if not hasattr(app, "data_fit_window_inputs"):
        return
    for (window, which) in app.data_fit_window_inputs:
        _refresh_x_from_pct(app, window, which)


def region_mode_changed(app, _button=None):
    _update_band_states(app)


def sync_region_to_inputs(app):
    _update_band_states(app)


def load_metadata_from_tdms(app):
    controller = app.data_fit_controller
    if not controller.channel_metadata:
        QMessageBox.information(app, "Data Fitting", "Load a TDMS recording first.")
        return
    for axis in ("time", "x", "y"):
        _load_axis_metadata_from_tdms(app, axis)
    _apply_voltage_tap_from_metadata(app)
    refresh_preview(app)


def _load_axis_metadata_from_tdms(app, axis: str) -> None:
    """Load scale/offset for one axis from the currently selected channel."""
    combo = getattr(app, f"data_fit_{axis}_cb", None)
    scale_input = getattr(app, f"data_fit_{axis}_scale", None)
    offset_input = getattr(app, f"data_fit_{axis}_offset", None)
    if combo is None or scale_input is None or offset_input is None:
        return
    name = combo.currentText()
    if not name or name == "Time":
        # Built-in Time option is not a real data channel metadata entry.
        scale_input.setText("1")
        offset_input.setText("0")
        return
    meta = app.data_fit_controller.get_metadata(name)
    scale_input.setText(f"{meta['scale']:g}")
    offset_input.setText(f"{meta['offset']:g}")


def _on_channel_selection_changed(app, axis: str) -> None:
    """Whenever a channel dropdown changes, refresh that axis transform."""
    _load_axis_metadata_from_tdms(app, axis)
    if axis == "y":
        _apply_voltage_tap_from_metadata(app)
    _on_transform_inputs_changed(app)


def _apply_voltage_tap_from_metadata(app) -> None:
    """Pick up the voltage-tap distance from the selected Y channel only.

    This keeps tap distance channel-specific: every Y-channel selection can
    carry its own Voltage_Tab_Distance from TDMS metadata.
    """
    controller = getattr(app, "data_fit_controller", None)
    if controller is None:
        return
    y_name = app.data_fit_y_cb.currentText()
    v_tap: Optional[float] = None
    if y_name:
        v_tap = controller.get_metadata(y_name).get("voltage_tap_cm")
    cb = app.data_fit_use_length_cb
    cb.blockSignals(True)
    try:
        if v_tap and v_tap > 0:
            cb.setChecked(True)
            app.data_fit_length_input.setText(f"{float(v_tap):g}")
            app.data_fit_vc_input.setText(f"{DEFAULT_EC_V_PER_CM * 1.0e6:.6g}")
        else:
            cb.setChecked(False)
            # Clear stale value from the previously selected channel so the
            # operator immediately sees this channel has no tap metadata.
            app.data_fit_length_input.setText("")
    finally:
        cb.blockSignals(False)
    _on_use_length_changed(app)


_ENG_PREFIXES = [
    (1e-24, "y"), (1e-21, "z"), (1e-18, "a"), (1e-15, "f"), (1e-12, "p"),
    (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, ""), (1e3, "k"),
    (1e6, "M"), (1e9, "G"), (1e12, "T"), (1e15, "P"), (1e18, "E"),
]


def _format_engineering(value: float, unit: str, decimals: int = 2) -> str:
    """Format value with an SI prefix: 1.23e-8 Ω → "12.30 nΩ"."""
    if value == 0 or not np.isfinite(value):
        return f"{value:.{decimals}f} {unit}".strip()
    import math
    exp = int(math.floor(math.log10(abs(value)) / 3) * 3)
    exp = max(-24, min(18, exp))
    factor = 10 ** exp
    prefix = next((p for mag, p in _ENG_PREFIXES if abs(mag - factor) / factor < 1e-3), f"e{exp}")
    scaled = value / factor
    return f"{scaled:.{decimals}f} {prefix}{unit}".strip()


_ENG_PREFIX_BY_EXP = {
    -24: "y", -21: "z", -18: "a", -15: "f", -12: "p",
    -9: "n", -6: "µ", -3: "m", 0: "",
    3: "k", 6: "M", 9: "G", 12: "T", 15: "P", 18: "E",
}


def _eng_tick_string(value: float) -> str:
    """Format a tick value with an SI prefix but no unit (e.g. 1.2e-6 → "1.2µ").

    Axis labels keep the unit — putting it on every tick clutters the plot.
    """
    import math
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    mag = abs(value)
    exp = int(math.floor(math.log10(mag) / 3) * 3)
    exp = max(-24, min(18, exp))
    prefix = _ENG_PREFIX_BY_EXP.get(exp, f"e{exp}")
    scaled = mag / (10.0 ** exp)
    # 3 sig figs max, trim trailing zeros
    if scaled >= 100:
        txt = f"{scaled:.0f}"
    elif scaled >= 10:
        txt = f"{scaled:.1f}"
    else:
        txt = f"{scaled:.2f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return f"{sign}{txt}{prefix}" if prefix else f"{sign}{txt}"


class EngineeringAxisItem(pg.AxisItem):
    """AxisItem whose tick labels use SI prefixes in both linear and log10 mode.

    In log mode pyqtgraph passes log10(value) to tickStrings; we exponentiate
    back so "-6" becomes "1µ" (= 10⁻⁶). enableAutoSIPrefix is turned off so
    pyqtgraph doesn't also append a global "(x1e-06)" factor to the axis title.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        out = []
        for v in values:
            if getattr(self, "logMode", False):
                try:
                    lin = 10.0 ** float(v)
                except (OverflowError, ValueError):
                    out.append("")
                    continue
            else:
                lin = float(v) * float(scale)
            out.append(_eng_tick_string(lin))
        return out


def _format_result(result) -> str:
    lines = []
    r_name = "Rho" if result.uses_sample_length else "R"
    r_unit = "Ω/cm" if result.uses_sample_length else "Ω"
    v_name = "Ec" if result.uses_sample_length else "Vc"
    v_unit = "V/cm" if result.uses_sample_length else "V"
    is_loglog = getattr(result, "fit_method", FIT_METHOD_NONLINEAR) == FIT_METHOD_LOG_LOG
    method_label = "Log E vs log I (IEC 61788)" if is_loglog else "Non-linear V-I"
    weight_mode = str(getattr(result, "weighting_mode", WEIGHT_MODE_EQUAL) or WEIGHT_MODE_EQUAL)
    weight_label = {
        WEIGHT_MODE_EQUAL: "Equal",
        WEIGHT_MODE_WEIGHTED: "Weighted",
        WEIGHT_MODE_ROBUST: "Robust",
    }.get(weight_mode, weight_mode)
    baseline_mode = str(getattr(result, "baseline_mode", DEFAULT_BASELINE_MODE) or DEFAULT_BASELINE_MODE)
    baseline_label = {
        BASELINE_MODE_OLS: "OLS (legacy)",
        BASELINE_MODE_HUBER: "Huber robust",
        BASELINE_MODE_THEIL_SEN: "Theil-Sen robust",
    }.get(baseline_mode, baseline_mode)
    lines.append(f"method        = {method_label}")
    lines.append(f"weighting     = {weight_label}")
    lines.append(f"baseline mode = {baseline_label}")
    lines.append(f"di/dt         = {_format_engineering(result.di_dt, 'A/s', 2)}")
    # Split the constant baseline into its three physical contributions:
    # V_ofs (thermal) + L·dI/dt (inductive) + R·I (resistive).
    vofs = getattr(result, "V_ofs", 0.0)
    vofs_note = "" if getattr(result, "thermal_offset_applied", False) else "  (not subtracted — no I = 0 segment)"
    lines.append(f"V_ofs         = {_format_engineering(vofs, v_unit, 2)}{vofs_note}")
    lines.append(f"L·di/dt       = {_format_engineering(result.V0, v_unit, 2)}  (= V0 from baseline fit)")
    lines.append(f"L             = {_format_engineering(result.inductance_L, 'H', 2)}  (= L·dI/dt / di_dt)")
    lines.append(f"{r_name:<13} = {_format_engineering(result.R, r_unit, 2)}")
    lines.append(f"R·Ic          = {_format_engineering(result.R * result.Ic, v_unit, 2)}")
    sigma_Ic = getattr(result, "sigma_Ic", 0.0)
    sigma_n = getattr(result, "sigma_n", 0.0)
    if sigma_Ic > 0:
        lines.append(f"Ic            = {result.Ic:.6g} ± {sigma_Ic:.3g} A")
    else:
        lines.append(f"Ic            = {result.Ic:.6g} A")
    if sigma_n > 0:
        lines.append(f"n-value       = {result.n_value:.2f} ± {sigma_n:.2f}")
    else:
        lines.append(f"n-value       = {result.n_value:.2f}")
    lines.append(f"{v_name:<13} = {_format_engineering(result.criterion, v_unit, 2)}")
    r_squared = getattr(result, "r_squared", 0.0)
    if r_squared != 0.0:
        lines.append(f"R²            = {r_squared:.6f}")
    lines.append(f"chi-squared   = {result.chi_sqr:.3g}")
    ratio = getattr(result, "ramp_inductive_ratio", 0.0)
    lines.append(
        f"|L·dI/dt| / (Ec·L_v) = {ratio:.4f}"
        + ("  (ramp too fast!)" if getattr(result, "ramp_too_fast", False) else "")
    )
    if is_loglog:
        lines.append(
            f"n window      = [Ec1={_format_engineering(result.ec1, v_unit, 2)}, "
            f"Ec2={_format_engineering(result.ec2, v_unit, 2)}]"
        )
        lines.append(
            f"I window      = [{result.n_window_I[0]:.4g}, {result.n_window_I[1]:.4g}] A, "
            f"N={result.n_points_used}"
            + ("  (too few — noisy)" if getattr(result, "insufficient_n_points", False) else "")
        )
    else:
        lines.append(f"iterations    = {result.iterations}")
        lines.append(f"Ic history    = [{', '.join(f'{v:.4g}' for v in result.ic_history)}]")
        lines.append(
            f"power window  = [{result.power_fit_window[0]:.4g}, "
            f"{result.power_fit_window[1]:.4g}]"
        )
    lines.append(f"linear window = [{result.linear_fit_window[0]:.4g}, {result.linear_fit_window[1]:.4g}]")
    return "\n".join(lines)


_FIT_PROPERTY_PREFIX = "fit_"
_FIT_PROPERTY_KEYS = (
    "method", "method_compliant", "fit_status", "fit_message",
    "fit_timestamp", "Ic_A", "sigma_Ic_A", "n_value", "sigma_n",
    "r_squared", "chi_squared", "di_dt_A_per_s", "criterion_value",
    "criterion_unit", "criterion_name", "Ec1", "Ec2",
    "n_window_I_lo_A", "n_window_I_hi_A", "n_points_used",
    "V_ofs", "V0_inductive", "inductance_L_H", "R_or_rho", "R_unit",
    "ramp_inductive_ratio", "ramp_too_fast", "insufficient_n_points",
    "thermal_offset_applied", "uses_sample_length",
    "fit_method",
    "weighting_mode",
    "baseline_mode",
)


def _fit_result_properties(result) -> dict:
    """Serialise a FitResult into a flat dict of TDMS-writable properties.

    Failed results are also serialised (with ``fit_status = "failed"`` and the
    originating message) so a user opening the recording later can see that an
    attempt was made and why it didn't produce numbers.
    """
    is_loglog = getattr(result, "fit_method", FIT_METHOD_NONLINEAR) == FIT_METHOD_LOG_LOG
    ok = bool(getattr(result, "ok", False))
    n_window = getattr(result, "n_window_I", (0.0, 0.0)) or (0.0, 0.0)
    props = {
        "method": "IEC 61788 log-log (decade n-value)" if is_loglog else "Non-linear V-I",
        "method_compliant": "IEC 61788-3" if is_loglog else "legacy",
        "fit_status": "ok" if ok else "failed",
        "fit_message": str(getattr(result, "message", "") or ""),
        "fit_timestamp": datetime.now().isoformat(timespec="seconds"),
        # ``method_id`` is the machine-readable identifier ("log_log" /
        # "nonlinear"); ``method`` is the human-readable label. They are
        # kept under different keys so the same-group writer can prefix
        # both with ``fit_`` without collapsing them onto one final key.
        "method_id": str(getattr(result, "fit_method", FIT_METHOD_LOG_LOG)),
        # Primary outputs per IEC 61788 / user request.
        "Ic_A": float(getattr(result, "Ic", 0.0)),
        "sigma_Ic_A": float(getattr(result, "sigma_Ic", 0.0)),
        "n_value": float(getattr(result, "n_value", 0.0)),
        "sigma_n": float(getattr(result, "sigma_n", 0.0)),
        "r_squared": float(getattr(result, "r_squared", 0.0)),
        "chi_squared": float(getattr(result, "chi_sqr", 0.0)),
        "di_dt_A_per_s": float(getattr(result, "di_dt", 0.0)),
        # Criterion: Ec (V/cm) when uses_sample_length, else Vc (V).
        "criterion_value": float(getattr(result, "criterion", 0.0)),
        "criterion_unit": "V/cm" if getattr(result, "uses_sample_length", False) else "V",
        "criterion_name": "Ec" if getattr(result, "uses_sample_length", False) else "Vc",
        # IEC decade window (0 for non-linear fits).
        "Ec1": float(getattr(result, "ec1", 0.0)),
        "Ec2": float(getattr(result, "ec2", 0.0)),
        "n_window_I_lo_A": float(n_window[0]),
        "n_window_I_hi_A": float(n_window[1]),
        "n_points_used": int(getattr(result, "n_points_used", 0)),
        # Baseline decomposition: V_total = V_ofs + L·dI/dt + R·I + Vc·(I/Ic)^n.
        "V_ofs": float(getattr(result, "V_ofs", 0.0)),
        "V0_inductive": float(getattr(result, "V0", 0.0)),
        "inductance_L_H": float(getattr(result, "inductance_L", 0.0)),
        "R_or_rho": float(getattr(result, "R", 0.0)),
        "R_unit": "Ω/cm" if getattr(result, "uses_sample_length", False) else "Ω",
        # Diagnostic flags the user asked to propagate to channel metadata.
        "ramp_inductive_ratio": float(getattr(result, "ramp_inductive_ratio", 0.0)),
        "ramp_too_fast": bool(getattr(result, "ramp_too_fast", False)),
        "insufficient_n_points": bool(getattr(result, "insufficient_n_points", False)),
        "thermal_offset_applied": bool(getattr(result, "thermal_offset_applied", False)),
        "uses_sample_length": bool(getattr(result, "uses_sample_length", False)),
        "weighting_mode": str(getattr(result, "weighting_mode", WEIGHT_MODE_EQUAL)),
        "baseline_mode": str(getattr(result, "baseline_mode", DEFAULT_BASELINE_MODE)),
    }
    # Booleans round-trip more reliably as strings in TDMS consumers (LabVIEW,
    # Origin) — keep human-readable "True"/"False" instead of raw bool.
    for k in ("ramp_too_fast", "insufficient_n_points",
              "thermal_offset_applied", "uses_sample_length"):
        props[k] = "True" if props[k] else "False"
    return props


def _prefix_fit_props(props: dict) -> dict:
    """Prefix per-channel fit properties with ``fit_`` so they don't collide
    with the channel's existing metadata when written into the same group."""
    out = {}
    for key, value in props.items():
        if key.startswith(_FIT_PROPERTY_PREFIX):
            out[key] = value
        else:
            out[f"{_FIT_PROPERTY_PREFIX}{key}"] = value
    return out


def _channel_to_group(tfile, name: str) -> Optional[str]:
    """Find which group a channel belongs to in a TDMS file."""
    for grp in tfile.groups():
        for ch in grp.channels():
            if getattr(ch, "name", "") == name:
                return grp.name
    return None


def _coerce_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_bool(value, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return fallback


def _prop_lookup(props: dict, *names, default=None):
    """Look up a property by any of several names — supports both
    same-group reads (where keys may be ``status`` after the ``fit_``
    prefix has been stripped) and FitResults-group reads (where keys
    keep their original ``fit_status``/``Ic_A`` form).
    """
    for name in names:
        if name in props:
            return props[name]
    return default


def _fit_result_from_props(props: dict):
    """Best-effort reconstruction of a FitResult from saved TDMS metadata.

    Produces a usable result for plotting (``ok``, ``Ic``, ``n_value``, ``R``,
    ``V0``, ``criterion``, ``uses_sample_length``) and synthesises a smooth
    fit curve on the same V-I model the run-time fitter uses, so re-opening a
    file shows the same overlay the user saw on the original run.
    """
    from .service import FitResult, FIT_METHOD_LOG_LOG, FIT_METHOD_NONLINEAR
    method = str(_prop_lookup(props, "fit_method", "method_id") or "").strip()
    if method not in (FIT_METHOD_LOG_LOG, FIT_METHOD_NONLINEAR):
        # Older recordings stored the human-readable label only; infer the
        # method from the descriptive "method" property.
        descriptive = str(_prop_lookup(props, "method") or "")
        method = FIT_METHOD_LOG_LOG if "log-log" in descriptive.lower() else FIT_METHOD_NONLINEAR
    uses_length = _coerce_bool(_prop_lookup(props, "uses_sample_length"), False)
    n_window = (
        _coerce_float(_prop_lookup(props, "n_window_I_lo_A"), 0.0),
        _coerce_float(_prop_lookup(props, "n_window_I_hi_A"), 0.0),
    )
    status = str(_prop_lookup(props, "fit_status", "status", default="ok")).lower()
    ok = status == "ok"
    return FitResult(
        ok=ok,
        message=str(_prop_lookup(props, "fit_message", "message") or ""),
        di_dt=_coerce_float(_prop_lookup(props, "di_dt_A_per_s")),
        inductance_L=_coerce_float(_prop_lookup(props, "inductance_L_H")),
        V_ofs=_coerce_float(_prop_lookup(props, "V_ofs")),
        V0=_coerce_float(_prop_lookup(props, "V0_inductive")),
        R=_coerce_float(_prop_lookup(props, "R_or_rho")),
        Ic=_coerce_float(_prop_lookup(props, "Ic_A")),
        n_value=_coerce_float(_prop_lookup(props, "n_value")),
        criterion=_coerce_float(_prop_lookup(props, "criterion_value")),
        chi_sqr=_coerce_float(_prop_lookup(props, "chi_squared")),
        linear_fit_window=(0.0, 0.0),
        power_fit_window=n_window,
        uses_sample_length=uses_length,
        fit_method=method,
        ec1=_coerce_float(_prop_lookup(props, "Ec1")),
        ec2=_coerce_float(_prop_lookup(props, "Ec2")),
        n_window_I=n_window,
        n_points_used=int(_coerce_float(_prop_lookup(props, "n_points_used"), 0)),
        sigma_Ic=_coerce_float(_prop_lookup(props, "sigma_Ic_A")),
        sigma_n=_coerce_float(_prop_lookup(props, "sigma_n")),
        r_squared=_coerce_float(_prop_lookup(props, "r_squared")),
        ramp_inductive_ratio=_coerce_float(_prop_lookup(props, "ramp_inductive_ratio")),
        ramp_too_fast=_coerce_bool(_prop_lookup(props, "ramp_too_fast")),
        insufficient_n_points=_coerce_bool(_prop_lookup(props, "insufficient_n_points")),
        thermal_offset_applied=_coerce_bool(_prop_lookup(props, "thermal_offset_applied")),
        weighting_mode=str(_prop_lookup(props, "weighting_mode", default=WEIGHT_MODE_EQUAL) or WEIGHT_MODE_EQUAL),
        baseline_mode=str(_prop_lookup(props, "baseline_mode", default=DEFAULT_BASELINE_MODE) or DEFAULT_BASELINE_MODE),
    )


def _build_fit_curve(result, x_data, *, length_cm: Optional[float] = None):
    """Reconstruct a smooth (fit_x, fit_y) curve from a FitResult.

    Mirrors the model used by ``run_full_fit`` so re-loaded fits overlay
    cleanly on the source data. Returns ``(None, None)`` when the result
    parameters are degenerate (e.g. failed-fit metadata).
    """
    if not getattr(result, "ok", False):
        return None, None
    Ic = float(getattr(result, "Ic", 0.0))
    if Ic <= 0:
        return None, None
    x_arr = np.asarray(x_data, dtype=float)
    if x_arr.size < 2:
        return None, None
    # Cover the full source range so callers that resample the smooth
    # curve onto x_arr (np.interp) don't get NaN at the endpoints. The
    # power-law term is well-defined at I = 0 (clipped to 1e-30/Ic ≈ 0)
    # so we no longer need to clamp x_min away from zero.
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    if x_max <= x_min:
        return None, None
    fit_x = np.linspace(x_min, x_max, 400)
    crit = float(getattr(result, "criterion", 0.0))
    V0 = float(getattr(result, "V0", 0.0))
    R = float(getattr(result, "R", 0.0))
    n_val = float(getattr(result, "n_value", 0.0))
    # ``np.maximum`` keeps the ratio non-negative so fractional powers
    # behave for the rare cases where x dips below zero on noisy data.
    ratio = np.maximum(fit_x / Ic, 0.0)
    fit_y = V0 + R * fit_x + crit * np.power(ratio, n_val)
    if getattr(result, "thermal_offset_applied", False):
        fit_y = fit_y + float(getattr(result, "V_ofs", 0.0))
    # If the saved fit was per-unit-length but the user is loading without
    # the voltage-tap path active, leave the curve in the same units the
    # caller plotted x_data in. The caller has already applied the correct
    # transform via ``_apply_transforms``.
    return fit_x, fit_y


def _write_fit_report_same_group(report_path: Path,
                                 new_entries: dict[str, dict]) -> Optional[str]:
    """Attach fit properties as channel metadata on the matching source channels.

    Reads the source TDMS, merges each curve's fit properties into the
    properties of the channel sharing its label, and rewrites the file with
    the original groups, channels and data preserved. Properties from prior
    fits on the same channel are overwritten; properties on channels that
    weren't fitted in this run are kept untouched.

    Channels whose label has no match in the file (e.g. user renamed a curve
    before fitting) fall back to a ``FitResults`` group inside the source
    file, so nothing is silently dropped.
    """
    if not report_path.exists():
        return None
    try:
        existing_groups: list[GroupObject] = []
        existing_channels: list[ChannelObject] = []
        unmatched: dict[str, dict] = dict(new_entries)
        with TdmsFile.read(str(report_path)) as tfile:
            for grp in tfile.groups():
                if grp.name == "FitResults":
                    continue
                existing_groups.append(GroupObject(grp.name, properties=dict(grp.properties)))
                for ch in grp.channels():
                    props = dict(ch.properties)
                    fit_props = unmatched.pop(ch.name, None)
                    if fit_props is not None:
                        props.update(_prefix_fit_props(fit_props))
                    existing_channels.append(
                        ChannelObject(grp.name, ch.name,
                                      np.asarray(ch[:]),
                                      properties=props)
                    )
        out_objects: list = list(existing_groups) + list(existing_channels)
        if unmatched:
            out_objects.append(GroupObject("FitResults"))
            for name, props in unmatched.items():
                data = np.array([np.nan], dtype=np.float64)
                out_objects.append(ChannelObject("FitResults", name, data, properties=props))
        with TdmsWriter(str(report_path)) as writer:
            writer.write_segment(out_objects)
        return str(report_path)
    except Exception as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        return None


def _write_fit_report_tdms(app, results: list[tuple[str, object]],
                           *, force: bool = False) -> Optional[str]:
    """Persist fit results into the source TDMS or a side-car file.

    Behaviour follows the checkboxes in the Settings dialog:

    * ``Automatically save fitting parameters as metadata`` (default on) —
      gates the writer entirely. When off, post-fit autosave is skipped
      (used by ``run_fit``); the manual ``Save metadata`` button passes
      ``force=True`` to bypass this gate.
    * ``Save fit results to a separate TDMS file`` → write into
      ``<source>_fit_report.tdms`` next to the source. Always uses a
      ``FitResults`` group regardless of the same-group toggle.
    * ``Use the same group/channel for fit metadata`` (default) → attach
      the fit properties to the fitted voltage channel itself, in its
      original group; no extra ``FitResults`` group is created. When
      unchecked, write a ``FitResults`` group into the source TDMS as
      before.

    Failed fits are also persisted (with ``fit_status = "failed"``) so users
    can see that an attempt was made.

    Returns the path written, or ``None`` if there's nothing to write, the
    source TDMS is unknown, or autosave is off (and ``force`` is False).
    I/O errors are logged and swallowed.
    """
    controller = getattr(app, "data_fit_controller", None)
    src_path = getattr(controller, "tdms_path", "") if controller is not None else ""
    persistent = [(lbl, r) for lbl, r in results if r is not None]
    if not src_path or not persistent:
        return None
    autosave_on = bool(
        _safe_checkbox_checked(app, "data_fit_autosave_cb", default=True)
    )
    if not autosave_on and not force:
        return None
    src = Path(src_path)
    save_separate = bool(
        getattr(app, "data_fit_save_separate_cb", None) is not None
        and app.data_fit_save_separate_cb.isChecked()
    )
    same_group = bool(
        getattr(app, "data_fit_same_group_cb", None) is None
        or app.data_fit_same_group_cb.isChecked()
    )
    report_path = src.with_name(f"{src.stem}_fit_report.tdms") if save_separate else src

    new_entries = {lbl: _fit_result_properties(r) for lbl, r in persistent}

    if not save_separate and same_group:
        return _write_fit_report_same_group(report_path, new_entries)

    if save_separate:
        # Preserve FitResults channels from prior runs that this fit didn't touch
        # by reading the existing side-car and rewriting it with merged entries.
        existing: dict[str, dict] = {}
        if report_path.exists():
            try:
                with TdmsFile.read(str(report_path)) as tfile:
                    for grp in tfile.groups():
                        if grp.name != "FitResults":
                            continue
                        for ch in grp.channels():
                            existing[ch.name] = dict(ch.properties)
            except Exception as exc:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                existing = {}
        merged = {**existing, **new_entries}
        objects = [GroupObject("FitResults")]
        for name, props in merged.items():
            data = np.array([np.nan], dtype=np.float64)
            objects.append(ChannelObject("FitResults", name, data, properties=props))
        try:
            with TdmsWriter(str(report_path)) as writer:
                writer.write_segment(objects)
        except Exception as exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            return None
        return str(report_path)

    # Append a FitResults segment to the source TDMS. nptdms' append mode
    # adds a new segment; readers see the latest properties, which is the
    # overwrite semantics requested for re-runs.
    objects = [GroupObject("FitResults")]
    for name, props in new_entries.items():
        data = np.array([np.nan], dtype=np.float64)
        objects.append(ChannelObject("FitResults", name, data, properties=props))
    try:
        with TdmsWriter(str(report_path), mode="a") as writer:
            writer.write_segment(objects)
    except TypeError:
        # Older nptdms versions don't support mode='a' on TdmsWriter; fall
        # back to rewriting the file with merged contents preserved.
        try:
            existing_groups: list[GroupObject] = []
            existing_channels: list[ChannelObject] = []
            existing_fit: dict[str, dict] = {}
            if report_path.exists():
                with TdmsFile.read(str(report_path)) as tfile:
                    for grp in tfile.groups():
                        if grp.name == "FitResults":
                            for ch in grp.channels():
                                existing_fit[ch.name] = dict(ch.properties)
                            continue
                        existing_groups.append(GroupObject(grp.name, properties=dict(grp.properties)))
                        for ch in grp.channels():
                            existing_channels.append(
                                ChannelObject(grp.name, ch.name, np.asarray(ch[:]),
                                              properties=dict(ch.properties))
                            )
            merged = {**existing_fit, **new_entries}
            out_objects = list(existing_groups) + list(existing_channels)
            out_objects.append(GroupObject("FitResults"))
            for name, props in merged.items():
                data = np.array([np.nan], dtype=np.float64)
                out_objects.append(ChannelObject("FitResults", name, data, properties=props))
            with TdmsWriter(str(report_path)) as writer:
                writer.write_segment(out_objects)
        except Exception as exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            return None
    except Exception as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        return None
    return str(report_path)


def run_fit(app):
    controller = app.data_fit_controller
    app.data_fit_power_window_manual = False

    def _loglog_window_fields_present() -> bool:
        try:
            lo_txt = (app.data_fit_power_low_x.text() or "").strip()
            hi_txt = (app.data_fit_power_high_x.text() or "").strip()
            if not lo_txt or not hi_txt:
                return False
            lo_v = float(lo_txt)
            hi_v = float(hi_txt)
        except (TypeError, ValueError):
            return False
        return np.isfinite(lo_v) and np.isfinite(hi_v) and hi_v > lo_v

    def _recompute_loglog_i_window_for_entry(result_obj, entry_obj, entry_settings_obj) -> None:
        """Recompute IEC I-window for one fitted curve and persist to its profile."""
        if getattr(result_obj, "fit_method", "") != FIT_METHOD_LOG_LOG:
            return
        x_raw = np.asarray(entry_obj.get("x", []), dtype=float)
        y_raw = np.asarray(entry_obj.get("y", []), dtype=float)
        n = int(min(x_raw.size, y_raw.size))
        if n == 0:
            return
        x_arr = x_raw[:n]
        y_arr = y_raw[:n]
        y_corr = y_arr - (float(result_obj.V0) + float(result_obj.R) * x_arr)
        y_sm = _adaptive_smooth_visual(y_corr, float(entry_settings_obj.ec1), float(entry_settings_obj.ec2))

        valid = np.isfinite(x_arr) & np.isfinite(y_sm) & (x_arr > 0)
        if not np.any(valid):
            return
        x_arr = x_arr[valid]
        y_sm = y_sm[valid]
        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_sm = y_sm[order]
        ec1 = max(float(entry_settings_obj.ec1), 1.0e-30)
        ec2 = max(float(entry_settings_obj.ec2), ec1 * 1.000001)
        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        span = max(0.0, x_max - x_min)
        x_lo, x_hi = pick_loglog_i_window_from_thresholds(
            x_arr, y_sm, ec1=ec1, ec2=ec2, guard_fraction=DEFAULT_EC_WINDOW_GUARD_FRAC,
        )
        if not np.isfinite(x_lo):
            x_lo = x_min
        if not np.isfinite(x_hi):
            x_hi = x_max
        if x_hi <= x_lo:
            x_hi = x_lo + max(1e-12, 0.01 * (span if span > 0 else 1.0))

        result_obj.n_window_I = (float(x_lo), float(x_hi))
        result_obj.power_fit_window = (float(x_lo), float(x_hi))
        profiles = getattr(app, "data_fit_curve_profiles", {}) or {}
        key = _profile_key_for_entry(entry_obj)
        profile = dict(profiles.get(key, {})) if isinstance(profiles, dict) else {}
        profile["power_low_x"] = f"{x_lo:.6g}"
        profile["power_high_x"] = f"{x_hi:.6g}"
        if isinstance(profiles, dict):
            profiles[key] = profile
            app.data_fit_curve_profiles = profiles

    def _apply_step4_window_from_reference(result_obj) -> None:
        """For log-log fits, recompute Step-4 window from the same
        corrected+smoothed reference used by the Add/Show helper curve."""
        if getattr(result_obj, "fit_method", "") != FIT_METHOD_LOG_LOG:
            return
        controller.last_result = result_obj
        if not _update_loglog_power_x_from_ec(app, auto_run_fit=False):
            return
        try:
            lo_w = float(app.data_fit_power_low_x.text())
            hi_w = float(app.data_fit_power_high_x.text())
        except (TypeError, ValueError):
            return
        if np.isfinite(lo_w) and np.isfinite(hi_w) and hi_w > lo_w:
            result_obj.n_window_I = (float(lo_w), float(hi_w))
            result_obj.power_fit_window = (float(lo_w), float(hi_w))
    if not controller.channel_names:
        QMessageBox.warning(app, "Data Fitting", "Load a recording first.")
        return
    try:
        # Live UI snapshot — used for the single-curve fallback and as the
        # last-resort settings source for any entry without a saved profile.
        settings = _settings_from_inputs(app)
    except Exception as exc:
        QMessageBox.critical(app, "Data Fitting", f"Invalid input: {exc}")
        return

    # Multi-curve mode: fit only curves explicitly marked "include in fit".
    curves = getattr(app, "data_fit_curves", [])
    included = [c for c in curves if c.get("include_in_fit", True)]
    transformed = _apply_transforms(app)
    has_preview = bool(getattr(app, "data_fit_preview_visible", True))
    preview_included = bool(getattr(app, "data_fit_preview_include_in_fit", True))
    if has_preview and preview_included:
        px = transformed.get("x")
        py = transformed.get("y")
        pt = transformed.get("time")
        if px is not None and py is not None and pt is not None and px.size and py.size and pt.size:
            included = [{
                # Tag the preview entry with the well-known profile key so
                # _settings_for_entry can look up its per-curve fit memory.
                "signature": "__preview__",
                "label": app.data_fit_y_cb.currentText(),
                "t": pt,
                "x": px,
                "y": py,
            }] + included
    has_any_plot_entry = has_preview or bool(curves)
    if has_any_plot_entry and not included:
        QMessageBox.warning(app, "Data Fitting", "No curve is selected in 'Include in fit'.")
        return
    if included:
        lines = []
        last_ok = None
        last_ok_settings: Optional[FitSettings] = None
        ok_results = []
        all_results: list[tuple[str, object]] = []
        last_show_criterion = True
        last_show_ic = False
        for entry in included:
            label = entry.get("label", "Curve")
            # Each curve carries its own fit method / windows / criterion via
            # the per-curve profile stored in app.data_fit_curve_profiles.
            try:
                entry_settings = _settings_for_entry(app, entry)
            except Exception as exc:
                traceback.print_exc()
                from .service import FitResult
                result = FitResult(ok=False, message=f"Bad settings for [{label}]: {exc}")
                entry["fit_result"] = result
                all_results.append((label, result))
                lines.append(f"[{label}] FIT FAILED: {result.message}")
                continue
            try:
                result = run_full_fit(entry["t"], entry["x"], entry["y"], entry_settings)
            except Exception as exc:
                traceback.print_exc()
                # Synthesise a failed FitResult so the channel still appears
                # in the report with the exception text — matches the user's
                # request to record an attempt even on total failure.
                from .service import FitResult
                result = FitResult(ok=False, message=f"Fit raised: {exc}")
            entry["fit_result"] = result
            all_results.append((label, result))
            if result.ok:
                _recompute_loglog_i_window_for_entry(result, entry, entry_settings)
                last_ok = result
                last_ok_settings = entry_settings
                ok_results.append((label, result))
                last_show_criterion = bool(entry.get("show_criterion", False))
                last_show_ic = bool(entry.get("show_ic", False))
                _upsert_fit_curve_entry(app, entry, result)
                lines.append(f"[{label}]\n" + _format_result(result) + "\n")
            else:
                lines.append(f"[{label}] FIT FAILED: {result.message}")
        app.data_fit_result_text.setPlainText("\n".join(lines) or "No curves included in fit.")
        if last_ok is not None:
            controller.last_result = last_ok
            if getattr(last_ok, "fit_method", "") == FIT_METHOD_LOG_LOG:
                n_window = getattr(last_ok, "n_window_I", None) or (0.0, 0.0)
                try:
                    lo_w = float(n_window[0])
                    hi_w = float(n_window[1])
                except (TypeError, ValueError, IndexError):
                    lo_w = hi_w = 0.0
                if (not _loglog_window_fields_present()) and np.isfinite(lo_w) and np.isfinite(hi_w) and hi_w > lo_w:
                    _set_silently(app.data_fit_power_low_x, f"{lo_w:.6g}")
                    _set_silently(app.data_fit_power_high_x, f"{hi_w:.6g}")
                    app.data_fit_power_window_manual = False
            _show_fit_overlays(
                app, last_ok, table_entries=ok_results,
                show_criterion=last_show_criterion, show_ic=last_show_ic,
            )
            _plot_residuals(app, last_ok)
            _post_fit_warnings(app, last_ok, last_ok_settings or settings)
        else:
            _hide_fit_overlays(app)
            _show_warning(app, "No curve produced a successful fit.", severity="error")
        # Any fit attempt — successful or not — enables the manual
        # "Save metadata" button so the user can re-persist the result
        # after toggling layout in Settings without re-running the fit.
        if hasattr(app, "data_fit_save_metadata_btn"):
            app.data_fit_save_metadata_btn.setEnabled(True)
        # Cache results on the controller so the manual Save metadata
        # button can find them even when the fit ran against the preview
        # (no entry stored in ``data_fit_curves``).
        controller_obj = getattr(app, "data_fit_controller", None)
        if controller_obj is not None:
            controller_obj.last_fit_results = list(all_results)
            if last_ok is not None:
                controller_obj.last_result = last_ok
        # Persist every attempt — including failed ones — so users can see a
        # paper trail in the TDMS even when the fit didn't converge. The
        # writer respects the Autosave toggle in Settings; a no-op return
        # here is normal when autosave is off.
        report_path = _write_fit_report_tdms(app, all_results)
        if report_path:
            current = app.data_fit_result_text.toPlainText()
            app.data_fit_result_text.setPlainText(
                current + f"\nFit report written to: {report_path}"
            )
        return

    # Single-curve mode: use current channel selection.
    t_scale, t_offset = _scale_offset_from_inputs(app, "time")
    x_scale, x_offset = _scale_offset_from_inputs(app, "x")
    y_scale, y_offset = _scale_offset_from_inputs(app, "y")
    try:
        result, msg = controller.compute_fit(
            time_sig=app.data_fit_time_cb.currentText(),
            x_sig=app.data_fit_x_cb.currentText(),
            y_sig=app.data_fit_y_cb.currentText(),
            x_scale=x_scale, x_offset=x_offset,
            y_scale=y_scale, y_offset=y_offset,
            t_scale=t_scale, t_offset=t_offset,
            settings=settings,
            avg_window=_active_avg_window(app),
        )
    except Exception as exc:
        traceback.print_exc()
        QMessageBox.critical(app, "Data Fitting", f"Fit failed: {exc}")
        return
    if result is None or not result.ok:
        app.data_fit_result_text.setPlainText(msg or "Fit failed.")
        app.data_fit_model_curve.setData([], [])
        _hide_fit_overlays(app)
        _show_warning(app, msg or "Fit failed.", severity="error")
        if result is not None:
            if hasattr(app, "data_fit_save_metadata_btn"):
                app.data_fit_save_metadata_btn.setEnabled(True)
            label = app.data_fit_y_cb.currentText() or "Curve"
            controller.last_fit_results = [(label, result)]
            controller.last_result = result
            # Even a totally failed fit leaves a record in the TDMS so the
            # user can tell, when re-opening the file later, that an attempt
            # was made and what error it produced.
            failed_path = _write_fit_report_tdms(
                app, [(label, result)]
            )
            if failed_path:
                current = app.data_fit_result_text.toPlainText()
                app.data_fit_result_text.setPlainText(
                    current + f"\nFailed-fit metadata written to: {failed_path}"
                )
        return
    controller.last_result = result
    if not _loglog_window_fields_present():
        _apply_step4_window_from_reference(result)
    app.data_fit_result_text.setPlainText(_format_result(result))
    if getattr(result, "fit_method", "") == FIT_METHOD_LOG_LOG:
        n_window = getattr(result, "n_window_I", None) or (0.0, 0.0)
        try:
            lo_w = float(n_window[0])
            hi_w = float(n_window[1])
        except (TypeError, ValueError, IndexError):
            lo_w = hi_w = 0.0
        if (not _loglog_window_fields_present()) and np.isfinite(lo_w) and np.isfinite(hi_w) and hi_w > lo_w:
            _set_silently(app.data_fit_power_low_x, f"{lo_w:.6g}")
            _set_silently(app.data_fit_power_high_x, f"{hi_w:.6g}")
            app.data_fit_power_window_manual = False
    if result.fit_x is not None and result.fit_y is not None:
        app.data_fit_model_curve.setData(result.fit_x, result.fit_y)
    _upsert_fit_curve_entry(
        app,
        {
            "signature": ("__single__", app.data_fit_y_cb.currentText()),
            "label": app.data_fit_y_cb.currentText(),
            "color": getattr(app, "data_fit_preview_color", "#888"),
            "alpha_pct": int(getattr(app, "data_fit_preview_alpha_pct", 100)),
        },
        result,
    )
    _show_fit_overlays(
        app, result,
        table_entries=[(app.data_fit_y_cb.currentText(), result)],
        show_criterion=True,
        show_ic=False,
    )
    _plot_residuals(app, result)
    _post_fit_warnings(app, result, settings)
    if hasattr(app, "data_fit_save_metadata_btn"):
        app.data_fit_save_metadata_btn.setEnabled(True)
    # Cache the single-curve result so manual Save metadata can re-persist
    # it after the user toggles the layout in Settings.
    label = app.data_fit_y_cb.currentText() or "Curve"
    controller.last_fit_results = [(label, result)]
    controller.last_result = result
    report_path = _write_fit_report_tdms(
        app, [(label, result)]
    )
    if report_path:
        current = app.data_fit_result_text.toPlainText()
        app.data_fit_result_text.setPlainText(
            current + f"\nFit report written to: {report_path}"
        )


def _hide_fit_overlays(app) -> None:
    app.data_fit_ic_line.setVisible(False)
    app.data_fit_criterion_line.setVisible(False)
    # Ec1/Ec2 lines stay driven by _update_fit_bands (they reflect the Step 4
    # editors, not a fit result), so we don't force them off here.
    app.data_fit_resid_plot.setVisible(False)
    app.data_fit_resid_curve.setData([], [])
    if hasattr(app, "data_fit_param_table"):
        app.data_fit_param_table.setVisible(False)
        app.data_fit_param_table.clear_parameters()


def _show_fit_overlays(
    app,
    result,
    table_entries: Optional[list[tuple[str, object]]] = None,
    show_criterion: bool = True,
    show_ic: bool = False,
) -> None:
    ic_label = f"Ic = {result.Ic:.6g} A"
    crit_unit = "V/cm" if result.uses_sample_length else "V"
    crit_name = "Ec" if result.uses_sample_length else "Vc"
    crit_label = f"{crit_name} = {_format_engineering(result.criterion, crit_unit, 2)}"
    is_loglog = _plot_is_loglog(app)
    app.data_fit_ic_line.setValue(_xform_for_view(result.Ic, is_loglog))
    app.data_fit_ic_line.label.setText(ic_label)
    app.data_fit_ic_line.setVisible(bool(show_ic))
    y_level = result.V0 + result.R * result.Ic + result.criterion
    app.data_fit_criterion_line.setValue(_xform_for_view(y_level, is_loglog))
    app.data_fit_criterion_line.label.setText(crit_label)
    app.data_fit_criterion_line.setVisible(bool(show_criterion))
    # Parameter table overlay.
    table = app.data_fit_param_table
    if table_entries is None:
        table.set_parameters(result)
    else:
        table.set_parameters_for_curves(table_entries)
    # Place near the top-left of the view on first show, then leave where the user drags it.
    if table.pos().x() == 0 and table.pos().y() == 0:
        vb = app.data_fit_plot.getPlotItem().getViewBox()
        vr = vb.viewRange()
        table.setPos(vr[0][0], vr[1][1])
    table.setVisible(True)


def _plot_residuals(app, result) -> None:
    if not getattr(app.data_fit_graph_settings, "show_residuals", False):
        app.data_fit_resid_plot.setVisible(False)
        app.data_fit_resid_curve.setData([], [])
        return
    transformed = _apply_transforms(app)
    x = transformed["x"]
    y = transformed["y"]
    if x is None or y is None or x.size == 0 or y.size == 0:
        app.data_fit_resid_plot.setVisible(False)
        return
    lo, hi = result.power_fit_window
    mask = (x >= lo) & (x <= hi)
    xm = x[mask]
    ym = y[mask]
    if xm.size == 0:
        app.data_fit_resid_plot.setVisible(False)
        return
    model = result.V0 + result.R * xm + result.criterion * np.power(
        np.clip(xm / result.Ic, 1e-30, None), result.n_value,
    )
    app.data_fit_resid_curve.setData(xm, ym - model)
    app.data_fit_resid_plot.setVisible(True)


def _show_warning(app, text: str, *, severity: str = "warning") -> None:
    label = getattr(app, "data_fit_warning_label", None)
    if label is None:
        return
    if severity == "error":
        label.setStyleSheet(
            "background-color: #ffd6d6; color: #7a0000; border: 1px solid #cc0000; padding: 6px;"
        )
    else:
        label.setStyleSheet(
            "background-color: #fff7c2; color: #665200; border: 1px solid #e6cc00; padding: 6px;"
        )
    label.setText(text)
    label.setVisible(bool(text))


def _clear_warning(app) -> None:
    _show_warning(app, "")


def _post_fit_warnings(app, result, settings) -> None:
    warnings = []
    is_loglog = getattr(result, "fit_method", FIT_METHOD_NONLINEAR) == FIT_METHOD_LOG_LOG
    if not is_loglog and result.iterations >= settings.max_iterations:
        warnings.append(
            f"Fit reached the iteration cap ({settings.max_iterations}) without "
            f"meeting the Ic tolerance ({settings.ic_tolerance * 100:.3g}%)."
        )
    # IEC-specific point-count check (Step 4, log-log method).
    if is_loglog and getattr(result, "insufficient_n_points", False):
        warnings.append(
            f"Only {result.n_points_used} samples fall inside the IEC n-value "
            f"window [I(Ec1), I(Ec2)] — below the recommended minimum of "
            f"{MIN_N_WINDOW_POINTS}. Slow the ramp or lower the averaging for a "
            f"less noisy n estimate."
        )
    # Quasi-static ramp check (applies to both fit methods).
    if getattr(result, "ramp_too_fast", False):
        ratio = getattr(result, "ramp_inductive_ratio", 0.0)
        warnings.append(
            f"|L·dI/dt| / (Ec·L_v) = {ratio:.3f} > "
            f"{RAMP_INDUCTIVE_WARN_RATIO:.2f}: the inductive voltage drop is "
            f"more than 10 % of the Ic criterion voltage. IEC 61788 expects a "
            f"quasi-static measurement — slow the ramp to reduce this ratio."
        )
    # Non-linear: fall back to generic too-few-samples warning on the power window.
    if not is_loglog:
        lo, hi = result.power_fit_window
        transformed = _apply_transforms(app)
        x = transformed["x"]
        if x is not None and x.size:
            mask = (x >= lo) & (x <= hi)
            n_points = int(np.count_nonzero(mask))
            if n_points < 30:
                warnings.append(
                    f"Power-law window contains only {n_points} samples — consider "
                    f"widening it or lowering the averaging factor for a more reliable fit."
                )
    if not warnings:
        _clear_warning(app)
        return
    _show_warning(app, " \n".join(warnings), severity="warning")


def _open_graph_settings(app) -> None:
    def _apply_live(settings: GraphSettings) -> None:
        app.data_fit_graph_settings = settings
        refresh_preview(app)

    dialog = GraphSettingsDialog(app.data_fit_graph_settings, app, on_apply=_apply_live)
    if dialog.exec_() == dialog.Accepted:
        app.data_fit_graph_settings = dialog.result_settings()
        refresh_preview(app)


def _open_export_dialog(app) -> None:
    default_dir = _preset_dir(app)
    dialog = ExportPlotDialog(app.data_fit_plot.getPlotItem(), app, default_dir)
    dialog.exec_()


def _refresh_save_settings_enabled(app) -> None:
    """Apply the dependency rules between the four metadata-saving checkboxes.

    Rules:
    * Autosave unchecked → save-separate and same-group are greyed out: the
      writer never runs in that mode, so the layout choice is moot.
    * Save-separate checked → same-group is greyed out: the side-car file
      always uses the FitResults group regardless of the same-group setting.
    """
    autosave_cb = getattr(app, "data_fit_autosave_cb", None)
    save_separate_cb = getattr(app, "data_fit_save_separate_cb", None)
    same_group_cb = getattr(app, "data_fit_same_group_cb", None)
    if autosave_cb is None or save_separate_cb is None or same_group_cb is None:
        return
    autosave_on = bool(autosave_cb.isChecked())
    save_separate_on = bool(save_separate_cb.isChecked())
    save_separate_cb.setEnabled(autosave_on)
    same_group_cb.setEnabled(autosave_on and not save_separate_on)


def _open_settings_dialog(app) -> None:
    """Open the Data Fitting settings dialog.

    The dialog hosts the four metadata-saving checkboxes that used to clutter
    the main Data Fitting panel. The QCheckBox instances live on ``app`` so
    the rest of the tab can read their state without a reference to the
    dialog; the dialog only re-parents them for the duration of its lifetime
    and returns them to the main panel as hidden widgets when it closes.
    """
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QPushButton, QHBoxLayout

    dialog = QDialog(app)
    dialog.setWindowTitle("Data Fitting — Settings")
    dialog.setModal(True)
    dialog.resize(520, 280)

    root = QVBoxLayout(dialog)

    load_group = QGroupBox("Loading")
    load_layout = QVBoxLayout(load_group)
    load_layout.addWidget(app.data_fit_auto_load_cb)
    root.addWidget(load_group)

    save_group = QGroupBox("Saving fit metadata")
    save_layout = QVBoxLayout(save_group)
    save_layout.addWidget(app.data_fit_autosave_cb)
    save_layout.addWidget(app.data_fit_save_separate_cb)
    save_layout.addWidget(app.data_fit_same_group_cb)
    root.addWidget(save_group)

    # Re-evaluate dependency rules every time the dialog opens, in case the
    # user has changed checkbox state via a preset since the last open.
    _refresh_save_settings_enabled(app)

    button_row = QHBoxLayout()
    button_row.addStretch(1)
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.accept)
    button_row.addWidget(close_btn)
    root.addLayout(button_row)

    dialog.exec_()


def _save_metadata_clicked(app) -> None:
    """Manual handler for the 'Save metadata' button next to Run Fit.

    Persists the most recent fit attempt for each curve into the loaded
    TDMS file, following the layout selected in the Settings dialog
    (separate file vs. same group/channel). No-op when there's no fit to
    save — the button is disabled in that case but the guard is here too
    so programmatic callers stay safe.
    """
    if not _collect_persisted_fit_results(app):
        QMessageBox.information(
            app, "Data Fitting",
            "No fit results to save yet. Run a fit first.",
        )
        return
    results = _collect_persisted_fit_results(app)
    report_path = _write_fit_report_tdms(app, results, force=True)
    if report_path:
        existing = app.data_fit_result_text.toPlainText()
        suffix = f"\nFit metadata saved to: {report_path}"
        if suffix.strip() not in existing:
            app.data_fit_result_text.setPlainText(existing + suffix)
        _show_warning(
            app, f"Fit metadata saved to {report_path}", severity="warning",
        )
    else:
        QMessageBox.warning(
            app, "Data Fitting",
            "Could not save fit metadata. Check that the source TDMS still exists "
            "and is writable.",
        )


def _collect_persisted_fit_results(app) -> list[tuple[str, object]]:
    """Return the fit results to persist when the user clicks Save metadata.

    Priority order:
    1. ``controller.last_fit_results`` — the (label, FitResult) tuples cached
       by the most recent ``run_fit`` call. This is the authoritative source
       because it covers preview-only fits that don't get a stored entry in
       ``data_fit_curves``.
    2. Stored curves with attached ``fit_result`` (excluding the
       ``is_fit_result`` overlay rows).
    3. ``controller.last_result`` as a final fallback for very old code paths.
    """
    controller = getattr(app, "data_fit_controller", None)
    cached = list(getattr(controller, "last_fit_results", []) or []) if controller is not None else []
    if cached:
        return cached
    seen_labels: set[str] = set()
    out: list[tuple[str, object]] = []
    for entry in getattr(app, "data_fit_curves", []) or []:
        if bool(entry.get("is_fit_result", False)):
            continue
        result = entry.get("fit_result")
        if result is None:
            continue
        label = str(entry.get("label") or "Curve")
        if label in seen_labels:
            continue
        seen_labels.add(label)
        out.append((label, result))
    last_result = getattr(controller, "last_result", None) if controller is not None else None
    if not out and last_result is not None:
        out.append((app.data_fit_y_cb.currentText() or "Curve", last_result))
    return out


# ----------------------------------------------------------------------------
# Multi-curve plot workflow
# ----------------------------------------------------------------------------

_PASTEL_COLORS = [
    "#8ecae6", "#ffb4a2", "#b8e0d2", "#ffd6a5", "#cdb4db",
    "#c1fba4", "#ffc8dd", "#a0c4ff", "#fde68a", "#e9c46a",
]


def _curve_signature(app) -> tuple:
    """A tuple uniquely identifying a plot entry's source + transform.

    Two "Plot" clicks that would produce identical curves are considered duplicates.
    """
    return (
        app.data_fit_time_cb.currentText(),
        app.data_fit_x_cb.currentText(),
        app.data_fit_y_cb.currentText(),
        _float_from(app.data_fit_x_scale, 1.0),
        _float_from(app.data_fit_x_offset, 0.0),
        _float_from(app.data_fit_y_scale, 1.0),
        _float_from(app.data_fit_y_offset, 0.0),
        _float_from(app.data_fit_time_scale, 1.0),
        _float_from(app.data_fit_time_offset, 0.0),
        _active_avg_window(app),
        _active_sample_length(app),
    )


def _next_pastel_color(app) -> str:
    n = len(getattr(app, "data_fit_curves", []))
    return _PASTEL_COLORS[n % len(_PASTEL_COLORS)]


def _entry_color_qcolor(entry: dict) -> QColor:
    base = pg.mkColor(entry.get("color", "#1f77b4"))
    alpha_pct = int(entry.get("alpha_pct", 100) or 100)
    alpha_pct = max(0, min(100, alpha_pct))
    base.setAlpha(int(round((alpha_pct / 100.0) * 255.0)))
    return base


def _button_bg_css(qcolor: QColor) -> str:
    return f"background: rgba({qcolor.red()}, {qcolor.green()}, {qcolor.blue()}, {qcolor.alpha()}); color:white;"


def _add_corrected_curve_from_last_fit(app) -> None:
    """Add Y_corrected = Y - (V0 + R*I) using Step-1/2/3 values."""
    resolved = _resolve_or_compute_step123_reference(app)
    if resolved is None:
        QMessageBox.warning(
            app,
            "Data Fitting",
            "Could not build corrected curve from Step-1/2/3 values.",
        )
        return
    result, _parent_entry, base_sig, base_label, x, y, t = resolved
    if int(min(x.size, y.size)) == 0:
        QMessageBox.warning(app, "Data Fitting", "No points available to build corrected curve.")
        return
    y_corr = y - (float(result.V0) + float(result.R) * x)

    sig = ("__corrected__", str(base_sig))
    existing = None
    for entry in getattr(app, "data_fit_curves", []):
        if entry.get("signature") == sig:
            existing = entry
            break
    if existing is None:
        color = "#ff7f0e"
        item = app.data_fit_plot.plot([], [], pen=pg.mkPen(color, width=1.8), symbol=None)
        existing = {
            "signature": sig,
            "label": f"{base_label} corrected",
            "color": color,
            "alpha_pct": 100,
            "skip_points": 1,
            "include_in_fit": False,
            "x": np.asarray([]),
            "y": np.asarray([]),
            "t": np.asarray([]),
            "plot_item": item,
            "fit_result": result,
            "curve_style": {"draw_mode": "Lines only", "line_width": 1.8, "point_size": 3},
            "avg_window": 1,
            "show_criterion": False,
            "show_ic": False,
            "source": {},
            "is_corrected_curve": True,
        }
        app.data_fit_curves.append(existing)
    existing["x"] = x
    existing["y"] = y_corr
    existing["t"] = t
    existing["fit_result"] = result
    existing["label"] = f"{base_label} corrected"
    _refresh_curve_item(existing)
    _refresh_curve_profile_selector(app)


def _resolve_fit_parent_and_result(app):
    """Return (result, parent_entry, base_sig, base_label, x, y, t) or None."""
    result = None
    parent_entry = None
    active_key = _curve_profile_key_from_ui(app)
    for entry in getattr(app, "data_fit_curves", []):
        if str(entry.get("signature")) != str(active_key):
            continue
        fit_result = entry.get("fit_result")
        if fit_result is not None and getattr(fit_result, "ok", False):
            result = fit_result
            parent_entry = entry
            break

    if result is None:
        for entry in getattr(app, "data_fit_curves", []):
            if bool(entry.get("is_fit_result", False)):
                continue
            fit_result = entry.get("fit_result")
            if fit_result is not None and getattr(fit_result, "ok", False):
                result = fit_result
                parent_entry = entry
                break

    if result is None:
        fit_result = getattr(getattr(app, "data_fit_controller", None), "last_result", None)
        if fit_result is not None and getattr(fit_result, "ok", False):
            result = fit_result

    if result is None:
        return None

    if parent_entry is not None:
        x = np.asarray(parent_entry.get("x", []), dtype=float)
        y = np.asarray(parent_entry.get("y", []), dtype=float)
        t = np.asarray(parent_entry.get("t", []), dtype=float)
        base_sig = parent_entry.get("signature", parent_entry.get("label", "curve"))
        base_label = parent_entry.get("label", "Curve")
    else:
        transformed = _apply_transforms(app)
        x = np.asarray(transformed.get("x", []), dtype=float)
        y = np.asarray(transformed.get("y", []), dtype=float)
        t = np.asarray(transformed.get("time", []), dtype=float)
        base_sig = ("__preview__", app.data_fit_y_cb.currentText())
        base_label = app.data_fit_y_cb.currentText() or "Preview"

    n = int(min(x.size, y.size))
    if n == 0:
        return None
    return result, parent_entry, base_sig, base_label, x[:n], y[:n], (t[:n] if t.size else np.asarray([]))


def _ensure_fit_for_reference(app, *, force_recompute: bool = False) -> bool:
    """Ensure Step-3/Step-4 fit values exist.

    When ``force_recompute`` is True, always re-run the fit so Step-1/2/3 are
    refreshed from the current data/settings before building helper curves.
    """
    if not force_recompute and _resolve_fit_parent_and_result(app) is not None:
        return True
    run_fit(app)
    return _resolve_fit_parent_and_result(app) is not None


def _resolve_reference_curve_data(app):
    """Return (entry, base_sig, base_label, x, y, t) for helper-curve building."""
    active_key = _curve_profile_key_from_ui(app)
    for entry in getattr(app, "data_fit_curves", []):
        if bool(entry.get("is_fit_result", False)):
            continue
        if str(entry.get("signature")) != str(active_key):
            continue
        x = np.asarray(entry.get("x", []), dtype=float)
        y = np.asarray(entry.get("y", []), dtype=float)
        t = np.asarray(entry.get("t", []), dtype=float)
        n = int(min(x.size, y.size))
        if n <= 0:
            continue
        return (
            entry,
            entry.get("signature", entry.get("label", "curve")),
            entry.get("label", "Curve"),
            x[:n],
            y[:n],
            (t[:n] if t.size else np.asarray([])),
        )

    transformed = _apply_transforms(app)
    x = np.asarray(transformed.get("x", []), dtype=float)
    y = np.asarray(transformed.get("y", []), dtype=float)
    t = np.asarray(transformed.get("time", []), dtype=float)
    n = int(min(x.size, y.size))
    if n <= 0:
        return None
    return (
        None,
        ("__preview__", app.data_fit_y_cb.currentText()),
        app.data_fit_y_cb.currentText() or "Preview",
        x[:n],
        y[:n],
        (t[:n] if t.size else np.asarray([])),
    )


def _compute_step123_result(t: np.ndarray, x: np.ndarray, y: np.ndarray, settings: FitSettings):
    """Compute Step-1/2/3 only, without running Step-4 fit."""
    t_arr = np.asarray(t, dtype=float).ravel()
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = int(min(t_arr.size, x_arr.size, y_arr.size))
    if n < 8:
        raise ValueError("Not enough valid samples to fit.")
    t_raw, x_raw, y_raw = t_arr[:n], x_arr[:n], y_arr[:n]
    t_arr, x_arr, y_arr = trim_vi_curve(t_raw, x_raw, y_raw, settings)
    if x_arr.size < 8:
        raise ValueError("Not enough valid samples to fit.")
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    if x_max <= x_min:
        raise ValueError("Current range is empty or degenerate.")

    V_ofs = 0.0
    if getattr(settings, "subtract_thermal_offset", True):
        # Match full-fit behavior: Step 1 uses the original cleaned curve so
        # V_ofs remains available even when the beginning is trimmed away.
        V_ofs, n_zero = estimate_thermal_offset(x_raw, y_raw, settings.zero_i_frac)
        if n_zero > 0:
            y_arr = y_arr - V_ofs
        else:
            V_ofs = 0.0
    di_dt = estimate_di_dt(t_arr, x_arr, settings.didt_low_frac, settings.didt_high_frac)
    lin_lo = x_min + settings.linear_low_frac * (x_max - x_min)
    lin_hi = x_min + settings.linear_high_frac * (x_max - x_min)
    baseline_mode = str(getattr(settings, "baseline_mode", DEFAULT_BASELINE_MODE) or DEFAULT_BASELINE_MODE)
    V0, R = fit_linear_baseline(x_arr, y_arr, lin_lo, lin_hi, mode=baseline_mode)
    uses_length = settings.sample_length_cm is not None and settings.sample_length_cm > 0
    return SimpleNamespace(
        ok=True,
        message="Step-1/2/3 succeeded.",
        V0=float(V0),
        R=float(R),
        V_ofs=float(V_ofs),
        di_dt=float(di_dt),
        uses_sample_length=bool(uses_length),
    )


def _resolve_or_compute_step123_reference(app):
    """Resolve helper-curve source and compute Step-1/2/3 if needed."""
    fit_resolved = _resolve_fit_parent_and_result(app)
    if fit_resolved is not None:
        return fit_resolved
    curve_data = _resolve_reference_curve_data(app)
    if curve_data is None:
        return None
    entry, base_sig, base_label, x, y, t = curve_data
    try:
        settings = _settings_for_entry(app, entry) if entry is not None else _settings_from_inputs(app)
        result = _compute_step123_result(t, x, y, settings)
    except Exception:
        return None
    return result, entry, base_sig, base_label, x, y, t


def _ensure_step4_reference_curve(app, *, create_plot_entry: bool, auto_run_fit: bool) -> bool:
    """Build corrected+smoothed Step-4 reference from latest successful fit."""
    resolved = _resolve_fit_parent_and_result(app)
    if resolved is None and auto_run_fit:
        _ensure_fit_for_reference(app)
        resolved = _resolve_fit_parent_and_result(app)
    if resolved is None:
        return False

    result, _parent_entry, base_sig, base_label, x, y, t = resolved
    y_corr = y - (float(result.V0) + float(result.R) * x)
    has_length = app.data_fit_use_length_cb.isChecked()
    to_si = 1.0e-6 if has_length else 1.0e-3
    ec1 = _float_from(app.data_fit_power_low, DEFAULT_EC1_V_PER_CM * 1.0e6) * to_si
    ec2 = _float_from(app.data_fit_power_vfrac, DEFAULT_EC2_V_PER_CM * 1.0e6) * to_si
    y_sm = _adaptive_smooth_visual(y_corr, ec1, ec2)
    app.data_fit_power_ref_curve = {"x": np.asarray(x), "y": np.asarray(y_sm)}

    if not create_plot_entry:
        return True

    sig = (
        "__smoothed_corrected__",
        str(base_sig),
        round(float(ec1), 18),
        round(float(ec2), 18),
    )
    existing = None
    for entry in getattr(app, "data_fit_curves", []):
        if entry.get("signature") == sig:
            existing = entry
            break
    if existing is None:
        color = "#ff9f1a"
        item = app.data_fit_plot.plot([], [], pen=pg.mkPen(color, width=1.6, style=Qt.DashLine), symbol=None)
        existing = {
            "signature": sig,
            "label": f"{base_label} smoothed+corrected",
            "color": color,
            "alpha_pct": 100,
            "skip_points": 1,
            "include_in_fit": False,
            "x": np.asarray([]),
            "y": np.asarray([]),
            "t": np.asarray([]),
            "plot_item": item,
            "fit_result": None,
            "curve_style": {"draw_mode": "Lines only", "line_width": 1.6, "point_size": 3},
            "avg_window": 1,
            "show_criterion": False,
            "show_ic": False,
            "source": {},
            "is_smoothed_curve": True,
            "is_step4_reference_curve": True,
        }
        app.data_fit_curves.append(existing)
    existing["x"] = x
    existing["y"] = y_sm
    existing["t"] = t
    existing["label"] = f"{base_label} smoothed+corrected"
    existing["fit_result"] = result
    _refresh_curve_item(existing)
    _refresh_curve_profile_selector(app)
    return True


def _add_smoothed_curve_from_current(app) -> None:
    """Create corrected+smoothed Step-4 reference curve and plot a copy."""
    resolved = _resolve_or_compute_step123_reference(app)
    if resolved is None:
        ok = False
    else:
        result, _parent_entry, base_sig, base_label, x, y, t = resolved
        y_corr = y - (float(result.V0) + float(result.R) * x)
        has_length = app.data_fit_use_length_cb.isChecked()
        to_si = 1.0e-6 if has_length else 1.0e-3
        ec1 = _float_from(app.data_fit_power_low, DEFAULT_EC1_V_PER_CM * 1.0e6) * to_si
        ec2 = _float_from(app.data_fit_power_vfrac, DEFAULT_EC2_V_PER_CM * 1.0e6) * to_si
        y_sm = _adaptive_smooth_visual(y_corr, ec1, ec2)
        app.data_fit_power_ref_curve = {"x": np.asarray(x), "y": np.asarray(y_sm)}

        sig = ("__smoothed_corrected__", str(base_sig), round(float(ec1), 18), round(float(ec2), 18))
        existing = None
        for entry in getattr(app, "data_fit_curves", []):
            if entry.get("signature") == sig:
                existing = entry
                break
        if existing is None:
            color = "#ff9f1a"
            item = app.data_fit_plot.plot([], [], pen=pg.mkPen(color, width=1.6, style=Qt.DashLine), symbol=None)
            existing = {
                "signature": sig,
                "label": f"{base_label} smoothed+corrected",
                "color": color,
                "alpha_pct": 100,
                "skip_points": 1,
                "include_in_fit": False,
                "x": np.asarray([]),
                "y": np.asarray([]),
                "t": np.asarray([]),
                "plot_item": item,
                "fit_result": result,
                "curve_style": {"draw_mode": "Lines only", "line_width": 1.6, "point_size": 3},
                "avg_window": 1,
                "show_criterion": False,
                "show_ic": False,
                "source": {},
                "is_smoothed_curve": True,
                "is_step4_reference_curve": True,
            }
            app.data_fit_curves.append(existing)
        existing["x"] = x
        existing["y"] = y_sm
        existing["t"] = t
        existing["label"] = f"{base_label} smoothed+corrected"
        existing["fit_result"] = result
        _refresh_curve_item(existing)
        _refresh_curve_profile_selector(app)
        ok = True
    if not ok:
        QMessageBox.warning(
            app,
            "Data Fitting",
            "Could not build smoothed+corrected curve from Step-1/2/3 values.",
        )


def _add_plot_from_current(app) -> None:
    """Snapshot the current inputs and add a curve entry to ``app.data_fit_curves``."""
    if not getattr(app, "data_fit_plot_dirty", True):
        return
    controller = app.data_fit_controller
    if not controller.channel_names:
        QMessageBox.warning(app, "Data Fitting", "Load a recording first.")
        return
    if not hasattr(app, "data_fit_curves"):
        app.data_fit_curves = []
    sig = _curve_signature(app)
    for entry in app.data_fit_curves:
        if entry["signature"] == sig:
            transformed = _apply_transforms(app)
            entry["x"] = transformed["x"]
            entry["y"] = transformed["y"]
            entry["t"] = transformed["time"]
            entry["avg_window"] = _active_avg_window(app)
            entry["label"] = app.data_fit_y_cb.currentText()
            profiles = getattr(app, "data_fit_curve_profiles", {})
            profiles[str(sig)] = _capture_fit_window_profile(app)
            app.data_fit_curve_profiles = profiles
            _refresh_curve_item(entry)
            _refresh_curve_profile_selector(app)
            app.data_fit_plot_dirty = False
            return
    transformed = _apply_transforms(app)
    x = transformed["x"]
    y = transformed["y"]
    if x is None or y is None or x.size == 0 or y.size == 0:
        QMessageBox.warning(app, "Data Fitting", "Could not read X/Y data for this selection.")
        return
    color = _next_pastel_color(app)
    label = app.data_fit_y_cb.currentText()
    # Plot item lives on the main plot.
    item = app.data_fit_plot.plot(
        [], [], pen=None, symbol="o", symbolSize=4,
        symbolBrush=pg.mkColor(color), symbolPen=pg.mkColor(color), name=label,
    )
    n_points = int(min(len(x), len(y)))
    adaptive_skip = 1
    entry = {
        "signature": sig,
        "label": label,
        "color": color,
        "alpha_pct": 100,
        "skip_points": adaptive_skip,
        "include_in_fit": True,
        "x": x,
        "y": y,
        "t": transformed["time"],
        "plot_item": item,
        "fit_result": None,
        "curve_style": {"draw_mode": "Auto", "line_width": 1.0, "point_size": 4},
        "avg_window": _active_avg_window(app),
        "show_criterion": False,
        "show_ic": False,
        "source": {
            "time_sig": app.data_fit_time_cb.currentText(),
            "x_sig": app.data_fit_x_cb.currentText(),
            "y_sig": app.data_fit_y_cb.currentText(),
            "t_scale": _float_from(app.data_fit_time_scale, 1.0),
            "t_offset": _float_from(app.data_fit_time_offset, 0.0),
            "x_scale": _float_from(app.data_fit_x_scale, 1.0),
            "x_offset": _float_from(app.data_fit_x_offset, 0.0),
            "y_scale": _float_from(app.data_fit_y_scale, 1.0),
            "y_offset": _float_from(app.data_fit_y_offset, 0.0),
            "use_length": bool(app.data_fit_use_length_cb.isChecked()),
            "length_cm": _float_from(app.data_fit_length_input, 1.0),
        },
    }
    app.data_fit_curves.append(entry)
    profiles = getattr(app, "data_fit_curve_profiles", {})
    profiles[str(sig)] = _capture_fit_window_profile(app)
    app.data_fit_curve_profiles = profiles
    _refresh_curve_item(entry)
    _refresh_curve_profile_selector(app)
    app.data_fit_plot_dirty = False


def _refresh_curve_item(entry: dict) -> None:
    step = max(1, int(entry.get("skip_points", 1) or 1))
    n = int(min(len(entry["x"]), len(entry["y"])))
    max_points = 20_000
    if n // step > max_points:
        step = max(step, n // max_points)
    x = entry["x"][::step]
    y = entry["y"][::step]
    color = _entry_color_qcolor(entry)
    style = entry.get("curve_style", {}) or {}
    draw_mode = style.get("draw_mode", "Auto")
    line_width = float(style.get("line_width", 1.0))
    point_size = int(style.get("point_size", 4))
    if draw_mode == "Points + lines":
        use_symbols = True
        use_line = True
    elif draw_mode == "Points only":
        use_symbols = True
        use_line = False
    elif draw_mode == "Lines only":
        use_symbols = False
        use_line = True
    else:
        use_symbols = x.size <= 8_000
        use_line = not use_symbols
    entry["plot_item"].setData(
        x, y,
        pen=pg.mkPen(color, width=line_width) if use_line else None,
        symbol=("o" if use_symbols else None),
        symbolSize=(point_size if use_symbols else 0),
        symbolBrush=color if use_symbols else None,
        symbolPen=color if use_symbols else None,
        name=entry["label"],
    )


def _remove_curve(app, entry: dict) -> None:
    app.data_fit_plot.removeItem(entry["plot_item"])
    try:
        app.data_fit_curves.remove(entry)
    except ValueError:
        pass
    _refresh_curve_profile_selector(app)


def _upsert_fit_curve_entry(app, source_entry: dict, result) -> None:
    fit_sig = ("__fit__", str(source_entry.get("signature", source_entry.get("label", "curve"))))
    parent_color = _entry_color_qcolor(source_entry)
    fit_color = parent_color.darker(118)
    fit_color_name = fit_color.name(QColor.HexRgb)
    fit_alpha_pct = int(round((fit_color.alpha() / 255.0) * 100.0))
    curves = getattr(app, "data_fit_curves", [])
    existing = None
    for c in curves:
        if c.get("signature") == fit_sig:
            existing = c
            break
    if existing is None:
        fit_item = app.data_fit_plot.plot([], [], pen=pg.mkPen(fit_color, width=2), symbol=None)
        existing = {
            "signature": fit_sig,
            "label": f"{source_entry.get('label', 'curve')} fit",
            "color": fit_color_name,
            "alpha_pct": fit_alpha_pct,
            "skip_points": 1,
            "include_in_fit": False,
            "x": np.asarray([]),
            "y": np.asarray([]),
            "t": np.asarray([]),
            "plot_item": fit_item,
            "fit_result": result,
            "curve_style": {"draw_mode": "Lines only", "line_width": 2.0, "point_size": 3},
            "avg_window": 1,
            "show_criterion": False,
            "show_ic": False,
            "source": {},
            "is_fit_result": True,
            "fit_parent_signature": source_entry.get("signature"),
        }
        curves.append(existing)
    existing["color"] = fit_color_name
    existing["alpha_pct"] = fit_alpha_pct
    existing["x"] = np.asarray(result.fit_x if result.fit_x is not None else [])
    existing["y"] = np.asarray(result.fit_y if result.fit_y is not None else [])
    existing["t"] = np.asarray([])
    existing["fit_result"] = result
    _refresh_curve_item(existing)


def _recompute_curve_from_source(app, entry: dict) -> None:
    src = entry.get("source") or {}
    controller = app.data_fit_controller
    time_sig = src.get("time_sig", "Time")
    t_raw = controller.get_channel(time_sig) if time_sig and time_sig != "Time" else controller.time_array
    x_raw = controller.get_channel(src.get("x_sig", ""))
    y_raw = controller.get_channel(src.get("y_sig", ""))
    t = controller.apply_transform(t_raw, float(src.get("t_scale", 1.0)), float(src.get("t_offset", 0.0)))
    x = controller.apply_transform(x_raw, float(src.get("x_scale", 1.0)), float(src.get("x_offset", 0.0)))
    y = controller.apply_transform(y_raw, float(src.get("y_scale", 1.0)), float(src.get("y_offset", 0.0)))
    avg_window = max(1, int(entry.get("avg_window", 1) or 1))
    if avg_window > 1:
        t = _block_average(t, avg_window)
        x = _block_average(x, avg_window)
        y = _block_average(y, avg_window)
    if bool(src.get("use_length")):
        length_cm = float(src.get("length_cm", 1.0))
        if length_cm > 0:
            y = y / length_cm
    entry["t"] = t
    entry["x"] = x
    entry["y"] = y


def _open_curve_settings_dialog(app, entry: dict, parent) -> None:
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QColorDialog, QDialogButtonBox, QCheckBox
    dialog = QDialog(parent)
    dialog.setWindowTitle("Curve settings")
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    style = dict(entry.get("curve_style", {}))
    draw_cb = QComboBox()
    draw_cb.addItems(["Auto", "Points + lines", "Points only", "Lines only"])
    draw_cb.setCurrentText(style.get("draw_mode", "Auto"))
    width_sb = QDoubleSpinBox()
    width_sb.setRange(0.2, 8.0)
    width_sb.setValue(float(style.get("line_width", 1.0)))
    point_sb = QSpinBox()
    point_sb.setRange(1, 20)
    point_sb.setValue(int(style.get("point_size", 4)))
    color_btn = QPushButton(entry.get("color", "#1f77b4"))
    color_btn.setStyleSheet(_button_bg_css(_entry_color_qcolor(entry)))
    alpha_sb = QSpinBox()
    alpha_sb.setRange(0, 100)
    alpha_sb.setSuffix(" %")
    initial_color = pg.mkColor(entry.get("color", "#1f77b4"))
    alpha_sb.setValue(int(entry.get("alpha_pct", int(round((initial_color.alpha() / 255.0) * 100.0)))))

    def pick_color():
        c = QColorDialog.getColor(
            pg.mkColor(entry.get("color", "#1f77b4")),
            dialog,
            "Curve color",
            options=QColorDialog.ShowAlphaChannel,
        )
        if c.isValid():
            entry["color"] = c.name(QColor.HexRgb)
            alpha_sb.setValue(int(round((c.alpha() / 255.0) * 100.0)))
            color_btn.setText(entry["color"])
            preview = QColor(c)
            preview.setAlpha(int(round((alpha_sb.value() / 100.0) * 255.0)))
            color_btn.setStyleSheet(_button_bg_css(preview))

    color_btn.clicked.connect(pick_color)
    show_crit_cb = QCheckBox("Show Ec/Vc criterion line for this curve")
    show_crit_cb.setChecked(bool(entry.get("show_criterion", False)))
    show_ic_cb = QCheckBox("Show Ic dashed line for this curve")
    show_ic_cb.setChecked(bool(entry.get("show_ic", False)))
    form.addRow("Color", color_btn)
    form.addRow("Transparency", alpha_sb)
    form.addRow("Draw mode", draw_cb)
    form.addRow("Line width", width_sb)
    form.addRow("Point size", point_sb)
    form.addRow(show_crit_cb)
    form.addRow(show_ic_cb)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addWidget(buttons)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    if dialog.exec_() != QDialog.Accepted:
        return
    entry["curve_style"] = {
        "draw_mode": draw_cb.currentText(),
        "line_width": float(width_sb.value()),
        "point_size": int(point_sb.value()),
    }
    entry["alpha_pct"] = int(alpha_sb.value())
    chosen = _entry_color_qcolor(entry)
    entry["color"] = chosen.name(QColor.HexRgb)
    entry["show_criterion"] = bool(show_crit_cb.isChecked())
    entry["show_ic"] = bool(show_ic_cb.isChecked())
    if entry.get("is_preview"):
        app.data_fit_preview_style = dict(entry["curve_style"])
        app.data_fit_preview_color = entry.get("color", "#1f77b4")
        app.data_fit_preview_alpha_pct = int(entry.get("alpha_pct", 100))
    _refresh_curve_item(entry)


def _open_plot_summary(app) -> None:
    _refresh_curve_profile_selector(app)
    from PyQt5.QtWidgets import (
        QDialog, QTableWidget, QVBoxLayout, QPushButton, QHBoxLayout,
        QCheckBox, QLineEdit, QHeaderView,
    )
    curves = list(getattr(app, "data_fit_curves", []))
    transformed = _apply_transforms(app)
    preview_x = transformed.get("x")
    preview_y = transformed.get("y")
    if (
        getattr(app, "data_fit_preview_visible", True)
        and preview_x is not None and preview_y is not None and preview_x.size and preview_y.size
    ):
        curves = [{
            "signature": ("__preview__",),
            "label": app.data_fit_y_cb.currentText(),
            "color": getattr(app, "data_fit_preview_color", "#1f77b4"),
            "alpha_pct": int(getattr(app, "data_fit_preview_alpha_pct", 100)),
            "skip_points": 1,
            "include_in_fit": bool(getattr(app, "data_fit_preview_include_in_fit", True)),
            "x": preview_x,
            "y": preview_y,
            "t": transformed.get("time"),
            "plot_item": app.data_fit_raw_curve,
            "fit_result": None,
            "is_preview": True,
            "curve_style": dict(getattr(app, "data_fit_preview_style", {})),
            "avg_window": 1,
            "show_criterion": False,
            "show_ic": False,
        }] + curves
    dialog = QDialog(app)
    dialog.setWindowTitle("Plot summary")
    dialog.resize(1080, 360)
    root = QVBoxLayout(dialog)
    table = QTableWidget(len(curves), 10)
    table.setHorizontalHeaderLabels([
        "Color", "Label", "Skip pts", "Avg", "Use tap", "Tap dist (cm)",
        "Effective rate", "Show", "Include", "Actions",
    ])
    table.horizontalHeader().setStretchLastSection(False)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    table.setColumnWidth(0, 90)
    table.setColumnWidth(2, 72)
    table.setColumnWidth(3, 70)
    table.setColumnWidth(4, 70)
    table.setColumnWidth(5, 100)
    table.setColumnWidth(6, 130)
    table.setColumnWidth(7, 60)
    table.setColumnWidth(8, 70)
    table.setColumnWidth(9, 250)

    def row_widgets(entry, row):
        is_preview = bool(entry.get("is_preview", False))
        color_btn = QPushButton(entry["color"])
        color_btn.setStyleSheet(_button_bg_css(_entry_color_qcolor(entry)) + " min-width:70px;")
        def pick_color():
            from PyQt5.QtWidgets import QColorDialog
            c = QColorDialog.getColor(
                _entry_color_qcolor(entry),
                dialog,
                "Curve color",
                options=QColorDialog.ShowAlphaChannel,
            )
            if c.isValid():
                entry["color"] = c.name(QColor.HexRgb)
                entry["alpha_pct"] = int(round((c.alpha() / 255.0) * 100.0))
                if is_preview:
                    app.data_fit_preview_color = entry["color"]
                    app.data_fit_preview_alpha_pct = int(entry["alpha_pct"])
                color_btn.setText(entry["color"])
                color_btn.setStyleSheet(_button_bg_css(_entry_color_qcolor(entry)) + " min-width:70px;")
                _refresh_curve_item(entry)
        color_btn.clicked.connect(pick_color)
        table.setCellWidget(row, 0, color_btn)
        lbl = QLineEdit(entry["label"])
        lbl.editingFinished.connect(lambda: (entry.update(label=lbl.text()), _refresh_curve_item(entry)))
        table.setCellWidget(row, 1, lbl)
        skip = QLineEdit(str(entry.get("skip_points", 1)))
        skip.setMaximumWidth(60)
        def on_skip():
            try:
                entry["skip_points"] = max(1, int(float(skip.text())))
            except ValueError:
                entry["skip_points"] = 1
                skip.setText("1")
            _refresh_curve_item(entry)
        skip.editingFinished.connect(on_skip)
        table.setCellWidget(row, 2, skip)
        avg = QLineEdit(str(entry.get("avg_window", 1)))
        avg.setMaximumWidth(60)
        rate_item = QLabel("—")
        tap = QLineEdit("—")
        tap.setMaximumWidth(85)
        src = entry.setdefault("source", {})
        tap_value = src.get("length_cm")
        if tap_value is not None:
            tap.setText(f"{float(tap_value):g}")
        is_fit_result = bool(entry.get("is_fit_result", False))

        def _rate_text_for_entry() -> str:
            t_local = entry.get("t")
            if t_local is None or np.asarray(t_local).size <= 1:
                return "—"
            t_arr_local = np.asarray(t_local, dtype=float)
            dt_local = float(np.mean(np.diff(t_arr_local)))
            if dt_local <= 0:
                return "—"
            return _format_rate(1.0 / dt_local)

        def on_avg():
            try:
                entry["avg_window"] = max(1, int(float(avg.text())))
            except ValueError:
                entry["avg_window"] = 1
                avg.setText("1")
            if is_preview:
                _set_silently(app.data_fit_avg_input, str(entry["avg_window"]))
                refresh_preview(app)
            else:
                _recompute_curve_from_source(app, entry)
                _refresh_curve_item(entry)
            rate_item.setText(_rate_text_for_entry())
        avg.editingFinished.connect(on_avg)
        table.setCellWidget(row, 3, avg)
        use_tap_cb = QCheckBox()
        if is_preview:
            use_tap_cb.setChecked(bool(app.data_fit_use_length_cb.isChecked()))
        else:
            use_tap_cb.setChecked(bool(src.get("use_length", False)))
        if is_fit_result:
            use_tap_cb.setEnabled(False)

        def _apply_tap_enabled_state() -> None:
            tap.setEnabled(bool(use_tap_cb.isChecked()) and not is_fit_result)

        def on_use_tap(v):
            checked = bool(v)
            _apply_tap_enabled_state()
            if is_fit_result:
                return
            if is_preview:
                app.data_fit_use_length_cb.setChecked(checked)
                if checked:
                    try:
                        updated = float(tap.text())
                        if updated <= 0:
                            raise ValueError
                    except ValueError:
                        updated = _float_from(app.data_fit_length_input, 1.0)
                        tap.setText(f"{updated:g}")
                    _set_silently(app.data_fit_length_input, f"{updated:g}")
                refresh_preview(app)
                return
            src["use_length"] = checked
            _recompute_curve_from_source(app, entry)
            _refresh_curve_item(entry)
            if _curve_profile_key_from_ui(app) == str(entry.get("signature", "")):
                _sync_active_length_settings(
                    app,
                    use_length=checked,
                    length_cm=float(src.get("length_cm", 1.0) or 1.0),
                )

        use_tap_cb.toggled.connect(on_use_tap)
        table.setCellWidget(row, 4, use_tap_cb)

        def on_tap():
            if is_fit_result:
                tap.setText("—")
                return
            if not use_tap_cb.isChecked():
                return
            previous = src.get("length_cm", 1.0)
            try:
                updated = float(tap.text())
                if updated <= 0:
                    raise ValueError
            except ValueError:
                tap.setText(f"{float(previous):g}")
                return
            if is_preview:
                app.data_fit_use_length_cb.setChecked(True)
                _set_silently(app.data_fit_length_input, f"{updated:g}")
                refresh_preview(app)
                return
            src["use_length"] = True
            src["length_cm"] = updated
            _recompute_curve_from_source(app, entry)
            _refresh_curve_item(entry)
            if _curve_profile_key_from_ui(app) == str(entry.get("signature", "")):
                _sync_active_length_settings(app, use_length=True, length_cm=updated)
        tap.editingFinished.connect(on_tap)
        _apply_tap_enabled_state()
        table.setCellWidget(row, 5, tap)
        rate_item.setText(_rate_text_for_entry())
        table.setCellWidget(row, 6, rate_item)
        show_cb = QCheckBox()
        show_cb.setToolTip(
            "Show or hide this curve on the plot. Hidden curves are still "
            "kept in memory and can be reshown — useful when an automatic "
            "fit added several voltage channels and you only want to "
            "compare two of them at a time."
        )
        if is_preview:
            show_cb.setChecked(bool(getattr(app, "data_fit_preview_visible", True)))
        else:
            show_cb.setChecked(bool(entry.get("visible", True)))

        def on_show(v, _entry=entry, _is_preview=is_preview):
            checked = bool(v)
            if _is_preview:
                app.data_fit_preview_visible = checked
                if checked:
                    refresh_preview(app)
                else:
                    app.data_fit_raw_curve.setData([], [])
                return
            _entry["visible"] = checked
            _apply_curve_visibility(_entry)
            # When this curve has a paired fit overlay, follow it.
            sig = _entry.get("signature")
            if sig is None:
                return
            for paired in getattr(app, "data_fit_curves", []):
                if (
                    paired.get("is_fit_result")
                    and paired.get("fit_parent_signature") == sig
                ):
                    paired["visible"] = checked
                    _apply_curve_visibility(paired)

        show_cb.toggled.connect(on_show)
        table.setCellWidget(row, 7, show_cb)
        include = QCheckBox()
        include.setChecked(entry.get("include_in_fit", True))
        if is_preview:
            include.toggled.connect(lambda v: setattr(app, "data_fit_preview_include_in_fit", bool(v)))
        else:
            include.toggled.connect(lambda v: entry.update(include_in_fit=bool(v)))
        table.setCellWidget(row, 8, include)
        actions = QHBoxLayout()
        actions_widget = QWidget()
        actions_widget.setLayout(actions)
        settings_btn = QPushButton("Curve settings")
        settings_btn.setMinimumWidth(120)
        settings_btn.clicked.connect(lambda: _open_curve_settings_dialog(app, entry, dialog))
        remove_btn = QPushButton("Remove")
        if is_preview:
            remove_btn.clicked.connect(
                lambda: (
                    setattr(app, "data_fit_preview_visible", False),
                    app.data_fit_raw_curve.setData([], []),
                    _refresh_curve_profile_selector(app),
                    dialog.accept(),
                    _open_plot_summary(app),
                )
            )
        else:
            remove_btn.clicked.connect(lambda: (_remove_curve(app, entry), dialog.accept(), _open_plot_summary(app)))
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(settings_btn)
        actions.addWidget(remove_btn)
        table.setCellWidget(row, 9, actions_widget)

    for i, entry in enumerate(curves):
        row_widgets(entry, i)

    root.addWidget(table)

    clear_btn = QPushButton("Remove all curves")
    def remove_all():
        for e in list(curves):
            if e.get("is_preview"):
                app.data_fit_preview_visible = False
                app.data_fit_raw_curve.setData([], [])
                _refresh_curve_profile_selector(app)
                continue
            _remove_curve(app, e)
        dialog.accept()
    clear_btn.clicked.connect(remove_all)
    root.addWidget(clear_btn)

    close = QPushButton("Close")
    close.clicked.connect(dialog.accept)
    root.addWidget(close)

    dialog.exec_()


def _fit_single_curve(app, entry: dict) -> None:
    try:
        settings = _settings_from_inputs(app)
    except Exception as exc:
        QMessageBox.critical(app, "Data Fitting", f"Invalid input: {exc}")
        return
    try:
        result = run_full_fit(entry["t"], entry["x"], entry["y"], settings)
    except Exception as exc:
        traceback.print_exc()
        QMessageBox.critical(app, "Data Fitting", f"Fit failed: {exc}")
        return
    entry["fit_result"] = result
    if result.ok:
        _show_fit_overlays(app, result)
        app.data_fit_result_text.setPlainText(f"[{entry['label']}]\n" + _format_result(result))
        _post_fit_warnings(app, result, settings)
        report_path = _write_fit_report_tdms(app, [(entry["label"], result)])
        if report_path:
            current = app.data_fit_result_text.toPlainText()
            app.data_fit_result_text.setPlainText(
                current + f"\nFit report written to: {report_path}"
            )
    else:
        _show_warning(app, result.message or "Fit failed.", severity="error")


def _settings_to_preset(app) -> FitPreset:
    save_separate = bool(
        getattr(app, "data_fit_save_separate_cb", None) is not None
        and app.data_fit_save_separate_cb.isChecked()
    )
    same_group = bool(
        getattr(app, "data_fit_same_group_cb", None) is None
        or app.data_fit_same_group_cb.isChecked()
    )
    auto_load = bool(
        _safe_checkbox_checked(app, "data_fit_auto_load_cb", default=True)
    )
    autosave = bool(
        _safe_checkbox_checked(app, "data_fit_autosave_cb", default=True)
    )
    return FitPreset(
        didt_low=_float_from(app.data_fit_didt_low, DEFAULT_DIDT_LOW_FRAC * 100),
        didt_high=_float_from(app.data_fit_didt_high, DEFAULT_DIDT_HIGH_FRAC * 100),
        linear_low=_float_from(app.data_fit_linear_low, DEFAULT_LINEAR_LOW_FRAC * 100),
        linear_high=_float_from(app.data_fit_linear_high, DEFAULT_LINEAR_HIGH_FRAC * 100),
        power_low=_float_from(app.data_fit_power_low, DEFAULT_POWER_LOW_FRAC * 100),
        power_vfrac=_float_from(app.data_fit_power_vfrac, DEFAULT_POWER_V_FRAC * 100),
        max_iter=int(_float_from(app.data_fit_max_iter, DEFAULT_MAX_ITERATIONS)),
        ic_tol_pct=_float_from(app.data_fit_ic_tol, DEFAULT_IC_TOLERANCE * 100),
        chi_tol=_float_from(app.data_fit_chi_tol, DEFAULT_CHI_SQR_TOL),
        use_length=app.data_fit_use_length_cb.isChecked(),
        sample_length_cm=_float_from(app.data_fit_length_input, 1.0),
        criterion_value=_float_from(app.data_fit_vc_input, DEFAULT_VC_VOLTS * 1000),
        avg_window=_active_avg_window(app),
        x_channel=app.data_fit_x_cb.currentText(),
        y_channel=app.data_fit_y_cb.currentText(),
        time_channel=app.data_fit_time_cb.currentText(),
        fit_method=_active_fit_method(app),
        save_to_separate_tdms=save_separate,
        save_fit_in_same_group=same_group,
        auto_load_after_acquisition=auto_load,
        autosave_fit_metadata=autosave,
    )


def _apply_preset(app, preset: FitPreset) -> None:
    _set_silently(app.data_fit_didt_low, f"{preset.didt_low:g}")
    _set_silently(app.data_fit_didt_high, f"{preset.didt_high:g}")
    _set_silently(app.data_fit_linear_low, f"{preset.linear_low:g}")
    _set_silently(app.data_fit_linear_high, f"{preset.linear_high:g}")
    _set_silently(app.data_fit_power_low, f"{preset.power_low:g}")
    _set_silently(app.data_fit_power_vfrac, f"{preset.power_vfrac:g}")
    _set_silently(app.data_fit_max_iter, f"{preset.max_iter}")
    _set_silently(app.data_fit_ic_tol, f"{preset.ic_tol_pct:g}")
    _set_silently(app.data_fit_chi_tol, f"{preset.chi_tol:g}")
    _set_silently(app.data_fit_length_input, f"{preset.sample_length_cm:g}")
    _set_silently(app.data_fit_vc_input, f"{preset.criterion_value:g}")
    _set_silently(app.data_fit_avg_input, f"{preset.avg_window}")
    app.data_fit_use_length_cb.blockSignals(True)
    app.data_fit_use_length_cb.setChecked(preset.use_length)
    app.data_fit_use_length_cb.blockSignals(False)
    _on_use_length_changed(app)
    # Restore the fit method (log-log vs non-linear) before refreshing the
    # mode-dependent UI so widget enabled-states and labels match the preset.
    method = getattr(preset, "fit_method", FIT_METHOD_LOG_LOG)
    if method == FIT_METHOD_LOG_LOG:
        app.data_fit_method_loglog_rb.setChecked(True)
    else:
        app.data_fit_method_nonlinear_rb.setChecked(True)
    _update_method_mode_ui(app)
    if hasattr(app, "data_fit_save_separate_cb"):
        app.data_fit_save_separate_cb.setChecked(
            bool(getattr(preset, "save_to_separate_tdms", False))
        )
    if hasattr(app, "data_fit_same_group_cb"):
        app.data_fit_same_group_cb.setChecked(
            bool(getattr(preset, "save_fit_in_same_group", True))
        )
    if hasattr(app, "data_fit_auto_load_cb"):
        app.data_fit_auto_load_cb.setChecked(
            bool(getattr(preset, "auto_load_after_acquisition", True))
        )
    if hasattr(app, "data_fit_autosave_cb"):
        app.data_fit_autosave_cb.setChecked(
            bool(getattr(preset, "autosave_fit_metadata", True))
        )
    # The four save-settings checkboxes drive each other's enabled state;
    # re-evaluate after every preset apply so the UI reflects the new values.
    _refresh_save_settings_enabled(app)
    # Attempt to restore channel selections if they exist in the current file.
    for combo, name in (
        (app.data_fit_x_cb, preset.x_channel),
        (app.data_fit_y_cb, preset.y_channel),
        (app.data_fit_time_cb, preset.time_channel),
    ):
        idx = combo.findText(name)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    refresh_preview(app)


def get_data_fit_preset_dict(app) -> dict:
    """Public helper: snapshot the Data Fitting tab state as a JSON-able dict.

    Used by DAQUniversal to embed Data Fitting settings inside daq_config.json.
    Returns an empty dict if the tab hasn't been built yet.
    """
    if not hasattr(app, "data_fit_vc_input"):
        return {}
    try:
        from dataclasses import asdict
        return asdict(_settings_to_preset(app))
    except Exception:
        return {}


def apply_data_fit_preset_dict(app, payload: dict) -> None:
    """Public helper: restore Data Fitting tab state from a dict produced by
    :func:`get_data_fit_preset_dict`. No-op if the tab isn't built yet or the
    payload is empty/invalid.
    """
    if not payload or not hasattr(app, "data_fit_vc_input"):
        return
    try:
        from dataclasses import fields
        allowed = {f.name for f in fields(FitPreset)}
        clean = {k: v for k, v in dict(payload).items() if k in allowed}
        _apply_preset(app, FitPreset(**clean))
    except Exception:
        pass


def _preset_dir(app) -> Path:
    controller = getattr(app, "data_fit_controller", None)
    base = None
    if controller is not None and controller.tdms_path:
        base = Path(controller.tdms_path).parent
    return base or Path.home()


def _save_preset(app) -> None:
    preset = _settings_to_preset(app)
    default_dir = _preset_dir(app)
    path_str, _ = QFileDialog.getSaveFileName(
        app, "Save fit preset", str(default_dir / "data_fit_preset.json"),
        "JSON (*.json)",
    )
    if not path_str:
        return
    try:
        save_preset_to_file(Path(path_str), preset)
    except OSError as exc:
        QMessageBox.critical(app, "Data Fitting", f"Failed to save preset: {exc}")
        return
    _show_warning(app, f"Saved preset to {path_str}", severity="warning")


def _load_preset(app) -> None:
    default_dir = _preset_dir(app)
    path_str, _ = QFileDialog.getOpenFileName(
        app, "Load fit preset", str(default_dir), "JSON (*.json)",
    )
    if not path_str:
        return
    try:
        preset = load_preset_from_file(Path(path_str))
    except (OSError, ValueError) as exc:
        QMessageBox.critical(app, "Data Fitting", f"Failed to load preset: {exc}")
        return
    _apply_preset(app, preset)
    _show_warning(app, f"Loaded preset from {path_str}", severity="warning")


def _build_fit_diagram_pixmap():
    """Render a schematic V-I curve highlighting offset, linear and power-law parts."""
    from PyQt5.QtCore import Qt, QRectF, QPointF
    from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QPainterPath

    W, H = 880, 500
    pix = QPixmap(W, H)
    pix.fill(QColor("white"))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)

    margin_l, margin_r, margin_t, margin_b = 80, 210, 60, 70
    px, py = margin_l, margin_t
    pw = W - margin_l - margin_r
    ph = H - margin_t - margin_b

    painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
    painter.setPen(QColor("#003a75"))
    painter.drawText(QRectF(0, 10, W, 28), Qt.AlignHCenter,
                     "Anatomy of a V–I curve")

    Ic_n, Vc, R, Vofs, n_val = 0.78, 0.085, 0.075, 0.045, 24

    def V_of(i: float) -> float:
        return Vofs + R * i + Vc * (max(i / Ic_n, 0.0) ** n_val)

    n_pts = 400
    curve = [(k / (n_pts - 1), V_of(k / (n_pts - 1))) for k in range(n_pts)]
    # Cap the y-axis so the offset, the linear slope, and the V_c criterion
    # line are all clearly visible — the power-law branch is allowed to exit
    # the top of the plot rectangle (it is then clipped at the frame).
    vmax = max(Vofs + R + 2.0 * Vc, 3.0 * Vc)

    def mx(i: float) -> float:
        return px + i * pw

    def my(v: float) -> float:
        return py + ph - (v / vmax) * ph

    bands = [
        (0.00, 0.10, QColor(255, 178, 90, 70),  "#a85a00", "offset"),
        (0.10, 0.34, QColor(120, 180, 255, 80), "#0a4a8c", "linear  R · I"),
        (0.45, 0.97, QColor(255, 120, 120, 80), "#8a1f2a", "power-law  V_c·(I/I_c)ⁿ"),
    ]

    painter.setPen(Qt.NoPen)
    for lo, hi, fill, _, _ in bands:
        painter.setBrush(fill)
        painter.drawRect(QRectF(mx(lo), py, mx(hi) - mx(lo), ph))

    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor("#1c2733"), 1.8))
    painter.drawRect(QRectF(px, py, pw, ph))

    painter.setPen(QPen(QColor("#a85a00"), 2.0, Qt.DashLine))
    painter.drawLine(QPointF(px, my(Vofs)), QPointF(px + pw, my(Vofs)))
    painter.setPen(QPen(QColor("#0a4a8c"), 2.0, Qt.DashLine))
    painter.drawLine(QPointF(mx(0.0), my(Vofs)),
                     QPointF(mx(1.0), my(Vofs + R)))

    painter.setPen(QPen(QColor("#1c2733"), 3.2))
    path = QPainterPath()
    path.moveTo(mx(curve[0][0]), my(curve[0][1]))
    for i, v in curve[1:]:
        path.lineTo(mx(i), my(v))
    painter.save()
    painter.setClipRect(QRectF(px, py, pw, ph))
    painter.drawPath(path)
    painter.restore()

    icx = mx(Ic_n)
    vcrit = V_of(Ic_n)
    painter.setPen(QPen(QColor("#cc2244"), 1.8, Qt.DashLine))
    painter.drawLine(QPointF(icx, py), QPointF(icx, py + ph))
    painter.drawLine(QPointF(px, my(vcrit)), QPointF(px + pw, my(vcrit)))

    painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
    painter.setPen(QColor("#cc2244"))
    painter.drawText(QPointF(icx + 8, py + 20), "I_c")
    painter.drawText(QPointF(px + 10, my(vcrit) - 6), "V_c criterion")

    painter.setPen(QColor("#1c2733"))
    painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
    painter.drawText(QRectF(px, py + ph + 22, pw, 24),
                     Qt.AlignHCenter, "Current  I  →")
    painter.save()
    painter.translate(22, py + ph / 2)
    painter.rotate(-90)
    painter.drawText(QRectF(-120, -10, 240, 20),
                     Qt.AlignHCenter, "Voltage  V  →")
    painter.restore()

    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
    for lo, hi, _, ink, name in bands:
        painter.setPen(QColor(ink))
        painter.drawText(
            QRectF(mx(lo), py - 20, mx(hi) - mx(lo), 16),
            Qt.AlignCenter, name,
        )

    legend_x = px + pw + 18
    legend_y = py + 6
    legend_items = [
        (QColor("#1c2733"), Qt.SolidLine, "V(I) total"),
        (QColor("#a85a00"), Qt.DashLine,  "V_ofs"),
        (QColor("#0a4a8c"), Qt.DashLine,  "V_ofs + R·I"),
        (QColor("#cc2244"), Qt.DashLine,  "I_c / V_c"),
    ]
    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
    for k, (color, style, name) in enumerate(legend_items):
        y = legend_y + k * 24
        painter.setPen(QPen(color, 2.4, style))
        painter.drawLine(QPointF(legend_x, y),
                         QPointF(legend_x + 36, y))
        painter.setPen(QColor("#1c2733"))
        painter.drawText(QPointF(legend_x + 44, y + 5), name)

    note_y = legend_y + len(legend_items) * 24 + 18
    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
    painter.setPen(QColor("#003a75"))
    painter.drawText(QPointF(legend_x, note_y), "Model")
    painter.setFont(QFont("Consolas", 10, QFont.Bold))
    painter.setPen(QColor("#1c2733"))
    eq_lines = ["V = V_ofs", "   + L · dI/dt",
                "   + R · I", "   + V_c·(I/I_c)ⁿ"]
    for k, ln in enumerate(eq_lines):
        painter.drawText(QPointF(legend_x, note_y + 20 + k * 16), ln)

    painter.end()
    return pix


def _build_loglog_diagram_pixmap():
    """Render a schematic of the IEC log-log linear fit.

    Shows log10(V - V_ofs - R·I) vs log10(I/Ic): a straight line whose
    slope is n, with the IEC decade window between Ec1 and Ec2 shaded.
    """
    import random
    from PyQt5.QtCore import Qt, QRectF, QPointF
    from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont

    W, H = 880, 500
    pix = QPixmap(W, H)
    pix.fill(QColor("white"))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)

    margin_l, margin_r, margin_t, margin_b = 140, 220, 60, 70
    px, py = margin_l, margin_t
    pw = W - margin_l - margin_r
    ph = H - margin_t - margin_b

    painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
    painter.setPen(QColor("#003a75"))
    painter.drawText(QRectF(0, 10, W, 28), Qt.AlignHCenter,
                     "Log–log linear fit (IEC 61788)")

    # Realistic-ish HTS tape sample (2 mm REBCO @ 77 K, Ls = 1 cm):
    # Ic ≈ 100 A, n ≈ 28, criterion Ec2 = 1 µV/cm, Ec1 = 0.1 µV/cm,
    # voltage noise floor ≈ 30 nV (post-averaging) → log10 ≈ −7.5.
    x_min, x_max = -0.45, 0.06        # log10(I / Ic): ~0.36·Ic .. 1.15·Ic
    y_min, y_max = -8.0, -5.4         # log10(V′)  [V or V/cm]
    log_Ec1, log_Ec2 = -7.0, -6.0     # 0.1 µV/cm and 1 µV/cm
    log_floor = -7.55                 # noise floor of V′ (≈ 28 nV/cm)
    n_slope = 28.0                    # n-value of the modelled tape

    def line_y(log_I_norm: float) -> float:
        return n_slope * log_I_norm + log_Ec2

    def mx(log_I: float) -> float:
        return px + (log_I - x_min) / (x_max - x_min) * pw

    def my(log_V: float) -> float:
        return py + ph - (log_V - y_min) / (y_max - y_min) * ph

    # Decade window shading.
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(120, 200, 130, 70))
    painter.drawRect(QRectF(px, my(log_Ec2),
                            pw, my(log_Ec1) - my(log_Ec2)))

    # Plot frame.
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor("#1c2733"), 1.8))
    painter.drawRect(QRectF(px, py, pw, ph))

    # Ec1 / Ec2 horizontal references.
    painter.setPen(QPen(QColor("#2d8a3a"), 1.8, Qt.DashLine))
    painter.drawLine(QPointF(px, my(log_Ec1)), QPointF(px + pw, my(log_Ec1)))
    painter.drawLine(QPointF(px, my(log_Ec2)), QPointF(px + pw, my(log_Ec2)))
    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
    painter.setPen(QColor("#1f5b2c"))
    painter.drawText(QPointF(px - 78, my(log_Ec1) + 5), "log E_c1")
    painter.drawText(QPointF(px - 78, my(log_Ec2) + 5), "log E_c2")

    # Vertical line at I = Ic (log_I_norm = 0).
    painter.setPen(QPen(QColor("#cc2244"), 1.8, Qt.DashLine))
    painter.drawLine(QPointF(mx(0.0), py), QPointF(mx(0.0), py + ph))
    painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
    painter.setPen(QColor("#cc2244"))
    painter.drawText(QPointF(mx(0.0) + 8, py + 22), "I = I_c")

    # Realistic measurement-style data:
    # • Below the noise floor V′ is dominated by amplifier noise → points
    #   scatter around log_floor with σ ≈ 0.13 in log space.
    # • Above the floor V′ tracks the power-law line with multiplicative
    #   noise σ_log ≈ 0.06 (~14 % on V).
    # • Real ramps are linear in I, so the X spacing is uniform and a
    #   given decade contains many samples — typical of a 1–5 A/s ramp
    #   sampled at 1 kHz over a few hundred ms of transition.
    random.seed(11)
    n_data = 240
    sigma_above = 0.06
    sigma_floor = 0.13
    # log_I_norm at which the underlying signal first overtakes the noise
    # floor; below this, even a "blue" scatter would not really be a
    # transition-segment point (mirrors how the IEC routine selects the
    # monotonic rise).
    log_I_onset = (log_floor - log_Ec2) / n_slope
    pts = []
    for k in range(n_data):
        log_I_norm = x_min + (x_max - x_min) * (k / (n_data - 1))
        v_true = line_y(log_I_norm)
        if v_true >= log_floor + 0.40:
            v_obs = v_true + random.gauss(0.0, sigma_above)
        elif v_true >= log_floor - 0.40:
            # Soft-knee: a smooth blend of true signal and noise floor.
            blend = (v_true - (log_floor - 0.40)) / 0.80
            v_obs = (
                blend * (v_true + random.gauss(0.0, sigma_above))
                + (1.0 - blend) * (log_floor + random.gauss(0.0, sigma_floor))
            )
        else:
            v_obs = log_floor + random.gauss(0.0, sigma_floor)
        pts.append((log_I_norm, v_obs))

    painter.save()
    painter.setClipRect(QRectF(px, py, pw, ph))
    painter.setPen(Qt.NoPen)
    for x_n, y_n in pts:
        if not (y_min <= y_n <= y_max):
            continue
        on_transition = x_n > log_I_onset
        in_band = (log_Ec1 <= y_n <= log_Ec2) and on_transition
        if in_band:
            painter.setBrush(QColor("#0a4a8c"))
            r = 3.8
        else:
            painter.setBrush(QColor(135, 138, 148, 215))
            r = 2.8
        painter.drawEllipse(QPointF(mx(x_n), my(y_n)), r, r)
    painter.restore()

    # Fitted line, clipped to the plot rectangle. Draw it on top of the
    # data with a slight white halo so it stays legible across the dots.
    painter.save()
    painter.setClipRect(QRectF(px, py, pw, ph))
    painter.setPen(QPen(QColor(255, 255, 255, 180), 5.0))
    painter.drawLine(
        QPointF(mx(x_min), my(line_y(x_min))),
        QPointF(mx(x_max), my(line_y(x_max))),
    )
    painter.setPen(QPen(QColor("#cc2244"), 2.8))
    painter.drawLine(
        QPointF(mx(x_min), my(line_y(x_min))),
        QPointF(mx(x_max), my(line_y(x_max))),
    )
    painter.restore()

    # Slope annotation, placed above the fitted line near the band.
    painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
    painter.setPen(QColor("#cc2244"))
    painter.drawText(QPointF(mx(-0.18) + 10, my(line_y(-0.18)) - 10),
                     "slope = n")

    # Axis labels.
    painter.setPen(QColor("#1c2733"))
    painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
    painter.drawText(QRectF(px, py + ph + 24, pw, 24),
                     Qt.AlignHCenter, "log₁₀ ( I / I_c )")
    painter.save()
    painter.translate(28, py + ph / 2)
    painter.rotate(-90)
    painter.drawText(QRectF(-180, -10, 360, 20),
                     Qt.AlignHCenter, "log₁₀ ( V − V_ofs − R · I )")
    painter.restore()

    # Decade-window label inside the band.
    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
    painter.setPen(QColor("#1f5b2c"))
    painter.drawText(QPointF(px + 10, my(log_Ec2) - 5),
                     "IEC decade window  [E_c1 , E_c2]")

    # Legend on the right.
    legend_x = px + pw + 18
    legend_y = py + 6
    items = [
        (QColor("#0a4a8c"), "in-window data"),
        (QColor(140, 140, 150), "outside window"),
        (QColor("#cc2244"), "linear fit"),
        (QColor("#2d8a3a"), "Ec1 / Ec2"),
        (QColor("#cc2244"), "I_c at E = Ec2"),
    ]
    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
    for k, (color, name) in enumerate(items):
        y = legend_y + k * 24
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(legend_x + 12, y), 5.5, 5.5)
        painter.setPen(QColor("#1c2733"))
        painter.drawText(QPointF(legend_x + 28, y + 5), name)

    note_y = legend_y + len(items) * 24 + 18
    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
    painter.setPen(QColor("#003a75"))
    painter.drawText(QPointF(legend_x, note_y), "How it fits")
    painter.setFont(QFont("Consolas", 10, QFont.Bold))
    painter.setPen(QColor("#1c2733"))
    eq_lines = [
        "V′ = V − V_ofs − R·I",
        "log V′ = n·log I + b",
        "I_c : V′(I_c) = E_c2",
    ]
    for k, ln in enumerate(eq_lines):
        painter.drawText(QPointF(legend_x, note_y + 20 + k * 16), ln)

    painter.end()
    return pix


def _open_help_dialog(app) -> None:
    """Open a sizable, tabbed help window explaining the fitting workflow."""
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QTextDocument
    from PyQt5.QtWidgets import (
        QDialog, QTabWidget, QTextBrowser, QVBoxLayout, QHBoxLayout,
        QPushButton, QSizePolicy,
    )

    existing = getattr(app, "_data_fit_help_dialog", None)
    if existing is not None:
        try:
            existing.raise_()
            existing.activateWindow()
            existing.show()
            return
        except RuntimeError:
            pass

    dialog = QDialog(app)
    dialog.setWindowTitle("Data Fitting — Help & Overview")
    dialog.setModal(False)
    dialog.resize(960, 720)
    dialog.setSizeGripEnabled(True)

    layout = QVBoxLayout(dialog)
    tabs = QTabWidget()
    tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    base_css = """
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt;
                   color: #1c2733; line-height: 1.45; }
            h1 { font-size: 18pt; color: #003a75; margin: 0 0 6px 0; }
            h2 { font-size: 14pt; color: #0a4a8c; margin: 14px 0 4px 0;
                 border-bottom: 1px solid #cfd8e3; padding-bottom: 2px; }
            h3 { font-size: 12pt; color: #1a4f7a; margin: 12px 0 2px 0; }
            p  { margin: 4px 0 6px 0; }
            ul, ol { margin: 2px 0 8px 18px; }
            li { margin: 2px 0; }
            code, .mono { font-family: Consolas, 'Courier New', monospace;
                          background: #f1f4f8; padding: 1px 4px; border-radius: 3px; }
            .pill { display: inline-block; padding: 1px 8px; border-radius: 10px;
                    background: #e6f2ff; color: #003a75; font-weight: bold;
                    font-size: 10pt; }
            .warn { background: #fff7c2; border: 1px solid #e6cc00;
                    padding: 8px 10px; border-radius: 4px; color: #5b4a00; }
            .tip  { background: #e8f7ee; border: 1px solid #58b56a;
                    padding: 8px 10px; border-radius: 4px; color: #1f5b2c; }
            table { border-collapse: collapse; margin: 6px 0 10px 0;
                    width: 100%; }
            th, td { border: 1px solid #cfd8e3; padding: 5px 8px;
                     text-align: left; vertical-align: top; }
            th { background: #eef3f8; color: #003a75; }
            .key { color: #0a4a8c; font-weight: bold; }
        </style>
    """

    overview_html = base_css + """
    <h1>Superconductor V&ndash;I Fitting</h1>
    <p>This tool extracts the <b>critical current <span class="mono">I<sub>c</sub></span></b>,
    the <b>n&#8209;value</b> and the <b>resistive baseline</b> from a recorded
    voltage&ndash;current ramp. Two complementary fitting methods are available
    (selectable on the left panel).</p>

    <p style="text-align:center; margin: 6px 0 2px 0;">
      <img src="fit-diagram://overview" width="820">
    </p>
    <p style="text-align:center; color:#5a6472; font-size:10pt; margin: 0 0 10px 0;">
      <i>Schematic in <b>linear axes</b>: the three additive parts of the model
      and the <span class="mono">I<sub>c</sub></span> /
      <span class="mono">V<sub>c</sub></span> criterion.
      The log&ndash;log view of the same data is on the next tab.</i>
    </p>

    <h2>The two fitting methods</h2>
    <table>
      <tr>
        <th style="width:50%;">Log&ndash;log linear&nbsp;
          <span class="pill">IEC 61788 standard</span></th>
        <th style="width:50%;">Non&#8209;linear (Levenberg&ndash;Marquardt)</th>
      </tr>
      <tr>
        <td>
          <ul>
            <li>Subtracts the linear baseline first
              (<code>V′ = V − V<sub>ofs</sub> − R·I</code>).</li>
            <li>Fits <code>log V′</code> vs <code>log I</code> with a
              straight line on the IEC <b>decade window</b>
              <code>[E<sub>c1</sub>, E<sub>c2</sub>]</code>.</li>
            <li>Slope = <span class="mono">n</span>;
              <span class="mono">I<sub>c</sub></span> at <code>V′ = E<sub>c2</sub></code>.</li>
          </ul>
        </td>
        <td>
          <ul>
            <li>Fits the full model
              <code>V = V<sub>ofs</sub> + L·dI/dt + R·I + V<sub>c</sub>·(I/I<sub>c</sub>)<sup>n</sup></code>
              directly to the raw V&ndash;I data.</li>
            <li>Solves for all parameters simultaneously inside an outer
              self&#8209;consistency loop on
              <span class="mono">I<sub>c</sub></span>.</li>
          </ul>
        </td>
      </tr>
      <tr>
        <th>Pros</th><th>Pros</th>
      </tr>
      <tr>
        <td>
          <ul>
            <li>Reference method per IEC 61788 → reproducible across labs.</li>
            <li>Closed&#8209;form linear regression: very fast, no convergence issues.</li>
            <li>Robust against multiplicative noise on V.</li>
          </ul>
        </td>
        <td>
          <ul>
            <li>Returns proper σ on every parameter from the covariance matrix.</li>
            <li>Handles non&#8209;negligible inductive / baseline coupling natively.</li>
            <li>Uses the entire ramp, not only the decade window.</li>
          </ul>
        </td>
      </tr>
      <tr>
        <th>Cons</th><th>Cons</th>
      </tr>
      <tr>
        <td>
          <ul>
            <li>Sensitive to errors in the pre&#8209;subtracted baseline
              <code>(V<sub>ofs</sub>, R)</code>.</li>
            <li>Only the data inside the decade window is used.</li>
            <li>Needs ≥ a few dozen samples between
              <code>E<sub>c1</sub></code> and <code>E<sub>c2</sub></code>.</li>
          </ul>
        </td>
        <td>
          <ul>
            <li>Slower; can fail to converge if windows or seeds are poor.</li>
            <li>Baseline and transition compete in the residual &mdash; sensitive
              to a wrongly chosen power&#8209;law window.</li>
            <li>No standardized acceptance criterion.</li>
          </ul>
        </td>
      </tr>
    </table>
    <div class="tip"><b>Default &amp; recommendation:</b> the tab starts in
      <b>log&ndash;log linear</b> mode because that is the IEC&nbsp;61788
      reference method for reporting <span class="mono">I<sub>c</sub></span> and
      <span class="mono">n</span>. Switch to <b>non&#8209;linear</b> only when the
      log&ndash;log fit reports <i>insufficient n&#8209;window points</i> or
      when the ramp is noticeably inductive.</div>

    <h2>The model</h2>
    <p>Without a sample length the fitted relation is:</p>
    <p style="font-size:13pt;"><code>V(I, dI/dt) = V<sub>ofs</sub> + L · dI/dt + R · I + V<sub>c</sub> · (I / I<sub>c</sub>)<sup>n</sup></code></p>
    <p>If a sample length <span class="mono">Lₛ</span> is provided, voltages are
    converted to electric field and the equivalent equation is fit:</p>
    <p style="font-size:13pt;"><code>E(I, dI/dt) = (L / L<sub>s</sub>) · dI/dt + ρ · I + E<sub>c</sub> · (I / I<sub>c</sub>)<sup>n</sup></code></p>

    <h2>Three&#8209;step procedure</h2>
    <ol>
      <li><b>dI/dt window</b> &mdash; a slope is extracted from the rising part of the
        current to estimate the inductive offset <code>L · dI/dt</code>.</li>
      <li><b>Linear baseline</b> &mdash; in a low&#8209;current window the residual
        voltage is fit to <code>V<sub>ofs</sub> + R · I</code> (or ρ · I).</li>
      <li><b>Power&#8209;law fit</b> &mdash; in the high&#8209;current window the
        residual is fit to <code>V<sub>c</sub> · (I / I<sub>c</sub>)<sup>n</sup></code>
        to extract <span class="mono">Iₒ</span> and <span class="mono">n</span>.</li>
    </ol>

    <h2>What's on the tab</h2>
    <ul>
      <li><span class="pill">Left panel</span> data source, channel transforms,
        fit windows, iteration knobs, presets and <b>Run Fit</b>.</li>
      <li><span class="pill">Right panel</span> equation reminder, channels,
        the live preview plot, fit results and curve management.</li>
    </ul>

    <div class="tip"><b>Tip:</b> Hover any field to see a tooltip. Each curve can
    be saved as a profile so its window settings persist between sessions.</div>
    """

    loglog_html = base_css + """
    <h1>Log&ndash;log linear fit
      <span class="pill">IEC 61788</span></h1>

    <p>The <b>log&ndash;log</b> method is the standard way to report
    <span class="mono">I<sub>c</sub></span> and the <span class="mono">n</span>&#8209;value.
    It splits the V&ndash;I curve into two clean steps and fits a straight line in
    log space &mdash; very robust as long as the linear baseline is correct.</p>

    <p style="text-align:center; margin: 6px 0 2px 0;">
      <img src="fit-diagram://loglog" width="820">
    </p>
    <p style="text-align:center; color:#5a6472; font-size:10pt; margin: 0 0 10px 0;">
      <i>Same curve as on the Overview tab, but plotted as
      <code>log V′</code> vs <code>log I</code>. The shaded band is the IEC
      decade window <code>[E<sub>c1</sub>, E<sub>c2</sub>]</code>.</i>
    </p>

    <h2>Procedure</h2>
    <ol>
      <li><b>Linear baseline</b> in the low&#8209;current window
        <code>V<sub>ofs</sub> + R·I</code> (or ρ&middot;I) is fit on points
        well below <span class="mono">I<sub>c</sub></span>.</li>
      <li>The baseline is subtracted from the raw signal:
        <code>V′(I) = V − V<sub>ofs</sub> − R·I</code>.</li>
      <li>Inside the <b>IEC decade window</b>
        <code>[E<sub>c1</sub>, E<sub>c2</sub>]</code> a straight line is fit to
        <code>log<sub>10</sub> V′</code> vs <code>log<sub>10</sub> I</code>
        using <code>numpy.polyfit</code> (closed form, no iterations).</li>
      <li>The slope is the <b>n&#8209;value</b>;
        <span class="mono">I<sub>c</sub></span> is the current at which
        <code>V′ = E<sub>c2</sub></code>.</li>
    </ol>

    <h2>IEC 61788 defaults</h2>
    <ul>
      <li><b>E<sub>c2</sub> = 1&nbsp;µV/cm</b> &mdash; the <span class="pill">I<sub>c</sub></span>
        criterion (HTS at 77&nbsp;K).</li>
      <li><b>E<sub>c1</sub> = 0.1&nbsp;µV/cm</b> &mdash; the lower edge of the
        decade window.</li>
      <li><b>≥&nbsp;50</b> samples inside the decade are recommended for a
        reliable n&#8209;value.</li>
      <li>Without a sample length the equivalent <span class="mono">V<sub>c</sub></span>
        criterion (default <b>1&nbsp;mV</b>) is used; <span class="mono">E<sub>c1</sub></span>
        and <span class="mono">E<sub>c2</sub></span> scale by the same ratio.</li>
    </ul>

    <h2>What you get</h2>
    <ul>
      <li><span class="mono">I<sub>c</sub></span>, <span class="mono">n</span></li>
      <li><b>σ(I<sub>c</sub>)</b>, <b>σ(n)</b> from the polyfit covariance.</li>
      <li><b>R²</b> on <code>log V′</code> vs the linear model.</li>
      <li><b>n&#8209;window points used</b> &mdash; the actual count of samples
        inside the decade.</li>
    </ul>

    <div class="warn"><b>Common failure modes</b>
      <ul>
        <li><i>“Insufficient n&#8209;window points”</i> &mdash; widen the
          baseline window so the residual is centred, slow the ramp, or use a
          longer recording so the decade contains more samples.</li>
        <li><i>“Data never reaches E<sub>c2</sub>”</i> &mdash; the ramp ended
          before the criterion was reached. Increase the maximum current.</li>
        <li>If the linear baseline is wrong, the residual <code>V′</code> may
          dip below zero in the decade window and the log fit will fail.
          Recheck the linear&#8209;baseline window first.</li>
      </ul>
    </div>
    """

    workflow_html = base_css + """
    <h1>How to fit a curve properly</h1>

    <h2>Step&#8209;by&#8209;step</h2>
    <ol>
      <li><b>Load a TDMS recording</b> via <i>Load TDMS…</i> and (optionally)
        <i>Load metadata from TDMS</i> to pull the per&#8209;channel
        scale/offset and voltage&#8209;tap distance.</li>
      <li><b>Pick the channels</b>:
        <ul>
          <li><span class="key">Time</span> &mdash; usually <code>t (s)</code>.</li>
          <li><span class="key">Current X</span> &mdash; the ramped current, in&nbsp;A.</li>
          <li><span class="key">Voltage Y</span> &mdash; the tap voltage, in&nbsp;V.</li>
        </ul>
      </li>
      <li><b>Set scale &amp; offset</b> for each channel so the displayed values
        are in physical units (<code>shown = raw · scale &minus; offset</code>).</li>
      <li><b>Provide the sample length</b> <code>Lₛ</code> if you want
        E&#8209;field results (ρ, E<sub>c</sub>); leave it blank for V&#8209;based
        results (R, V<sub>c</sub>).</li>
      <li><b>Inspect the preview plot</b> &mdash; the three coloured bands show
        the dI/dt, linear and power&#8209;law fit windows. Drag their handles or
        edit the percentages so each window covers a clean part of the curve.</li>
      <li><b>Choose the criterion</b> <code>V<sub>c</sub></code> /
        <code>E<sub>c</sub></code> (1&nbsp;µV/cm is the IEC default for HTS at 77&nbsp;K).</li>
      <li><b>Press <span class="pill">Run Fit</span></b> and read the result block
        on the right. The fit overlay shows <span class="mono">Iₒ</span>, the
        criterion line and the n&#8209;window points actually used.</li>
    </ol>

    <h2>Choosing fit windows well</h2>
    <p><b>Common to both methods:</b></p>
    <ul>
      <li><b>dI/dt window</b> &mdash; pick a region where the current ramps
        linearly (typically <b>40&nbsp;%&ndash;60&nbsp;%</b> of the trace). Avoid the
        switch&#8209;on transient and the transition region.</li>
      <li><b>Linear baseline window</b> &mdash; well below
        <span class="mono">I<sub>c</sub></span> (default
        <b>5&nbsp;%&ndash;30&nbsp;%</b>&nbsp;of <code>I<sub>max</sub></code>),
        where <code>V</code> is dominated by the resistive baseline and noise.</li>
    </ul>

    <p><b>Power&#8209;law window depends on the method:</b></p>
    <ul>
      <li><span class="pill">Log&ndash;log (IEC 61788)</span> &mdash; the
        effective fit window is the <b>decade
        <code>[E<sub>c1</sub>, E<sub>c2</sub>]</code></b> on the
        baseline&#8209;subtracted residual (defaults
        <b>0.1&nbsp;µV/cm</b> &ndash; <b>1&nbsp;µV/cm</b>). The numeric
        “Power low” / “Power V frac” fields only act as a sanity envelope
        on top of the IEC window.</li>
      <li><span class="pill">Non&#8209;linear (LM)</span> &mdash; uses the
        full <b>“Power&#8209;law window”</b> from a low fraction of
        <span class="mono">I<sub>c</sub></span> up to the largest voltage you
        trust (<b>≤&nbsp;80&nbsp;%</b> of <code>V<sub>max</sub></code> by
        default), so the transition is captured but flux&#8209;flow / runaway
        is excluded.</li>
    </ul>

    <h2>Reading the result</h2>
    <ul>
      <li><b>Iₒ</b> &mdash; current at which V reaches V<sub>c</sub>
        (or E<sub>c</sub> · L<sub>s</sub>).</li>
      <li><b>n</b> &mdash; sharpness of the transition; larger ⇒ sharper.</li>
      <li><b>R / ρ</b> &mdash; resistive baseline (joints, leads, contact).</li>
      <li><b>L</b> &mdash; inductive offset coefficient from the dI/dt step.</li>
      <li><b>R²</b> and <b>σ(Iₒ), σ(n)</b> &mdash; goodness of the power&#8209;law fit.</li>
    </ul>

    <div class="warn"><b>Watch out:</b>
      <ul>
        <li>If the warning bar lights up <i>“ramp too fast”</i> the inductive
          term swamps the n&#8209;value &mdash; lower <code>dI/dt</code> on the
          measurement.</li>
        <li><i>“Insufficient n&#8209;window points”</i> means the power&#8209;law
          window contains too few samples; widen it or sample faster.</li>
        <li>If R&sup2; is &lt;&nbsp;0.99 or σ(n)/n is &gt;&nbsp;5&nbsp;%, recheck
          your windows.</li>
      </ul>
    </div>
    """

    options_html = base_css + """
    <h1>Custom settings &amp; options</h1>

    <h2>Fit method (top of the iteration box)</h2>
    <ul>
      <li><span class="pill">Log&ndash;log linear</span> &mdash; default,
        IEC 61788 reference. Fits <code>log V′</code> vs <code>log I</code>
        after baseline subtraction; closed&#8209;form, no iteration.</li>
      <li><span class="pill">Non&#8209;linear&nbsp;(LM)</span> &mdash; fits the
        full coupled model in one shot using Levenberg&ndash;Marquardt inside an
        outer self&#8209;consistency loop on
        <span class="mono">I<sub>c</sub></span>.</li>
    </ul>

    <h2>Parameters that depend on the method</h2>

    <h3>Log&ndash;log linear &mdash; <i>only these inputs are used</i></h3>
    <table>
      <tr><th>Field</th><th>Meaning</th><th>Default</th></tr>
      <tr><td><code>E<sub>c1</sub></code></td>
        <td>Lower edge of the IEC decade window.</td>
        <td>0.1&nbsp;µV/cm</td></tr>
      <tr><td><code>E<sub>c2</sub></code> /
        <code>V<sub>c</sub></code></td>
        <td>Upper edge of the decade and the
          <span class="mono">I<sub>c</sub></span> criterion.</td>
        <td>1&nbsp;µV/cm (or 1&nbsp;mV without L<sub>s</sub>)</td></tr>
      <tr><td><code>Linear low / high</code></td>
        <td>Window for fitting <code>V<sub>ofs</sub> + R·I</code> &mdash;
          the baseline that is subtracted before the log fit.</td>
        <td>5&nbsp;%&ndash;30&nbsp;% of I<sub>max</sub></td></tr>
      <tr><td><code>dI/dt low / high</code></td>
        <td>Window for the inductive&#8209;ratio diagnostic
          (does not change the fitted <span class="mono">I<sub>c</sub></span>).</td>
        <td>40&nbsp;%&ndash;60&nbsp;% of t</td></tr>
    </table>
    <p style="color:#5a6472; font-size:10pt;"><i>The
      <code>Max iterations</code>, <code>I<sub>c</sub> stop tol</code> and
      <code>Chi&#8209;sqr tol</code> fields are <b>greyed out / ignored</b> in
      this mode &mdash; the polyfit is one&#8209;shot.</i></p>

    <h3>Non&#8209;linear (LM) &mdash; <i>iteration knobs apply here</i></h3>
    <table>
      <tr><th>Field</th><th>Meaning</th><th>Default</th></tr>
      <tr><td><code>Max iterations</code></td>
        <td>Outer&#8209;loop cap for the self&#8209;consistent
          <span class="mono">I<sub>c</sub></span> refinement.</td>
        <td>20</td></tr>
      <tr><td><code>I<sub>c</sub> stop tol (%)</code></td>
        <td>Stop when
          |ΔI<sub>c</sub>|/I<sub>c</sub> falls below this.</td>
        <td>0.1&nbsp;%</td></tr>
      <tr><td><code>Chi&#8209;sqr tol</code></td>
        <td><code>ftol</code>/<code>xtol</code>/<code>gtol</code> passed to the
          inner LM solver.</td>
        <td>1e&minus;6</td></tr>
      <tr><td><code>V<sub>c</sub></code> / <code>E<sub>c</sub></code></td>
        <td>Criterion voltage / electric field used to define
          <span class="mono">I<sub>c</sub></span>.</td>
        <td>1&nbsp;mV / 1&nbsp;µV/cm</td></tr>
      <tr><td><code>Power low</code> /
        <code>Power V frac</code></td>
        <td>Lower current bound and upper voltage bound of the
          <b>power&#8209;law fit window</b>.</td>
        <td>0.05·I<sub>max</sub> / 0.80·V<sub>max</sub></td></tr>
    </table>
    <p style="color:#5a6472; font-size:10pt;"><i>
      <code>E<sub>c1</sub></code> is irrelevant in this mode (no decade window).</i></p>

    <h2>Shared diagnostics</h2>
    <ul>
      <li><b>Zero&#8209;I fraction</b> &mdash; default <b>2&nbsp;%</b>; used by
        both methods to pin the thermal offset away from any switch&#8209;on
        glitch.</li>
      <li><b>dI/dt window</b> &mdash; both methods compute the inductive
        ratio (<code>L·dI/dt</code> / criterion) here and raise the
        <i>“ramp too fast”</i> warning if it dominates.</li>
    </ul>

    <h2>Presets &amp; profiles</h2>
    <ul>
      <li><span class="pill">Save preset…</span> writes every numeric setting,
        the criterion, and the chosen fit method to a JSON file.</li>
      <li><span class="pill">Load preset…</span> restores them in one click —
        ideal for sharing reproducible recipes between operators.</li>
      <li><b>Per&#8209;curve profiles</b> remember each curve's individual
        windows so you can fit several samples sequentially without
        re&#8209;adjusting.</li>
    </ul>

    <h2>Plot &amp; export</h2>
    <ul>
      <li><b>Show dI/dt / Linear / Power bands</b> toggle the coloured overlays
        on the preview plot.</li>
      <li><b>Add plot</b> stores the active curve in the curve list,
        <b>Add corrected curve</b> stores the baseline&#8209;subtracted version
        of the last fit.</li>
      <li><b>Plot scale</b> switches between linear and log&#8209;log axes
        without changing the underlying fit.</li>
      <li><b>Export…</b> writes a publication&#8209;quality PNG/PDF and a
        TDMS side&#8209;car (<code>*_fit_report.tdms</code>) with all metadata.</li>
    </ul>
    """

    metadata_html = base_css + """
    <h1>Fit results stored in metadata</h1>
    <p>Every successful fit is written as a channel under the
    <code>FitResults</code> group of the side&#8209;car
    <code>&lt;source&gt;_fit_report.tdms</code> (or, when "same group" is
    enabled, prefixed with <code>fit_</code> on the source channel itself).
    Each channel has the parameters below attached as TDMS properties, so they
    survive round&#8209;trips through LabVIEW, OriginLab and Python
    consumers.</p>

    <h2>Run identification</h2>
    <table>
      <tr><th>Property</th><th>Type</th><th>Description</th></tr>
      <tr><td><code>method</code></td><td>str</td>
        <td>Human&#8209;readable label, e.g. <i>"IEC 61788 log-log
          (decade n-value)"</i> or <i>"Non-linear V-I"</i>.</td></tr>
      <tr><td><code>method_id</code></td><td>str</td>
        <td>Machine&#8209;readable id: <code>log_log</code> or
          <code>nonlinear</code>. Used by re&#8209;loaders.</td></tr>
      <tr><td><code>method_compliant</code></td><td>str</td>
        <td><code>IEC 61788-3</code> for the log&ndash;log method, otherwise
          <code>legacy</code>.</td></tr>
      <tr><td><code>fit_status</code></td><td>str</td>
        <td><code>ok</code> or <code>failed</code>.</td></tr>
      <tr><td><code>fit_message</code></td><td>str</td>
        <td>Diagnostic / error string (empty on success).</td></tr>
      <tr><td><code>fit_timestamp</code></td><td>ISO&#8209;8601</td>
        <td>Local time at which Run Fit produced this record.</td></tr>
    </table>

    <h2>Primary parameters</h2>
    <table>
      <tr><th>Property</th><th>Unit</th><th>Description</th></tr>
      <tr><td><code>Ic_A</code></td><td>A</td>
        <td>Critical current at the chosen criterion.</td></tr>
      <tr><td><code>sigma_Ic_A</code></td><td>A</td>
        <td>1&#8209;σ uncertainty on <span class="mono">I<sub>c</sub></span>
          (covariance matrix / propagated polyfit error).</td></tr>
      <tr><td><code>n_value</code></td><td>&mdash;</td>
        <td>n&#8209;value (transition sharpness).</td></tr>
      <tr><td><code>sigma_n</code></td><td>&mdash;</td>
        <td>1&#8209;σ uncertainty on <span class="mono">n</span>.</td></tr>
      <tr><td><code>r_squared</code></td><td>&mdash;</td>
        <td>Coefficient of determination of the power&#8209;law fit (in
          log&ndash;log space for the IEC method).</td></tr>
      <tr><td><code>chi_squared</code></td><td>&mdash;</td>
        <td>Sum of squared residuals (LM) or log&#8209;space residual sum
          (log&ndash;log). Smaller is better; always read together with
          <code>r_squared</code>.</td></tr>
      <tr><td><code>di_dt_A_per_s</code></td><td>A/s</td>
        <td>Mean ramp rate measured inside the dI/dt window.</td></tr>
    </table>

    <h2>Criterion &amp; n&#8209;window</h2>
    <table>
      <tr><th>Property</th><th>Unit</th><th>Description</th></tr>
      <tr><td><code>criterion_value</code></td><td>V or V/cm</td>
        <td>The applied <span class="mono">V<sub>c</sub></span> /
          <span class="mono">E<sub>c</sub></span>.</td></tr>
      <tr><td><code>criterion_name</code></td><td>str</td>
        <td><code>Ec</code> when a sample length is given, else
          <code>Vc</code>.</td></tr>
      <tr><td><code>criterion_unit</code></td><td>str</td>
        <td><code>V/cm</code> for E&#8209;field fits, <code>V</code>
          otherwise.</td></tr>
      <tr><td><code>Ec1</code>, <code>Ec2</code></td><td>V/cm</td>
        <td>IEC decade window (only set for log&ndash;log fits; 0 for
          non&#8209;linear).</td></tr>
      <tr><td><code>n_window_I_lo_A</code>, <code>n_window_I_hi_A</code></td>
        <td>A</td>
        <td>Current bounds of the n&#8209;value window actually used.</td></tr>
      <tr><td><code>n_points_used</code></td><td>int</td>
        <td>Number of samples that entered the power&#8209;law fit.</td></tr>
    </table>

    <h2>Baseline decomposition</h2>
    <p><code>V<sub>total</sub> = V<sub>ofs</sub> + L · dI/dt + R · I +
       V<sub>c</sub> · (I / I<sub>c</sub>)<sup>n</sup></code></p>
    <table>
      <tr><th>Property</th><th>Unit</th><th>Description</th></tr>
      <tr><td><code>V_ofs</code></td><td>V</td>
        <td>Thermal/instrumental offset (median of the I&nbsp;≈&nbsp;0
          segment).</td></tr>
      <tr><td><code>V0_inductive</code></td><td>V</td>
        <td>Inductive voltage at the dI/dt&nbsp;window center
          (<code>L · dI/dt</code>).</td></tr>
      <tr><td><code>inductance_L_H</code></td><td>H</td>
        <td>Effective lead/sample inductance.</td></tr>
      <tr><td><code>R_or_rho</code></td><td>Ω or Ω/cm</td>
        <td>Resistive baseline; unit follows <code>R_unit</code>.</td></tr>
      <tr><td><code>R_unit</code></td><td>str</td>
        <td><code>Ω/cm</code> for E&#8209;field fits, <code>Ω</code>
          otherwise.</td></tr>
    </table>

    <h2>Diagnostic flags</h2>
    <table>
      <tr><th>Property</th><th>Type</th><th>Meaning</th></tr>
      <tr><td><code>ramp_inductive_ratio</code></td><td>float</td>
        <td>Inductive&nbsp;voltage / criterion&nbsp;voltage
          (<code>|L · dI/dt| / V<sub>c</sub></code>).</td></tr>
      <tr><td><code>ramp_too_fast</code></td><td>True/False</td>
        <td>Set when the inductive term dominates &mdash; lower dI/dt or
          extend the ramp.</td></tr>
      <tr><td><code>insufficient_n_points</code></td><td>True/False</td>
        <td>Set if the power&#8209;law window has too few samples
          (&lt;&nbsp;50 per IEC 61788).</td></tr>
      <tr><td><code>thermal_offset_applied</code></td><td>True/False</td>
        <td>Set if a non&#8209;zero <code>V<sub>ofs</sub></code> was
          subtracted.</td></tr>
      <tr><td><code>uses_sample_length</code></td><td>True/False</td>
        <td>True for E&#8209;field fits (with L<sub>s</sub>), False for
          V&#8209;based fits.</td></tr>
    </table>

    <h2>Recommended boundaries for a good fit</h2>
    <p>Use these ranges as a quality gate when reviewing a fit. They are
    derived from IEC&nbsp;61788&#8209;3 acceptance criteria and from typical
    HTS / LTS conductor performance at 77&nbsp;K&nbsp;/&nbsp;4.2&nbsp;K.</p>
    <table>
      <tr><th>Quantity</th><th>Excellent</th><th>Acceptable</th>
        <th>Suspect &mdash; recheck windows</th></tr>
      <tr><td><b>n&#8209;value</b><br><span style="color:#5a6472;">HTS tape
        (e.g. REBCO, BSCCO) at 77&nbsp;K</span></td>
        <td>30 – 60</td><td>20 – 30</td><td>&lt;&nbsp;15 or &gt;&nbsp;80</td></tr>
      <tr><td><b>n&#8209;value</b><br><span style="color:#5a6472;">LTS wire
        (e.g. NbTi, Nb<sub>3</sub>Sn) at 4.2&nbsp;K</span></td>
        <td>40 – 80</td><td>25 – 40</td><td>&lt;&nbsp;20 or &gt;&nbsp;120</td></tr>
      <tr><td><b>σ(n)&nbsp;/&nbsp;n</b><br>relative uncertainty on the
        n&#8209;value</td>
        <td>&lt;&nbsp;1&nbsp;%</td><td>1&nbsp;%&nbsp;–&nbsp;5&nbsp;%</td>
        <td>&gt;&nbsp;5&nbsp;%</td></tr>
      <tr><td><b>σ(I<sub>c</sub>)&nbsp;/&nbsp;I<sub>c</sub></b><br>relative
        uncertainty on the critical current</td>
        <td>&lt;&nbsp;0.1&nbsp;%</td><td>0.1&nbsp;%&nbsp;–&nbsp;0.5&nbsp;%</td>
        <td>&gt;&nbsp;1&nbsp;%</td></tr>
      <tr><td><b>R²</b> in log&#8209;log space</td>
        <td>&gt;&nbsp;0.999</td><td>0.99&nbsp;–&nbsp;0.999</td>
        <td>&lt;&nbsp;0.99</td></tr>
      <tr><td><b>n&#8209;window points</b> (<code>n_points_used</code>)</td>
        <td>≥&nbsp;200</td><td>50&nbsp;–&nbsp;200</td>
        <td>&lt;&nbsp;50 (IEC minimum)</td></tr>
      <tr><td><b>Ramp inductive ratio</b><br>
        (<code>ramp_inductive_ratio</code>)</td>
        <td>&lt;&nbsp;0.01</td><td>0.01&nbsp;–&nbsp;0.1</td>
        <td>&gt;&nbsp;0.1 → <code>ramp_too_fast</code></td></tr>
      <tr><td><b>Resistive baseline R</b><br>before the transition</td>
        <td>≲&nbsp;1&nbsp;µΩ (clean joint)</td>
        <td>1&nbsp;–&nbsp;100&nbsp;µΩ</td>
        <td>&gt;&nbsp;1&nbsp;mΩ (joint or contact problem)</td></tr>
      <tr><td><b>V<sub>ofs</sub></b> (thermal offset)</td>
        <td>|V<sub>ofs</sub>| &lt; 0.1·V<sub>c</sub></td>
        <td>0.1·V<sub>c</sub> – 0.5·V<sub>c</sub></td>
        <td>&gt;&nbsp;0.5·V<sub>c</sub> → recheck zero&#8209;I window</td></tr>
      <tr><td><b>I<sub>c</sub>&nbsp;/&nbsp;I<sub>max</sub></b></td>
        <td>0.6&nbsp;–&nbsp;0.9</td><td>0.4&nbsp;–&nbsp;0.95</td>
        <td>&lt;&nbsp;0.3 (under&#8209;driven) or &gt;&nbsp;0.98
          (no headroom above the criterion)</td></tr>
    </table>

    <div class="warn"><b>Reading these together:</b>
      <ul>
        <li>A small σ(n) but R² &lt; 0.99 usually means a wrong baseline,
          not a noisy n&#8209;value &mdash; widen the linear window.</li>
        <li>A large σ(I<sub>c</sub>) with a normal n typically points at too
          few points in the decade &mdash; sample faster or slow the
          ramp.</li>
        <li><code>ramp_too_fast = True</code> invalidates the n&#8209;value
          regardless of σ(n); the inductive term is contaminating the
          transition.</li>
      </ul>
    </div>

    <div class="tip"><b>Note:</b> Booleans are stored as the strings
    <code>"True"</code>/<code>"False"</code> for round&#8209;trip safety with
    LabVIEW and Origin readers.</div>
    """

    overview_pixmap = _build_fit_diagram_pixmap()
    loglog_pixmap = _build_loglog_diagram_pixmap()

    for title, html, resources in (
        ("Overview", overview_html,
         (("fit-diagram://overview", overview_pixmap),)),
        ("Log–log linear fit", loglog_html,
         (("fit-diagram://loglog", loglog_pixmap),)),
        ("How to fit", workflow_html, ()),
        ("Settings & options", options_html, ()),
        ("Metadata fields", metadata_html, ()),
    ):
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        for url, image in resources:
            browser.document().addResource(
                QTextDocument.ImageResource,
                QUrl(url),
                image,
            )
        browser.setHtml(html)
        tabs.addTab(browser, title)

    layout.addWidget(tabs)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.accept)
    btn_row.addWidget(close_btn)
    layout.addLayout(btn_row)

    app._data_fit_help_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

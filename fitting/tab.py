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
    DEFAULT_CHI_SQR_TOL,
    DEFAULT_DIDT_HIGH_FRAC,
    DEFAULT_DIDT_LOW_FRAC,
    DEFAULT_EC_V_PER_CM,
    DEFAULT_EC1_V_PER_CM,
    DEFAULT_EC2_V_PER_CM,
    DEFAULT_FIT_METHOD,
    DEFAULT_IC_TOLERANCE,
    DEFAULT_LINEAR_HIGH_FRAC,
    DEFAULT_LINEAR_LOW_FRAC,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_POWER_LOW_FRAC,
    DEFAULT_POWER_V_FRAC,
    DEFAULT_VC_VOLTS,
    DEFAULT_ZERO_I_FRAC,
    FIT_METHOD_LOG_LOG,
    FIT_METHOD_NONLINEAR,
    FitSettings,
    MIN_N_WINDOW_POINTS,
    RAMP_INDUCTIVE_WARN_RATIO,
    robust_view_range,
    run_full_fit,
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
                      high_label, high_pct, high_x, base_row: int = 0):
    """Show checkbox on base_row, percents on base_row+1, X values on base_row+2."""
    layout.addWidget(show_cb, base_row, 0, 1, 4)
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


def _capture_fit_window_profile(app) -> dict:
    return {
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
    }


def _apply_fit_window_profile(app, profile: dict) -> None:
    if not profile:
        return
    for widget, key in (
        (app.data_fit_didt_low, "didt_low"),
        (app.data_fit_didt_high, "didt_high"),
        (app.data_fit_linear_low, "linear_low"),
        (app.data_fit_linear_high, "linear_high"),
        (app.data_fit_power_low, "power_low"),
        (app.data_fit_power_vfrac, "power_vfrac"),
        (app.data_fit_max_iter, "max_iter"),
        (app.data_fit_ic_tol, "ic_tol"),
        (app.data_fit_chi_tol, "chi_tol"),
        (app.data_fit_vc_input, "vc"),
    ):
        if key in profile:
            _set_silently(widget, str(profile[key]))
    if "zero_i_frac" in profile and getattr(app, "data_fit_zero_i_frac", None) is not None:
        _set_silently(app.data_fit_zero_i_frac, str(profile["zero_i_frac"]))
    if "subtract_vofs" in profile and getattr(app, "data_fit_subtract_vofs_cb", None) is not None:
        app.data_fit_subtract_vofs_cb.setChecked(bool(profile["subtract_vofs"]))
    sync_region_to_inputs(app)


def _float_from(widget: QLineEdit, fallback: float, as_fraction: bool = False) -> float:
    try:
        v = float(widget.text())
    except (TypeError, ValueError):
        return fallback
    if as_fraction:
        return v / 100.0
    return v


def _read_time_channel(tdms_file):
    if "RawData" in tdms_file and "Time" in tdms_file["RawData"]:
        return np.asarray(tdms_file["RawData"]["Time"][:], dtype=float)
    if "Time" in tdms_file and "Time" in tdms_file["Time"]:
        return np.asarray(tdms_file["Time"]["Time"][:], dtype=float)
    return None


def _read_channel_metadata(channel) -> dict:
    """Extract Scale_Factor / Offset / Voltage_Tab_Distance metadata.

    Voltage_Tab_Distance is written by the Tape and Cable presets of
    DAQUniversal (see TDMS metadata spec). When present, the Data Fitting
    tab auto-populates the voltage-tap separation and enables the E-field
    path so Ic is computed per IEC 61788 with a 1 uV/cm criterion.
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
    v_tap_raw = props.get("Voltage_Tab_Distance", "")
    v_tap: Optional[float] = None
    if v_tap_raw not in ("", None):
        try:
            v_tap = float(v_tap_raw)
            if v_tap <= 0:
                v_tap = None
        except (TypeError, ValueError):
            v_tap = None
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

    # --- data source -----------------------------------------------------
    def load_recording(self, path: str) -> tuple[bool, str]:
        self.channel_cache.clear()
        self.channel_metadata.clear()
        self.channel_names = []
        self.time_array = None
        if not path or not os.path.exists(path):
            return False, "No recording found. Click 'Load File…' to choose a TDMS."
        try:
            with TdmsFile.read(path) as tdms_file:
                self.time_array = _read_time_channel(tdms_file)
                names: list[str] = []
                for group in tdms_file.groups():
                    for channel in group.channels():
                        name = getattr(channel, "name", "")
                        if not name or name.lower() == "time":
                            continue
                        if name in self.channel_cache:
                            continue
                        self.channel_cache[name] = np.asarray(channel[:], dtype=float)
                        self.channel_metadata[name] = _read_channel_metadata(channel)
                        names.append(name)
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
        app.data_fit_time_cb, app.data_fit_x_cb, app.data_fit_y_cb,
    ):
        if hasattr(widget, "editingFinished"):
            widget.editingFinished.connect(lambda: _on_transform_inputs_changed(app))
        else:
            widget.currentIndexChanged.connect(lambda _: _on_transform_inputs_changed(app))
    for w in (app.data_fit_max_iter, app.data_fit_ic_tol, app.data_fit_chi_tol, app.data_fit_vc_input):
        w.editingFinished.connect(lambda: _save_active_curve_profile(app))
    app.data_fit_graph_btn.clicked.connect(lambda: _open_graph_settings(app))
    app.data_fit_save_preset_btn.clicked.connect(lambda: _save_preset(app))
    app.data_fit_load_preset_btn.clicked.connect(lambda: _load_preset(app))
    app.data_fit_show_didt.toggled.connect(lambda _: _update_band_states(app))
    app.data_fit_show_linear.toggled.connect(lambda _: _update_band_states(app))
    app.data_fit_show_power.toggled.connect(lambda _: (_update_band_states(app), refresh_preview(app)))
    app.data_fit_export_btn.clicked.connect(lambda: _open_export_dialog(app))
    app.data_fit_add_plot_btn.clicked.connect(lambda: (_add_plot_from_current(app), robust_view(app)))
    app.data_fit_add_corrected_btn.clicked.connect(lambda: (_add_corrected_curve_from_last_fit(app), robust_view(app)))
    app.data_fit_plot_summary_btn.clicked.connect(lambda: _open_plot_summary(app))
    app.data_fit_curve_profile_cb.currentIndexChanged.connect(lambda _: _on_curve_profile_changed(app))
    app.data_fit_method_loglog_rb.toggled.connect(lambda _: _on_fit_method_changed(app))
    app.data_fit_method_nonlinear_rb.toggled.connect(lambda _: _on_fit_method_changed(app))
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
    if getattr(app, "data_fit_subtract_vofs_cb", None) is not None:
        app.data_fit_subtract_vofs_cb.setChecked(True)
    if getattr(app, "data_fit_zero_i_frac", None) is not None:
        app.data_fit_zero_i_frac.setText(f"{DEFAULT_ZERO_I_FRAC * 100:.2f}")
    app.data_fit_didt_low.setText(f"{DEFAULT_DIDT_LOW_FRAC * 100:.2f}")
    app.data_fit_didt_high.setText(f"{DEFAULT_DIDT_HIGH_FRAC * 100:.2f}")
    app.data_fit_linear_low.setText(f"{DEFAULT_LINEAR_LOW_FRAC * 100:.2f}")
    app.data_fit_linear_high.setText(f"{DEFAULT_LINEAR_HIGH_FRAC * 100:.2f}")
    if DEFAULT_FIT_METHOD == FIT_METHOD_LOG_LOG:
        app.data_fit_method_loglog_rb.setChecked(True)
        app.data_fit_power_low.setText(f"{DEFAULT_EC1_V_PER_CM * 1.0e6:g}")
        app.data_fit_power_vfrac.setText(f"{DEFAULT_EC2_V_PER_CM * 1.0e6:g}")
    else:
        app.data_fit_method_nonlinear_rb.setChecked(True)
        app.data_fit_power_low.setText(f"{DEFAULT_POWER_LOW_FRAC * 100:.2f}")
        app.data_fit_power_vfrac.setText(f"{DEFAULT_POWER_V_FRAC * 100:.2f}")
    app.data_fit_show_didt.setChecked(False)
    app.data_fit_show_linear.setChecked(False)
    app.data_fit_show_power.setChecked(False)
    for entry in list(getattr(app, "data_fit_curves", [])):
        item = entry.get("plot_item")
        if item is not None:
            app.data_fit_plot.removeItem(item)
        fit_item = entry.get("fit_plot_item")
        if fit_item is not None:
            app.data_fit_plot.removeItem(fit_item)
    app.data_fit_curves = []
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


def _curve_profile_key_from_ui(app) -> str:
    key = app.data_fit_curve_profile_cb.currentData()
    return str(key) if key else "__preview__"


def _save_active_curve_profile(app) -> None:
    key = _curve_profile_key_from_ui(app)
    profiles = getattr(app, "data_fit_curve_profiles", {})
    profiles[key] = _capture_fit_window_profile(app)
    app.data_fit_curve_profiles = profiles


def _on_curve_profile_changed(app) -> None:
    key = _curve_profile_key_from_ui(app)
    profiles = getattr(app, "data_fit_curve_profiles", {})
    if key not in profiles:
        profiles[key] = _capture_fit_window_profile(app)
        app.data_fit_curve_profiles = profiles
    _apply_fit_window_profile(app, profiles.get(key, {}))


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


def setup_data_fitting_tab_layout(app):
    root = QHBoxLayout(app.ui_state.data_fitting_tab)

    left_widget = QWidget()
    left = QVBoxLayout(left_widget)
    left_widget.setMaximumWidth(504)

    file_row = QHBoxLayout()
    app.data_fit_path_label = QLabel("No file loaded.")
    app.data_fit_path_label.setStyleSheet("color: gray;")
    app.data_fit_clear_btn = QPushButton("Clear all")
    app.data_fit_clear_btn.setToolTip("Reset data fitting to defaults and clear loaded curves/preview state.")
    app.data_fit_load_btn = QPushButton("Load File…")
    app.data_fit_refresh_btn = QPushButton("Use Current Recording")
    app.data_fit_export_btn = QPushButton("Export plot…")
    app.data_fit_export_btn.setToolTip("Export the plot in a chosen image or data format (PNG, SVG, CSV, …).")
    file_row.addWidget(app.data_fit_clear_btn)
    file_row.addWidget(app.data_fit_load_btn)
    file_row.addWidget(app.data_fit_refresh_btn)
    file_row.addWidget(app.data_fit_export_btn)
    left.addLayout(file_row)
    left.addWidget(app.data_fit_path_label)

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
    for compact_btn in (app.data_fit_load_metadata_btn, app.data_fit_add_plot_btn, app.data_fit_plot_summary_btn):
        compact_btn.setMinimumHeight(24)
        compact_btn.setStyleSheet("padding:2px 8px; font-size:11px;")
    ch_grid.addWidget(app.data_fit_load_metadata_btn, 5, 0)
    ch_grid.addWidget(app.data_fit_add_plot_btn, 5, 1)
    ch_grid.addWidget(app.data_fit_plot_summary_btn, 5, 2, 1, 2)

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
    _fill_window_grid(
        linear_layout, app.data_fit_show_linear,
        low_label="Low (%)", low_pct=app.data_fit_linear_low, low_x=app.data_fit_linear_low_x,
        high_label="High (%)", high_pct=app.data_fit_linear_high, high_x=app.data_fit_linear_high_x,
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
    power_layout.addWidget(app.data_fit_add_corrected_btn, 5, 3)

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
        base_row=2,
    )
    # Keep references to the text labels so the method-mode handler can
    # swap them when the user switches between IEC and non-linear modes.
    # (row 3 holds the Low/High value labels; row 4 holds the X-value row.)
    app.data_fit_power_low_label = power_layout.itemAtPosition(3, 0).widget()
    app.data_fit_power_high_label = power_layout.itemAtPosition(3, 2).widget()
    app.data_fit_power_low_x_label = power_layout.itemAtPosition(4, 0).widget()
    app.data_fit_power_high_x_label = power_layout.itemAtPosition(4, 2).widget()
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

    app.data_fit_run_btn = QPushButton("Run Fit")
    app.data_fit_run_btn.setStyleSheet("font-weight: bold; background-color: #0078D7; color: white; padding: 8px;")
    left.addWidget(app.data_fit_run_btn)

    preset_row = QHBoxLayout()
    app.data_fit_save_preset_btn = QPushButton("Save preset…")
    app.data_fit_save_preset_btn.setToolTip("Save the current fit-window preset to a JSON file.")
    app.data_fit_load_preset_btn = QPushButton("Load preset…")
    app.data_fit_load_preset_btn.setToolTip("Load a fit-window preset from a JSON file.")
    preset_row.addWidget(app.data_fit_save_preset_btn)
    preset_row.addWidget(app.data_fit_load_preset_btn)
    left.addLayout(preset_row)

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

    toolbar = QGridLayout()
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
    toolbar.addWidget(app.data_fit_graph_btn, 0, 0)
    toolbar.addWidget(app.data_fit_robust_view_btn, 0, 1)
    toolbar.addWidget(app.data_fit_zoom_mode_btn, 1, 0)
    toolbar.addWidget(app.data_fit_reset_view_btn, 1, 1)
    left_header.addLayout(toolbar)

    app.data_fit_result_text = QTextEdit()
    app.data_fit_result_text.setReadOnly(True)
    app.data_fit_result_text.setPlaceholderText("Fit results will appear here.")
    app.data_fit_result_text.setMinimumHeight(60)
    app.data_fit_result_text.setMaximumHeight(80)
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
    if method == FIT_METHOD_LOG_LOG:
        criterion_value = ec2  # Ic is reported at E = Ec2.

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
    )
    return settings


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


def open_file_dialog(app):
    runtime_state = getattr(app, "runtime_state", None)
    start_dir = getattr(runtime_state, "output_folder", "") or ""
    path, _ = QFileDialog.getOpenFileName(app, "Select TDMS recording", start_dir, "TDMS Files (*.tdms);;All Files (*)")
    if not path:
        return
    ok, msg = app.data_fit_controller.load_recording(path)
    app.data_fit_path_label.setText(msg)
    app.data_fit_path_label.setStyleSheet("color: black;" if ok else "color: #b35a00;")
    if ok:
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


def refresh_current_recording(app):
    path = getattr(app, "current_tdms_filepath", "") or ""
    ok, msg = app.data_fit_controller.load_recording(path)
    app.data_fit_path_label.setText(msg)
    app.data_fit_path_label.setStyleSheet("color: black;" if ok else "color: #b35a00;")
    if ok:
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


_Y_TITLE_VOLTAGE = "Voltage (V)"
_Y_TITLE_E_FIELD = "Electric field (V/cm)"


def _update_y_axis_label(app):
    """Swap the Y-axis title. Also update graph_settings.title_left.text so the
    graph-settings dialog / apply_graph_settings stay in sync.
    If the user has customised the title to something other than one of the two
    auto values, we leave it alone.
    """
    desired = _Y_TITLE_E_FIELD if app.data_fit_use_length_cb.isChecked() else _Y_TITLE_VOLTAGE
    if _current_plot_scale(app) == _PLOT_SCALE_LOGLOG:
        app.data_fit_plot.setLabel("left", f"{desired}  [log scale]")
    else:
        app.data_fit_plot.setLabel("left", desired)
    settings = getattr(app, "data_fit_graph_settings", None)
    if settings is not None and settings.title_left.text in ("", _Y_TITLE_VOLTAGE, _Y_TITLE_E_FIELD):
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
    # Vc input: disabled in log-log mode (criterion is Ec2 from the Step 4
    # editors) and when the E-field path is active it is Ec not Vc anyway.
    app.data_fit_vc_input.setEnabled(not is_loglog)
    app.data_fit_vc_label.setEnabled(not is_loglog)
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
    """Apply IEC-standard defaults when switching into log-log mode; restore
    the legacy fractions when switching into non-linear mode. Only rewrite
    the editors when their current content matches the other mode's default,
    so user-customised values survive toggling back and forth.
    """
    method = _active_fit_method(app)
    low_txt = app.data_fit_power_low.text().strip()
    high_txt = app.data_fit_power_vfrac.text().strip()
    if method == FIT_METHOD_LOG_LOG:
        if low_txt in ("", f"{DEFAULT_POWER_LOW_FRAC * 100:.2f}"):
            _set_silently(app.data_fit_power_low, f"{DEFAULT_EC1_V_PER_CM * 1.0e6:g}")
        if high_txt in ("", f"{DEFAULT_POWER_V_FRAC * 100:.2f}"):
            _set_silently(app.data_fit_power_vfrac, f"{DEFAULT_EC2_V_PER_CM * 1.0e6:g}")
    else:
        if low_txt in ("", f"{DEFAULT_EC1_V_PER_CM * 1.0e6:g}"):
            _set_silently(app.data_fit_power_low, f"{DEFAULT_POWER_LOW_FRAC * 100:.2f}")
        if high_txt in ("", f"{DEFAULT_EC2_V_PER_CM * 1.0e6:g}"):
            _set_silently(app.data_fit_power_vfrac, f"{DEFAULT_POWER_V_FRAC * 100:.2f}")
    _update_method_mode_ui(app)
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


def _apply_axis_labels_for_scale(app, to_loglog: bool) -> None:
    """Rewrite axis titles to make the log transform explicit.

    Tick labels are handled by EngineeringAxisItem (which shows SI-prefixed
    values in both linear and log mode), so here we only adjust the title.
    """
    plot_widget = getattr(app, "data_fit_plot", None)
    if plot_widget is None:
        return
    plot_item = plot_widget.getPlotItem()
    use_length = (
        getattr(app, "data_fit_use_length_cb", None)
        and app.data_fit_use_length_cb.isChecked()
    )
    y_base = _Y_TITLE_E_FIELD if use_length else _Y_TITLE_VOLTAGE
    if to_loglog:
        plot_item.setLabel("bottom", "Current (A)  [log scale]")
        plot_item.setLabel("left", f"{y_base}  [log scale]")
    else:
        plot_item.setLabel("bottom", "Current (A)")
        plot_item.setLabel("left", y_base)


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
    t = transformed["time"]
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
    t_view = t[:n] if (t is not None and np.asarray(t).size >= n) else None
    _update_fit_bands(app, x[:n], y[:n], t=t_view)
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


def _update_fit_bands(app, x: np.ndarray, y: np.ndarray, *, t: Optional[np.ndarray] = None) -> None:
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
    ec1 = ec2 = None
    if _active_fit_method(app) == FIT_METHOD_LOG_LOG:
        # The Step 4 editors hold Ec1/Ec2 (µV/cm or µV). A full band would
        # require baseline-subtracted E_sc = y − V0 − R·I which the user
        # does not have until Step 3 has run. As a visual hint, shade the
        # current range where y alone exceeds Ec1·L_v / Ec1 and Ec2·L_v /
        # Ec2 — fine for coarse feedback, exact band is shown post-fit.
        has_length = app.data_fit_use_length_cb.isChecked()
        to_si = 1.0e-6 if has_length else 1.0e-3
        ec1 = _float_from(app.data_fit_power_low, DEFAULT_EC1_V_PER_CM * 1.0e6) * to_si
        ec2 = _float_from(app.data_fit_power_vfrac, DEFAULT_EC2_V_PER_CM * 1.0e6) * to_si
        pow_pair = None
        # Prefer the exact IEC window from the same Step 1..4 pipeline used by
        # the fit engine. This keeps the orange "Show / edit" band aligned with
        # the I-window reported after Run Fit.
        if t is not None and y is not None and np.asarray(t).size == np.asarray(x).size:
            try:
                preview_result = run_full_fit(np.asarray(t), np.asarray(x), np.asarray(y), _settings_from_inputs(app))
                if preview_result.ok:
                    lo, hi = preview_result.n_window_I
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        pow_pair = (float(lo), float(hi))
            except Exception:
                pow_pair = None
        # Fallback for cases where a preview fit is not possible yet.
        if pow_pair is None and y is not None and y.size:
            above_1 = np.where(y >= ec1)[0]
            above_2 = np.where(y >= ec2)[0]
            pow_lo = float(x[above_1[0]]) if above_1.size else x_max
            pow_hi = float(x[above_2[0]]) if above_2.size else x_max
            if pow_hi <= pow_lo:
                pow_hi = pow_lo + max(1e-12, 0.01 * span)
            pow_pair = (pow_lo, pow_hi)
        if pow_pair is not None:
            band_pairs.append((app.data_fit_band_power, pow_pair))
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
        if y is None or y.size == 0:
            return
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        idx_lo = int(np.argmin(np.abs(x_arr - lo)))
        idx_hi = int(np.argmin(np.abs(x_arr - hi)))
        ec1 = max(float(y_arr[idx_lo]), 1.0e-30)
        ec2 = max(float(y_arr[idx_hi]), ec1 * 1.000001)
        from_si = 1.0e6 if app.data_fit_use_length_cb.isChecked() else 1.0e3
        _set_silently(app.data_fit_power_low, f"{ec1 * from_si:.6g}")
        _set_silently(app.data_fit_power_vfrac, f"{ec2 * from_si:.6g}")
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
    pairs = [
        ("time", app.data_fit_time_cb, app.data_fit_time_scale, app.data_fit_time_offset),
        ("x", app.data_fit_x_cb, app.data_fit_x_scale, app.data_fit_x_offset),
        ("y", app.data_fit_y_cb, app.data_fit_y_scale, app.data_fit_y_offset),
    ]
    for _, combo, scale_input, offset_input in pairs:
        name = combo.currentText()
        if not name or name == "Time":
            continue
        meta = controller.get_metadata(name)
        scale_input.setText(f"{meta['scale']:g}")
        offset_input.setText(f"{meta['offset']:g}")
    _apply_voltage_tap_from_metadata(app)
    refresh_preview(app)


def _apply_voltage_tap_from_metadata(app) -> None:
    """Pick up Voltage_Tab_Distance from the Y channel (Tape/Cable preset).

    If present and > 0: check the voltage-tap-separation box, populate the
    distance, and keep Ec at the IEC default (1 uV/cm). If absent:
    uncheck the box so the tool does not fit in E-field mode with a stale
    sample length.
    """
    controller = getattr(app, "data_fit_controller", None)
    if controller is None:
        return
    y_name = app.data_fit_y_cb.currentText()
    if not y_name:
        return
    meta = controller.get_metadata(y_name)
    v_tap = meta.get("voltage_tap_cm")
    cb = app.data_fit_use_length_cb
    cb.blockSignals(True)
    try:
        if v_tap and v_tap > 0:
            cb.setChecked(True)
            app.data_fit_length_input.setText(f"{float(v_tap):g}")
            app.data_fit_vc_input.setText(f"{DEFAULT_EC_V_PER_CM * 1.0e6:.6g}")
        else:
            cb.setChecked(False)
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
    lines.append(f"method        = {method_label}")
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


def _fit_result_properties(result) -> dict:
    """Serialise a FitResult into a flat dict of TDMS-writable properties."""
    is_loglog = getattr(result, "fit_method", FIT_METHOD_NONLINEAR) == FIT_METHOD_LOG_LOG
    props = {
        "method": "IEC 61788 log-log (decade n-value)" if is_loglog else "Non-linear V-I",
        "method_compliant": "IEC 61788-3" if is_loglog else "legacy",
        "fit_timestamp": datetime.now().isoformat(timespec="seconds"),
        # Primary outputs per IEC 61788 / user request.
        "Ic_A": float(result.Ic),
        "sigma_Ic_A": float(getattr(result, "sigma_Ic", 0.0)),
        "n_value": float(result.n_value),
        "sigma_n": float(getattr(result, "sigma_n", 0.0)),
        "r_squared": float(getattr(result, "r_squared", 0.0)),
        "chi_squared": float(result.chi_sqr),
        "di_dt_A_per_s": float(result.di_dt),
        # Criterion: Ec (V/cm) when uses_sample_length, else Vc (V).
        "criterion_value": float(result.criterion),
        "criterion_unit": "V/cm" if result.uses_sample_length else "V",
        "criterion_name": "Ec" if result.uses_sample_length else "Vc",
        # IEC decade window (0 for non-linear fits).
        "Ec1": float(getattr(result, "ec1", 0.0)),
        "Ec2": float(getattr(result, "ec2", 0.0)),
        "n_window_I_lo_A": float(result.n_window_I[0]),
        "n_window_I_hi_A": float(result.n_window_I[1]),
        "n_points_used": int(getattr(result, "n_points_used", 0)),
        # Baseline decomposition: V_total = V_ofs + L·dI/dt + R·I + Vc·(I/Ic)^n.
        "V_ofs": float(getattr(result, "V_ofs", 0.0)),
        "V0_inductive": float(result.V0),
        "inductance_L_H": float(result.inductance_L),
        "R_or_rho": float(result.R),
        "R_unit": "Ω/cm" if result.uses_sample_length else "Ω",
        # Diagnostic flags the user asked to propagate to channel metadata.
        "ramp_inductive_ratio": float(getattr(result, "ramp_inductive_ratio", 0.0)),
        "ramp_too_fast": bool(getattr(result, "ramp_too_fast", False)),
        "insufficient_n_points": bool(getattr(result, "insufficient_n_points", False)),
        "thermal_offset_applied": bool(getattr(result, "thermal_offset_applied", False)),
        "uses_sample_length": bool(result.uses_sample_length),
    }
    # Booleans round-trip more reliably as strings in TDMS consumers (LabVIEW,
    # Origin) — keep human-readable "True"/"False" instead of raw bool.
    for k in ("ramp_too_fast", "insufficient_n_points",
              "thermal_offset_applied", "uses_sample_length"):
        props[k] = "True" if props[k] else "False"
    return props


def _write_fit_report_tdms(app, results: list[tuple[str, object]]) -> Optional[str]:
    """Write fit results to ``<source_stem>_fit_report.tdms`` next to the source.

    Each OK ``FitResult`` becomes one channel in the ``FitResults`` group, with
    the fit parameters (Ic, n, Ec, dI/dt, σ(Ic), σ(n), R², warnings, …) attached
    as channel properties. Channels for the same label are overwritten; other
    channels in an existing side-car are preserved.

    Returns the path written (or ``None`` if no source file or no OK results).
    Any I/O error is swallowed after logging — a failed report must not break
    an otherwise successful fit.
    """
    controller = getattr(app, "data_fit_controller", None)
    src_path = getattr(controller, "tdms_path", "") if controller is not None else ""
    ok_results = [(lbl, r) for lbl, r in results if r is not None and getattr(r, "ok", False)]
    if not src_path or not ok_results:
        return None
    src = Path(src_path)
    report_path = src.with_name(f"{src.stem}_fit_report.tdms")

    # Preserve channel entries from prior runs that this fit didn't touch.
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

    new_entries = {lbl: _fit_result_properties(r) for lbl, r in ok_results}
    merged = {**existing, **new_entries}

    # nptdms' TdmsWriter takes a flat list of objects per segment. Empty
    # channels aren't allowed by the spec, so we include a single NaN sample
    # as a sentinel — consumers that only read properties ignore the data.
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


def run_fit(app):
    controller = app.data_fit_controller
    if not controller.channel_names:
        QMessageBox.warning(app, "Data Fitting", "Load a recording first.")
        return
    try:
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
        ok_results = []
        last_show_criterion = True
        last_show_ic = False
        for entry in included:
            try:
                result = run_full_fit(entry["t"], entry["x"], entry["y"], settings)
            except Exception as exc:
                traceback.print_exc()
                lines.append(f"[{entry['label']}] FAILED: {exc}")
                continue
            entry["fit_result"] = result
            if result.ok:
                last_ok = result
                ok_results.append((entry.get("label", "Curve"), result))
                last_show_criterion = bool(entry.get("show_criterion", False))
                last_show_ic = bool(entry.get("show_ic", False))
                _upsert_fit_curve_entry(app, entry, result)
                lines.append(f"[{entry['label']}]\n" + _format_result(result) + "\n")
            else:
                lines.append(f"[{entry['label']}] {result.message}")
        app.data_fit_result_text.setPlainText("\n".join(lines) or "No curves included in fit.")
        if last_ok is not None:
            _show_fit_overlays(
                app, last_ok, table_entries=ok_results,
                show_criterion=last_show_criterion, show_ic=last_show_ic,
            )
            _plot_residuals(app, last_ok)
            _post_fit_warnings(app, last_ok, settings)
            report_path = _write_fit_report_tdms(app, ok_results)
            if report_path:
                current = app.data_fit_result_text.toPlainText()
                app.data_fit_result_text.setPlainText(
                    current + f"\nFit report written to: {report_path}"
                )
        else:
            _hide_fit_overlays(app)
            _show_warning(app, "No curve produced a successful fit.", severity="error")
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
        return
    app.data_fit_result_text.setPlainText(_format_result(result))
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
    report_path = _write_fit_report_tdms(
        app, [(app.data_fit_y_cb.currentText(), result)]
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
    dialog = GraphSettingsDialog(app.data_fit_graph_settings, app)
    if dialog.exec_() == dialog.Accepted:
        app.data_fit_graph_settings = dialog.result_settings()
        refresh_preview(app)


def _open_export_dialog(app) -> None:
    default_dir = _preset_dir(app)
    dialog = ExportPlotDialog(app.data_fit_plot.getPlotItem(), app, default_dir)
    dialog.exec_()


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
    """Add Y_corrected = Y - (V0 + R*I) using the most relevant successful fit."""
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
        QMessageBox.warning(
            app,
            "Data Fitting",
            "Run a successful fit first, then click 'Add corrected curve'.",
        )
        return

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
        QMessageBox.warning(app, "Data Fitting", "No points available to build corrected curve.")
        return
    x = x[:n]
    y = y[:n]
    t = t[:n] if t.size else np.asarray([])
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
    table = QTableWidget(len(curves), 8)
    table.setHorizontalHeaderLabels(["Color", "Label", "Skip pts", "Avg", "Tap dist (cm)", "Effective rate", "Include", "Actions"])
    table.horizontalHeader().setStretchLastSection(False)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    table.setColumnWidth(0, 90)
    table.setColumnWidth(2, 72)
    table.setColumnWidth(3, 70)
    table.setColumnWidth(4, 100)
    table.setColumnWidth(5, 130)
    table.setColumnWidth(6, 70)
    table.setColumnWidth(7, 250)

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
        if bool(entry.get("is_fit_result", False)):
            tap.setEnabled(False)

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
        def on_tap():
            if bool(entry.get("is_fit_result", False)):
                tap.setText("—")
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
        tap.editingFinished.connect(on_tap)
        table.setCellWidget(row, 4, tap)
        rate_item.setText(_rate_text_for_entry())
        table.setCellWidget(row, 5, rate_item)
        include = QCheckBox()
        include.setChecked(entry.get("include_in_fit", True))
        if is_preview:
            include.toggled.connect(lambda v: setattr(app, "data_fit_preview_include_in_fit", bool(v)))
        else:
            include.toggled.connect(lambda v: entry.update(include_in_fit=bool(v)))
        table.setCellWidget(row, 6, include)
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
        table.setCellWidget(row, 7, actions_widget)

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

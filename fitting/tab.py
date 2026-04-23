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
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from nptdms import TdmsFile
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .service import (
    DEFAULT_CHI_SQR_TOL,
    DEFAULT_DIDT_HIGH_FRAC,
    DEFAULT_DIDT_LOW_FRAC,
    DEFAULT_IC_TOLERANCE,
    DEFAULT_LINEAR_HIGH_FRAC,
    DEFAULT_LINEAR_LOW_FRAC,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_POWER_LOW_FRAC,
    DEFAULT_POWER_V_FRAC,
    DEFAULT_VC_VOLTS,
    FitSettings,
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
                      high_label, high_pct, high_x):
    """Show checkbox on row 0, percents row 1, X values row 2."""
    layout.addWidget(show_cb, 0, 0, 1, 4)
    layout.addWidget(QLabel(low_label), 1, 0)
    layout.addWidget(low_pct, 1, 1)
    layout.addWidget(QLabel(high_label), 1, 2)
    layout.addWidget(high_pct, 1, 3)
    layout.addWidget(QLabel("Low (X)"), 2, 0)
    layout.addWidget(low_x, 2, 1)
    layout.addWidget(QLabel("High (X)"), 2, 2)
    layout.addWidget(high_x, 2, 3)


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
    """Extract Scale_Factor / Offset metadata written by the TDMS writer.

    The TDMS data is already stored in scaled engineering units (see
    math_processing_worker), so returning the metadata here is informational:
    the user can apply an additional scale/offset on top if needed.
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
    return {"scale": scale, "offset": offset}


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
        return self.channel_metadata.get(name, {"scale": 1.0, "offset": 0.0})

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
    app.data_fit_show_power.toggled.connect(lambda _: _update_band_states(app))
    app.data_fit_export_btn.clicked.connect(lambda: _open_export_dialog(app))
    app.data_fit_add_plot_btn.clicked.connect(lambda: (_add_plot_from_current(app), robust_view(app)))
    app.data_fit_plot_summary_btn.clicked.connect(lambda: _open_plot_summary(app))
    app.data_fit_curve_profile_cb.currentIndexChanged.connect(lambda _: _on_curve_profile_changed(app))


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
    app.data_fit_use_length_cb.setChecked(False)
    app.data_fit_length_input.setText("1.0")
    app.data_fit_vc_input.setText(f"{DEFAULT_VC_VOLTS * 1000:.6g}")
    app.data_fit_didt_low.setText(f"{DEFAULT_DIDT_LOW_FRAC * 100:.2f}")
    app.data_fit_didt_high.setText(f"{DEFAULT_DIDT_HIGH_FRAC * 100:.2f}")
    app.data_fit_linear_low.setText(f"{DEFAULT_LINEAR_LOW_FRAC * 100:.2f}")
    app.data_fit_linear_high.setText(f"{DEFAULT_LINEAR_HIGH_FRAC * 100:.2f}")
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
    left_widget.setMaximumWidth(420)

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
    app.data_fit_load_metadata_btn = QPushButton("Load TDMS scale/offset")
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
    ch_grid.addWidget(app.data_fit_add_plot_btn, 6, 0, 1, 2)
    ch_grid.addWidget(app.data_fit_plot_summary_btn, 6, 2, 1, 2)

    # Sample length + criterion widgets.
    app.data_fit_use_length_cb = QCheckBox("Y is E (V/cm)")
    app.data_fit_length_input = QLineEdit("1.0")
    app.data_fit_length_input.setMaximumWidth(80)
    app.data_fit_vc_input = QLineEdit(f"{DEFAULT_VC_VOLTS * 1000:.6g}")
    app.data_fit_vc_input.setMaximumWidth(80)
    app.data_fit_vc_label = QLabel("Vc (mV):")
    ch_grid.addWidget(app.data_fit_use_length_cb, 4, 1)
    ch_grid.addWidget(QLabel("Length (cm):"), 4, 2)
    ch_grid.addWidget(app.data_fit_length_input, 4, 3)
    ch_grid.addWidget(app.data_fit_load_metadata_btn, 5, 0, 1, 2)
    app.data_fit_avg_rate_label.setVisible(False)

    profile_group = QGroupBox("Active fitting settings")
    profile_layout = QHBoxLayout(profile_group)
    profile_layout.addWidget(QLabel("Curve label:"))
    app.data_fit_curve_profile_cb = QComboBox()
    app.data_fit_curve_profile_cb.setToolTip("Select curve label to edit/load fit-window settings for that curve.")
    profile_layout.addWidget(app.data_fit_curve_profile_cb, stretch=1)
    left.addWidget(profile_group)

    didt_group = QGroupBox("Step 1: di/dt window (fraction of Imax)")
    didt_layout = QGridLayout(didt_group)
    didt_goal = QLabel('<b>Goal:</b> estimate <b>dI/dt</b> from the linear current ramp.')
    didt_goal.setTextFormat(Qt.RichText)
    didt_layout.addWidget(didt_goal, 3, 0, 1, 4)
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

    linear_group = QGroupBox("Step 2: Linear baseline window (fraction of Imax)")
    linear_layout = QGridLayout(linear_group)
    app.data_fit_linear_goal = QLabel('<b>Goal:</b> fit the linear part to get <b>R</b> and <b>L</b>.')
    app.data_fit_linear_goal.setTextFormat(Qt.RichText)
    linear_layout.addWidget(app.data_fit_linear_goal, 3, 0, 1, 4)
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

    power_group = QGroupBox("Step 3: Power-law window")
    power_goal = QLabel('<b>Goal:</b> fit the superconducting transition to get <b>Ic</b> and <b>n</b>.')
    power_goal.setTextFormat(Qt.RichText)
    power_layout = QGridLayout(power_group)
    app.data_fit_power_low = _percent_edit(DEFAULT_POWER_LOW_FRAC)
    app.data_fit_power_vfrac = _percent_edit(DEFAULT_POWER_V_FRAC)
    app.data_fit_power_low_x = _xvalue_edit()
    app.data_fit_power_high_x = _xvalue_edit()
    app.data_fit_show_power = QCheckBox("Show / edit")
    app.data_fit_show_power.setChecked(False)
    app.data_fit_show_power.setToolTip("Show/hide the orange band for this window on the plot.")
    _fill_window_grid(
        power_layout, app.data_fit_show_power,
        low_label="Low (% of Imax)", low_pct=app.data_fit_power_low, low_x=app.data_fit_power_low_x,
        high_label="High (% of Vmax)", high_pct=app.data_fit_power_vfrac, high_x=app.data_fit_power_high_x,
    )
    power_layout.addWidget(power_goal, 3, 0, 1, 4)
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

    app.data_fit_result_text = QTextEdit()
    app.data_fit_result_text.setReadOnly(True)
    app.data_fit_result_text.setPlaceholderText("Fit results will appear here.")
    app.data_fit_result_text.setMaximumHeight(240)
    left.addWidget(app.data_fit_result_text)

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
    left_header.addStretch()
    header.addLayout(left_header, stretch=1)

    # Channels (displayed = raw * scale - offset) — moved from left panel.
    header.addWidget(app.data_fit_channels_group, stretch=2)

    right.addLayout(header)

    app.data_fit_plot = pg.PlotWidget(title="V-I preview")
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

    # Draggable parameter-table overlay (visible after a successful fit).
    app.data_fit_param_table = _FitParamTable()
    app.data_fit_param_table.setVisible(False)
    app.data_fit_plot.addItem(app.data_fit_param_table, ignoreBounds=True)

    right.addWidget(app.data_fit_plot, stretch=3)

    # Residuals sub-plot, shown after a fit.
    app.data_fit_resid_plot = pg.PlotWidget(title="Residuals (data − model)")
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

    root.addWidget(right_widget, stretch=1)

    app.data_fit_controller = DataFittingController(app)
    app.data_fit_preview_visible = True
    app.data_fit_preview_include_in_fit = True
    app.data_fit_preview_color = "#1f77b4"
    app.data_fit_preview_alpha_pct = 100
    app.data_fit_preview_style = {"draw_mode": "Auto", "line_width": 1.5, "point_size": 4}
    app.data_fit_curve_profiles = {"__preview__": _capture_fit_window_profile(app)}
    _on_use_length_changed(app)
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
    start_dir = app.runtime_state.output_folder or ""
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
    app.data_fit_plot.setLabel("left", desired)
    settings = getattr(app, "data_fit_graph_settings", None)
    if settings is not None and settings.title_left.text in ("", _Y_TITLE_VOLTAGE, _Y_TITLE_E_FIELD):
        settings.title_left.text = desired


def _update_equation_label(app):
    if getattr(app, "data_fit_use_length_cb", None) and app.data_fit_use_length_cb.isChecked():
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


def _on_use_length_changed(app):
    app.data_fit_plot_dirty = True
    app.data_fit_length_input.setEnabled(app.data_fit_use_length_cb.isChecked())
    if app.data_fit_use_length_cb.isChecked():
        app.data_fit_vc_label.setText("Ec (µV/cm):")
        app.data_fit_vc_input.setText("1")
        app.data_fit_linear_goal.setText(
            '<b>Goal:</b> fit the linear part to get <b>Rho</b> and <b>L</b>.'
        )
    else:
        app.data_fit_vc_label.setText("Vc (mV):")
        app.data_fit_vc_input.setText(f"{DEFAULT_VC_VOLTS * 1000:.6g}")
        app.data_fit_linear_goal.setText(
            '<b>Goal:</b> fit the linear part to get <b>R</b> and <b>L</b>.'
        )
    _update_equation_label(app)
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
    x_min_full = float(np.min(x[:n]))
    x_max_full = float(np.max(x[:n]))
    _apply_robust_view(app, x[:n], y[:n])
    _refresh_all_x_values(app)
    _update_fit_bands(app, x[:n], y[:n])
    _update_band_states(app)
    app.data_fit_xrange_label.setText(f"X window: [{x_min_full:.6g}, {x_max_full:.6g}]")


def _update_fit_bands(app, x: np.ndarray, y: np.ndarray) -> None:
    """Update the three semi-transparent bands that show the configured windows."""
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
    pow_lo = from_pct(app.data_fit_power_low, DEFAULT_POWER_LOW_FRAC)
    v_f = _float_from(app.data_fit_power_vfrac, DEFAULT_POWER_V_FRAC * 100, as_fraction=True)
    y_max = float(np.max(y)) if (y is not None and y.size) else 0.0
    threshold = v_f * y_max
    above = np.where(y >= threshold)[0] if y is not None else np.array([], dtype=int)
    pow_hi = float(x[above[0]]) if above.size else x_max

    for band, pair in (
        (app.data_fit_band_didt, (didt_lo, didt_hi)),
        (app.data_fit_band_linear, (lin_lo, lin_hi)),
        (app.data_fit_band_power, (pow_lo, pow_hi)),
    ):
        band.blockSignals(True)
        try:
            band.setRegion(pair)
        finally:
            band.blockSignals(False)


def _apply_robust_view(app, x: np.ndarray, y: np.ndarray) -> None:
    x_lo, x_hi = robust_view_range(x)
    y_lo, y_hi = robust_view_range(y)
    view_box = app.data_fit_plot.getPlotItem().getViewBox()
    view_box.setRange(xRange=(x_lo, x_hi), yRange=(y_lo, y_hi), padding=0.0)


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
    """Show and allow dragging for every window whose Show checkbox is enabled."""
    for window, band, show_cb in (
        ("didt", app.data_fit_band_didt, app.data_fit_show_didt),
        ("linear", app.data_fit_band_linear, app.data_fit_show_linear),
        ("power", app.data_fit_band_power, app.data_fit_show_power),
    ):
        enabled = bool(show_cb.isChecked())
        band.setMovable(enabled)
        band.setVisible(enabled)


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
    ctx = _data_ctx(app)
    if ctx is None:
        return
    x_min, x_max, x, y, y_max = ctx
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
    refresh_preview(app)


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


def _format_result(result) -> str:
    lines = []
    r_name = "Rho" if result.uses_sample_length else "R"
    r_unit = "Ω/cm" if result.uses_sample_length else "Ω"
    v_name = "Ec" if result.uses_sample_length else "Vc"
    v_unit = "V/cm" if result.uses_sample_length else "V"
    lines.append(f"di/dt         = {_format_engineering(result.di_dt, 'A/s', 2)}")
    lines.append(f"L             = {_format_engineering(result.inductance_L, 'H', 2)}  (= V0 / di_dt)")
    lines.append(f"V0            = {_format_engineering(result.V0, v_unit, 2)}")
    lines.append(f"{r_name:<13} = {_format_engineering(result.R, r_unit, 2)}")
    lines.append(f"Ic            = {result.Ic:.6g} A")
    lines.append(f"n-value       = {result.n_value:.2f}")
    lines.append(f"{v_name:<13} = {_format_engineering(result.criterion, v_unit, 2)}")
    lines.append(f"chi-squared   = {result.chi_sqr:.3g}")
    lines.append(f"iterations    = {result.iterations}")
    lines.append(f"Ic history    = [{', '.join(f'{v:.4g}' for v in result.ic_history)}]")
    lines.append(f"linear window = [{result.linear_fit_window[0]:.4g}, {result.linear_fit_window[1]:.4g}]")
    lines.append(f"power window  = [{result.power_fit_window[0]:.4g}, {result.power_fit_window[1]:.4g}]")
    return "\n".join(lines)


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


def _hide_fit_overlays(app) -> None:
    app.data_fit_ic_line.setVisible(False)
    app.data_fit_criterion_line.setVisible(False)
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
    app.data_fit_ic_line.setValue(result.Ic)
    app.data_fit_ic_line.label.setText(ic_label)
    app.data_fit_ic_line.setVisible(bool(show_ic))
    y_level = result.V0 + result.R * result.Ic + result.criterion
    app.data_fit_criterion_line.setValue(y_level)
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
    if result.iterations >= settings.max_iterations:
        warnings.append(
            f"Fit reached the iteration cap ({settings.max_iterations}) without "
            f"meeting the Ic tolerance ({settings.ic_tolerance * 100:.3g}%)."
        )
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
    existing["t"] = np.asarray(result.fit_x if result.fit_x is not None else [])
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
    table = QTableWidget(len(curves), 7)
    table.setHorizontalHeaderLabels(["Color", "Label", "Skip pts", "Avg", "Effective rate", "Include", "Actions"])
    table.horizontalHeader().setStretchLastSection(False)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    table.setColumnWidth(0, 90)
    table.setColumnWidth(2, 72)
    table.setColumnWidth(3, 70)
    table.setColumnWidth(4, 130)
    table.setColumnWidth(5, 70)
    table.setColumnWidth(6, 250)

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
        rate_item.setText(_rate_text_for_entry())
        table.setCellWidget(row, 4, rate_item)
        include = QCheckBox()
        include.setChecked(entry.get("include_in_fit", True))
        if is_preview:
            include.toggled.connect(lambda v: setattr(app, "data_fit_preview_include_in_fit", bool(v)))
        else:
            include.toggled.connect(lambda v: entry.update(include_in_fit=bool(v)))
        table.setCellWidget(row, 5, include)
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
        table.setCellWidget(row, 6, actions_widget)

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

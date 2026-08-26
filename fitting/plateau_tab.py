"""Builds and wires the "Plateau R calculation" tab.

Has its own TDMS file loader (same file format as the Ic fitting tab, via
the same :class:`fitting.tab.DataFittingController`, but its own independent
instance/state) and plots voltage/current channels vs time. Reports, per
user-added draggable time window, the number of points and the average
voltage / resistance (V / I) over that window.

Call :func:`setup_plateau_tab_layout` once on a host app whose
``ui_state.plateau_tab`` is a ``QWidget`` to populate.
"""

from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .service import (
    DEFAULT_EC_V_PER_CM,
    DEFAULT_PLATEAU_CURRENT_BAND_A,
    find_current_plateaus,
    fit_reduced_ic,
    window_average_stats,
)
from .tab import DataFittingController, EngineeringAxisItem, _float_from

_WINDOW_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_VOLTAGE_CURVE_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
]


class _CtrlRectZoomViewBox(pg.ViewBox):
    """ViewBox that zooms to a dragged rectangle while Ctrl is held.

    Plain (no-Ctrl) drags keep pyqtgraph's default pan behavior — the mouse
    mode is only switched to RectMode for the duration of a Ctrl-held drag.
    """

    def mouseDragEvent(self, ev, axis=None):
        prev_mode = self.state["mouseMode"]
        if ev.modifiers() & Qt.ControlModifier:
            self.state["mouseMode"] = pg.ViewBox.RectMode
        try:
            super().mouseDragEvent(ev, axis=axis)
        finally:
            self.state["mouseMode"] = prev_mode


def _enable_boxed_border(plot_widget) -> None:
    """Give a plot a real border on all four sides, not just bottom/left.

    pyqtgraph only draws an actual AxisItem line on the bottom/left sides by
    default — top/right are ``showAxis``-disabled entirely. What looked like
    a full box before any data loaded was just the grid lines from
    ``showGrid`` happening to coincide with the default empty (0, 1) view's
    edges; once real data autoranges the view (with padding), those grid
    lines no longer land exactly on the boundary, so the top/right "border"
    vanishes — because it was never a real axis line to begin with. Turning
    the top/right axes on (with their tick *values* hidden, so we don't get
    duplicate numbers) gives a permanent, data-independent border line.
    """
    plot_item = plot_widget.getPlotItem()
    for side in ("top", "right"):
        plot_item.showAxis(side)
        plot_item.getAxis(side).setStyle(showValues=False)


class _PlateauWindow:
    __slots__ = ("wid", "color", "region_v", "region_i", "row_label", "remove_btn", "row_widget")

    def __init__(self, wid, color, region_v, region_i, row_label, remove_btn, row_widget):
        self.wid = wid
        self.color = color
        self.region_v = region_v
        self.region_i = region_i
        self.row_label = row_label
        self.remove_btn = remove_btn
        self.row_widget = row_widget


def setup_plateau_tab_layout(app) -> None:
    """Populate ``app.ui_state.plateau_tab`` with the Plateau R calculation UI."""
    root = QHBoxLayout(app.ui_state.plateau_tab)

    app.plateau_windows: list[_PlateauWindow] = []
    app.plateau_next_id = 0
    app.plateau_voltage_checks: dict[str, QCheckBox] = {}
    app.plateau_voltage_curves: dict[str, object] = {}
    app.plateau_defaults_applied = False
    app.plateau_controller = DataFittingController(app)

    # ---- Left: controls ----
    left_widget = QWidget()
    left_widget.setMaximumWidth(320)
    left = QVBoxLayout(left_widget)

    app.plateau_status_label = QLabel("No file loaded.")
    app.plateau_status_label.setWordWrap(True)
    left.addWidget(app.plateau_status_label)

    app.plateau_load_btn = QPushButton("Load File…")
    left.addWidget(app.plateau_load_btn)

    app.plateau_reset_view_btn = QPushButton("Reset view")
    app.plateau_reset_view_btn.setToolTip("Reset both plots to the default (full-data) view.")
    left.addWidget(app.plateau_reset_view_btn)

    threshold_row = QHBoxLayout()
    threshold_row.addWidget(QLabel("Detection threshold (± A):"))
    app.plateau_band_input = QLineEdit(f"{DEFAULT_PLATEAU_CURRENT_BAND_A:g}")
    app.plateau_band_input.setMaximumWidth(70)
    app.plateau_band_input.setToolTip(
        "A run of samples counts as one plateau while its current stays "
        "within this many amps of its own min/max over that run. Raise it "
        "if noisy recordings split a real plateau into several short "
        "windows or miss it entirely; lower it if separate steps are being "
        "merged into one window."
    )
    threshold_row.addWidget(app.plateau_band_input)
    left.addLayout(threshold_row)

    app.plateau_detect_btn = QPushButton("Detect plateaus")
    app.plateau_detect_btn.setToolTip(
        "Clear existing windows and re-run plateau detection with the "
        "threshold above."
    )
    left.addWidget(app.plateau_detect_btn)

    left.addWidget(QLabel("Voltage channel(s):"))
    app.plateau_voltage_scroll = QScrollArea()
    app.plateau_voltage_scroll.setWidgetResizable(True)
    voltage_container = QWidget()
    app.plateau_voltage_list_layout = QVBoxLayout(voltage_container)
    app.plateau_voltage_list_layout.setSpacing(2)
    app.plateau_voltage_list_layout.setContentsMargins(4, 4, 4, 4)
    app.plateau_voltage_list_layout.addStretch()
    app.plateau_voltage_scroll.setWidget(voltage_container)
    # Size for a minimum of 4 visible rows, comfortably fitting ~6-7 before
    # scrolling — measured from a real QCheckBox so it adapts to the active
    # font/DPI instead of a hard-coded pixel guess.
    _measure_cb = QCheckBox()
    row_height = _measure_cb.sizeHint().height()
    _measure_cb.deleteLater()
    app.plateau_voltage_scroll.setMinimumHeight(row_height * 4 + 16)
    app.plateau_voltage_scroll.setMaximumHeight(row_height * 7 + 20)
    left.addWidget(app.plateau_voltage_scroll)

    left.addWidget(QLabel("Current channel:"))
    app.plateau_current_cb = QComboBox()
    left.addWidget(app.plateau_current_cb)

    app.plateau_add_window_btn = QPushButton("+ Add window")
    left.addWidget(app.plateau_add_window_btn)

    left.addWidget(QLabel("Windows:"))
    app.plateau_windows_scroll = QScrollArea()
    app.plateau_windows_scroll.setWidgetResizable(True)
    windows_container = QWidget()
    app.plateau_windows_list_layout = QVBoxLayout(windows_container)
    app.plateau_windows_list_layout.addStretch()
    app.plateau_windows_scroll.setWidget(windows_container)
    left.addWidget(app.plateau_windows_scroll, stretch=1)

    app.plateau_generate_btn = QPushButton("Generate reduced plots")
    app.plateau_generate_btn.setToolTip(
        "Build R-vs-I and V-vs-I plots from every window's averaged values."
    )
    left.addWidget(app.plateau_generate_btn)

    root.addWidget(left_widget)

    # ---- Right: tabbed plots ----
    app.plateau_reduced_data: dict[str, tuple] = {}
    app.plateau_r_curves: dict[str, object] = {}
    app.plateau_vi_curves: dict[str, object] = {}

    right_tabs = QTabWidget()
    app.plateau_right_tabs = right_tabs

    # -- Time plots tab (unchanged from before) --
    time_widget = QWidget()
    time_layout = QVBoxLayout(time_widget)

    app.plateau_v_plot = pg.PlotWidget(
        title="Voltage vs Time",
        viewBox=_CtrlRectZoomViewBox(),
        axisItems={
            "bottom": EngineeringAxisItem(orientation="bottom"),
            "left": EngineeringAxisItem(orientation="left"),
        },
    )
    app.plateau_v_plot.setLabel("bottom", "Time (s)")
    app.plateau_v_plot.setLabel("left", "Voltage (V)")
    app.plateau_v_plot.showGrid(x=True, y=True)
    app.plateau_v_plot.getPlotItem().setContentsMargins(4, 4, 4, 4)
    _enable_boxed_border(app.plateau_v_plot)
    app.plateau_v_plot.addLegend()
    time_layout.addWidget(app.plateau_v_plot, stretch=1)

    app.plateau_i_plot = pg.PlotWidget(
        title="Current vs Time",
        viewBox=_CtrlRectZoomViewBox(),
        axisItems={
            "bottom": EngineeringAxisItem(orientation="bottom"),
            "left": EngineeringAxisItem(orientation="left"),
        },
    )
    app.plateau_i_plot.setLabel("bottom", "Time (s)")
    app.plateau_i_plot.setLabel("left", "Current (A)")
    app.plateau_i_plot.showGrid(x=True, y=True)
    app.plateau_i_plot.getPlotItem().setContentsMargins(4, 4, 4, 4)
    _enable_boxed_border(app.plateau_i_plot)
    app.plateau_i_plot.setXLink(app.plateau_v_plot)
    time_layout.addWidget(app.plateau_i_plot, stretch=1)

    right_tabs.addTab(time_widget, "Time plots")

    # -- Derived R vs I tab --
    r_widget = QWidget()
    r_layout = QVBoxLayout(r_widget)
    app.plateau_r_plot = pg.PlotWidget(
        title="R vs I (plateau averages)",
        axisItems={
            "bottom": EngineeringAxisItem(orientation="bottom"),
            "left": EngineeringAxisItem(orientation="left"),
        },
    )
    app.plateau_r_plot.setLabel("bottom", "Current (A)")
    app.plateau_r_plot.setLabel("left", "Resistance (Ω)")
    app.plateau_r_plot.showGrid(x=True, y=True)
    app.plateau_r_plot.getPlotItem().setContentsMargins(4, 4, 4, 4)
    _enable_boxed_border(app.plateau_r_plot)
    app.plateau_r_plot.addLegend()
    r_layout.addWidget(app.plateau_r_plot)
    app.plateau_r_tab_widget = r_widget
    right_tabs.addTab(r_widget, "Derived R vs I")

    # -- Derived V vs I tab --
    app.plateau_fit_results: dict = {}
    app.plateau_fit_curves: dict[str, object] = {}

    vi_widget = QWidget()
    vi_layout = QVBoxLayout(vi_widget)
    fit_controls = QHBoxLayout()
    app.plateau_fit_btn = QPushButton("Run Ic fit (all channels)")
    app.plateau_fit_btn.setToolTip(
        "Fit every channel in the reduced dataset independently and add its "
        "result to the 'Ic fit results table' tab."
    )
    fit_controls.addWidget(app.plateau_fit_btn)
    fit_controls.addStretch()
    vi_layout.addLayout(fit_controls)
    app.plateau_fit_result_label = QLabel("Click 'Generate reduced plots', then 'Run Ic fit'.")
    app.plateau_fit_result_label.setWordWrap(True)
    vi_layout.addWidget(app.plateau_fit_result_label)
    app.plateau_vi_plot = pg.PlotWidget(
        title="V vs I (plateau averages)",
        axisItems={
            "bottom": EngineeringAxisItem(orientation="bottom"),
            "left": EngineeringAxisItem(orientation="left"),
        },
    )
    app.plateau_vi_plot.setLabel("bottom", "Current (A)")
    app.plateau_vi_plot.setLabel("left", "Voltage (V)")
    app.plateau_vi_plot.showGrid(x=True, y=True)
    app.plateau_vi_plot.getPlotItem().setContentsMargins(4, 4, 4, 4)
    _enable_boxed_border(app.plateau_vi_plot)
    app.plateau_vi_plot.addLegend()
    app.plateau_vi_plot.setXLink(app.plateau_r_plot)
    vi_layout.addWidget(app.plateau_vi_plot, stretch=1)
    right_tabs.addTab(vi_widget, "Derived V vs I")

    # -- Ic fit results table tab --
    table_widget = QWidget()
    table_layout = QVBoxLayout(table_widget)
    app.plateau_fit_table = QTableWidget(0, 12)
    app.plateau_fit_table.setHorizontalHeaderLabels([
        "Channel", "Ic (A)", "σIc", "n", "σn", "R (Ω)", "R/l (nΩ/m)",
        "V0 (V)", "Vc (V)", "R²", "N pts", "Status",
    ])
    app.plateau_fit_table.horizontalHeader().setStretchLastSection(True)
    app.plateau_fit_table.setEditTriggers(QTableWidget.NoEditTriggers)
    table_layout.addWidget(app.plateau_fit_table)
    right_tabs.addTab(table_widget, "Ic fit results table")

    root.addWidget(right_tabs, stretch=1)

    app.plateau_current_curve = app.plateau_i_plot.plot(pen=pg.mkPen("k", width=1.5))

    app.plateau_load_btn.clicked.connect(lambda: _open_file_dialog(app))
    app.plateau_reset_view_btn.clicked.connect(lambda: _reset_view(app))
    app.plateau_detect_btn.clicked.connect(lambda: _redetect_windows(app))
    app.plateau_add_window_btn.clicked.connect(lambda: _add_window(app))
    app.plateau_current_cb.currentIndexChanged.connect(lambda _: _on_selection_changed(app))
    app.plateau_generate_btn.clicked.connect(lambda: _generate_reduced_plots(app))
    app.plateau_fit_btn.clicked.connect(lambda: _run_ic_fit(app))

    _sync_ui(app)


# ---------------------------------------------------------------------------
# Data refresh
# ---------------------------------------------------------------------------

def _open_file_dialog(app) -> None:
    """Prompt for a TDMS file and load it into this tab's own controller.

    Independent of whatever recording the Ic fitting tab has loaded — this
    tab keeps its own :class:`~fitting.tab.DataFittingController` instance.
    On success, existing windows are cleared and windows are auto-added for
    every detected stable-current plateau.
    """
    runtime_state = getattr(app, "runtime_state", None)
    start_dir = getattr(runtime_state, "output_folder", "") or ""
    path, _ = QFileDialog.getOpenFileName(
        app, "Select TDMS recording", start_dir, "TDMS Files (*.tdms);;All Files (*)",
    )
    if not path:
        return
    ok, msg = app.plateau_controller.load_recording(path)
    if not ok:
        app.plateau_status_label.setText(msg)
        app.plateau_status_label.setStyleSheet("color: #b35a00;")
        return
    _clear_all_windows(app)
    _sync_ui(app)
    n_detected = _auto_detect_windows(app)
    if n_detected:
        app.plateau_status_label.setText(
            app.plateau_status_label.text() + f" — {n_detected} plateau window(s) auto-detected"
        )
    _reset_view(app)


def _reset_view(app) -> None:
    """Reset both plots to the default (auto-ranged, full-data) view."""
    app.plateau_v_plot.getPlotItem().getViewBox().autoRange(padding=0.05)
    app.plateau_i_plot.getPlotItem().getViewBox().autoRange(padding=0.05)


def _detection_band(app) -> float:
    """Current detection threshold (± A) from the UI, falling back to the default."""
    return max(0.0, _float_from(app.plateau_band_input, DEFAULT_PLATEAU_CURRENT_BAND_A))


def _auto_detect_windows(app) -> int:
    """Add a window for every stable-current plateau found in the recording.

    Uses :func:`fitting.service.find_current_plateaus` against the currently
    selected current channel, with the noise-tolerance band read from
    ``app.plateau_band_input`` so the user can loosen/tighten detection
    without touching code. Returns the number of windows added.
    """
    controller = getattr(app, "plateau_controller", None)
    if controller is None or controller.time_array is None:
        return 0
    current_name = app.plateau_current_cb.currentText()
    current_values = _calibrated(controller, current_name)
    if current_values is None:
        return 0
    plateaus = find_current_plateaus(
        controller.time_array, current_values, band=_detection_band(app)
    )
    for t0, t1 in plateaus:
        _add_window(app, t0, t1)
    return len(plateaus)


def _redetect_windows(app) -> None:
    """Re-run plateau auto-detection with the current threshold, replacing existing windows."""
    controller = getattr(app, "plateau_controller", None)
    if controller is None or controller.time_array is None:
        app.plateau_status_label.setText("No file loaded. Click 'Load File…' to choose a TDMS.")
        app.plateau_status_label.setStyleSheet("color: gray;")
        return
    _clear_all_windows(app)
    n_detected = _auto_detect_windows(app)
    base = f"{os.path.basename(controller.tdms_path)} ({len(controller.channel_names)} channels)"
    suffix = f" — {n_detected} plateau window(s) detected" if n_detected else " — no plateaus detected"
    app.plateau_status_label.setText(base + suffix)
    app.plateau_status_label.setStyleSheet("color: black;")


def _clear_all_windows(app) -> None:
    for win in list(app.plateau_windows):
        _remove_window(app, win)


def _sync_ui(app) -> None:
    """Refresh channel widgets, curves and window readouts from ``app.plateau_controller``."""
    controller = getattr(app, "plateau_controller", None)
    if controller is None or not controller.tdms_path:
        app.plateau_status_label.setText("No file loaded. Click 'Load File…' to choose a TDMS.")
        app.plateau_status_label.setStyleSheet("color: gray;")
        _rebuild_channel_widgets(app, None)
    else:
        app.plateau_status_label.setText(
            f"{os.path.basename(controller.tdms_path)} ({len(controller.channel_names)} channels)"
        )
        app.plateau_status_label.setStyleSheet("color: black;")
        _rebuild_channel_widgets(app, controller)
    _redraw_curves(app)
    _update_all_readouts(app)


def _calibrated(controller, name: str):
    if controller is None or not name:
        return None
    raw = controller.get_channel(name)
    if raw is None:
        return None
    meta = controller.get_metadata(name)
    return controller.apply_transform(raw, meta.get("scale", 1.0), meta.get("offset", 0.0))


def _rebuild_channel_widgets(app, controller) -> None:
    names = list(controller.channel_names) if controller is not None else []

    prev_checked = {name for name, cb in app.plateau_voltage_checks.items() if cb.isChecked()}
    prev_current = app.plateau_current_cb.currentText()

    layout = app.plateau_voltage_list_layout
    for cb in app.plateau_voltage_checks.values():
        layout.removeWidget(cb)
        cb.deleteLater()
    app.plateau_voltage_checks.clear()

    for name in names:
        cb = QCheckBox(name)
        if name in prev_checked:
            cb.setChecked(True)
        elif not app.plateau_defaults_applied and "volt" in name.lower():
            cb.setChecked(True)
        cb.stateChanged.connect(lambda _=None: _on_selection_changed(app))
        layout.insertWidget(layout.count() - 1, cb)
        app.plateau_voltage_checks[name] = cb
    if names:
        app.plateau_defaults_applied = True

    combo = app.plateau_current_cb
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(names)
    idx = combo.findText(prev_current) if prev_current else -1
    if idx < 0:
        for i, name in enumerate(names):
            low = name.lower()
            if "current" in low or "dcct" in low or "imon" in low or low in ("i", "ai0"):
                idx = i
                break
    if idx < 0 and names:
        idx = 0
    if idx >= 0:
        combo.setCurrentIndex(idx)
    combo.blockSignals(False)


def _on_selection_changed(app) -> None:
    _redraw_curves(app)
    _update_all_readouts(app)


def _redraw_curves(app) -> None:
    controller = getattr(app, "plateau_controller", None)
    time = controller.time_array if controller is not None else None

    checked_names = [name for name, cb in app.plateau_voltage_checks.items() if cb.isChecked()]

    legend = app.plateau_v_plot.plotItem.legend
    for name in list(app.plateau_voltage_curves.keys()):
        if name not in checked_names:
            curve = app.plateau_voltage_curves.pop(name)
            app.plateau_v_plot.removeItem(curve)
            if legend is not None:
                try:
                    legend.removeItem(name)
                except Exception:
                    pass

    for i, name in enumerate(checked_names):
        values = _calibrated(controller, name)
        curve = app.plateau_voltage_curves.get(name)
        if curve is None:
            color = _VOLTAGE_CURVE_COLORS[i % len(_VOLTAGE_CURVE_COLORS)]
            curve = app.plateau_v_plot.plot(pen=pg.mkPen(color, width=1.5), name=name)
            app.plateau_voltage_curves[name] = curve
        if time is not None and values is not None:
            curve.setData(time, values)
        else:
            curve.setData([], [])

    current_name = app.plateau_current_cb.currentText()
    current_values = _calibrated(controller, current_name)
    if time is not None and current_values is not None:
        app.plateau_current_curve.setData(time, current_values)
    else:
        app.plateau_current_curve.setData([], [])


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _make_synced_handlers(app, win: _PlateauWindow):
    state = {"updating": False}

    def on_v_changed():
        if state["updating"]:
            return
        state["updating"] = True
        try:
            win.region_i.setRegion(win.region_v.getRegion())
        finally:
            state["updating"] = False
        _update_window_readout(app, win)

    def on_i_changed():
        if state["updating"]:
            return
        state["updating"] = True
        try:
            win.region_v.setRegion(win.region_i.getRegion())
        finally:
            state["updating"] = False
        _update_window_readout(app, win)

    return on_v_changed, on_i_changed


def _add_window(app, t0: float | None = None, t1: float | None = None) -> None:
    """Add a window region. Defaults to the middle 10% of the visible time
    range when ``t0``/``t1`` aren't given (manual "+ Add window" click)."""
    if t0 is None or t1 is None:
        controller = getattr(app, "plateau_controller", None)
        time = controller.time_array if controller is not None else None
        if time is not None and len(time):
            lo, hi = float(np.min(time)), float(np.max(time))
            span = hi - lo
            t0 = lo + 0.45 * span
            t1 = lo + 0.55 * span
            if t0 == t1:
                t1 = t0 + 1.0
        else:
            t0, t1 = 0.0, 1.0

    wid = app.plateau_next_id
    app.plateau_next_id += 1
    color = _WINDOW_COLORS[wid % len(_WINDOW_COLORS)]
    qc = pg.mkColor(color)
    brush = pg.mkBrush(qc.red(), qc.green(), qc.blue(), 45)
    pen = pg.mkPen(qc.red(), qc.green(), qc.blue(), 200)

    region_v = pg.LinearRegionItem(values=(t0, t1), brush=brush, pen=pen)
    region_i = pg.LinearRegionItem(values=(t0, t1), brush=brush, pen=pen)
    region_v.setZValue(5)
    region_i.setZValue(5)
    app.plateau_v_plot.addItem(region_v, ignoreBounds=True)
    app.plateau_i_plot.addItem(region_i, ignoreBounds=True)

    row_widget = QWidget()
    row = QHBoxLayout(row_widget)
    row.setContentsMargins(2, 2, 2, 2)
    swatch = QLabel()
    swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #333;")
    swatch.setFixedSize(14, 14)
    row.addWidget(swatch)
    row_label = QLabel("...")
    row_label.setWordWrap(True)
    row.addWidget(row_label, stretch=1)
    remove_btn = QPushButton("✕")
    remove_btn.setFixedWidth(24)
    row.addWidget(remove_btn)

    win = _PlateauWindow(wid, color, region_v, region_i, row_label, remove_btn, row_widget)
    app.plateau_windows.append(win)

    layout = app.plateau_windows_list_layout
    layout.insertWidget(layout.count() - 1, row_widget)

    on_v_changed, on_i_changed = _make_synced_handlers(app, win)
    region_v.sigRegionChanged.connect(on_v_changed)
    region_i.sigRegionChanged.connect(on_i_changed)
    remove_btn.clicked.connect(lambda: _remove_window(app, win))

    _update_window_readout(app, win)


def _remove_window(app, win: _PlateauWindow) -> None:
    app.plateau_v_plot.removeItem(win.region_v)
    app.plateau_i_plot.removeItem(win.region_i)
    win.row_widget.setParent(None)
    win.row_widget.deleteLater()
    app.plateau_windows.remove(win)


def _update_all_readouts(app) -> None:
    for win in app.plateau_windows:
        _update_window_readout(app, win)


def _update_window_readout(app, win: _PlateauWindow) -> None:
    controller = getattr(app, "plateau_controller", None)
    if controller is None or controller.time_array is None:
        win.row_label.setText("No recording loaded.")
        return

    t0, t1 = win.region_v.getRegion()
    checked_names = [name for name, cb in app.plateau_voltage_checks.items() if cb.isChecked()]
    voltage_channels = {}
    for name in checked_names:
        values = _calibrated(controller, name)
        if values is not None:
            voltage_channels[name] = values

    current_name = app.plateau_current_cb.currentText()
    current_values = _calibrated(controller, current_name)

    stats = window_average_stats(controller.time_array, current_values, voltage_channels, t0, t1)

    i_str = f"{stats['avg_i']:.4g} A" if stats["avg_i"] is not None else "—"
    lines = [
        f"{stats['n_points']} pts, t = [{t0:.4g}, {t1:.4g}] s",
        f"{current_name or 'current'}: I_avg={i_str}",
    ]
    for name, ch in stats["channels"].items():
        v_str = f"{ch['avg_v']:.4g} V" if ch["avg_v"] is not None else "—"
        r_str = f"{ch['avg_r']:.4g} Ω" if ch["avg_r"] is not None else "—"
        lines.append(f"{name}: V={v_str}, R={r_str}")
    win.row_label.setText("\n".join(lines))


# ---------------------------------------------------------------------------
# Reduced (plateau-averaged) R-I / V-I plots + simplified Ic fit
# ---------------------------------------------------------------------------

def _collect_reduced_points(app) -> dict:
    """Build ``{channel_name: (I_array, V_array, R_array)}`` from every
    window's averaged values, sorted by current. Windows with no valid
    average (empty region, missing channel data) are skipped per-channel."""
    controller = getattr(app, "plateau_controller", None)
    if controller is None or controller.time_array is None:
        return {}

    checked_names = [name for name, cb in app.plateau_voltage_checks.items() if cb.isChecked()]
    current_name = app.plateau_current_cb.currentText()
    current_values = _calibrated(controller, current_name)
    voltage_channels = {}
    for name in checked_names:
        values = _calibrated(controller, name)
        if values is not None:
            voltage_channels[name] = values

    per_channel: dict = {name: [] for name in voltage_channels}
    for win in app.plateau_windows:
        t0, t1 = win.region_v.getRegion()
        stats = window_average_stats(controller.time_array, current_values, voltage_channels, t0, t1)
        avg_i = stats["avg_i"]
        if avg_i is None:
            continue
        for name, ch in stats["channels"].items():
            if ch["avg_v"] is None:
                continue
            per_channel[name].append((avg_i, ch["avg_v"], ch["avg_r"]))

    result = {}
    for name, points in per_channel.items():
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        arr = np.array(points, dtype=float)
        result[name] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return result


def _generate_reduced_plots(app) -> None:
    """"Generate reduced plots" button handler: rebuild the reduced dataset
    from every current window and redraw the R-vs-I / V-vs-I tabs.

    Clears any previous fit results/overlays/table rows, since they were
    computed against the old reduced dataset and no longer apply.
    """
    app.plateau_reduced_data = _collect_reduced_points(app)
    _redraw_reduced_plot(app, app.plateau_r_plot, app.plateau_r_curves, value_index=2)
    _redraw_reduced_plot(app, app.plateau_vi_plot, app.plateau_vi_curves, value_index=1)
    app.plateau_fit_results.clear()
    _redraw_fit_curves(app)
    _redraw_fit_table(app)
    app.plateau_fit_result_label.setText(
        "Click 'Run Ic fit' to fit every channel."
        if app.plateau_reduced_data
        else "No reduced data — add windows with valid averages, then try again."
    )
    app.plateau_right_tabs.setCurrentWidget(app.plateau_r_tab_widget)


def _redraw_reduced_plot(app, plot, curves: dict, value_index: int) -> None:
    """Redraw one of the two reduced plots (R-vs-I or V-vs-I) from
    ``app.plateau_reduced_data``. ``value_index`` selects V (1) or R (2)
    from each channel's ``(I, V, R)`` tuple."""
    legend = plot.plotItem.legend
    for name, curve in list(curves.items()):
        plot.removeItem(curve)
        if legend is not None:
            try:
                legend.removeItem(name)
            except Exception:
                pass
    curves.clear()

    for i, (name, values) in enumerate(app.plateau_reduced_data.items()):
        I = values[0]
        y = values[value_index]
        color = _VOLTAGE_CURVE_COLORS[i % len(_VOLTAGE_CURVE_COLORS)]
        curve = plot.plot(
            I, y, pen=pg.mkPen(color, width=1), symbol="o", symbolSize=7,
            symbolBrush=pg.mkBrush(color), symbolPen=None, name=name,
        )
        curves[name] = curve


def _run_ic_fit(app) -> None:
    """"Run Ic fit" button handler: fit every channel in the reduced dataset
    independently to V = V0 + R*I + Vc*(I/Ic)^n, Vc fixed per-channel from
    its voltage-tap metadata (tap_length_cm x 1 uV/cm, IEC 61788).

    Results are keyed by channel name in ``app.plateau_fit_results`` — a
    channel's entry is simply overwritten on a re-run, so re-fitting never
    creates duplicate table rows.
    """
    if not app.plateau_reduced_data:
        app.plateau_fit_result_label.setText("No reduced data. Click 'Generate reduced plots' first.")
        return

    controller = app.plateau_controller
    successes, failures = [], []
    for name, points in app.plateau_reduced_data.items():
        I, V, _R = points
        v_tap = controller.get_metadata(name).get("voltage_tap_cm")
        if not v_tap or float(v_tap) <= 0:
            app.plateau_fit_results[name] = {
                "ok": False,
                "message": "no Voltage_Tab_Distance_cm metadata",
            }
            failures.append(name)
            continue

        Vc = float(v_tap) * DEFAULT_EC_V_PER_CM
        result = fit_reduced_ic(I, V, Vc)
        if not result.ok:
            app.plateau_fit_results[name] = {"ok": False, "message": result.message}
            failures.append(name)
            continue

        app.plateau_fit_results[name] = {
            "ok": True,
            "Ic": result.Ic, "sigma_Ic": result.sigma_Ic,
            "n_value": result.n_value, "sigma_n": result.sigma_n,
            "R": result.R, "V0": result.V0, "criterion": Vc,
            "tap_length_cm": float(v_tap),
            "r_squared": result.r_squared, "n_points_used": result.n_points_used,
            "fit_x": result.fit_x, "fit_y": result.fit_y,
        }
        successes.append(name)

    _redraw_fit_table(app)
    _redraw_fit_curves(app)

    total = len(app.plateau_reduced_data)
    summary = f"Fit {len(successes)}/{total} channel(s) successfully."
    if failures:
        summary += " Failed: " + ", ".join(failures) + "."
    app.plateau_fit_result_label.setText(summary)


def _redraw_fit_curves(app) -> None:
    """Overlay every successfully-fit channel's curve on the V-vs-I plot,
    color-matched to that channel's scatter (dashed to distinguish the fit
    from the raw plateau-averaged points)."""
    plot = app.plateau_vi_plot
    for curve in app.plateau_fit_curves.values():
        plot.removeItem(curve)
    app.plateau_fit_curves.clear()

    names = list(app.plateau_reduced_data.keys())
    for i, name in enumerate(names):
        res = app.plateau_fit_results.get(name)
        if not res or not res.get("ok"):
            continue
        color = _VOLTAGE_CURVE_COLORS[i % len(_VOLTAGE_CURVE_COLORS)]
        curve = plot.plot(res["fit_x"], res["fit_y"], pen=pg.mkPen(color, width=2, style=Qt.DashLine))
        app.plateau_fit_curves[name] = curve


def _redraw_fit_table(app) -> None:
    """Rebuild the "Ic fit results table" tab from ``app.plateau_fit_results``.

    Always fully rebuilt from a dict keyed by channel name, sorted by name —
    so there is exactly one row per channel (re-fitting overwrites its entry
    rather than appending) and the row order stays consistent as channels
    are added.
    """
    table = app.plateau_fit_table
    names = sorted(app.plateau_fit_results.keys())
    table.setRowCount(len(names))
    for row, name in enumerate(names):
        res = app.plateau_fit_results[name]
        if res.get("ok"):
            # R/l: resistance per unit tap length, in nOhm/m (tap length is
            # stored in cm, so R/(l_cm/100) converted from Ohm/m to nOhm/m).
            r_per_l = res["R"] * 1.0e11 / res["tap_length_cm"]
            values = [
                name,
                f"{res['Ic']:.4g}", f"{res['sigma_Ic']:.2g}",
                f"{res['n_value']:.4g}", f"{res['sigma_n']:.2g}",
                f"{res['R']:.4g}", f"{r_per_l:.4g}", f"{res['V0']:.4g}",
                f"{res['criterion']:.4g}", f"{res['r_squared']:.4g}",
                str(res["n_points_used"]), "OK",
            ]
        else:
            values = [name] + ["—"] * 10 + [res.get("message", "Failed")]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, col, item)

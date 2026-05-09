"""Standalone Capacitor Testing window.

Decouples the Capacitor Testing workflow from the V-I fitting tab into a
free-standing top-level window. Inside it the user can:

* Load a TDMS or ASCII recording (.txt/.dat/.csv/.tsv/.asc) — or pull the
  active book's data straight from the Data Workspace.
* Browse channels in a small preview plot with X/Y combos and a draggable
  purple region for "Step 1: Select data range on X-axis".
* Step 2: multiply two rows for ``Power on a Resistor (W)``.
* Step 3: cumulatively integrate a row over a time row for
  ``Integrated Power (J)``.
* Step 4: compute capacity vs voltage at every 10 V step (Coulomb counting
  and Energy Integration), stored as a separate ``Capacity vs Voltage``
  dataset selectable from the Channels-group Dataset combo.
* Save the in-memory Power / Integrated Power back into the source file
  (TDMS or ASCII), into a brand-new TDMS, or save Capacity vs Voltage to
  a TDMS.

Wired into the Data Workspace toolbar — see ``tab._open_workspace_window``
which adds a ``🔋 Capacitor Testing`` action that calls
``open_capacitor_testing_window``.
"""

from __future__ import annotations

import os
import traceback
from typing import Optional

import numpy as np
import pyqtgraph as pg
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


_ASCII_RECORDING_EXTS = {".txt", ".dat", ".csv", ".tsv", ".asc"}

CAPACITOR_POWER_CHANNEL_NAME = "Power on a Resistor (W)"
CAPACITOR_ENERGY_CHANNEL_NAME = "Integrated Power (J)"
CAPACITY_DATASET_NAME = "Capacity vs Voltage"
CAPACITY_VOLTAGE_COLUMN = "Voltage (V)"
CAPACITY_COULOMB_COLUMN = "Capacity [F] Coulomb counting method"
CAPACITY_ENERGY_COLUMN = "Capacity [F] Energy Integration method"
CAPACITY_COULOMB_AVG_COLUMN = "Capacity [F] Coulomb counting Avg. over"
CAPACITY_ENERGY_AVG_COLUMN = "Capacity [F] Energy integration Avg. over"
CAPACITY_VOLTAGE_STEP_V = 10.0


# ---------------------------------------------------------------------------
# Source-file readers (mirrored from the CapacitorTesting branch's tab.py).
# ---------------------------------------------------------------------------


def _read_time_channel_from_tdms(tdms_file) -> Optional[np.ndarray]:
    for group in tdms_file.groups():
        for channel in group.channels():
            name = getattr(channel, "name", "")
            if name and name.lower() == "time":
                return np.asarray(channel[:], dtype=float)
    return None


def _read_ascii_recording(path: str):
    """Read a tab- or whitespace-separated ASCII data file with a header row.

    Returns ``(time_array, channels_dict, time_column_name)`` where
    ``channels_dict`` maps the verbatim header text to a numpy array.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header_line = ""
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue
            header_line = stripped
            break
    if not header_line:
        raise ValueError("File is empty or has no header line.")
    if path.lower().endswith(".csv") or ("," in header_line and "\t" not in header_line):
        delim = ","
    elif "\t" in header_line:
        delim = "\t"
    else:
        delim = None
    if delim is None:
        column_names = [c.strip() for c in header_line.split()]
    else:
        column_names = [c.strip() for c in header_line.split(delim)]
    column_names = [c for c in column_names if c]
    if not column_names:
        raise ValueError("Could not parse header columns.")
    try:
        data = np.loadtxt(
            path, skiprows=1, dtype=float, delimiter=delim, comments=("#", "%"),
        )
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric data: {exc}") from exc
    if data.ndim == 1:
        data = data.reshape(1, -1) if data.size == len(column_names) else data.reshape(-1, 1)
    if data.shape[1] != len(column_names):
        raise ValueError(
            f"Header has {len(column_names)} columns but data has {data.shape[1]}."
        )
    time_idx = -1
    for k, name in enumerate(column_names):
        if name.lower().startswith("time"):
            time_idx = k
            break
    time_array = data[:, time_idx] if time_idx >= 0 else None
    time_name = column_names[time_idx] if time_idx >= 0 else "time"
    channels: dict[str, np.ndarray] = {}
    for k, name in enumerate(column_names):
        if k == time_idx:
            continue
        channels[name] = np.asarray(data[:, k], dtype=float)
    return time_array, channels, time_name


# ---------------------------------------------------------------------------
# Controller: holds the source recording + derived datasets.
# ---------------------------------------------------------------------------


class CapTestController:
    """In-memory state of one Capacitor Testing session.

    Holds the source recording (TDMS or ASCII) plus any derived in-memory
    channels (``Power on a Resistor (W)``, ``Integrated Power (J)``) and the
    derived ``Capacity vs Voltage`` dataset.
    """

    def __init__(self) -> None:
        self.source_path: str = ""
        self.time_array: Optional[np.ndarray] = None
        self.time_column_name: str = "Time"
        self.channel_cache: dict[str, np.ndarray] = {}
        self.channel_names: list[str] = []
        # Multi-dataset: the recording is the primary dataset; Step 4 adds
        # ``Capacity vs Voltage`` as a derived dataset selectable from the
        # Channels-group Dataset combo.
        self.datasets: dict[str, dict] = {}
        self.active_dataset_name: str = ""

    def reset(self) -> None:
        self.source_path = ""
        self.time_array = None
        self.time_column_name = "Time"
        self.channel_cache = {}
        self.channel_names = []
        self.datasets = {}
        self.active_dataset_name = ""

    def load_recording(self, path: str) -> tuple[bool, str]:
        self.reset()
        if not path or not os.path.exists(path):
            return False, "No recording found. Click 'Load file…' to choose a file."
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".tdms":
                self._load_from_tdms(path)
            elif ext in _ASCII_RECORDING_EXTS:
                self._load_from_ascii(path)
            else:
                return False, (
                    f"Unsupported file extension '{ext}'. "
                    "Use .tdms or an ASCII data file (.txt, .dat, .csv, .tsv, .asc)."
                )
        except Exception as exc:
            return False, f"Could not read file: {exc}"
        if self.time_array is None:
            return False, "Recording has no time column."
        self.source_path = path
        self.register_recording_dataset(os.path.basename(path))
        return True, f"Loaded {os.path.basename(path)} with {len(self.channel_names)} channels."

    def _load_from_tdms(self, path: str) -> None:
        with TdmsFile.read(path) as tdms_file:
            self.time_array = _read_time_channel_from_tdms(tdms_file)
            names: list[str] = []
            for group in tdms_file.groups():
                if group.name == "FitResults":
                    continue
                for channel in group.channels():
                    name = getattr(channel, "name", "")
                    if not name or name.lower() == "time":
                        continue
                    if name in self.channel_cache:
                        continue
                    self.channel_cache[name] = np.asarray(channel[:], dtype=float)
                    names.append(name)
            self.channel_names = names

    def _load_from_ascii(self, path: str) -> None:
        time_arr, channels, time_name = _read_ascii_recording(path)
        self.time_array = time_arr
        self.time_column_name = time_name
        self.channel_cache.update(channels)
        self.channel_names = list(channels.keys())

    def adopt_from_book(self, controller, source_label: str) -> tuple[bool, str]:
        """Copy data from a Data Workspace book ``DataFittingController`` instance.

        Pulls in the channel cache, names, time array and source path so the
        Capacitor Testing window starts from the same data the workspace
        book is showing. Edits stay local to this controller.
        """
        path = getattr(controller, "tdms_path", "") or ""
        names = list(getattr(controller, "channel_names", []) or [])
        cache = dict(getattr(controller, "channel_cache", {}) or {})
        time_arr = getattr(controller, "time_array", None)
        if time_arr is None or not names:
            return False, "Selected book has no usable channels yet."
        self.reset()
        self.source_path = path
        self.time_array = np.asarray(time_arr, dtype=float)
        self.time_column_name = getattr(controller, "time_column_name", None) or "Time"
        # Deep-copy each array so capacitor-side mutations never bleed into
        # the workspace book's cache.
        self.channel_cache = {k: np.asarray(v, dtype=float).copy() for k, v in cache.items()}
        self.channel_names = names
        self.register_recording_dataset(source_label or os.path.basename(path) or "Book")
        return True, f"Imported {len(self.channel_names)} channels from {source_label}."

    def register_recording_dataset(self, name: str) -> None:
        self.datasets = {
            name: {
                "channels": self.channel_cache,
                "time": self.time_array,
                "is_recording": True,
            }
        }
        self.active_dataset_name = name

    def add_derived_dataset(self, name: str, channels: dict, time=None) -> None:
        self.datasets[name] = {
            "channels": dict(channels),
            "time": time,
            "is_recording": False,
        }

    def active_dataset(self) -> Optional[dict]:
        return self.datasets.get(self.active_dataset_name)

    def get_channel(self, name: str):
        if not name:
            return None
        return self.channel_cache.get(name)


# ---------------------------------------------------------------------------
# Math helpers.
# ---------------------------------------------------------------------------


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if y.size < 2:
        return np.zeros_like(y)
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    return np.concatenate(([0.0], np.cumsum(avg * dx)))


def _capacitor_discharge_segment(t: np.ndarray, v: np.ndarray, i: np.ndarray):
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    n = int(min(t.size, v.size, i.size))
    if n < 3:
        return None, None, None
    t = t[:n]
    v = v[:n]
    i = i[:n]
    finite = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    if not np.any(finite):
        return None, None, None
    t = t[finite]
    v = v[finite]
    i = i[finite]
    if t.size < 3:
        return None, None, None
    idx_peak = int(np.argmax(v))
    t_seg = t[idx_peak:]
    v_seg = v[idx_peak:]
    i_seg = i[idx_peak:]
    if v_seg.size < 2 or float(v_seg[0]) <= float(np.min(v_seg)):
        return None, None, None
    return t_seg, v_seg, i_seg


def _compute_capacity_results(
    t: np.ndarray, v: np.ndarray, i: np.ndarray,
    step_v: float = CAPACITY_VOLTAGE_STEP_V,
):
    if step_v <= 0:
        return None
    t_seg, v_seg, i_seg = _capacitor_discharge_segment(t, v, i)
    if t_seg is None:
        return None
    q = _cumulative_trapezoid(i_seg, t_seg)
    e = _cumulative_trapezoid(v_seg * i_seg, t_seg)
    v_max_seg = float(v_seg[0])
    v_min_seg = float(np.min(v_seg))
    top = float(np.floor(v_max_seg / step_v) * step_v)
    if top > v_max_seg:
        top -= step_v
    levels: list[float] = []
    cur = top
    while cur >= v_min_seg - 1.0e-9:
        levels.append(cur)
        cur -= step_v
    if len(levels) < 2:
        return None
    levels_arr = np.asarray(levels, dtype=float)
    v_rev = v_seg[::-1]
    q_rev = q[::-1]
    e_rev = e[::-1]
    keep = np.concatenate(([True], np.diff(v_rev) > 0))
    v_rev = v_rev[keep]
    q_rev = q_rev[keep]
    e_rev = e_rev[keep]
    if v_rev.size < 2:
        return None
    q_levels = np.interp(levels_arr, v_rev, q_rev)
    e_levels = np.interp(levels_arr, v_rev, e_rev)
    voltages: list[float] = []
    coulomb: list[float] = []
    energy: list[float] = []
    for k in range(len(levels_arr) - 1):
        v_high = float(levels_arr[k])
        v_low = float(levels_arr[k + 1])
        dv = v_high - v_low
        denom_e = v_high * v_high - v_low * v_low
        if dv <= 0 or denom_e <= 0:
            continue
        dq = abs(float(q_levels[k + 1] - q_levels[k]))
        de = abs(float(e_levels[k + 1] - e_levels[k]))
        voltages.append(0.5 * (v_high + v_low))
        coulomb.append(dq / dv)
        energy.append(2.0 * de / denom_e)
    if not voltages:
        return None
    v_start = float(v_seg[0])
    v_end = float(v_seg[-1])
    q_total = abs(float(q[-1]))
    e_total = abs(float(e[-1]))
    coulomb_avg = q_total / (v_start - v_end) if v_start > v_end else None
    denom_full = v_start * v_start - v_end * v_end
    energy_avg = (2.0 * e_total / denom_full) if denom_full > 0 else None
    return {
        "voltages": np.asarray(voltages, dtype=float),
        "coulomb": np.asarray(coulomb, dtype=float),
        "energy": np.asarray(energy, dtype=float),
        "coulomb_avg": coulomb_avg,
        "energy_avg": energy_avg,
        "v_start": v_start,
        "v_end": v_end,
    }


# ---------------------------------------------------------------------------
# Small UI helpers.
# ---------------------------------------------------------------------------


def _refill_combo(combo: QComboBox, items) -> None:
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    for item in items:
        combo.addItem(item)
    restored = combo.findText(current) if current else -1
    if restored >= 0:
        combo.setCurrentIndex(restored)
    combo.blockSignals(False)


def _try_select(combo: QComboBox, preferred_substrings) -> None:
    if combo.count() == 0:
        return
    for i in range(combo.count()):
        text = combo.itemText(i)
        for needle in preferred_substrings:
            if needle.lower() in text.lower():
                combo.setCurrentIndex(i)
                return


def _set_silently(widget: QLineEdit, text: str) -> None:
    widget.blockSignals(True)
    try:
        widget.setText(text)
    finally:
        widget.blockSignals(False)


def _safe_float(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _suggested_capacity_save_path(controller: CapTestController) -> str:
    src = controller.source_path or ""
    if src:
        src_dir = os.path.dirname(src)
        src_base = os.path.splitext(os.path.basename(src))[0]
        return os.path.join(src_dir, f"{src_base}_CapacityvsVoltage.tdms")
    return "CapacityvsVoltage.tdms"


def _suggested_source_with_computed_path(controller: CapTestController) -> str:
    src = controller.source_path or ""
    if src:
        src_dir = os.path.dirname(src)
        src_base = os.path.splitext(os.path.basename(src))[0]
        return os.path.join(src_dir, f"{src_base}_with_computed.tdms")
    return "with_computed.tdms"


# ---------------------------------------------------------------------------
# The window itself.
# ---------------------------------------------------------------------------


class CapacitorTestingWindow(QMainWindow):
    """Free-floating Capacitor Testing window.

    Owns its own ``CapTestController``: edits and computations stay local to
    the window. The optional ``parent`` reference is used solely for Qt
    parent-child semantics (so the OS keeps the window grouped with the main
    application) — it is never inspected for shared state.
    """

    def __init__(self, parent=None) -> None:
        # QMainWindow's parent must be a QWidget (or None). Fall back to None
        # when the caller passes a non-widget host object — the host is still
        # remembered as ``_host`` so 'Use active book' can read its
        # ``data_fit_controller`` for cross-window data import.
        widget_parent = parent if isinstance(parent, QWidget) else None
        super().__init__(widget_parent)
        self._host = parent
        self.setWindowTitle("Capacitor Testing")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(1280, 760)

        self.controller = CapTestController()

        self._build_ui()
        self._connect_actions()
        self._refresh_all_enabled()

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        # --- top toolbar --------------------------------------------------
        tb = QToolBar("Capacitor Testing")
        tb.setMovable(False)
        self.addToolBar(tb)
        self._act_load = tb.addAction("📂 Load file…")
        self._act_load.setToolTip(
            "Open a recording from disk: a TDMS file or an ASCII data file\n"
            "(.txt/.dat/.csv/.tsv/.asc) whose first row is column names."
        )
        self._act_use_book = tb.addAction("📚 Use active book")
        self._act_use_book.setToolTip(
            "Import channels from the Data Workspace's currently active book.\n"
            "Edits inside this window stay local — the workspace book is not\n"
            "modified."
        )
        tb.addSeparator()
        self._act_clear = tb.addAction("🧹 Clear")
        self._act_clear.setToolTip("Reset every Step group and remove the loaded data.")

        # --- left panel ---------------------------------------------------
        left_widget = QWidget()
        left_widget.setMaximumWidth(520)
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)

        self.path_label = QLabel("No file loaded.")
        self.path_label.setStyleSheet("color: gray;")
        self.path_label.setWordWrap(True)
        left.addWidget(self.path_label)

        # Channels group (Dataset / X / Y).
        ch_group = QGroupBox("Channels")
        ch_grid = QGridLayout(ch_group)
        ch_grid.addWidget(QLabel("Dataset:"), 0, 0)
        self.dataset_cb = QComboBox()
        self.dataset_cb.setToolTip(
            "Active dataset. Defaults to the loaded recording. Step 4 adds a\n"
            "derived 'Capacity vs Voltage' dataset that can be selected here;\n"
            "switching repopulates the X- and Y-axis dropdowns below."
        )
        ch_grid.addWidget(self.dataset_cb, 0, 1, 1, 3)
        ch_grid.addWidget(QLabel("X axis:"), 1, 0)
        self.x_cb = QComboBox()
        ch_grid.addWidget(self.x_cb, 1, 1, 1, 3)
        ch_grid.addWidget(QLabel("Y axis:"), 2, 0)
        self.y_cb = QComboBox()
        ch_grid.addWidget(self.y_cb, 2, 1, 1, 3)
        self.robust_view_btn = QPushButton("Robust Auto-Range")
        self.robust_view_btn.setToolTip(
            "Set the view to the 1st-99th percentile of the data with a 10% margin."
        )
        self.full_view_btn = QPushButton("Full View")
        self.full_view_btn.setToolTip("Show the complete data range.")
        ch_grid.addWidget(self.robust_view_btn, 3, 0, 1, 2)
        ch_grid.addWidget(self.full_view_btn, 3, 2, 1, 2)
        left.addWidget(ch_group)

        # Step 1.
        step1 = QGroupBox("Step 1: Select data range on X-axis")
        s1 = QGridLayout(step1)
        self.show_range_cb = QCheckBox("Show / edit")
        self.show_range_cb.setToolTip(
            "Show/hide the purple X-range selector on the plot. While shown,\n"
            "drag its edges to update the Low / High textboxes (and vice versa)."
        )
        s1.addWidget(self.show_range_cb, 0, 0, 1, 4)
        s1.addWidget(QLabel("Low (X):"), 1, 0)
        self.range_low = QLineEdit()
        self.range_low.setMaximumWidth(120)
        self.range_low.setPlaceholderText("auto")
        s1.addWidget(self.range_low, 1, 1)
        s1.addWidget(QLabel("High (X):"), 1, 2)
        self.range_high = QLineEdit()
        self.range_high.setMaximumWidth(120)
        self.range_high.setPlaceholderText("auto")
        s1.addWidget(self.range_high, 1, 3)
        left.addWidget(step1)

        # Step 2.
        step2 = QGroupBox("Step 2: Compute Power on a Resistor (W)")
        s2 = QGridLayout(step2)
        s2.addWidget(QLabel("Row A:"), 0, 0)
        self.row_a_cb = QComboBox()
        self.row_a_cb.setToolTip("First row of the multiplication.")
        s2.addWidget(self.row_a_cb, 0, 1)
        s2.addWidget(QLabel("Row B:"), 1, 0)
        self.row_b_cb = QComboBox()
        self.row_b_cb.setToolTip("Second row of the multiplication.")
        s2.addWidget(self.row_b_cb, 1, 1)
        self.compute_btn = QPushButton("Power on a Resistor (W) calculation")
        self.compute_btn.setToolTip(
            "Compute the elementwise product of Row A and Row B and store it\n"
            f'in memory as "{CAPACITOR_POWER_CHANNEL_NAME}". The new row\n'
            "becomes selectable in the X- and Y-axis dropdowns above."
        )
        self.compute_btn.setStyleSheet(
            "font-weight: bold; background-color: #6f42c1; color: white; padding: 6px;"
        )
        self.compute_btn.setEnabled(False)
        s2.addWidget(self.compute_btn, 2, 0, 1, 2)
        self.compute_status = QLabel("")
        self.compute_status.setStyleSheet("color: gray;")
        self.compute_status.setWordWrap(True)
        s2.addWidget(self.compute_status, 3, 0, 1, 2)
        left.addWidget(step2)

        # Step 3.
        step3 = QGroupBox("Step 3: Integrated power over time (J)")
        s3 = QGridLayout(step3)
        s3.addWidget(QLabel("Time row:"), 0, 0)
        self.int_time_cb = QComboBox()
        self.int_time_cb.setToolTip(
            "Row used as the time axis for the cumulative integral. Defaults\n"
            'to "Time".'
        )
        s3.addWidget(self.int_time_cb, 0, 1)
        s3.addWidget(QLabel("Integrand row:"), 1, 0)
        self.int_value_cb = QComboBox()
        self.int_value_cb.setToolTip(
            f'Row to integrate over time. Defaults to "{CAPACITOR_POWER_CHANNEL_NAME}"\n'
            "when present, but any row may be selected."
        )
        s3.addWidget(self.int_value_cb, 1, 1)
        self.integrate_btn = QPushButton("Integrate over time (J) calculation")
        self.integrate_btn.setToolTip(
            "Compute the cumulative trapezoidal integral of the integrand row\n"
            f'with respect to the time row and store the result as "{CAPACITOR_ENERGY_CHANNEL_NAME}".'
        )
        self.integrate_btn.setStyleSheet(
            "font-weight: bold; background-color: #6f42c1; color: white; padding: 6px;"
        )
        self.integrate_btn.setEnabled(False)
        s3.addWidget(self.integrate_btn, 2, 0, 1, 2)
        self.integrate_status = QLabel("")
        self.integrate_status.setStyleSheet("color: gray;")
        self.integrate_status.setWordWrap(True)
        s3.addWidget(self.integrate_status, 3, 0, 1, 2)
        left.addWidget(step3)

        # Step 4.
        step4 = QGroupBox(f"Step 4: Capacity at every {CAPACITY_VOLTAGE_STEP_V:.0f} V")
        s4 = QGridLayout(step4)
        s4.addWidget(QLabel("Time row:"), 0, 0)
        self.q_time_cb = QComboBox()
        self.q_time_cb.setToolTip(
            'Row used as the time axis for the cumulative-charge integral.'
        )
        s4.addWidget(self.q_time_cb, 0, 1)
        s4.addWidget(QLabel("Capacitor bank current:"), 1, 0)
        self.q_current_cb = QComboBox()
        self.q_current_cb.setToolTip(
            "Discharge current row. The cumulative charge ∫ i dt is sampled at\n"
            "every voltage step to give the capacity via C = ΔQ / ΔV."
        )
        s4.addWidget(self.q_current_cb, 1, 1)
        s4.addWidget(QLabel("Capacitor bank voltage:"), 2, 0)
        self.q_voltage_cb = QComboBox()
        self.q_voltage_cb.setToolTip(
            f"Voltage row. Capacity is computed at every {CAPACITY_VOLTAGE_STEP_V:.0f} V step\n"
            "from the peak down."
        )
        s4.addWidget(self.q_voltage_cb, 2, 1)
        self.q_compute_btn = QPushButton(
            f'Compute capacity vs voltage ({CAPACITY_VOLTAGE_STEP_V:.0f} V steps)'
        )
        self.q_compute_btn.setToolTip(
            f'Compute capacity by Coulomb counting at every {CAPACITY_VOLTAGE_STEP_V:.0f} V step\n'
            f'and store the result as a separate "{CAPACITY_DATASET_NAME}" dataset.'
        )
        self.q_compute_btn.setStyleSheet(
            "font-weight: bold; background-color: #6f42c1; color: white; padding: 6px;"
        )
        self.q_compute_btn.setEnabled(False)
        s4.addWidget(self.q_compute_btn, 3, 0, 1, 2)
        self.q_compute_status = QLabel("")
        self.q_compute_status.setStyleSheet("color: gray;")
        self.q_compute_status.setWordWrap(True)
        s4.addWidget(self.q_compute_status, 4, 0, 1, 2)
        left.addWidget(step4)

        # Save derived data.
        save_group = QGroupBox("Save derived data")
        sg = QGridLayout(save_group)
        self.save_source_btn = QPushButton(
            "Save Power && Integrated Power into source file"
        )
        self.save_source_btn.setToolTip(
            f'Append "{CAPACITOR_POWER_CHANNEL_NAME}" and "{CAPACITOR_ENERGY_CHANNEL_NAME}"\n'
            "back into the loaded recording. TDMS files get a separate\n"
            "'Computed' group; ASCII files get two new columns appended."
        )
        self.save_source_btn.setEnabled(False)
        sg.addWidget(self.save_source_btn, 0, 0, 1, 2)
        self.save_source_tdms_btn = QPushButton(
            "Save Power && Integrated Power as TDMS…"
        )
        self.save_source_tdms_btn.setToolTip(
            "Open a Save-As dialog and write the source channels plus Power\n"
            "and Integrated Power into a new TDMS file. The source file is\n"
            "left untouched."
        )
        self.save_source_tdms_btn.setEnabled(False)
        sg.addWidget(self.save_source_tdms_btn, 1, 0, 1, 2)
        self.save_capacity_btn = QPushButton(
            f'Save "{CAPACITY_DATASET_NAME}" as TDMS…'
        )
        self.save_capacity_btn.setToolTip(
            "Open a Save-As dialog and write the Capacity vs Voltage dataset to\n"
            "a TDMS file."
        )
        self.save_capacity_btn.setEnabled(False)
        sg.addWidget(self.save_capacity_btn, 2, 0, 1, 2)
        self.save_status = QLabel("")
        self.save_status.setStyleSheet("color: gray;")
        self.save_status.setWordWrap(True)
        sg.addWidget(self.save_status, 3, 0, 1, 2)
        left.addWidget(save_group)

        left.addStretch()
        root.addWidget(left_widget)

        # --- right panel: plot --------------------------------------------
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)

        try:
            from .tab import EngineeringAxisItem
            self.plot = pg.PlotWidget(
                title="Capacitor preview",
                axisItems={
                    "bottom": EngineeringAxisItem(orientation="bottom"),
                    "left": EngineeringAxisItem(orientation="left"),
                },
            )
        except Exception:
            self.plot = pg.PlotWidget(title="Capacitor preview")
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True)
        self.preview_curve = self.plot.plot(pen=pg.mkPen("b", width=1.5))

        self.range_band = pg.LinearRegionItem(
            brush=pg.mkBrush(170, 80, 200, 45),
            pen=pg.mkPen(170, 80, 200, 180),
            movable=False,
        )
        self.range_band.setZValue(5)
        self.range_band.setVisible(False)
        self.plot.addItem(self.range_band, ignoreBounds=True)

        right.addWidget(self.plot, stretch=1)
        root.addWidget(right_widget, stretch=1)

    # -- signal wiring ---------------------------------------------------

    def _connect_actions(self) -> None:
        self._act_load.triggered.connect(self._on_load_file)
        self._act_use_book.triggered.connect(self._on_use_active_book)
        self._act_clear.triggered.connect(self._on_clear)

        self.dataset_cb.currentIndexChanged.connect(lambda _i: self._on_dataset_changed())
        self.x_cb.currentIndexChanged.connect(lambda _i: self._refresh_preview_curve())
        self.y_cb.currentIndexChanged.connect(lambda _i: self._refresh_preview_curve())
        self.robust_view_btn.clicked.connect(self._on_robust_view)
        self.full_view_btn.clicked.connect(self._on_full_view)

        self.show_range_cb.toggled.connect(self._on_show_range_toggled)
        self.range_low.editingFinished.connect(self._on_range_text_edited)
        self.range_high.editingFinished.connect(self._on_range_text_edited)
        self.range_band.sigRegionChanged.connect(self._on_band_dragged)

        for combo in (self.row_a_cb, self.row_b_cb):
            combo.currentIndexChanged.connect(lambda _i: self._refresh_compute_enabled())
        self.compute_btn.clicked.connect(self._on_compute_power)

        for combo in (self.int_time_cb, self.int_value_cb):
            combo.currentIndexChanged.connect(lambda _i: self._refresh_integrate_enabled())
        self.integrate_btn.clicked.connect(self._on_integrate)

        for combo in (self.q_time_cb, self.q_current_cb, self.q_voltage_cb):
            combo.currentIndexChanged.connect(lambda _i: self._refresh_capacity_enabled())
        self.q_compute_btn.clicked.connect(self._on_compute_capacity)

        self.save_source_btn.clicked.connect(self._on_save_to_source)
        self.save_source_tdms_btn.clicked.connect(self._on_save_source_as_tdms)
        self.save_capacity_btn.clicked.connect(self._on_save_capacity)

    # -- file/book actions -----------------------------------------------

    def _on_load_file(self) -> None:
        start_dir = ""
        if self.controller.source_path:
            start_dir = os.path.dirname(self.controller.source_path)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load recording",
            start_dir,
            "Recording files (*.tdms *.txt *.dat *.csv *.tsv *.asc);;All Files (*)",
        )
        if not path:
            return
        ok, msg = self.controller.load_recording(path)
        if not ok:
            QMessageBox.warning(self, "Load failed", msg)
            return
        self._after_data_changed(msg)

    def _on_use_active_book(self) -> None:
        host = self._host if self._host is not None else self.parent()
        if host is None:
            QMessageBox.information(
                self, "Use active book",
                "Open this window from the Data Workspace toolbar to import\n"
                "the active book.",
            )
            return
        ctrl = self._resolve_active_book_controller(host)
        if ctrl is None or not getattr(ctrl, "channel_names", None):
            QMessageBox.information(
                self, "Use active book",
                "The Data Workspace has no active book yet — load a recording\n"
                "in the workspace first.",
            )
            return
        label = os.path.basename(getattr(ctrl, "tdms_path", "") or "") or "Active book"
        ok, msg = self.controller.adopt_from_book(ctrl, label)
        if not ok:
            QMessageBox.warning(self, "Use active book", msg)
            return
        self._after_data_changed(msg)

    @staticmethod
    def _resolve_active_book_controller(host):
        """Return the active book controller from the workspace, or None.

        Prefers the workspace's last-active book window so the import
        targets whichever book the user just clicked on. Falls back to the
        single ``data_fit_controller`` for legacy hosts that still expose
        only one controller.
        """
        try:
            from .tab import _workspace_active_book_state
        except ImportError:  # pragma: no cover - defensive
            _workspace_active_book_state = None
        if _workspace_active_book_state is not None:
            try:
                state = _workspace_active_book_state(host)
            except Exception:
                state = None
            if state is not None:
                ctrl = state.get("controller")
                if ctrl is not None:
                    return ctrl
        return getattr(host, "data_fit_controller", None)

    def _on_clear(self) -> None:
        self.controller.reset()
        self._after_data_changed("No file loaded.")

    def _after_data_changed(self, status_text: str) -> None:
        if self.controller.source_path:
            self.path_label.setText(status_text)
            self.path_label.setStyleSheet("color: black;")
        else:
            self.path_label.setText(status_text)
            self.path_label.setStyleSheet("color: gray;")
        # Hide the Step-1 band on every reload — the prior recording's
        # range is no longer meaningful.
        self.show_range_cb.blockSignals(True)
        self.show_range_cb.setChecked(False)
        self.show_range_cb.blockSignals(False)
        self.range_band.setVisible(False)
        self.range_band.setMovable(False)
        _set_silently(self.range_low, "")
        _set_silently(self.range_high, "")
        for status in (
            self.compute_status,
            self.integrate_status,
            self.q_compute_status,
            self.save_status,
        ):
            status.setText("")
            status.setStyleSheet("color: gray;")
        self._populate_dataset_combo()
        self._populate_step_combos()
        self._refresh_preview_curve()
        self._refresh_all_enabled()

    # -- dataset / channel combos ----------------------------------------

    def _populate_dataset_combo(self) -> None:
        names = list(self.controller.datasets.keys())
        _refill_combo(self.dataset_cb, names)
        if self.controller.active_dataset_name and self.controller.active_dataset_name in names:
            idx = self.dataset_cb.findText(self.controller.active_dataset_name)
            if idx >= 0:
                self.dataset_cb.setCurrentIndex(idx)
        self._refresh_xy_combos_from_active_dataset()

    def _refresh_xy_combos_from_active_dataset(self) -> None:
        active = self.controller.active_dataset()
        if active is None:
            _refill_combo(self.x_cb, [])
            _refill_combo(self.y_cb, [])
            return
        channels = active.get("channels", {}) or {}
        names = list(channels.keys())
        # Recording dataset offers Time as an X option too.
        x_options = (["Time"] + names) if active.get("is_recording") else names
        _refill_combo(self.x_cb, x_options)
        _refill_combo(self.y_cb, names)
        if active.get("is_recording"):
            _try_select(self.x_cb, ("AI0", "Current", "I", "current", "Time"))
            _try_select(self.y_cb, ("AI1", "Voltage", "V", "voltage"))

    def _on_dataset_changed(self) -> None:
        name = self.dataset_cb.currentText()
        if name and name in self.controller.datasets:
            self.controller.active_dataset_name = name
        self._refresh_xy_combos_from_active_dataset()
        self._refresh_preview_curve()

    def _populate_step_combos(self) -> None:
        names = list(self.controller.channel_names)
        options = ["Time"] + names
        _refill_combo(self.row_a_cb, options)
        _refill_combo(self.row_b_cb, options)
        _refill_combo(self.int_time_cb, options)
        _refill_combo(self.int_value_cb, options)
        idx_t = self.int_time_cb.findText("Time")
        if idx_t >= 0:
            self.int_time_cb.setCurrentIndex(idx_t)
        idx_p = self.int_value_cb.findText(CAPACITOR_POWER_CHANNEL_NAME)
        if idx_p >= 0:
            self.int_value_cb.setCurrentIndex(idx_p)
        _refill_combo(self.q_time_cb, options)
        idx_t = self.q_time_cb.findText("Time")
        if idx_t >= 0:
            self.q_time_cb.setCurrentIndex(idx_t)
        _refill_combo(self.q_current_cb, names)
        _refill_combo(self.q_voltage_cb, names)
        _try_select(self.q_current_cb, ("idisch", "discharge", "Current", "current", "I_", "I(", "I "))
        _try_select(self.q_voltage_cb, ("V(cap", "VR", "Voltage", "voltage", "V_", "V(", "U("))

    # -- preview plot -----------------------------------------------------

    def _resolve_channel(self, name: str) -> Optional[np.ndarray]:
        if not name:
            return None
        active = self.controller.active_dataset()
        if active is None:
            return None
        if name == "Time":
            t = active.get("time")
            return None if t is None else np.asarray(t, dtype=float)
        ch = (active.get("channels") or {}).get(name)
        return None if ch is None else np.asarray(ch, dtype=float)

    def _refresh_preview_curve(self) -> None:
        x = self._resolve_channel(self.x_cb.currentText())
        y = self._resolve_channel(self.y_cb.currentText())
        if x is None or y is None:
            self.preview_curve.setData([], [])
            return
        n = int(min(x.size, y.size))
        if n < 1:
            self.preview_curve.setData([], [])
            return
        self.preview_curve.setData(x[:n], y[:n])
        self.plot.setLabel("bottom", self.x_cb.currentText() or "X")
        self.plot.setLabel("left", self.y_cb.currentText() or "Y")

    def _on_robust_view(self) -> None:
        x = self._resolve_channel(self.x_cb.currentText())
        y = self._resolve_channel(self.y_cb.currentText())
        if x is None or y is None or x.size == 0 or y.size == 0:
            return
        try:
            x_lo, x_hi = float(np.nanpercentile(x, 1)), float(np.nanpercentile(x, 99))
            y_lo, y_hi = float(np.nanpercentile(y, 1)), float(np.nanpercentile(y, 99))
        except Exception:
            return
        x_pad = (x_hi - x_lo) * 0.10 if x_hi > x_lo else 1.0
        y_pad = (y_hi - y_lo) * 0.10 if y_hi > y_lo else 1.0
        vb = self.plot.getPlotItem().getViewBox()
        vb.setRange(xRange=(x_lo - x_pad, x_hi + x_pad), yRange=(y_lo - y_pad, y_hi + y_pad), padding=0)

    def _on_full_view(self) -> None:
        self.plot.getPlotItem().getViewBox().autoRange()

    # -- Step 1: range band ----------------------------------------------

    def _on_show_range_toggled(self, checked: bool) -> None:
        self.range_band.setMovable(bool(checked))
        self.range_band.setVisible(bool(checked))
        if checked:
            self._seed_range_band()

    def _seed_range_band(self) -> None:
        lo = _safe_float(self.range_low.text())
        hi = _safe_float(self.range_high.text())
        if lo is None or hi is None:
            vb = self.plot.getPlotItem().getViewBox()
            x_lo, x_hi = vb.viewRange()[0]
            span = x_hi - x_lo
            margin = span * 0.25 if span > 0 else 1.0
            if lo is None:
                lo = x_lo + margin
            if hi is None:
                hi = x_hi - margin
            if lo >= hi:
                lo, hi = x_lo, x_hi
        self.range_band.blockSignals(True)
        try:
            self.range_band.setRegion((float(lo), float(hi)))
        finally:
            self.range_band.blockSignals(False)
        _set_silently(self.range_low, f"{float(lo):.6g}")
        _set_silently(self.range_high, f"{float(hi):.6g}")

    def _on_band_dragged(self, *_args) -> None:
        if not self.range_band.movable:
            return
        lo, hi = self.range_band.getRegion()
        _set_silently(self.range_low, f"{float(lo):.6g}")
        _set_silently(self.range_high, f"{float(hi):.6g}")

    def _on_range_text_edited(self) -> None:
        lo = _safe_float(self.range_low.text())
        hi = _safe_float(self.range_high.text())
        if lo is None or hi is None:
            return
        if lo > hi:
            lo, hi = hi, lo
        self.range_band.blockSignals(True)
        try:
            self.range_band.setRegion((float(lo), float(hi)))
        finally:
            self.range_band.blockSignals(False)

    # -- Step 2: power on a resistor -------------------------------------

    def _resolve_step_row(self, name: str):
        if not name:
            return None
        if name == "Time":
            return self.controller.time_array
        return self.controller.get_channel(name)

    def _refresh_compute_enabled(self) -> None:
        self.compute_btn.setEnabled(
            bool(self.row_a_cb.currentText()) and bool(self.row_b_cb.currentText())
        )

    def _on_compute_power(self) -> None:
        name_a = self.row_a_cb.currentText()
        name_b = self.row_b_cb.currentText()
        a = self._resolve_step_row(name_a)
        b = self._resolve_step_row(name_b)
        if a is None or b is None:
            self._set_status(self.compute_status, "Could not resolve one of the selected rows.", warn=True)
            return
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        n = int(min(a_arr.size, b_arr.size))
        if n == 0:
            self._set_status(self.compute_status, "Selected rows are empty.", warn=True)
            return
        product = a_arr[:n] * b_arr[:n]
        self.controller.channel_cache[CAPACITOR_POWER_CHANNEL_NAME] = product
        if CAPACITOR_POWER_CHANNEL_NAME not in self.controller.channel_names:
            self.controller.channel_names.append(CAPACITOR_POWER_CHANNEL_NAME)
        self._refresh_xy_combos_from_active_dataset()
        self._populate_step_combos()
        self._refresh_preview_curve()
        self._refresh_all_enabled()
        self._set_status(
            self.compute_status,
            f'Computed "{CAPACITOR_POWER_CHANNEL_NAME}" = "{name_a}" * "{name_b}" '
            f'({n} samples). Available now in the Y-axis dropdown.',
            warn=False,
        )

    # -- Step 3: cumulative integral -------------------------------------

    def _refresh_integrate_enabled(self) -> None:
        self.integrate_btn.setEnabled(
            bool(self.int_time_cb.currentText()) and bool(self.int_value_cb.currentText())
        )

    def _on_integrate(self) -> None:
        time_name = self.int_time_cb.currentText()
        value_name = self.int_value_cb.currentText()
        t = self._resolve_step_row(time_name)
        v = self._resolve_step_row(value_name)
        if t is None or v is None:
            self._set_status(self.integrate_status, "Could not resolve one of the selected rows.", warn=True)
            return
        t_arr = np.asarray(t, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        n = int(min(t_arr.size, v_arr.size))
        if n < 2:
            self._set_status(self.integrate_status, "Need at least two samples to integrate.", warn=True)
            return
        energy = _cumulative_trapezoid(v_arr[:n], t_arr[:n])
        self.controller.channel_cache[CAPACITOR_ENERGY_CHANNEL_NAME] = energy
        if CAPACITOR_ENERGY_CHANNEL_NAME not in self.controller.channel_names:
            self.controller.channel_names.append(CAPACITOR_ENERGY_CHANNEL_NAME)
        self._refresh_xy_combos_from_active_dataset()
        self._populate_step_combos()
        self._refresh_preview_curve()
        self._refresh_all_enabled()
        self._set_status(
            self.integrate_status,
            f'Computed "{CAPACITOR_ENERGY_CHANNEL_NAME}" = ∫ "{value_name}" d"{time_name}" '
            f'({n} samples, total ≈ {energy[-1]:.6g} J).',
            warn=False,
        )

    # -- Step 4: capacity vs voltage -------------------------------------

    def _refresh_capacity_enabled(self) -> None:
        self.q_compute_btn.setEnabled(
            bool(self.q_time_cb.currentText())
            and bool(self.q_current_cb.currentText())
            and bool(self.q_voltage_cb.currentText())
        )

    def _on_compute_capacity(self) -> None:
        time_name = self.q_time_cb.currentText()
        current_name = self.q_current_cb.currentText()
        voltage_name = self.q_voltage_cb.currentText()
        if not time_name or not current_name or not voltage_name:
            self._set_status(self.q_compute_status, "Pick a time, current and voltage row.", warn=True)
            return
        t = self._resolve_step_row(time_name)
        i = self.controller.get_channel(current_name)
        v = self.controller.get_channel(voltage_name)
        if t is None or i is None or v is None:
            self._set_status(self.q_compute_status, "Could not resolve the selected rows.", warn=True)
            return
        results = _compute_capacity_results(t, v, i, step_v=CAPACITY_VOLTAGE_STEP_V)
        if results is None:
            self._set_status(
                self.q_compute_status,
                f"Not enough discharge for one {CAPACITY_VOLTAGE_STEP_V:.0f} V step. "
                "Check the selected rows.",
                warn=True,
            )
            return
        voltages = results["voltages"]
        coulomb = results["coulomb"]
        energy = results["energy"]
        coulomb_avg = results["coulomb_avg"]
        energy_avg = results["energy_avg"]
        channels: dict[str, np.ndarray] = {
            CAPACITY_VOLTAGE_COLUMN: voltages,
            CAPACITY_COULOMB_COLUMN: coulomb,
            CAPACITY_ENERGY_COLUMN: energy,
        }
        if coulomb_avg is not None:
            channels[CAPACITY_COULOMB_AVG_COLUMN] = np.asarray([coulomb_avg], dtype=float)
        if energy_avg is not None:
            channels[CAPACITY_ENERGY_AVG_COLUMN] = np.asarray([energy_avg], dtype=float)
        self.controller.add_derived_dataset(CAPACITY_DATASET_NAME, channels=channels, time=None)
        self._populate_dataset_combo()
        self._refresh_all_enabled()
        avg_parts = []
        if coulomb_avg is not None:
            avg_parts.append(f"Coulomb avg ≈ {coulomb_avg:.6g} F")
        if energy_avg is not None:
            avg_parts.append(f"Energy avg ≈ {energy_avg:.6g} F")
        avg_text = " / ".join(avg_parts) if avg_parts else "no avg"
        self._set_status(
            self.q_compute_status,
            f'Stored "{CAPACITY_DATASET_NAME}" ({voltages.size} steps; '
            f"V {results['v_start']:.6g}→{results['v_end']:.6g} V). "
            f"{avg_text}. Pick the dataset to plot.",
            warn=False,
        )

    # -- save handlers ---------------------------------------------------

    def _refresh_all_enabled(self) -> None:
        self._refresh_compute_enabled()
        self._refresh_integrate_enabled()
        self._refresh_capacity_enabled()
        has_power = CAPACITOR_POWER_CHANNEL_NAME in self.controller.channel_cache
        has_energy = CAPACITOR_ENERGY_CHANNEL_NAME in self.controller.channel_cache
        has_capacity = CAPACITY_DATASET_NAME in (self.controller.datasets or {})
        has_computed = has_power or has_energy
        src_path = self.controller.source_path or ""
        ext = os.path.splitext(src_path)[1].lower()
        src_is_supported = ext == ".tdms" or ext in _ASCII_RECORDING_EXTS
        self.save_source_btn.setEnabled(src_is_supported and has_computed)
        # Save-As-TDMS only needs computed data — works even without a path.
        self.save_source_tdms_btn.setEnabled(has_computed)
        self.save_capacity_btn.setEnabled(has_capacity)

    def _set_status(self, label: QLabel, text: str, *, warn: bool) -> None:
        label.setText(text)
        label.setStyleSheet("color: #b35a00;" if warn else "color: #2a7a2a;")

    def _on_save_source_as_tdms(self) -> None:
        controller = self.controller
        power = controller.channel_cache.get(CAPACITOR_POWER_CHANNEL_NAME)
        energy = controller.channel_cache.get(CAPACITOR_ENERGY_CHANNEL_NAME)
        if power is None and energy is None:
            self._set_status(self.save_status, "Nothing to save — compute Power or Integrated Power first.", warn=True)
            return
        suggested = _suggested_source_with_computed_path(controller)
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save source + computed channels as TDMS",
            suggested,
            "TDMS Files (*.tdms);;All Files (*)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".tdms"):
            out_path = out_path + ".tdms"
        try:
            objects: list = [RootObject(), GroupObject("RawData")]
            time_arr = controller.time_array
            if time_arr is not None:
                objects.append(
                    ChannelObject("RawData", "Time", np.asarray(time_arr, dtype=float))
                )
            skip_names = {CAPACITOR_POWER_CHANNEL_NAME, CAPACITOR_ENERGY_CHANNEL_NAME}
            for name in controller.channel_names:
                if name in skip_names:
                    continue
                arr = controller.channel_cache.get(name)
                if arr is not None:
                    objects.append(
                        ChannelObject("RawData", name, np.asarray(arr, dtype=float))
                    )
            objects.append(GroupObject("Computed"))
            added: list[str] = []
            if power is not None:
                objects.append(
                    ChannelObject(
                        "Computed", CAPACITOR_POWER_CHANNEL_NAME, np.asarray(power, dtype=float)
                    )
                )
                added.append(CAPACITOR_POWER_CHANNEL_NAME)
            if energy is not None:
                objects.append(
                    ChannelObject(
                        "Computed", CAPACITOR_ENERGY_CHANNEL_NAME, np.asarray(energy, dtype=float)
                    )
                )
                added.append(CAPACITOR_ENERGY_CHANNEL_NAME)
            with TdmsWriter(out_path) as w:
                w.write_segment(objects)
        except Exception as exc:
            self._set_status(self.save_status, f"Save failed: {exc}", warn=True)
            return
        self._set_status(
            self.save_status,
            f'Saved source + {", ".join(added)} to "{os.path.basename(out_path)}".',
            warn=False,
        )

    def _on_save_to_source(self) -> None:
        controller = self.controller
        path = controller.source_path
        ext = os.path.splitext(path)[1].lower() if path else ""
        if not path or not os.path.exists(path) or (
            ext != ".tdms" and ext not in _ASCII_RECORDING_EXTS
        ):
            self._set_status(self.save_status, "Source file is not a supported recording.", warn=True)
            return
        power = controller.channel_cache.get(CAPACITOR_POWER_CHANNEL_NAME)
        energy = controller.channel_cache.get(CAPACITOR_ENERGY_CHANNEL_NAME)
        if power is None and energy is None:
            self._set_status(self.save_status, "Nothing to save — compute Power or Integrated Power first.", warn=True)
            return
        try:
            if ext == ".tdms":
                added = _save_computed_into_tdms_source(controller, path, power, energy)
                location = f'group "Computed" of "{os.path.basename(path)}"'
            else:
                added = _save_computed_into_ascii_source(controller, path, power, energy)
                location = f'"{os.path.basename(path)}"'
        except Exception as exc:
            self._set_status(self.save_status, f"Save failed: {exc}", warn=True)
            return
        self._set_status(self.save_status, f'Saved {", ".join(added)} into {location}.', warn=False)

    def _on_save_capacity(self) -> None:
        controller = self.controller
        dataset = controller.datasets.get(CAPACITY_DATASET_NAME) if controller.datasets else None
        if dataset is None:
            self._set_status(self.save_status, "Compute capacity first (Step 4).", warn=True)
            return
        suggested = _suggested_capacity_save_path(controller)
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            f'Save "{CAPACITY_DATASET_NAME}"',
            suggested,
            "TDMS Files (*.tdms);;All Files (*)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".tdms"):
            out_path = out_path + ".tdms"
        try:
            objects = [RootObject(), GroupObject("CapacityvsVoltage")]
            for col_name, arr in dataset["channels"].items():
                objects.append(
                    ChannelObject("CapacityvsVoltage", col_name, np.asarray(arr, dtype=float))
                )
            with TdmsWriter(out_path) as w:
                w.write_segment(objects)
        except Exception as exc:
            self._set_status(self.save_status, f"Save failed: {exc}", warn=True)
            return
        self._set_status(self.save_status, f'Saved capacity dataset to "{os.path.basename(out_path)}".', warn=False)


# ---------------------------------------------------------------------------
# Module-level save helpers (used by the window).
# ---------------------------------------------------------------------------


def _save_computed_into_tdms_source(controller: CapTestController, path, power, energy):
    objects = [RootObject()]
    groups_seen: set[str] = set()
    with TdmsFile.read(path) as src:
        for group in src.groups():
            if group.name == "Computed":
                continue
            if group.name not in groups_seen:
                objects.append(GroupObject(group.name))
                groups_seen.add(group.name)
            for channel in group.channels():
                objects.append(
                    ChannelObject(
                        group.name, channel.name, np.asarray(channel[:], dtype=float)
                    )
                )
    objects.append(GroupObject("Computed"))
    added: list[str] = []
    if power is not None:
        objects.append(
            ChannelObject("Computed", CAPACITOR_POWER_CHANNEL_NAME, np.asarray(power, dtype=float))
        )
        added.append(CAPACITOR_POWER_CHANNEL_NAME)
    if energy is not None:
        objects.append(
            ChannelObject("Computed", CAPACITOR_ENERGY_CHANNEL_NAME, np.asarray(energy, dtype=float))
        )
        added.append(CAPACITOR_ENERGY_CHANNEL_NAME)
    tmp_path = path + ".tmp"
    with TdmsWriter(tmp_path) as w:
        w.write_segment(objects)
    os.replace(tmp_path, path)
    return added


def _save_computed_into_ascii_source(controller: CapTestController, path, power, energy):
    delim = "," if path.lower().endswith(".csv") else "\t"
    time_arr = controller.time_array
    if time_arr is None:
        raise RuntimeError("source has no time array to align against")
    columns: list[tuple[str, np.ndarray]] = [
        (controller.time_column_name or "time", np.asarray(time_arr, dtype=float)),
    ]
    skip_names = {CAPACITOR_POWER_CHANNEL_NAME, CAPACITOR_ENERGY_CHANNEL_NAME}
    for name in controller.channel_names:
        if name in skip_names:
            continue
        arr = controller.channel_cache.get(name)
        if arr is None:
            continue
        columns.append((name, np.asarray(arr, dtype=float)))
    added: list[str] = []
    if power is not None:
        columns.append((CAPACITOR_POWER_CHANNEL_NAME, np.asarray(power, dtype=float)))
        added.append(CAPACITOR_POWER_CHANNEL_NAME)
    if energy is not None:
        columns.append((CAPACITOR_ENERGY_CHANNEL_NAME, np.asarray(energy, dtype=float)))
        added.append(CAPACITOR_ENERGY_CHANNEL_NAME)
    n = int(min(arr.size for _, arr in columns))
    if n == 0:
        raise RuntimeError("aligned column length is zero")
    matrix = np.column_stack([arr[:n] for _, arr in columns])
    header = delim.join(name for name, _ in columns)
    tmp_path = path + ".tmp"
    np.savetxt(tmp_path, matrix, delimiter=delim, header=header, comments="", fmt="%.15e")
    os.replace(tmp_path, path)
    return added


# ---------------------------------------------------------------------------
# Public entry point used by the Data Workspace toolbar.
# ---------------------------------------------------------------------------


def open_capacitor_testing_window(parent=None) -> CapacitorTestingWindow:
    """Open or raise the singleton Capacitor Testing window for ``parent``.

    A reference is stashed on the parent under ``_capacitor_testing_window``
    so subsequent toolbar clicks reuse the same window instead of opening a
    duplicate.
    """
    holder = parent if parent is not None else open_capacitor_testing_window
    existing = getattr(holder, "_capacitor_testing_window", None)
    if existing is not None:
        try:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        except RuntimeError:
            try:
                setattr(holder, "_capacitor_testing_window", None)
            except (AttributeError, TypeError):
                pass
    win = CapacitorTestingWindow(parent)
    try:
        setattr(holder, "_capacitor_testing_window", win)
    except (AttributeError, TypeError):
        # Module-level fallback when parent rejects new attributes.
        open_capacitor_testing_window._capacitor_testing_window = win  # type: ignore[attr-defined]
    win.show()
    win.raise_()
    win.activateWindow()
    return win

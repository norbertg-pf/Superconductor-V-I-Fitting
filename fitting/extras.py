"""Dialogs and helpers for the Data Fitting tab: graph formatting, presets, export.

Kept together in this folder so the whole tab can later be lifted out as a
standalone module: copy ``data_fitting_tab.py``, ``data_fitting_extras.py`` and
``src/services/data_fitting_service.py`` to a new project and the tab will
work with just PyQt5, pyqtgraph, numpy, scipy and nptdms as dependencies.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pyqtgraph as pg
from pyqtgraph import exporters
from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QGraphicsObject
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFontComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QDoubleValidator


# ----------------------------------------------------------------------------
# Settings model
# ----------------------------------------------------------------------------

LINE_STYLES = {
    "Solid": Qt.SolidLine,
    "Dash": Qt.DashLine,
    "Dot": Qt.DotLine,
    "DashDot": Qt.DashDotLine,
    "DashDotDot": Qt.DashDotDotLine,
}


class SciEdit(QLineEdit):
    """QLineEdit that accepts scientific-notation floats (e.g. ``1e-6``, ``2.5E3``)."""

    def __init__(self, value: float = 0.0, *, decimals: int = 12, parent=None):
        super().__init__(parent)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.ScientificNotation)
        validator.setDecimals(decimals)
        self.setValidator(validator)
        self.setMinimumWidth(110)
        self.setValue(value)

    def value(self) -> float:
        try:
            return float(self.text())
        except (ValueError, TypeError):
            return 0.0

    def setValue(self, value: float) -> None:
        if value == 0:
            self.setText("0")
        else:
            self.setText(f"{value:g}")


def _axis_icon(kind: str, size: int = 28) -> QIcon:
    """Tiny axis thumbnail: the selected side is drawn in bold red."""
    pm = QPixmap(size, size)
    pm.fill(Qt.white)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)

    frame_pen = QPen(QColor("#999999"))
    frame_pen.setWidthF(1.0)
    painter.setPen(frame_pen)
    margin = 4
    rect = (margin, margin, size - 2 * margin, size - 2 * margin)
    painter.drawRect(*rect)

    hot = QPen(QColor("#cc2222"))
    hot.setWidthF(2.6)
    painter.setPen(hot)
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    if kind == "Horizontal":
        painter.drawLine(x0, y1, x1, y1)
    elif kind == "Vertical":
        painter.drawLine(x0, y0, x0, y1)
    elif kind == "Bottom":
        painter.drawLine(x0, y1, x1, y1)
    elif kind == "Top":
        painter.drawLine(x0, y0, x1, y0)
    elif kind == "Left":
        painter.drawLine(x0, y0, x0, y1)
    elif kind == "Right":
        painter.drawLine(x1, y0, x1, y1)
    painter.end()
    return QIcon(pm)


@dataclass
class AxisScale:
    auto_range: bool = True
    from_val: float = 0.0
    to_val: float = 1.0
    scale_type: str = "Linear"            # "Linear" | "Log10"
    reverse: bool = False
    major_tick_type: str = "By Increment"  # "By Increment" | "By Count"
    major_tick_value: float = 0.0          # 0 = auto
    minor_tick_type: str = "By Count"
    minor_tick_count: int = 1


@dataclass
class TickLabels:
    show: bool = True
    divide_factor: float = 1.0
    decimal_places: int = 3
    set_decimal: bool = False
    prefix: str = ""
    suffix: str = ""


@dataclass
class AxisTitle:
    show: bool = True
    text: str = ""
    color: str = "#000000"
    size: int = 12
    font_family: str = ""


@dataclass
class GridLine:
    show: bool = False
    color: str = "#808080"
    style: str = "Solid"
    thickness: float = 0.5


@dataclass
class GridConfig:
    major: GridLine = field(default_factory=lambda: GridLine(
        show=True, style="Solid", color="#c8c8c8", thickness=0.4,
    ))
    minor: GridLine = field(default_factory=lambda: GridLine(
        show=True, style="Dash", color="#e0e0e0", thickness=0.3,
    ))


@dataclass
class LineAndTicks:
    show: bool = True
    line_color: str = "#000000"
    line_thickness: float = 1.0
    major_length: int = 6
    minor_length: int = 3


@dataclass
class RawCurveStyle:
    draw_mode: str = "Lines + points"     # "Lines + points" | "Points only" | "Lines only"
    line_color: str = "#0057b8"            # scientific-paper blue
    point_color: str = "#0057b8"
    line_width: float = 1.2
    point_size: int = 3
    alpha: int = 255


@dataclass
class GraphSettings:
    """Everything the Graph-settings dialog exposes."""

    curve: RawCurveStyle = field(default_factory=RawCurveStyle)
    # Scale: horizontal = bottom/x, vertical = left/y.
    scale_h: AxisScale = field(default_factory=AxisScale)
    scale_v: AxisScale = field(default_factory=AxisScale)
    # Tick label config per side.
    ticks_bottom: TickLabels = field(default_factory=TickLabels)
    ticks_top: TickLabels = field(default_factory=lambda: TickLabels(show=False))
    ticks_left: TickLabels = field(default_factory=TickLabels)
    ticks_right: TickLabels = field(default_factory=lambda: TickLabels(show=False))
    # Axis titles per side.
    title_bottom: AxisTitle = field(default_factory=lambda: AxisTitle(text="Current (A)"))
    title_top: AxisTitle = field(default_factory=lambda: AxisTitle(show=False))
    title_left: AxisTitle = field(default_factory=lambda: AxisTitle(text="Voltage (V)"))
    title_right: AxisTitle = field(default_factory=lambda: AxisTitle(show=False))
    # Grids (vertical = X ticks, horizontal = Y ticks).
    grid_v: GridConfig = field(default_factory=GridConfig)
    grid_h: GridConfig = field(default_factory=GridConfig)
    # Axis line + tick visibility per side.
    line_bottom: LineAndTicks = field(default_factory=LineAndTicks)
    line_top: LineAndTicks = field(default_factory=lambda: LineAndTicks(show=True))
    line_left: LineAndTicks = field(default_factory=LineAndTicks)
    line_right: LineAndTicks = field(default_factory=lambda: LineAndTicks(show=True))
    # Plot title.
    plot_title_text: str = "V-I preview"
    plot_title_size: int = 13
    plot_title_color: str = "#000000"
    plot_title_font_family: str = ""
    # Residuals sub-plot toggle (off by default — user enables via graph settings).
    show_residuals: bool = False


def _parse_color(value: str, fallback: str = "#1f77ff") -> QColor:
    c = QColor(value) if value else QColor(fallback)
    if not c.isValid():
        c = QColor(fallback)
    return c


# ----------------------------------------------------------------------------
# Applying settings to a PlotWidget
# ----------------------------------------------------------------------------

def _range_in_view_coords(from_val: float, to_val: float, is_log: bool):
    """Convert user-entered linear range bounds into pyqtgraph view coordinates.

    pyqtgraph's setXRange/setYRange interprets values in *view* space, which
    is log10(data) when the axis is in log mode. The dialog asks the user for
    linear values (e.g. ``1e-6`` to ``1e0``); in log mode we have to convert
    those before handing them to pyqtgraph, otherwise the range is wildly off.
    Returns ``None`` if the bounds can't be expressed in the current view.
    """
    if not is_log:
        return float(from_val), float(to_val)
    if from_val <= 0 or to_val <= 0:
        return None
    return math.log10(float(from_val)), math.log10(float(to_val))


def apply_graph_settings(plot_widget, raw_curve, x, y, settings: GraphSettings) -> None:
    """Update raw curve data + plot styling from settings.

    ``raw_curve``, ``x`` and ``y`` may be ``None`` (or ``x``/``y`` may be empty).
    In that case the curve data is left untouched and only the plot styling is
    applied — this is what lets the Graph-settings dialog reconfigure the plot
    even before the user has loaded any data.
    """

    curve = settings.curve
    line_color = _parse_color(curve.line_color)
    point_color = _parse_color(curve.point_color)
    line_color.setAlpha(max(0, min(255, int(curve.alpha))))
    point_color.setAlpha(max(0, min(255, int(curve.alpha))))

    has_curve_data = (
        raw_curve is not None
        and x is not None and y is not None
        and len(x) > 0 and len(y) > 0
    )
    if has_curve_data:
        if curve.draw_mode == "Points only":
            raw_curve.setData(
                x, y, pen=None, symbol="o", symbolSize=curve.point_size,
                symbolBrush=point_color, symbolPen=point_color,
            )
        elif curve.draw_mode == "Lines only":
            raw_curve.setData(x, y, pen=pg.mkPen(line_color, width=curve.line_width), symbol=None)
        else:
            raw_curve.setData(
                x, y,
                pen=pg.mkPen(line_color, width=curve.line_width),
                symbol="o", symbolSize=curve.point_size,
                symbolBrush=point_color, symbolPen=point_color,
            )

    plot_item = plot_widget.getPlotItem()

    # Log / linear.
    is_log_x = settings.scale_h.scale_type == "Log10"
    is_log_y = settings.scale_v.scale_type == "Log10"
    plot_item.setLogMode(x=is_log_x, y=is_log_y)

    # Range. setXRange/setYRange take values in view coordinates (log10 when
    # the axis is in log mode); the dialog stores user-entered linear bounds.
    vb = plot_item.getViewBox()
    if settings.scale_h.auto_range:
        vb.enableAutoRange(axis="x")
    else:
        bounds = _range_in_view_coords(settings.scale_h.from_val, settings.scale_h.to_val, is_log_x)
        if bounds is not None:
            vb.setXRange(bounds[0], bounds[1], padding=0)
    if settings.scale_v.auto_range:
        vb.enableAutoRange(axis="y")
    else:
        bounds = _range_in_view_coords(settings.scale_v.from_val, settings.scale_v.to_val, is_log_y)
        if bounds is not None:
            vb.setYRange(bounds[0], bounds[1], padding=0)
    vb.invertX(settings.scale_h.reverse)
    vb.invertY(settings.scale_v.reverse)

    # Plot title.
    plot_item.setTitle(_html_for_title(
        settings.plot_title_text,
        settings.plot_title_color,
        settings.plot_title_size,
        settings.plot_title_font_family,
    ))

    # Axis visibility is driven by the Line and Ticks panel, not the Title panel,
    # so the user can hide a side completely without losing its title text.
    side_visibility = {
        "bottom": settings.line_bottom.show,
        "top": settings.line_top.show,
        "left": settings.line_left.show,
        "right": settings.line_right.show,
    }
    for axis_name, visible in side_visibility.items():
        plot_item.showAxis(axis_name, bool(visible))

    # Per-axis title.
    for axis_name, title in (
        ("bottom", settings.title_bottom),
        ("top", settings.title_top),
        ("left", settings.title_left),
        ("right", settings.title_right),
    ):
        axis = plot_item.getAxis(axis_name)
        if title.show and title.text:
            axis.setLabel(_html_for_title(title.text, title.color, title.size, title.font_family))
        else:
            axis.setLabel("")

    # Tick spacing on bottom/left (the sides pyqtgraph draws by default). In log
    # mode pyqtgraph chooses tick positions in view (= log10) space, so a
    # user-entered linear increment can't map cleanly — fall back to auto.
    for axis_name, scale, is_log in (
        ("bottom", settings.scale_h, is_log_x),
        ("left", settings.scale_v, is_log_y),
    ):
        axis = plot_item.getAxis(axis_name)
        _apply_tick_spacing(axis, scale, is_log)

    # Tick-label formatting on every side; visibility from labels.show.
    for axis_name, labels in (
        ("bottom", settings.ticks_bottom),
        ("top", settings.ticks_top),
        ("left", settings.ticks_left),
        ("right", settings.ticks_right),
    ):
        axis = plot_item.getAxis(axis_name)
        _install_tick_formatter(axis, labels)
        axis.setStyle(showValues=bool(labels.show))

    # Axis line / tick styling on every side.
    for axis_name, lt in (
        ("bottom", settings.line_bottom),
        ("top", settings.line_top),
        ("left", settings.line_left),
        ("right", settings.line_right),
    ):
        axis = plot_item.getAxis(axis_name)
        pen = pg.mkPen(_parse_color(lt.line_color), width=lt.line_thickness)
        axis.setPen(pen)
        axis.setTextPen(pen)
        # tickLength sign controls direction: negative draws ticks outside.
        axis.setStyle(tickLength=-int(lt.major_length) if lt.show else 0)
        # Minor ticks share the major direction; pyqtgraph draws them at
        # max(major_length / 2, minor_length).
        try:
            axis.setStyle(tickTextOffset=2)
        except Exception:
            pass

    # Grids — replace pyqtgraph's built-in axis.setGrid (which only honours
    # opacity) with a custom overlay that respects color, style and thickness
    # for both major and minor grid lines.
    for axis in (plot_item.getAxis("bottom"), plot_item.getAxis("left"),
                 plot_item.getAxis("top"), plot_item.getAxis("right")):
        axis.setGrid(False)
    _apply_custom_grid(plot_item, settings.grid_v, settings.grid_h)


def _html_for_title(text: str, color: str, size: int, font_family: str) -> str:
    parts = [f"color:{_parse_color(color).name()}", f"font-size:{int(size)}pt"]
    if font_family:
        parts.append(f"font-family:'{font_family}'")
    return f'<span style="{";".join(parts)}">{text}</span>'


def _apply_tick_spacing(axis, scale: AxisScale, is_log: bool) -> None:
    """Set major/minor tick spacing on an axis according to ``scale``.

    "By Increment" sets a fixed linear step. "By Count" splits the current
    visible view range into N equal intervals. Log axes fall back to
    pyqtgraph's automatic tick selection because tick positions live in
    view (log10) space.
    """
    if is_log:
        axis.setTickSpacing(major=None, minor=None)
        return

    minor_step = None
    major_step = None
    if scale.major_tick_type == "By Increment":
        if scale.major_tick_value and scale.major_tick_value > 0:
            major_step = float(scale.major_tick_value)
    elif scale.major_tick_type == "By Count":
        count = int(scale.major_tick_value) if scale.major_tick_value else 0
        if count > 0:
            try:
                view_range = axis.linkedView().viewRange()
                vmin, vmax = view_range[0] if axis.orientation in ("bottom", "top") else view_range[1]
                span = float(vmax) - float(vmin)
                if span > 0:
                    major_step = span / count
            except Exception:
                major_step = None

    if major_step and major_step > 0:
        if scale.minor_tick_type == "By Count" and scale.minor_tick_count > 0:
            minor_step = major_step / float(scale.minor_tick_count + 1)
        elif scale.minor_tick_type == "None":
            minor_step = major_step  # equal => pyqtgraph draws no extra minor ticks
        else:
            minor_step = major_step / 5.0
        axis.setTickSpacing(major=major_step, minor=minor_step)
    else:
        axis.setTickSpacing(major=None, minor=None)


def _apply_custom_grid(plot_item, grid_v: GridConfig, grid_h: GridConfig) -> None:
    """Install or update a single ``_PlotGridOverlay`` on ``plot_item``.

    Stored on the ViewBox as ``_custom_grid_overlay`` so repeated calls just
    refresh the existing item instead of stacking up.
    """
    vb = plot_item.getViewBox()
    overlay = getattr(vb, "_custom_grid_overlay", None)
    if overlay is None:
        overlay = _PlotGridOverlay(plot_item)
        vb.addItem(overlay, ignoreBounds=True)
        vb._custom_grid_overlay = overlay
    overlay.set_config(grid_v, grid_h)


def _install_tick_formatter(axis, labels: TickLabels) -> None:
    """Override ``tickStrings`` on ``axis`` to apply divide-by-factor, prefix/suffix."""
    if not hasattr(axis, "_base_tick_strings"):
        axis._base_tick_strings = axis.tickStrings

    use_default_labels = (
        (labels.divide_factor == 1.0 or labels.divide_factor == 0.0)
        and not labels.set_decimal
        and labels.prefix == ""
        and labels.suffix == ""
    )
    if use_default_labels:
        axis.tickStrings = axis._base_tick_strings
        return

    divisor = labels.divide_factor if labels.divide_factor and labels.divide_factor != 0 else 1.0
    decimals = labels.decimal_places if labels.set_decimal else None

    def _tick_strings(values, scale, spacing):
        out = []
        for v in values:
            try:
                val = float(v)
            except Exception:
                out.append("")
                continue
            if getattr(axis, "logMode", False):
                try:
                    val = 10.0 ** val
                except (OverflowError, ValueError):
                    out.append("")
                    continue
            else:
                val *= float(scale)
            val /= divisor
            if decimals is not None:
                out.append(f"{labels.prefix}{val:.{decimals}f}{labels.suffix}")
            else:
                out.append(f"{labels.prefix}{val:g}{labels.suffix}")
        return out

    axis.tickStrings = _tick_strings


class _PlotGridOverlay(QGraphicsObject):
    """Custom grid overlay that respects color/style/thickness for both major
    and minor grids on each axis. Lives inside the plot's ViewBox so it follows
    pan/zoom; tick positions come from the bottom/left axis tick generators.
    """

    def __init__(self, plot_item, parent=None):
        super().__init__(parent)
        self._plot_item = plot_item
        self._grid_v: Optional[GridConfig] = None
        self._grid_h: Optional[GridConfig] = None
        self.setZValue(-100)  # Behind data curves.
        vb = plot_item.getViewBox()
        vb.sigRangeChanged.connect(lambda *_: self.update())
        vb.sigResized.connect(lambda *_: self.update())

    def set_config(self, grid_v: GridConfig, grid_h: GridConfig) -> None:
        self._grid_v = grid_v
        self._grid_h = grid_h
        self.update()

    def boundingRect(self) -> QRectF:
        vb = self._plot_item.getViewBox()
        return vb.viewRect()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        if self._grid_v is None or self._grid_h is None:
            return
        vb = self._plot_item.getViewBox()
        view_rect = vb.viewRect()
        if view_rect.width() <= 0 or view_rect.height() <= 0:
            return

        bottom_axis = self._plot_item.getAxis("bottom")
        left_axis = self._plot_item.getAxis("left")
        x_major, x_minor = _tick_values_in_range(bottom_axis, view_rect.left(), view_rect.right())
        y_major, y_minor = _tick_values_in_range(left_axis, view_rect.top(), view_rect.bottom())

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)

        # Vertical lines (controlled by grid_v) span the full y view.
        if self._grid_v is not None:
            self._paint_lines(painter, x_minor, view_rect, self._grid_v.minor, vertical=True)
            self._paint_lines(painter, x_major, view_rect, self._grid_v.major, vertical=True)
        # Horizontal lines (controlled by grid_h) span the full x view.
        if self._grid_h is not None:
            self._paint_lines(painter, y_minor, view_rect, self._grid_h.minor, vertical=False)
            self._paint_lines(painter, y_major, view_rect, self._grid_h.major, vertical=False)

        painter.restore()

    def _paint_lines(self, painter: QPainter, values, view_rect: QRectF,
                     line: GridLine, *, vertical: bool) -> None:
        if not line.show or not values:
            return
        pen = QPen(_parse_color(line.color))
        pen.setStyle(LINE_STYLES.get(line.style, Qt.SolidLine))
        pen.setWidthF(max(0.0, float(line.thickness)))
        pen.setCosmetic(True)  # Width in pixels regardless of view scale.
        painter.setPen(pen)
        if vertical:
            top, bottom = view_rect.top(), view_rect.bottom()
            for v in values:
                painter.drawLine(QPointF(v, top), QPointF(v, bottom))
        else:
            left, right = view_rect.left(), view_rect.right()
            for v in values:
                painter.drawLine(QPointF(left, v), QPointF(right, v))


def _tick_values_in_range(axis, lo: float, hi: float):
    """Return (major_values, minor_values) in view coordinates for ``axis``.

    Falls back to empty lists if pyqtgraph hasn't computed ticks yet.
    """
    if hi == lo:
        return [], []
    if hi < lo:
        lo, hi = hi, lo
    try:
        size_rect = axis.geometry()
        if axis.orientation in ("bottom", "top"):
            pixel_size = max(1.0, float(size_rect.width()))
        else:
            pixel_size = max(1.0, float(size_rect.height()))
    except Exception:
        pixel_size = max(hi - lo, 1.0)
    try:
        ticks = axis.tickValues(lo, hi, pixel_size)
    except Exception:
        return [], []
    if not ticks:
        return [], []
    major: list[float] = []
    minor: list[float] = []
    # tickValues returns [(spacing0, [vals0]), (spacing1, [vals1]), ...]
    # First entry is the largest spacing (major). Subsequent entries are minor.
    for idx, (_, vals) in enumerate(ticks):
        target = major if idx == 0 else minor
        for v in vals:
            if lo <= v <= hi:
                target.append(float(v))
    return major, minor


# ----------------------------------------------------------------------------
# Colour-picker button
# ----------------------------------------------------------------------------

class _ColorButton(QPushButton):
    def __init__(self, initial: str, parent=None):
        super().__init__("", parent)
        self._color = _parse_color(initial)
        self.setFixedWidth(80)
        self._repaint()
        self.clicked.connect(self._pick)

    def _repaint(self) -> None:
        self.setStyleSheet(f"background-color: {self._color.name()}; color: white;")
        self.setText(self._color.name())

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Choose color")
        if chosen.isValid():
            self._color = chosen
            self._repaint()

    def color_name(self) -> str:
        return self._color.name()


# ----------------------------------------------------------------------------
# Axis-selector side panel (OriginLab-style list on the left)
# ----------------------------------------------------------------------------

class _AxisSelectorTab(QWidget):
    """A tab: vertical icon list on the left (OriginLab-style) + stacked panels."""

    def __init__(self, keys, panel_factory, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.IconMode)
        self._list.setIconSize(QSize(32, 32))
        self._list.setMovement(QListWidget.Static)
        self._list.setFlow(QListWidget.TopToBottom)
        self._list.setSpacing(4)
        self._list.setUniformItemSizes(True)
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setFixedWidth(84)
        self._list.setStyleSheet(
            "QListWidget::item { padding: 6px; }"
            "QListWidget::item:selected { background: #cfe4ff; color: black; }"
        )
        self._stack = QStackedWidget()
        self.panels: dict[str, QWidget] = {}
        for key in keys:
            item = QListWidgetItem(_axis_icon(key), key)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            self._list.addItem(item)
            panel = panel_factory(key)
            self._stack.addWidget(panel)
            self.panels[key] = panel
        self._list.currentRowChanged.connect(self._stack.setCurrentIndex)
        if keys:
            self._list.setCurrentRow(0)
        layout.addWidget(self._list)
        layout.addWidget(self._stack, stretch=1)


# ----------------------------------------------------------------------------
# Per-axis editor panels
# ----------------------------------------------------------------------------

class _ScalePanel(QWidget):
    def __init__(self, scale: AxisScale, parent=None):
        super().__init__(parent)
        self._scale = scale
        form = QFormLayout(self)
        self.auto_cb = QCheckBox("Auto range")
        self.auto_cb.setChecked(scale.auto_range)
        self.from_sb = SciEdit(scale.from_val)
        self.to_sb = SciEdit(scale.to_val)
        self.type_cb = QComboBox()
        self.type_cb.addItems(["Linear", "Log10"])
        self.type_cb.setCurrentText(scale.scale_type)
        self.reverse_cb = QCheckBox("Reverse")
        self.reverse_cb.setChecked(scale.reverse)
        self.major_type_cb = QComboBox()
        self.major_type_cb.addItems(["By Increment", "By Count"])
        self.major_type_cb.setCurrentText(scale.major_tick_type)
        self.major_val_sb = SciEdit(scale.major_tick_value)
        self.minor_type_cb = QComboBox()
        self.minor_type_cb.addItems(["By Count", "None"])
        self.minor_type_cb.setCurrentText(scale.minor_tick_type)
        self.minor_count_sb = QSpinBox()
        self.minor_count_sb.setRange(0, 20)
        self.minor_count_sb.setValue(scale.minor_tick_count)
        form.addRow(self.auto_cb)
        form.addRow("From", self.from_sb)
        form.addRow("To", self.to_sb)
        form.addRow("Type", self.type_cb)
        form.addRow(self.reverse_cb)
        form.addRow(QLabel("<b>Major Ticks</b>"))
        form.addRow("Type", self.major_type_cb)
        self._major_value_label = QLabel("Increment (0 = auto)")
        form.addRow(self._major_value_label, self.major_val_sb)
        form.addRow(QLabel("<b>Minor Ticks</b>"))
        form.addRow("Type", self.minor_type_cb)
        form.addRow("Count between major", self.minor_count_sb)
        self.major_type_cb.currentTextChanged.connect(self._on_major_type_changed)
        self._on_major_type_changed(self.major_type_cb.currentText())

    def _on_major_type_changed(self, kind: str) -> None:
        if kind == "By Count":
            self._major_value_label.setText("Count (0 = auto)")
        else:
            self._major_value_label.setText("Increment (0 = auto)")

    def commit(self) -> None:
        self._scale.auto_range = self.auto_cb.isChecked()
        self._scale.from_val = self.from_sb.value()
        self._scale.to_val = self.to_sb.value()
        self._scale.scale_type = self.type_cb.currentText()
        self._scale.reverse = self.reverse_cb.isChecked()
        self._scale.major_tick_type = self.major_type_cb.currentText()
        self._scale.major_tick_value = self.major_val_sb.value()
        self._scale.minor_tick_type = self.minor_type_cb.currentText()
        self._scale.minor_tick_count = int(self.minor_count_sb.value())


class _TickLabelsPanel(QWidget):
    def __init__(self, labels: TickLabels, parent=None):
        super().__init__(parent)
        self._labels = labels
        form = QFormLayout(self)
        self.show_cb = QCheckBox("Show")
        self.show_cb.setChecked(labels.show)
        self.set_decimal_cb = QCheckBox("Set decimal places")
        self.set_decimal_cb.setChecked(labels.set_decimal)
        self.decimals_sb = QSpinBox()
        self.decimals_sb.setRange(0, 12)
        self.decimals_sb.setValue(labels.decimal_places)
        self.divide_sb = SciEdit(labels.divide_factor)
        self.prefix_edit = QLineEdit(labels.prefix)
        self.suffix_edit = QLineEdit(labels.suffix)
        form.addRow(self.show_cb)
        form.addRow(self.set_decimal_cb)
        form.addRow("Decimal places", self.decimals_sb)
        form.addRow("Divide by factor", self.divide_sb)
        form.addRow("Prefix", self.prefix_edit)
        form.addRow("Suffix", self.suffix_edit)

    def commit(self) -> None:
        self._labels.show = self.show_cb.isChecked()
        self._labels.set_decimal = self.set_decimal_cb.isChecked()
        self._labels.decimal_places = int(self.decimals_sb.value())
        self._labels.divide_factor = self.divide_sb.value()
        self._labels.prefix = self.prefix_edit.text()
        self._labels.suffix = self.suffix_edit.text()


class _TitlePanel(QWidget):
    def __init__(self, title: AxisTitle, parent=None):
        super().__init__(parent)
        self._title = title
        form = QFormLayout(self)
        self.show_cb = QCheckBox("Show")
        self.show_cb.setChecked(title.show)
        self.text_edit = QLineEdit(title.text)
        self.color_btn = _ColorButton(title.color)
        self.size_sb = QSpinBox()
        self.size_sb.setRange(6, 48)
        self.size_sb.setValue(title.size)
        self.font_cb = QFontComboBox()
        if title.font_family:
            self.font_cb.setCurrentFont(QFont(title.font_family))
        form.addRow(self.show_cb)
        form.addRow("Text", self.text_edit)
        form.addRow("Color", self.color_btn)
        form.addRow("Size", self.size_sb)
        form.addRow("Font", self.font_cb)

    def commit(self) -> None:
        self._title.show = self.show_cb.isChecked()
        self._title.text = self.text_edit.text()
        self._title.color = self.color_btn.color_name()
        self._title.size = int(self.size_sb.value())
        self._title.font_family = self.font_cb.currentFont().family()


class _GridLinePanel(QGroupBox):
    def __init__(self, heading: str, grid: GridLine, default_style: str, parent=None):
        super().__init__(heading, parent)
        self._grid = grid
        form = QFormLayout(self)
        self.show_cb = QCheckBox("Show")
        self.show_cb.setChecked(grid.show)
        self.color_btn = _ColorButton(grid.color)
        self.style_cb = QComboBox()
        self.style_cb.addItems(list(LINE_STYLES.keys()))
        self.style_cb.setCurrentText(grid.style or default_style)
        self.thickness_sb = QDoubleSpinBox()
        self.thickness_sb.setRange(0.1, 10.0)
        self.thickness_sb.setDecimals(2)
        self.thickness_sb.setValue(grid.thickness)
        form.addRow(self.show_cb)
        form.addRow("Color", self.color_btn)
        form.addRow("Style", self.style_cb)
        form.addRow("Thickness", self.thickness_sb)

    def commit(self) -> None:
        self._grid.show = self.show_cb.isChecked()
        self._grid.color = self.color_btn.color_name()
        self._grid.style = self.style_cb.currentText()
        self._grid.thickness = float(self.thickness_sb.value())


class _GridPanel(QWidget):
    def __init__(self, grid: GridConfig, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._major = _GridLinePanel("Major grid lines", grid.major, "Solid")
        self._minor = _GridLinePanel("Minor grid lines", grid.minor, "Dash")
        layout.addWidget(self._major)
        layout.addWidget(self._minor)
        layout.addStretch()

    def commit(self) -> None:
        self._major.commit()
        self._minor.commit()


class _LinePanel(QWidget):
    def __init__(self, lt: LineAndTicks, parent=None):
        super().__init__(parent)
        self._lt = lt
        form = QFormLayout(self)
        self.show_cb = QCheckBox("Show line and ticks")
        self.show_cb.setChecked(lt.show)
        self.line_color_btn = _ColorButton(lt.line_color)
        self.line_thickness_sb = QDoubleSpinBox()
        self.line_thickness_sb.setRange(0.1, 10.0)
        self.line_thickness_sb.setValue(lt.line_thickness)
        self.major_len_sb = QSpinBox()
        self.major_len_sb.setRange(0, 40)
        self.major_len_sb.setValue(lt.major_length)
        self.minor_len_sb = QSpinBox()
        self.minor_len_sb.setRange(0, 40)
        self.minor_len_sb.setValue(lt.minor_length)
        form.addRow(self.show_cb)
        form.addRow("Line color", self.line_color_btn)
        form.addRow("Line thickness", self.line_thickness_sb)
        form.addRow(QLabel("<b>Major Ticks</b>"))
        form.addRow("Length", self.major_len_sb)
        form.addRow(QLabel("<b>Minor Ticks</b>"))
        form.addRow("Length", self.minor_len_sb)

    def commit(self) -> None:
        self._lt.show = self.show_cb.isChecked()
        self._lt.line_color = self.line_color_btn.color_name()
        self._lt.line_thickness = float(self.line_thickness_sb.value())
        self._lt.major_length = int(self.major_len_sb.value())
        self._lt.minor_length = int(self.minor_len_sb.value())


class _LineAndTickLabelsPanel(QWidget):
    """Single side-panel that combines line/tick geometry with tick-label format."""

    def __init__(self, lt: LineAndTicks, labels: TickLabels, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._line_panel = _LinePanel(lt)
        self._labels_panel = _TickLabelsPanel(labels)
        self._labels_group = QGroupBox("Tick Labels")
        labels_layout = QVBoxLayout(self._labels_group)
        labels_layout.addWidget(self._labels_panel)
        layout.addWidget(self._line_panel)
        layout.addWidget(self._labels_group)
        layout.addStretch()

    def commit(self) -> None:
        self._line_panel.commit()
        self._labels_panel.commit()


class _CurvePanel(QWidget):
    def __init__(self, settings: GraphSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        curve = settings.curve
        form = QFormLayout(self)
        self.draw_cb = QComboBox()
        self.draw_cb.addItems(["Lines + points", "Points only", "Lines only"])
        self.draw_cb.setCurrentText(curve.draw_mode)
        self.line_width_sb = QDoubleSpinBox()
        self.line_width_sb.setRange(0.1, 10.0)
        self.line_width_sb.setDecimals(2)
        self.line_width_sb.setValue(curve.line_width)
        self.point_size_sb = QSpinBox()
        self.point_size_sb.setRange(1, 30)
        self.point_size_sb.setValue(curve.point_size)
        self.line_color_btn = _ColorButton(curve.line_color)
        self.point_color_btn = _ColorButton(curve.point_color)
        self.alpha_sb = QSpinBox()
        self.alpha_sb.setRange(0, 255)
        self.alpha_sb.setValue(curve.alpha)
        self.residuals_cb = QCheckBox("Show residuals sub-plot after a fit")
        self.residuals_cb.setChecked(settings.show_residuals)
        form.addRow("Draw mode", self.draw_cb)
        form.addRow("Line width", self.line_width_sb)
        form.addRow("Point size", self.point_size_sb)
        form.addRow("Line color", self.line_color_btn)
        form.addRow("Point color", self.point_color_btn)
        form.addRow("Alpha (0-255)", self.alpha_sb)
        form.addRow(self.residuals_cb)

    def commit(self) -> None:
        curve = self._settings.curve
        curve.draw_mode = self.draw_cb.currentText()
        curve.line_width = float(self.line_width_sb.value())
        curve.point_size = int(self.point_size_sb.value())
        curve.line_color = self.line_color_btn.color_name()
        curve.point_color = self.point_color_btn.color_name()
        curve.alpha = int(self.alpha_sb.value())
        self._settings.show_residuals = self.residuals_cb.isChecked()


class _PlotTitlePanel(QWidget):
    def __init__(self, settings: GraphSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        form = QFormLayout(self)
        self.text_edit = QLineEdit(settings.plot_title_text)
        self.size_sb = QSpinBox()
        self.size_sb.setRange(6, 48)
        self.size_sb.setValue(settings.plot_title_size)
        self.color_btn = _ColorButton(settings.plot_title_color)
        self.font_cb = QFontComboBox()
        if settings.plot_title_font_family:
            self.font_cb.setCurrentFont(QFont(settings.plot_title_font_family))
        form.addRow("Plot title", self.text_edit)
        form.addRow("Title size", self.size_sb)
        form.addRow("Title color", self.color_btn)
        form.addRow("Font", self.font_cb)

    def commit(self) -> None:
        self._settings.plot_title_text = self.text_edit.text()
        self._settings.plot_title_size = int(self.size_sb.value())
        self._settings.plot_title_color = self.color_btn.color_name()
        self._settings.plot_title_font_family = self.font_cb.currentFont().family()


# ----------------------------------------------------------------------------
# Tabbed dialog
# ----------------------------------------------------------------------------

class GraphSettingsDialog(QDialog):
    """Tabbed graph-settings dialog mirroring the familiar OriginLab layout."""

    def __init__(
        self,
        settings: GraphSettings,
        parent=None,
        on_apply: Optional[Callable[[GraphSettings], None]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Graph settings")
        self.resize(700, 600)
        self._settings = deepcopy(settings)
        self._on_apply = on_apply
        self._panels: list[Any] = []
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        tabs = QTabWidget()

        # Scale tab: Horizontal / Vertical.
        scale_keys = ("Horizontal", "Vertical")
        scale_map = {"Horizontal": self._settings.scale_h, "Vertical": self._settings.scale_v}

        def _scale_factory(key: str):
            panel = _ScalePanel(scale_map[key])
            self._panels.append(panel)
            return panel

        tabs.addTab(_AxisSelectorTab(scale_keys, _scale_factory), "Scale")

        # Axis lines + tick labels tab: Bottom / Top / Left / Right.
        side_keys = ("Bottom", "Top", "Left", "Right")
        line_map = {
            "Bottom": self._settings.line_bottom, "Top": self._settings.line_top,
            "Left": self._settings.line_left, "Right": self._settings.line_right,
        }
        ticks_map = {
            "Bottom": self._settings.ticks_bottom, "Top": self._settings.ticks_top,
            "Left": self._settings.ticks_left, "Right": self._settings.ticks_right,
        }

        def _axis_line_ticks_factory(key: str):
            panel = _LineAndTickLabelsPanel(line_map[key], ticks_map[key])
            self._panels.append(panel)
            return panel

        tabs.addTab(_AxisSelectorTab(side_keys, _axis_line_ticks_factory), "Line, Ticks, Labels")

        # Title tab.
        titles_map = {
            "Bottom": self._settings.title_bottom, "Top": self._settings.title_top,
            "Left": self._settings.title_left, "Right": self._settings.title_right,
        }

        def _title_factory(key: str):
            panel = _TitlePanel(titles_map[key])
            self._panels.append(panel)
            return panel

        tabs.addTab(_AxisSelectorTab(side_keys, _title_factory), "Title")

        # Grids tab.
        grid_keys = ("Vertical", "Horizontal")
        grid_map = {"Vertical": self._settings.grid_v, "Horizontal": self._settings.grid_h}

        def _grid_factory(key: str):
            panel = _GridPanel(grid_map[key])
            self._panels.append(panel)
            return panel

        tabs.addTab(_AxisSelectorTab(grid_keys, _grid_factory), "Grids")

        # Plot title tab.
        title_panel = _PlotTitlePanel(self._settings)
        self._panels.append(title_panel)
        tabs.addTab(title_panel, "Plot title")

        root.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply_clicked)
        root.addWidget(buttons)

    def _apply_clicked(self) -> None:
        self.result_settings()
        if self._on_apply is not None:
            self._on_apply(self._settings)

    def result_settings(self) -> GraphSettings:
        for p in self._panels:
            p.commit()
        return self._settings


# ----------------------------------------------------------------------------
# Preset save / load
# ----------------------------------------------------------------------------

@dataclass
class FitPreset:
    didt_low: float = 40.0
    didt_high: float = 60.0
    linear_low: float = 5.0
    linear_high: float = 40.0
    power_low: float = 5.0
    power_vfrac: float = 80.0
    max_iter: int = 10
    ic_tol_pct: float = 0.1
    chi_tol: float = 1.0e-9
    use_length: bool = False
    sample_length_cm: float = 1.0
    criterion_value: float = 1.0
    avg_window: int = 1
    x_channel: str = ""
    y_channel: str = ""
    time_channel: str = "Time"
    fit_method: str = "log_log"
    save_to_separate_tdms: bool = False
    # When True, fit metadata is attached as channel properties on the
    # fitted voltage channel itself (no separate FitResults group). When
    # False, a FitResults group is added with one channel per fitted curve.
    save_fit_in_same_group: bool = True
    # When True, loading a TDMS or finishing an acquisition auto-populates
    # the Data Fitting tab with the source curves and any saved fit overlays.
    auto_load_after_acquisition: bool = True
    # When True, every fit attempt (success or failure) is automatically
    # written into the loaded TDMS as fit metadata. When False, fits run
    # silently and the user must press the Save metadata button to persist.
    autosave_fit_metadata: bool = True
    # When True, the Settings dialog may generate a raw file name from tape
    # metadata as SupplierID_TapeID_SampleID and push it to the host app.
    generate_raw_filename_from_metadata: bool = False


def preset_to_dict(preset: FitPreset) -> dict[str, Any]:
    return asdict(preset)


def preset_from_dict(data: dict[str, Any]) -> FitPreset:
    allowed = {f.name for f in fields(FitPreset)}
    clean = {k: v for k, v in data.items() if k in allowed}
    return FitPreset(**clean)


def save_preset_to_file(path: Path, preset: FitPreset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(preset_to_dict(preset), fh, indent=2, sort_keys=True)


def load_preset_from_file(path: Path) -> FitPreset:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return preset_from_dict(data)


# ----------------------------------------------------------------------------
# Export dialog
# ----------------------------------------------------------------------------

class ExportPlotDialog(QDialog):
    """Export the V-I plot to an image/data format."""

    FORMAT_EXTS = {
        "PNG (raster image)": "png",
        "JPEG (raster image)": "jpg",
        "SVG (vector image)": "svg",
        "CSV (data points)": "csv",
    }

    def __init__(self, plot_item, parent=None, default_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("Export plot")
        self._plot_item = plot_item
        self._default_dir = default_dir or Path.home()
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.format_cb = QComboBox()
        self.format_cb.addItems(self.FORMAT_EXTS.keys())
        self.width_sb = QSpinBox()
        self.width_sb.setRange(100, 10000)
        self.width_sb.setValue(1600)
        self.height_sb = QSpinBox()
        self.height_sb.setRange(100, 10000)
        self.height_sb.setValue(1000)
        self.bg_color_btn = _ColorButton("#ffffff")
        form.addRow("Format", self.format_cb)
        form.addRow("Width (px, raster only)", self.width_sb)
        form.addRow("Height (px, raster only)", self.height_sb)
        form.addRow("Background color", self.bg_color_btn)
        root.addLayout(form)

        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Choose file and save…")
        self.save_btn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _save(self) -> None:
        fmt_label = self.format_cb.currentText()
        ext = self.FORMAT_EXTS[fmt_label]
        suggest = str(self._default_dir / f"vi_plot.{ext}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", suggest, f"{fmt_label} (*.{ext})",
        )
        if not path:
            return
        bg = _parse_color(self.bg_color_btn.color_name())
        try:
            if ext == "csv":
                exporter = exporters.CSVExporter(self._plot_item)
                exporter.export(path)
            elif ext == "svg":
                exporter = exporters.SVGExporter(self._plot_item)
                exporter.export(path)
            else:
                exporter = exporters.ImageExporter(self._plot_item)
                exporter.parameters()["width"] = int(self.width_sb.value())
                exporter.parameters()["height"] = int(self.height_sb.value())
                exporter.parameters()["background"] = bg
                exporter.export(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export plot", f"Failed to export: {exc}")
            return
        QMessageBox.information(self, "Export plot", f"Saved to {path}")
        self.accept()

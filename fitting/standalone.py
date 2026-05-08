"""Standalone Superconductor V-I fitting app.

The main window is the Data Workspace — a thin-toolbar QMainWindow that
hosts every loaded TDMS book and every plot as its own free-floating
top-level window. The original V-I fitting tab (Run fit / Ec / IEC /
presets / fit-window editing) opens in its own window from the
workspace's "🔬 Superconductor V-I fitting" toolbar button.

Run with ``python -m fitting`` from the project root.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
import pyqtgraph as pg

from . import tab as _tab
from .extras import FitPreset, load_preset_from_file, save_preset_to_file


# Default preset file co-located with the standalone package. When present it
# is loaded on startup so users get their saved Data Fitting parameters back
# without having to click "Load preset…" each time. When absent, the file is
# created on first launch with the hard-coded ``FitPreset`` defaults.
PRESET_FILENAME = "data_fit_preset.json"


def _app_base_directory() -> Path:
    """Return the directory used to read/write the standalone preset file.

    Mirrors DAQUniversal's ``get_app_base_directory``: PyInstaller frozen
    builds save next to the exe, source/dev runs save in the current working
    directory.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _preferred_preset_path() -> Path:
    """Canonical location for writing the preset file."""
    return _app_base_directory() / PRESET_FILENAME


def _candidate_preset_paths() -> list[Path]:
    """Return preset locations to try in order when reading on startup.

    The preferred path is checked first so it always wins on save. Legacy
    locations (next to the package, project root, fallback cwd) are kept so
    existing user files keep working.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        _preferred_preset_path(),
        here.parent / PRESET_FILENAME,
        here / PRESET_FILENAME,
        Path.cwd() / PRESET_FILENAME,
    ]
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _ensure_default_preset_file() -> Path:
    """Return the preset path to use, creating it with defaults if missing.

    Scans the candidate locations; if any of them exists, that file is used.
    Otherwise, the hard-coded ``FitPreset()`` defaults are written to the
    preferred path so subsequent launches read user-editable JSON.
    """
    for path in _candidate_preset_paths():
        if path.exists():
            return path
    target = _preferred_preset_path()
    try:
        save_preset_to_file(target, FitPreset())
    except OSError:
        traceback.print_exc()
    return target


class WorkspaceMainWindow(QMainWindow):
    """Standalone main window — the Data Workspace.

    The thin-toolbar workspace UI is built directly on this window. The
    original V-I fitting tab is constructed once on startup, lives inside
    a hidden child QMainWindow (``self._fitting_window``), and is shown
    on demand from the workspace's "🔬 Superconductor V-I fitting"
    button.

    The window itself acts as ``app`` for ``fitting.tab`` — its
    ``ui_state.data_fitting_tab`` is the QWidget the tab layout fills,
    and the various ``data_fitting_*`` delegates point at the same free
    functions ``DAQUniversal``'s ``window_feature_facade`` proxies in
    embedded mode.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Data workspace — Superconductor V-I fitting")

        # Required by ``setup_data_fitting_tab_layout``.
        self.ui_state = SimpleNamespace()
        self.runtime_state = SimpleNamespace(output_folder="")
        self._preset_path: Path | None = None

        # Build the V-I fitting tab into a child QWidget; that widget
        # becomes the central widget of a hidden QMainWindow shown on
        # demand from the workspace toolbar.
        fitting_central = QWidget()
        self.ui_state.data_fitting_tab = fitting_central

        # Wire the data_fitting_* method delegates the tab expects.
        self.data_fitting_open_file = lambda *_: _tab.open_file_dialog(self)
        self.data_fitting_refresh_current = lambda *_: _tab.refresh_current_recording(self)
        self.data_fitting_refresh_preview = lambda *_: _tab.refresh_preview(self)
        self.data_fitting_run = lambda *_: _tab.run_fit(self)
        self.data_fitting_load_metadata = lambda *_: _tab.load_metadata_from_tdms(self)
        self.data_fitting_robust_view = lambda *_: _tab.robust_view(self)
        self.data_fitting_reset_view = lambda *_: _tab.reset_view(self)
        self.data_fitting_toggle_zoom = lambda checked=False: _tab.toggle_zoom(self, bool(checked))
        self.data_fitting_region_mode_changed = lambda _btn=None: _tab.region_mode_changed(self)
        self.data_fitting_sync_region_to_inputs = lambda *_: _tab.sync_region_to_inputs(self)

        # Populate the V-I fitting tab. After this returns,
        # ``self.data_fit_*`` widgets exist on this window and the
        # ``fitting_central`` QWidget holds the assembled layout.
        _tab.setup_data_fitting_tab_layout(self)

        # Wrap the V-I fitting tab in a hidden top-level QMainWindow so
        # the user can toggle it from the workspace toolbar.
        self._fitting_window = QMainWindow()
        self._fitting_window.setWindowTitle("Superconductor V-I fitting")
        self._fitting_window.setWindowFlags(
            self._fitting_window.windowFlags() | Qt.Window
        )
        self._fitting_window.setAttribute(Qt.WA_DeleteOnClose, False)
        self._fitting_window.setCentralWidget(fitting_central)
        self._fitting_window.resize(1500, 950)
        # Hidden by default — the user opens it from the workspace.

        # Hook used by the workspace toolbar to show the V-I fitting
        # window. ``_data_fit_open_fitting_window`` is consumed by
        # ``_build_workspace_ui``; setting it on this window also marks
        # this app as standalone so the toolbar adds the launcher button.
        self._data_fit_open_fitting_window = self._open_fitting_window

        # Build the workspace toolbar on this window — make it the
        # primary workspace so subsequent ``Workspace…`` clicks just
        # raise it instead of spawning a duplicate.
        _tab._build_workspace_ui(self, self)

        # Restore the user's saved fit preset so the V-I fitting window
        # opens with the previous session's parameters.
        self._load_default_preset()

        self.resize(1100, 140)

    # --- workspace launcher hook ---------------------------------------
    def _open_fitting_window(self) -> None:
        """Show (and raise) the V-I fitting window."""
        try:
            self._fitting_window.show()
            self._fitting_window.raise_()
            self._fitting_window.activateWindow()
        except RuntimeError:
            traceback.print_exc()

    # --- preset persistence --------------------------------------------
    def _load_default_preset(self) -> None:
        path = _ensure_default_preset_file()
        self._preset_path = path
        try:
            preset = load_preset_from_file(path)
        except Exception:
            traceback.print_exc()
            preset = FitPreset()
        try:
            _tab._apply_preset(self, preset)
        except Exception:
            traceback.print_exc()

    def _save_current_preset(self) -> None:
        if self._preset_path is None:
            return
        try:
            preset = _tab._settings_to_preset(self)
        except Exception:
            traceback.print_exc()
            return
        try:
            save_preset_to_file(self._preset_path, preset)
        except OSError:
            traceback.print_exc()

    def closeEvent(self, event):  # noqa: N802 - Qt override name
        # Closing the workspace also closes the hidden V-I fitting
        # window so the Qt event loop has no remaining top-level windows
        # and exits cleanly.
        self._save_current_preset()
        try:
            self._fitting_window.close()
        except RuntimeError:
            pass
        super().closeEvent(event)


# Backwards-compatible alias — earlier embedders / scripts imported
# ``DataFittingWindow`` directly. The new entry point is the Data
# Workspace, which still hosts the V-I fitting tab on demand, so the
# alias keeps existing imports working without changes.
DataFittingWindow = WorkspaceMainWindow


def main() -> int:
    # Force a light plot theme in standalone mode.
    # In embedded mode, the host app can define its own pyqtgraph theme.
    pg.setConfigOptions(background="w", foreground="k")

    app = QApplication(sys.argv)
    win = WorkspaceMainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

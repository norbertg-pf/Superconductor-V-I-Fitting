"""Minimal QMainWindow that shows just the Data Fitting tab.

Run with ``python -m src.fitting`` from the project root.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
import pyqtgraph as pg

from . import tab as _tab
from .extras import load_preset_from_file


# Default preset file co-located with the standalone package. When present it
# is loaded on startup so users get their saved Data Fitting parameters back
# without having to click "Load preset…" each time.
PRESET_FILENAME = "data_fit_preset.json"


def _candidate_preset_paths() -> list[Path]:
    """Return preset locations to try in order for the standalone app."""
    here = Path(__file__).resolve().parent
    return [
        here / PRESET_FILENAME,
        here.parent / PRESET_FILENAME,
        Path.cwd() / PRESET_FILENAME,
    ]


class DataFittingWindow(QMainWindow):
    """A standalone window hosting only the Data Fitting tab.

    The tab's signal wiring expects ``data_fitting_*`` bound methods on the
    host object. Here we provide them as thin delegates to the free functions
    in :mod:`.tab`, mirroring what ``window_feature_facade`` does inside the
    full DAQUniversal app.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Superconductor V-I fitting")
        self.ui_state = SimpleNamespace()
        self.runtime_state = SimpleNamespace(output_folder="")
        central = QWidget()
        self.ui_state.data_fitting_tab = central
        self.setCentralWidget(central)

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

        _tab.setup_data_fitting_tab_layout(self)
        self._load_default_preset()
        self.resize(1500, 950)

    def _load_default_preset(self) -> None:
        for path in _candidate_preset_paths():
            if not path.exists():
                continue
            try:
                preset = load_preset_from_file(path)
            except Exception:
                traceback.print_exc()
                continue
            try:
                _tab._apply_preset(self, preset)
            except Exception:
                traceback.print_exc()
            return


def main() -> int:
    # Force a light plot theme in standalone mode.
    # In embedded mode, the host app can define its own pyqtgraph theme.
    pg.setConfigOptions(background="w", foreground="k")

    app = QApplication(sys.argv)
    win = DataFittingWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

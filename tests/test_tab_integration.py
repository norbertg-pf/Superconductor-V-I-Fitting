"""End-to-end: the standalone window loads a recording, plots, fits and reloads.

Runs the real tab code offscreen (QT_QPA_PLATFORM=offscreen, set in conftest).
Dialogs are replaced: the file dialog returns the test file, message boxes
return at once.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from fitting import plateau_tab
from fitting import tab as t


def write_vi_recording(path: str, n: int = 20_000, rate: float = 1000.0) -> dict:
    """A V-I ramp: I from 0 to 100 A, V = V0 + R I + Vc (I/Ic)^n, in DAQUniversal layout."""
    time = np.arange(n) / rate
    current = np.linspace(0.0, 100.0, n)
    voltage = 2e-6 + 1e-7 * current + 1e-3 * (current / 80.0) ** 20
    voltage += np.random.default_rng(3).standard_normal(n) * 2e-7
    with TdmsWriter(path) as w:
        for k, (a, b) in enumerate(((0, n // 2), (n // 2, n))):
            objs = []
            if k == 0:
                objs += [RootObject({"Sample_Rate_Hz": rate}), GroupObject("Time"), GroupObject("Dev")]
            objs.append(ChannelObject("Time", "Time", time[a:b], {"Unit": "s"} if k == 0 else None))
            objs.append(ChannelObject("Dev", "Current", current[a:b],
                                      {"Scale_Factor": 1.0, "Offset": 0.0, "Unit": "A"} if k == 0 else None))
            objs.append(ChannelObject("Dev", "Voltage", voltage[a:b],
                                      {"Scale_Factor": 1.0, "Offset": 0.0, "Unit": "V"} if k == 0 else None))
            w.write_segment(objs)
    return {"Time": time, "Current": current, "Voltage": voltage}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    import fitting.standalone as standalone

    monkeypatch.setattr(standalone, "_app_base_directory", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))
    win = standalone.DataFittingWindow()
    yield win
    win.close()


def load_via_dialog(win, path: str, monkeypatch) -> str:
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))
    t.open_file_dialog(win)
    return win.data_fit_path_label.text()


def test_load_plot_fit_and_reload(window, tmp_path, monkeypatch):
    path = str(tmp_path / "vi.tdms")
    data = write_vi_recording(path)
    msg = load_via_dialog(window, path, monkeypatch)
    assert msg == "Loaded vi.tdms with 2 channels.", msg
    c = window.data_fit_controller
    assert c.channel_names == ["Current", "Voltage"]
    assert window.data_fit_x_cb.currentText() == "Current" and window.data_fit_y_cb.currentText() == "Voltage"
    # The averaging factor (1000 S/s -> 100 S/s) was set before the first preview, and the preview
    # was streamed from the file: no full-resolution array is kept, no full-rate pass was made.
    assert window.data_fit_avg_input.text() == "10"
    assert not c.channel_cache.loaded("Current") and not c.channel_cache.loaded("Voltage")
    assert c._time_array is None
    assert c._transformed and all(key[3] == 10 for key in c._transformed), list(c._transformed)
    xs = window.data_fit_raw_curve.xData
    assert xs is not None and xs.size == 2000
    assert np.isclose(float(xs.max()), 100.0, atol=0.1)
    assert window.data_fit_avg_rate_label.text().startswith("Resampled to 100 Hz")

    # Add to plot, fit (autosave writes fit_* properties into the source file), reload.
    t._add_plot_from_current(window)
    assert len(window.data_fit_curves) == 1
    before = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(before)
    t.run_fit(window)
    assert os.path.exists(path)
    results = [r for _label, r in c.last_fit_results if r is not None]
    assert results, window.data_fit_result_text.toPlainText()
    assert results[0].ok and 70.0 < float(results[0].Ic) < 90.0
    with TdmsFile.read(path) as f:
        props = f["Dev"]["Voltage"].properties
        assert any(k.startswith("fit_") for k in props), list(props)
        assert np.array_equal(f["Dev"]["Voltage"][:], data["Voltage"])
    # The save appended a small metadata segment; the recorded bytes were not touched.
    after = os.path.getsize(path)
    assert before < after < before + 4096
    with open(path, "rb") as fh:
        assert fh.read(before) == head

    # The rewritten file is read again on demand; a reload finds the saved fit and can replay it.
    assert np.array_equal(c.get_channel("Current"), data["Current"])
    msg = load_via_dialog(window, path, monkeypatch)
    assert msg.startswith("Loaded vi.tdms with 2 channels"), msg
    c = window.data_fit_controller
    assert c.recording.n_segments == 3  # two data segments and the appended metadata segment
    assert "Voltage" in c.saved_fit_results
    t._replay_saved_fits_into_plot(window)  # what the "Auto-load fitted recording" option runs
    assert any(e.get("is_fit_result") for e in window.data_fit_curves), [e.get("label") for e in window.data_fit_curves]
    assert f"Voltage{t._FITTED_CHANNEL_SUFFIX}" in c.channel_names

    # Data book: Time + channels + the fitted reconstruction, arrays of full length.
    columns = t._build_data_book_columns(window)
    names = [col.name for col in columns]
    assert names[:3] == ["Time", "Current", "Voltage"]
    assert all(len(col.data) == data["Time"].size for col in columns)
    assert np.array_equal(columns[1].data, data["Current"])


def test_plateau_tab_uses_the_same_loader(window, tmp_path):
    path = str(tmp_path / "plateau.tdms")
    n = 45_000  # three plateaus of 15 s at 1 kS/s (the detector wants >= 10 s and >= 50 A)
    time = np.arange(n) / 1000.0
    current = np.where(time < 15.0, 60.0, np.where(time < 30.0, 120.0, 60.0))
    current = current + np.random.default_rng(1).standard_normal(n) * 0.01
    voltage = current * 1e-6
    with TdmsWriter(path) as w:
        w.write_segment([GroupObject("Time"), ChannelObject("Time", "Time", time),
                         GroupObject("Dev"), ChannelObject("Dev", "Current", current),
                         ChannelObject("Dev", "Voltage", voltage)])
    ok, msg = window.plateau_controller.load_recording(path)
    assert ok, msg
    plateau_tab._clear_all_windows(window)
    plateau_tab._sync_ui(window)
    window.plateau_current_cb.setCurrentText("Current")
    found = plateau_tab._auto_detect_windows(window)
    assert found >= 2, found
    assert window.plateau_controller.channel_cache.loaded("Current")
    assert len(window.plateau_windows) == found

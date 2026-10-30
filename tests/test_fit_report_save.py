"""Tests for the fit-metadata save: an appended segment, never a rewrite of the data."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter

from fitting import tab as t
from fitting.tdms_source import TdmsRecording

PROPS = {"Ic_A": 12.5, "n": 20.0, "status": "ok", "ramp_too_fast": "False"}


def write_file(path: str, index: bool = False, segments: int = 3, n: int = 3000) -> dict:
    rng = np.random.default_rng(2)
    time = np.arange(n) / 1000.0
    ai0 = rng.standard_normal(n)
    ai1 = rng.standard_normal(n)
    cuts = np.linspace(0, n, segments + 1).astype(int)
    with TdmsWriter(path, index_file=index) as w:
        for k in range(segments):
            a, b = cuts[k], cuts[k + 1]
            objs = [RootObject({"Sample_Rate_Hz": 1000.0}), GroupObject("Time"), GroupObject("Dev")] if k == 0 else []
            objs += [ChannelObject("Time", "Time", time[a:b], {"Unit": "s"} if k == 0 else None),
                     ChannelObject("Dev", "AI0", ai0[a:b], {"Scale_Factor": 2.0} if k == 0 else None),
                     ChannelObject("Dev", "AI1", ai1[a:b], {"Scale_Factor": 1.0} if k == 0 else None)]
            w.write_segment(objs)
    return {"Time": time, "AI0": ai0, "AI1": ai1}


def read_props(path: str, group: str, name: str) -> dict:
    with TdmsFile.read(path) as f:
        return dict(f[group][name].properties)


def test_same_group_save_appends_a_metadata_segment(tmp_path):
    path = str(tmp_path / "rec.tdms")
    data = write_file(path)
    before = os.path.getsize(path)
    head = Path(path).read_bytes()
    assert t._write_fit_report_same_group(Path(path), {"AI1": PROPS}) == path
    after = os.path.getsize(path)
    assert before < after < before + 2048
    assert Path(path).read_bytes()[:before] == head  # the recorded bytes are untouched
    props = read_props(path, "Dev", "AI1")
    assert props["fit_Ic_A"] == 12.5 and props["fit_status"] == "ok" and props["Scale_Factor"] == 1.0
    assert "fit_Ic_A" not in read_props(path, "Dev", "AI0")
    # A second save overrides the values; the data stays readable by the direct reader.
    assert t._write_fit_report_same_group(Path(path), {"AI1": {**PROPS, "Ic_A": 13.0}}) == path
    assert read_props(path, "Dev", "AI1")["fit_Ic_A"] == 13.0
    rec = TdmsRecording(path)
    assert rec.warnings == [] and rec.n_segments == 5
    ch = rec.find("Dev", "AI1")
    assert ch.fast and np.array_equal(rec.read_one(ch.path), data["AI1"])
    assert np.array_equal(rec.read_one(rec.find("Time", "Time").path), data["Time"])


def test_same_group_save_keeps_the_index_file_in_step(tmp_path):
    path = str(tmp_path / "rec.tdms")
    data = write_file(path, index=True)
    assert os.path.isfile(path + "_index")
    index_before = os.path.getsize(path + "_index")
    assert t._write_fit_report_same_group(Path(path), {"AI0": PROPS}) == path
    assert os.path.getsize(path + "_index") > index_before
    rec = TdmsRecording(path)
    assert rec.warnings == []  # the index still matches the data file
    assert read_props(path, "Dev", "AI0")["fit_n"] == 20.0
    assert np.array_equal(rec.read_one(rec.find("Dev", "AI0").path), data["AI0"])


def test_unmatched_label_goes_to_fit_results_group(tmp_path):
    path = str(tmp_path / "rec.tdms")
    write_file(path)
    assert t._write_fit_report_same_group(Path(path), {"AI1": PROPS, "renamed curve": PROPS}) == path
    with TdmsFile.read(path) as f:
        assert f["FitResults"]["renamed curve"].properties["Ic_A"] == 12.5
        assert f["Dev"]["AI1"].properties["fit_Ic_A"] == 12.5
    c = t.DataFittingController(None)
    assert c.load_recording(path)[0]
    assert set(c.saved_fit_results) == {"AI1", "renamed curve"}
    assert c.channel_names == ["AI0", "AI1"]


def test_torn_file_is_rewritten_instead(tmp_path):
    path = str(tmp_path / "torn.tdms")
    data = write_file(path)
    size = os.path.getsize(path)
    os.truncate(path, size - 100)  # the last segment is cut: no segment may be appended after it
    assert t._write_fit_report_same_group(Path(path), {"AI1": PROPS}) == path
    with TdmsFile.read(path) as f:
        assert f["Dev"]["AI1"].properties["fit_Ic_A"] == 12.5
        assert len(f["Dev"]["AI1"][:]) < data["AI1"].size  # the torn tail is gone, the rest is kept
        assert np.array_equal(f["Dev"]["AI0"][:], data["AI0"][: len(f["Dev"]["AI0"][:])])
    rec = TdmsRecording(path)
    assert rec.warnings == [] and rec.n_segments == 1


def test_fit_results_group_mode_appends_too(tmp_path):
    path = str(tmp_path / "rec.tdms")
    write_file(path)
    before = os.path.getsize(path)
    app = type("App", (), {})()
    app.data_fit_controller = t.DataFittingController(None)
    assert app.data_fit_controller.load_recording(path)[0]
    app.data_fit_autosave_cb = None
    app.data_fit_save_separate_cb = None
    app.data_fit_same_group_cb = type("Cb", (), {"isChecked": lambda self: False})()
    result = type("R", (), {})()
    out = t._write_fit_report_tdms(app, [("AI1", result)], force=True)
    assert out == path
    assert before < os.path.getsize(path) < before + 4096
    with TdmsFile.read(path) as f:
        assert "AI1" in [ch.name for ch in f["FitResults"].channels()]

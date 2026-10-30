"""Tests for DataFittingController.load_recording (fitting.tab) with on-demand channels.

Headless: fitting.tab imports PyQt5 but the controller creates no widgets.
"""

from __future__ import annotations

import os

import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter

from fitting.tab import DataFittingController, _block_average, _transform_blocks
from fitting.tdms_source import LazyChannelCache


def write_daqu_file(path: str, n: int = 2000, segments: int = 4, time_group: str = "Time",
                    extra=(), rate_hz: float = 1000.0) -> dict:
    """A recording like DAQUniversal writes: Time group, device group, FitResults group."""
    rng = np.random.default_rng(7)
    t = np.arange(n) / rate_hz
    ai0 = rng.standard_normal(n)
    ai1 = np.linspace(0, 1, n) ** 12
    cuts = np.linspace(0, n, segments + 1).astype(int)
    with TdmsWriter(path) as w:
        for k in range(segments):
            a, b = cuts[k], cuts[k + 1]
            objs = []
            if k == 0:
                objs += [RootObject({"Sample_Rate_Hz": rate_hz, "DAQ-U_Version": "test"}),
                         GroupObject(time_group), GroupObject("cDAQ1Mod1")]
            objs.append(ChannelObject(time_group, "Time", t[a:b], {"Unit": "s"} if k == 0 else None))
            objs.append(ChannelObject("cDAQ1Mod1", "AI0", ai0[a:b], {
                "Scale_Factor": 2.0, "Offset": 1.0, "Voltage_Tab_Distance": 3.5, "Unit": "A",
            } if k == 0 else None))
            objs.append(ChannelObject("cDAQ1Mod1", "AI1", ai1[a:b], {
                "Scale_Factor": 1.0, "Offset": 0.0, "fit_Ic_A": 12.5, "fit_n": 20.0,
            } if k == 0 else None))
            objs += [o for o in extra] if k == 0 else []
            w.write_segment(objs)
        w.write_segment([GroupObject("FitResults"),
                         ChannelObject("FitResults", "AI1", np.array([np.nan]), {"Ic_A": 13.0, "n": 21.0})])
    return {"Time": t, "AI0": ai0, "AI1": ai1}


def test_load_reads_metadata_and_time_only(tmp_path, monkeypatch):
    path = str(tmp_path / "rec.tdms")
    data = write_daqu_file(path)
    c = DataFittingController(None)
    ok, msg = c.load_recording(path)
    assert ok, msg
    assert msg == "Loaded rec.tdms with 2 channels."
    assert c.tdms_path == path and c.load_warnings == []
    assert c._time_array is None  # the Time channel is read on first access
    assert np.array_equal(c.time_array, data["Time"]) and c.time_array.dtype == np.float64
    assert c._time_array is not None
    assert c.channel_names == ["AI0", "AI1"]
    assert isinstance(c.channel_cache, LazyChannelCache)
    assert len(c.channel_cache) == 2 and "AI0" in c.channel_cache and "Time" not in c.channel_cache
    assert not c.channel_cache.loaded("AI0") and not c.channel_cache.loaded("AI1")
    assert c.channel_metadata["AI0"] == {"scale": 2.0, "offset": 1.0, "voltage_tap_cm": 3.5}
    assert c.channel_metadata["AI1"] == {"scale": 1.0, "offset": 0.0, "voltage_tap_cm": None}
    assert c.get_metadata("nope") == {"scale": 1.0, "offset": 0.0, "voltage_tap_cm": None}
    # Saved fits: the FitResults group wins over the fit_* properties of the channel.
    assert c.saved_fit_results == {"AI1": {"Ic_A": 13.0, "n": 21.0}}
    assert c.recording.properties["Sample_Rate_Hz"] == 1000.0
    # Values load on first access and are then kept.
    reads = []
    orig = c.recording.read

    def counting(paths):
        reads.append(list(paths))
        return orig(paths)

    monkeypatch.setattr(c.recording, "read", counting)
    assert np.array_equal(c.get_channel("AI0"), data["AI0"])
    assert c.channel_cache.loaded("AI0") and reads == [["/'cDAQ1Mod1'/'AI0'"]]
    assert c.get_channel("AI0") is c.channel_cache["AI0"] and len(reads) == 1
    assert c.get_channel("") is None and c.get_channel("missing") is None
    assert np.array_equal(c.channel_cache.get("AI1"), data["AI1"]) and len(reads) == 2


def test_preload_reads_many_channels_in_one_call(tmp_path, monkeypatch):
    path = str(tmp_path / "rec.tdms")
    data = write_daqu_file(path)
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    reads = []
    orig = c.recording.read

    def counting(paths):
        reads.append(list(paths))
        return orig(paths)

    monkeypatch.setattr(c.recording, "read", counting)
    c.preload(("Time", "AI0", "AI1", "", "nope"))
    assert reads == [["/'cDAQ1Mod1'/'AI0'", "/'cDAQ1Mod1'/'AI1'"]]
    assert np.array_equal(c.get_channel("AI1"), data["AI1"]) and len(reads) == 1
    c.preload(("AI0", "AI1"))  # nothing left to read
    assert len(reads) == 1
    # Transform uses the loaded values.
    assert np.allclose(c.apply_transform(c.get_channel("AI0"), 2.0, 1.0), data["AI0"] * 2.0 - 1.0)


def test_compute_fit_inputs_come_from_the_recording(tmp_path):
    path = str(tmp_path / "rec.tdms")
    data = write_daqu_file(path)
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    result, message = c.compute_fit(time_sig="Time", x_sig="AI9", y_sig="AI1", x_scale=1, x_offset=0,
                                    y_scale=1, y_offset=0, t_scale=1, t_offset=0, settings=None)
    assert result is None and "exist in the recording" in message
    # The inputs are streamed from the file: no full-resolution channel is kept in channel_cache.
    assert not c.channel_cache.loaded("AI1") and not c.channel_cache.loaded("AI0") and c._time_array is None
    assert np.array_equal(c.transformed("AI1", 1.0, 0.0, 1), data["AI1"])


def test_rawdata_time_group_and_old_layout(tmp_path):
    path = str(tmp_path / "old.tdms")
    data = write_daqu_file(path, time_group="RawData")
    c = DataFittingController(None)
    ok, msg = c.load_recording(path)
    assert ok, msg
    assert np.array_equal(c.time_array, data["Time"])


def test_missing_time_channel(tmp_path):
    path = str(tmp_path / "notime.tdms")
    with TdmsWriter(path) as w:
        w.write_segment([GroupObject("G"), ChannelObject("G", "AI0", np.arange(10.0))])
    c = DataFittingController(None)
    ok, msg = c.load_recording(path)
    assert not ok and msg == "Recording has no 'Time' channel."
    assert c.tdms_path == "" and c.time_array is None and c.channel_names == [] and c.recording is None


def test_missing_or_broken_file(tmp_path):
    c = DataFittingController(None)
    ok, msg = c.load_recording(str(tmp_path / "nope.tdms"))
    assert not ok and "No recording found" in msg
    ok, msg = c.load_recording("")
    assert not ok
    bad = tmp_path / "bad.tdms"
    bad.write_bytes(b"garbage" * 100)
    ok, msg = c.load_recording(str(bad))
    assert not ok and msg.startswith("Could not read TDMS:")


def test_string_and_timestamp_channels_are_skipped(tmp_path):
    path = str(tmp_path / "mixed.tdms")
    extra = [ChannelObject("cDAQ1Mod1", "Note", np.array(["a", "b"], dtype=object)),
             ChannelObject("cDAQ1Mod1", "Stamp", np.array(["2026-01-01T00:00:00", "2026-01-01T00:00:01"],
                                                          dtype="datetime64[us]"))]
    # Strings and timestamps go to their own segment: the builder of this file is npTDMS.
    write_daqu_file(path, extra=())
    with TdmsWriter(path, mode="a") as w:
        w.write_segment(extra)
    c = DataFittingController(None)
    ok, msg = c.load_recording(path)
    assert ok, msg
    assert c.channel_names == ["AI0", "AI1"]
    assert np.array_equal(c.get_channel("AI0"), write_daqu_file(str(tmp_path / "ref.tdms"))["AI0"])


def test_duplicate_channel_name_first_group_wins(tmp_path):
    path = str(tmp_path / "dup.tdms")
    with TdmsWriter(path) as w:
        w.write_segment([GroupObject("Time"), ChannelObject("Time", "Time", np.arange(5.0)),
                         GroupObject("A"), ChannelObject("A", "V", np.ones(5)),
                         GroupObject("B"), ChannelObject("B", "V", np.zeros(5)),
                         ChannelObject("B", "time", np.zeros(5))])
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    assert c.channel_names == ["V"]
    assert np.array_equal(c.get_channel("V"), np.ones(5))


def test_load_warning_is_in_the_message(tmp_path):
    path = str(tmp_path / "tail.tdms")
    data = write_daqu_file(path)
    with open(path, "ab") as fh:
        fh.write(b"\0" * 1000)
    c = DataFittingController(None)
    ok, msg = c.load_recording(path)
    assert ok and "Warning:" in msg and "ignored" in msg, msg
    assert c.load_warnings and np.array_equal(c.get_channel("AI1"), data["AI1"])


def test_clear_drops_everything(tmp_path):
    path = str(tmp_path / "rec.tdms")
    write_daqu_file(path)
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    c.get_channel("AI0")
    c.clear()
    assert c.tdms_path == "" and c.time_array is None and c.recording is None
    assert c.channel_cache == {} and c.channel_names == [] and c.channel_metadata == {}
    assert c.saved_fit_results == {} and c.load_warnings == []
    assert c.get_channel("AI0") is None
    c.preload(("AI0",))  # no recording: a no-op


def test_cache_edits_and_additions_are_kept(tmp_path):
    """The data book writes into channel_cache; replay adds '- fitted' channels."""
    path = str(tmp_path / "rec.tdms")
    data = write_daqu_file(path)
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    edited = data["AI0"] * 3.0
    c.channel_cache["AI0"] = edited
    assert c.get_channel("AI0") is edited
    c.channel_cache["AI1 - fitted"] = np.zeros(10)
    c.channel_names.append("AI1 - fitted")
    assert list(c.channel_cache) == ["AI0", "AI1", "AI1 - fitted"]
    assert {k: v.shape for k, v in c.channel_cache.items()} == {"AI0": (2000,), "AI1": (2000,), "AI1 - fitted": (10,)}


def test_file_rewritten_after_load_like_the_fit_report(tmp_path):
    """Same-group metadata save rewrites the file (os.replace): later reads use the new file."""
    path = str(tmp_path / "rec.tdms")
    data = write_daqu_file(path)
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    ai0 = c.get_channel("AI0")
    # Rewrite the file in one segment with new channel properties (as _write_fit_report_same_group).
    tmp = str(tmp_path / "rec.tmp")
    with TdmsFile.read(path) as src:
        objs = [RootObject(dict(src.properties))]
        for g in src.groups():
            objs.append(GroupObject(g.name, dict(g.properties)))
            for ch in g.channels():
                props = dict(ch.properties)
                if ch.name == "AI1":
                    props["fit_Ic_A"] = 99.0
                objs.append(ChannelObject(g.name, ch.name, np.asarray(ch[:]), props))
    with TdmsWriter(tmp) as w:
        w.write_segment(objs)
    os.replace(tmp, path)
    assert np.array_equal(c.get_channel("AI1"), data["AI1"])
    assert c.get_channel("AI0") is ai0  # loaded before the rewrite: kept
    assert c.recording.n_segments == 1


# -- transformed (block-wise, kept) ------------------------------------------------------------

def reference_transform(raw, scale, offset, window):
    return _block_average(DataFittingController.apply_transform(raw, scale, offset), window)


def test_transform_blocks_equals_old_pipeline_bit_for_bit():
    rng = np.random.default_rng(11)
    raw = rng.standard_normal(100_003) * 1e-3 + 0.5
    for window in (1, 7, 100, 4096, 100_003, 200_000):
        for scale, offset in ((1.0, 0.0), (2.5, -0.125), (1e-6, 3.0)):
            ref = reference_transform(raw, scale, offset, window)
            for block in (1, 7 * 13, 1 << 16, 10 ** 9):
                block = max(window, block // window * window)
                blocks = (raw[i:i + block] for i in range(0, raw.size, block))
                got = _transform_blocks(blocks, raw.size, scale, offset, window)
                assert got.dtype == np.float64 and got.shape == ref.shape, (window, block)
                assert got.tobytes() == ref.tobytes(), (window, scale, offset, block)


def test_transformed_streams_from_file_and_equals_ram(tmp_path):
    path = str(tmp_path / "rec.tdms")
    data = write_daqu_file(path, n=20_000, segments=7)
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    for name, raw in (("AI0", data["AI0"]), ("Time", data["Time"])):
        for window in (1, 3, 100, 20_000, 30_000):
            ref = reference_transform(raw, 2.0, 1.0, window)
            got = c.transformed(name, 2.0, 1.0, window)
            assert got.tobytes() == ref.tobytes(), (name, window)
            assert not got.flags.writeable
            assert c.transformed(name, 2.0, 1.0, window) is got  # kept, not computed again
    assert not c.channel_cache.loaded("AI0") and c._time_array is None  # streamed: nothing kept at full size
    # The same values once the channel is in RAM (data book), and after an edit the result changes.
    c.channel_cache.load(["AI0"])
    assert c.channel_cache.loaded("AI0")
    kept = c.transformed("AI0", 2.0, 1.0, 100)
    assert kept.tobytes() == reference_transform(data["AI0"], 2.0, 1.0, 100).tobytes()
    c.channel_cache["AI0"] = data["AI0"] * 3.0
    fresh = c.transformed("AI0", 2.0, 1.0, 100)
    assert fresh is not kept and fresh.tobytes() == reference_transform(data["AI0"] * 3.0, 2.0, 1.0, 100).tobytes()
    before = c.transformed("Time", 1.0, 0.0, 10)
    c.time_array = data["Time"] + 1.0
    assert c.transformed("Time", 1.0, 0.0, 10) is not before
    assert np.allclose(c.transformed("Time", 1.0, 0.0, 10), before + 1.0)
    assert c.transformed("", 1.0, 0.0, 1) is None and c.transformed("nope", 1.0, 0.0, 1) is None


def test_transformed_cache_is_bounded(tmp_path):
    path = str(tmp_path / "rec.tdms")
    write_daqu_file(path, n=3000)
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    for k in range(40):
        c.transformed("AI0", 1.0, float(k), 1)
    assert len(c._transformed) == c._TRANSFORMED_MAX
    assert ("AI0", 1.0, 39.0, 1) in c._transformed and ("AI0", 1.0, 0.0, 1) not in c._transformed
    c._TRANSFORMED_BYTES = 10
    c.transformed("AI1", 1.0, 0.0, 1)
    assert len(c._transformed) == c._TRANSFORMED_KEEP  # over the byte budget: only the newest stay
    c.clear()
    assert len(c._transformed) == 0


def test_sample_interval_uses_a_prefix(tmp_path, monkeypatch):
    path = str(tmp_path / "rec.tdms")
    data = write_daqu_file(path, n=20_000, rate_hz=2500.0)
    c = DataFittingController(None)
    assert c.load_recording(path)[0]
    reads = []
    orig = c.recording.read_range

    def spy(p, start, stop):
        reads.append((start, stop))
        return orig(p, start, stop)

    monkeypatch.setattr(c.recording, "read_range", spy)
    assert np.isclose(c.sample_interval("Time"), 1 / 2500.0)
    assert reads == [(0, 4096)] and c._time_array is None
    assert np.isclose(c.sample_interval(""), 1 / 2500.0)
    assert c.sample_interval("AI0") is not None  # any channel can serve as time
    assert c.sample_interval("nope") is None
    assert len(reads) == 3  # one short read per call, never the whole channel
    c.time_array = data["Time"] * 2.0
    assert np.isclose(c.sample_interval("Time"), 2 / 2500.0) and len(reads) == 3  # from RAM now

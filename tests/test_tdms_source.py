"""Tests for fitting.tdms_source (TdmsRecording and LazyChannelCache).

Every value read is compared with npTDMS TdmsFile.read (full read).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter

import tdms_builders as tb
from tdms_builders import Obj, TdmsBuilder, obj_path
from fitting import tdms_fastread as fastread
from fitting.tdms_source import LazyChannelCache, TdmsRecording

pytestmark = pytest.mark.usefixtures("read_path")  # POSIX and Windows read paths


def reference_values(path: str, use_index: bool = True) -> dict:
    """{channel path: float64 values} of every numeric channel, read by npTDMS."""
    out = {}
    tdms = TdmsFile.read(path) if use_index else TdmsFile.read(open(path, "rb"))
    with tdms:
        for g in tdms.groups():
            for c in g.channels():
                a = np.asarray(c[:])
                if a.dtype.kind in "fiub":
                    out[c.path] = a.astype(np.float64)
    return out


def assert_float_equal(got: np.ndarray, ref: np.ndarray, msg: str = "") -> None:
    assert got.dtype == np.float64, msg
    assert got.flags.c_contiguous and got.flags.writeable, msg
    assert got.shape == ref.shape, f"{msg}: shape {got.shape} != {ref.shape}"
    assert np.array_equal(got, ref, equal_nan=True), f"{msg}: values differ"


# -- values ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", tb.ALL_SCENARIOS)
def test_values_match_nptdms(name, scenario):
    path, builder, notes = scenario(name)
    rec = TdmsRecording(path)
    ref = reference_values(path)
    numeric = [ch for ch in rec.channels if ch.numeric]
    assert numeric
    for ch in rec.channels:
        assert ch.numeric == (ch.dtype.kind in "fiub"), (name, ch.path, ch.dtype)
        if not ch.numeric:  # strings, timestamps, complex: listed, never read
            assert not ch.fast
            with pytest.raises(ValueError):
                rec.read_one(ch.path)
            continue
        want_fast = notes["fast"].get(ch.path)
        if want_fast is True and ch.length:
            assert ch.fast, (name, ch.path)
        if want_fast is False:
            assert not ch.fast, (name, ch.path)
        assert_float_equal(rec.read_one(ch.path), ref[ch.path], f"{name} {ch.path}")
    # All channels in one call: same values, one pass per family.
    many = rec.read([ch.path for ch in numeric])
    for ch in numeric:
        assert_float_equal(many[ch.path], ref[ch.path], f"{name} {ch.path} (read many)")


def test_metadata_and_unknown_path(scenario):
    path, builder, notes = scenario("waveform")
    rec = TdmsRecording(path)
    assert rec.properties == {"name": "wave test", "author": "tests"}
    assert set(rec.group_properties) == {"Wave"}
    ch = rec.find("Wave", "sine")
    assert ch is not None and ch.label == "Wave/sine" and ch.length == 1000
    assert ch.properties["unit_string"] == "V" and ch.properties["wf_increment"] == 0.001
    assert rec.find("Wave", "missing") is None
    assert rec.n_segments == 1 and rec.size == os.path.getsize(path)
    with pytest.raises(KeyError):
        rec.read_one(obj_path("Wave", "missing"))
    assert rec.read([]) == {}


def test_empty_channel_reads_empty_float64(write_tdms):
    b = TdmsBuilder()
    b.segment(tb.header_objects(["G"]) + [Obj(obj_path("G", "e"), np.empty(0, np.float64))])
    rec = TdmsRecording(write_tdms(b))
    ch = rec.find("G", "e")
    assert ch.numeric and ch.length == 0 and not ch.fast
    got = rec.read_one(ch.path)
    assert got.shape == (0,) and got.dtype == np.float64


def test_direct_reader_result_is_independent_of_scratch(scenario, rng):
    """Reading the same channel twice gives equal, separate arrays."""
    path, builder, notes = scenario("chunks")
    rec = TdmsRecording(path)
    p = obj_path("Group", "small_a")
    a = rec.read_one(p)
    b = rec.read_one(p)
    assert a is not b and np.array_equal(a, b, equal_nan=True)
    a[:] = 0.0
    assert not np.array_equal(a, rec.read_one(p), equal_nan=True)


# -- fallback (npTDMS) channels ----------------------------------------------------------

def test_scaled_channels_use_nptdms_and_are_scaled(scenario):
    path, builder, notes = scenario("scaled")
    rec = TdmsRecording(path)
    ref = reference_values(path)
    scaled = rec.find("Group", "scaled_i16")
    assert scaled.numeric and not scaled.fast
    got = rec.read_one(scaled.path)
    assert_float_equal(got, ref[scaled.path], scaled.path)
    raw = np.asarray(builder.expected(scaled.path), dtype=np.float64)
    assert np.allclose(got, raw * 0.001 - 3.0)


def test_daqmx_file_one_pass_for_all_slow_channels(scenario, monkeypatch):
    """DAQmx (mixed) layout: the first npTDMS read fills every slow channel."""
    path, builder, notes = scenario("daqmx")
    rec = TdmsRecording(path)
    assert rec._mixed
    slow = [ch for ch in rec.channels if ch.numeric and not ch.fast and ch.length]
    assert len(slow) == 2
    calls = []
    orig = TdmsRecording._chunk_pass

    def counting(self, tdms):
        calls.append(1)
        return orig(self, tdms)

    monkeypatch.setattr(TdmsRecording, "_chunk_pass", counting)
    ref = reference_values(path)
    for ch in slow:
        assert_float_equal(rec.read_one(ch.path), ref[ch.path], ch.path)
    assert len(calls) == 1


# -- the file changes on disk ------------------------------------------------------------

def _write_nptdms_file(path: str, values: dict, seed_props=None) -> None:
    objs = [RootObject(seed_props or {}), GroupObject("G")]
    objs += [ChannelObject("G", n, v) for n, v in values.items()]
    with TdmsWriter(path) as w:
        w.write_segment(objs)


def test_rewritten_file_is_read_again(tmp_path, rng):
    path = str(tmp_path / "r.tdms")
    _write_nptdms_file(path, {"a": rng.standard_normal(500), "b": rng.standard_normal(500)})
    rec = TdmsRecording(path)
    a1 = rec.read_one(obj_path("G", "a"))
    # Rewrite with another layout (two segments) and other values, as the fit report does.
    tmp = str(tmp_path / "r.tmp")
    new_a = rng.standard_normal(700)
    new_b = rng.standard_normal(700)
    with TdmsWriter(tmp) as w:
        w.write_segment([GroupObject("G"), ChannelObject("G", "a", new_a[:300]), ChannelObject("G", "b", new_b[:300])])
        w.write_segment([ChannelObject("G", "a", new_a[300:]), ChannelObject("G", "b", new_b[300:])])
    os.replace(tmp, path)
    got = rec.read([obj_path("G", "a"), obj_path("G", "b")])
    assert rec.n_segments == 2
    assert np.array_equal(got[obj_path("G", "a")], new_a)
    assert np.array_equal(got[obj_path("G", "b")], new_b)
    assert a1.shape == (500,)  # the earlier array is untouched


def test_shrunken_file_gives_clean_error(scenario):
    path, builder, notes = scenario("contiguous")
    rec = TdmsRecording(path)
    p = obj_path("Group", "f64")
    rec.read_one(p)
    os.truncate(path, os.path.getsize(path) - 100)
    # The file changed: it is opened again; the truncated tail gives shorter channels, not garbage.
    ref = reference_values(path)
    assert_float_equal(rec.read_one(p), ref[p], p)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/fd is Linux only")
def test_no_file_handle_stays_open(scenario):
    path, builder, notes = scenario("mixed")
    rec = TdmsRecording(path)
    for ch in rec.channels:
        if ch.numeric:
            rec.read_one(ch.path)
    open_files = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            open_files.add(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            pass
    assert os.path.abspath(path) not in open_files


# -- damaged files --------------------------------------------------------------------------

def test_garbage_after_last_segment_is_ignored(scenario, rng):
    path, builder, notes = scenario("raw_only")
    good = reference_values(path)
    with open(path, "ab") as fh:
        fh.write(b"\0" * 4096 + rng.bytes(300))
    rec = TdmsRecording(path)
    assert any("ignored" in w for w in rec.warnings), rec.warnings
    for p, ref in good.items():
        assert_float_equal(rec.read_one(p), ref, p)


def test_stale_index_file_is_ignored(scenario):
    path, builder, notes = scenario("contiguous", index=3)  # index of the first 3 of 5 segments
    rec = TdmsRecording(path)
    assert any("index" in w for w in rec.warnings), rec.warnings
    ref = reference_values(path, use_index=False)
    for ch in rec.channels:
        if ch.numeric:
            assert_float_equal(rec.read_one(ch.path), ref[ch.path], ch.path)


def test_matching_index_file_is_used(scenario):
    path, builder, notes = scenario("contiguous", index=True)
    rec = TdmsRecording(path)
    assert not any("index" in w for w in rec.warnings), rec.warnings
    ref = reference_values(path)
    for ch in rec.channels:
        if ch.numeric:
            assert_float_equal(rec.read_one(ch.path), ref[ch.path], ch.path)


def test_truncated_file_reads_the_prefix(scenario):
    path, builder, notes = scenario("truncated_mid_chunk")
    rec = TdmsRecording(path)
    for ch in rec.channels:
        if ch.numeric:
            exp = np.asarray(builder.expected(ch.path), dtype=np.float64)[: ch.length]
            assert_float_equal(rec.read_one(ch.path), exp, ch.path)


def test_not_a_tdms_file_raises(tmp_path):
    path = tmp_path / "x.tdms"
    path.write_bytes(b"not a tdms file at all" * 10)
    with pytest.raises(Exception):
        TdmsRecording(str(path))


def test_direct_path_off_falls_back_to_nptdms(scenario, monkeypatch):
    """Without usable npTDMS internals every channel is read with npTDMS."""
    monkeypatch.setattr(fastread, "_nt", None)
    path, builder, notes = scenario("mixed")
    rec = TdmsRecording(path)
    assert not any(ch.fast for ch in rec.channels)
    ref = reference_values(path)
    for ch in rec.channels:
        if ch.numeric:
            assert_float_equal(rec.read_one(ch.path), ref[ch.path], ch.path)


# -- LazyChannelCache ------------------------------------------------------------------------

def _cache(values: dict, calls: list) -> LazyChannelCache:
    def loader(names):
        calls.append(list(names))
        return {n: values[n] for n in names}

    return LazyChannelCache(list(values), loader)


def test_lazy_cache_loads_on_first_access():
    calls = []
    c = _cache({"a": np.arange(3.0), "b": np.arange(4.0)}, calls)
    assert len(c) == 2 and list(c) == ["a", "b"] and "a" in c and "z" not in c
    assert not c.loaded("a") and calls == []
    assert np.array_equal(c["a"], np.arange(3.0)) and c.loaded("a") and calls == [["a"]]
    assert np.array_equal(c.get("a"), np.arange(3.0)) and calls == [["a"]]  # kept, not loaded again
    assert c.get("z") is None and c.get("z", 5) == 5
    with pytest.raises(KeyError):
        c["z"]
    assert len(calls) == 1


def test_lazy_cache_batch_and_bulk_access():
    calls = []
    c = _cache({"a": np.ones(2), "b": np.zeros(2), "d": np.ones(1)}, calls)
    c.load(["a", "b", "a", "zz"])  # unknown names are ignored, duplicates once
    assert calls == [["a", "b"]]
    assert sorted(k for k, _v in c.items()) == ["a", "b", "d"]  # items() loads the rest
    assert calls == [["a", "b"], ["d"]]
    assert len(list(c.values())) == 3 and calls[-1] == ["d"]


def test_lazy_cache_set_and_delete():
    calls = []
    c = _cache({"a": np.ones(2)}, calls)
    c["new"] = np.zeros(3)
    assert "new" in c and list(c) == ["a", "new"] and np.array_equal(c["new"], np.zeros(3))
    c["a"] = np.full(2, 7.0)  # replaced before it was loaded: never read from the file
    assert np.array_equal(c["a"], np.full(2, 7.0)) and calls == []
    del c["a"]
    assert "a" not in c and list(c) == ["new"] and len(c) == 1
    with pytest.raises(KeyError):
        del c["a"]
    assert dict(c) == {"new": c["new"]}


def test_lazy_cache_get_hides_load_errors(caplog):
    def loader(names):
        raise OSError("disk gone")

    c = LazyChannelCache(["a"], loader)
    assert c.get("a") is None
    assert "could not be loaded" in caplog.text
    with pytest.raises(OSError):
        c["a"]
    assert not c.loaded("a") and "a" in c  # still pending: the next access tries again


def test_bulk_mode_when_most_channels_of_a_family_are_read(write_tdms, rng, monkeypatch):
    """4 channels with 64 KiB pieces in 6 equal segments: 2 of 4 -> bulk, 1 of 4 -> not."""
    b = TdmsBuilder()
    paths = [obj_path("G", n) for n in "abcd"]
    n = 8192
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", n, rng)) for p in paths])
    for _ in range(5):
        b.raw_segment({p: tb.random_values("f8", n, rng) for p in paths})
    path = write_tdms(b)
    rec = TdmsRecording(path)
    assert all(ch.fast and not rec._readers[ch.path].fragmented for ch in rec.channels)
    calls = []
    orig = fastread.read_many

    def spy(readers, start, stop, outs=None, bulk=False):
        calls.append((len(readers), bulk))
        return orig(readers, start, stop, outs, bulk)

    monkeypatch.setattr(fastread, "read_many", spy)
    ref = reference_values(path)
    got = rec.read(paths[:2])
    assert calls == [(2, True)]
    for p, a in got.items():
        assert_float_equal(a, ref[p], p)
    calls.clear()
    assert_float_equal(rec.read_one(paths[3]), ref[paths[3]], paths[3])
    assert calls == [(1, False)]

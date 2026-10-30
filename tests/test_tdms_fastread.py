"""Tests for fitting.tdms_fastread (direct pread reader), ported from TDMSViewer.

Every fast read is compared byte by byte with npTDMS (full read, which
uses another npTDMS code path than channel.read_data).
"""

from __future__ import annotations

import os
import threading
from dataclasses import replace

import numpy as np
import pytest
from nptdms import TdmsFile

import tdms_builders as tb
from tdms_builders import Obj, TdmsBuilder, obj_path
from fitting import tdms_fastread as fastread
from fitting.tdms_fastread import FastPathError, build_parts, make_fast_reader

pytestmark = pytest.mark.usefixtures("read_path")  # POSIX and Windows read paths

FAST_SCENARIOS = ("contiguous", "mixed", "extra_bytes", "chunks", "single_channel_chunks", "interleaved",
                  "interleaved_be", "big_endian", "all_dtypes", "raw_only", "growing", "waveform") + tuple(
    "truncated_" + m for m in tb.TRUNCATED_MODES)


class Opened:
    """An npTDMS file opened like TdmsSource does, plus an fd for preads."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self.tdms = TdmsFile.open(self.path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.ref = tb.nptdms_full(self.path)

    def channel(self, path):
        for g in self.tdms.groups():
            for c in g.channels():
                if c.path == path:
                    return c
        raise KeyError(path)

    def ref_data(self, ch):
        return tb.nptdms_channel_data(self.ref, ch.group_name, ch.name)

    def close(self):
        self.tdms.close()
        os.close(self.fd)


@pytest.fixture
def opened():
    items = []

    def _open(path):
        o = Opened(path)
        items.append(o)
        return o

    yield _open
    for o in items:
        o.close()


def check_reader(reader, ref, rng, joints=(), msg=""):
    """Compare reader.read with npTDMS values for many windows (clamped like TdmsSource)."""
    n = len(ref)
    assert reader.length == n
    for a, b in tb.windows(n, rng, count=60, joints=joints):
        got = reader.read(a, b)
        lo, hi = tb.clamp_window(n, a, b)
        tb.assert_same_values(got, ref[lo:hi], f"{msg} [{a}, {b})")
        assert got.dtype.isnative
        assert got.flags.c_contiguous and got.flags.writeable


# -- cross check with npTDMS -------------------------------------------------------

@pytest.mark.parametrize("name", FAST_SCENARIOS)
def test_fast_reader_matches_nptdms(name, scenario, opened, rng):
    path, builder, notes = scenario(name)
    o = opened(path)
    checked = 0
    for p, want_fast in notes["fast"].items():
        ch = o.channel(p)
        if not want_fast:
            continue
        reader = make_fast_reader(o.tdms, ch, o.fd)
        parts, base = build_parts(o.tdms, ch)
        joints = [q.start for q in parts] + [q.start + q.count for q in parts]
        ref = o.ref_data(ch)
        check_reader(reader, ref, rng, joints, f"{name} {p}")
        if notes.get("exact", True):
            exp = np.asarray(builder.expected(p))
            tb.assert_same_values(reader.read(0, reader.length), exp.astype(exp.dtype.newbyteorder("=")), p)
        elif notes.get("prefix"):
            exp = np.asarray(builder.expected(p))[: reader.length]
            tb.assert_same_values(reader.read(0, reader.length), exp.astype(exp.dtype.newbyteorder("=")), p)
        checked += 1
    assert checked > 0


# -- layout details ----------------------------------------------------------------

def test_parts_cover_channel_contiguously(scenario, opened):
    path, builder, notes = scenario("growing")
    o = opened(path)
    size = os.path.getsize(path)
    for p in notes["fast"]:
        ch = o.channel(p)
        parts, base = build_parts(o.tdms, ch)
        pos = 0
        for q in parts:
            assert q.start == pos
            assert q.count > 0
            last = q.offset + ((q.count - 1) // q.npc) * q.chunk_stride + ((q.count - 1) % q.npc) * q.item_stride
            assert 0 < q.offset and last + q.dtype.itemsize <= size
            pos += q.count
        assert pos == len(ch)


def test_parts_of_chunked_segment(scenario, opened):
    path, builder, notes = scenario("chunks")
    o = opened(path)
    # small_a (f8, 50) and small_b (i2, 50) share 7 chunks, then 3 chunks.
    parts, _ = build_parts(o.tdms, o.channel(obj_path("Group", "small_b")))
    assert [(q.count, q.npc, q.chunk_stride, q.item_stride) for q in parts] == [
        (350, 50, 50 * 10, 2), (150, 50, 50 * 10, 2)]
    parts, _ = build_parts(o.tdms, o.channel(obj_path("Group", "small_a")))
    assert parts[1].offset - parts[0].offset > 7 * 500
    parts, _ = build_parts(o.tdms, o.channel(obj_path("Big", "large_b")))
    assert [(q.count, q.npc, q.chunk_stride) for q in parts] == [(20000, 5000, 5000 * 12)]


def test_parts_of_interleaved_segment(scenario, opened):
    path, builder, notes = scenario("interleaved")
    o = opened(path)
    row = 8 + 2 + 1 + 4 + 8 + 1
    parts, _ = build_parts(o.tdms, o.channel(obj_path("Group", "c_u1")))
    assert [(q.count, q.item_stride) for q in parts] == [(999, row), (666, row), (333, row)]
    assert all(q.npc == q.count and q.chunk_stride == 0 for q in parts)


def test_parts_of_truncated_segment(scenario, opened):
    path, builder, notes = scenario("truncated_mid_object")
    o = opened(path)
    # Last segment: 4 chunks of 90 values, cut inside the second object of chunk 4.
    counts = {n: [q.count for q in build_parts(o.tdms, o.channel(obj_path("Group", n)))[0]]
              for n in ("ta", "tb", "tc")}
    assert counts == {"ta": [90, 270, 90], "tb": [90, 270, 11], "tc": [90, 270]}


@pytest.mark.parametrize("block", [8, 24, 100, 4096])
@pytest.mark.parametrize("name", ["chunks", "interleaved", "interleaved_be", "mixed", "big_endian"])
def test_small_read_blocks(name, block, scenario, opened, rng, monkeypatch):
    """Tiny block sizes force the span/strided loops to split reads."""
    monkeypatch.setattr(fastread, "_BLOCK_BYTES", block)
    path, builder, notes = scenario(name)
    o = opened(path)
    for p, want_fast in notes["fast"].items():
        if want_fast:
            ch = o.channel(p)
            reader = make_fast_reader(o.tdms, ch, o.fd)
            check_reader(reader, o.ref_data(ch), rng, msg=f"{p} block={block}")


@pytest.mark.parametrize("piece_min", [1, 1 << 40])
def test_piece_read_threshold(piece_min, scenario, opened, rng, monkeypatch):
    """Per-chunk reads and span reads give the same values."""
    monkeypatch.setattr(fastread, "_PIECE_READ_MIN", piece_min)
    path, builder, notes = scenario("chunks")
    o = opened(path)
    for p in notes["fast"]:
        ch = o.channel(p)
        check_reader(make_fast_reader(o.tdms, ch, o.fd), o.ref_data(ch), rng, msg=p)


def test_read_is_clamped_and_native(scenario, opened):
    path, builder, notes = scenario("big_endian")
    o = opened(path)
    ch = o.channel(obj_path("Group", "be_i4"))
    reader = make_fast_reader(o.tdms, ch, o.fd)
    n = reader.length
    assert reader.dtype == np.dtype("=i4")
    assert reader.read(5, 3).shape == (0,)
    assert reader.read(n, n + 100).shape == (0,)
    assert reader.read(-100, -1).shape == (0,)
    full = reader.read(-10, n + 10)
    assert full.shape == (n,) and full.dtype.isnative
    exp = np.asarray(builder.expected(ch.path)).astype("=i4")
    assert np.array_equal(full, exp)


def test_reads_are_thread_safe(scenario, opened):
    path, builder, notes = scenario("chunks")
    o = opened(path)
    ch = o.channel(obj_path("Big", "large_a"))
    reader = make_fast_reader(o.tdms, ch, o.fd)
    ref = o.ref_data(ch)
    errors = []

    def work(seed):
        r = np.random.default_rng(seed)
        try:
            for _ in range(200):
                a = int(r.integers(0, reader.length))
                b = int(r.integers(a, reader.length + 1))
                if reader.read(a, b).tobytes() != ref[a:b].tobytes():
                    errors.append((a, b))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(repr(exc))

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# -- the fast path must be off for special channels -------------------------------------

@pytest.mark.parametrize("name", ["all_dtypes", "scaled", "daqmx"])
def test_build_parts_rejects_special_channels(name, scenario, opened):
    path, builder, notes = scenario(name)
    o = opened(path)
    for p, want_fast in notes["fast"].items():
        ch = o.channel(p)
        if want_fast:
            build_parts(o.tdms, ch)
            continue
        if len(ch) == 0:
            continue  # TdmsSource never asks for empty channels
        with pytest.raises(FastPathError):
            build_parts(o.tdms, ch)
        with pytest.raises(FastPathError):
            make_fast_reader(o.tdms, ch, o.fd)


def test_scaling_on_file_level_disables_fast_path(write_tdms, opened, rng):
    b = TdmsBuilder()
    b.segment([Obj("/", props=tb.linear_scale_props(3.0, 1.0)), Obj(obj_path("G")),
               Obj(obj_path("G", "x"), tb.random_values("i2", 100, rng))])
    o = opened(write_tdms(b))
    with pytest.raises(FastPathError, match="scaling"):
        build_parts(o.tdms, o.channel(obj_path("G", "x")))


def test_numeric_channel_next_to_string_channel(scenario, opened, rng):
    """Numeric data in a segment with strings: fast reader (if built) must be right."""
    path, builder, notes = scenario("strings_mixed")
    o = opened(path)
    for name in ("num_before", "num_after"):
        ch = o.channel(obj_path("Group", name))
        try:
            reader = make_fast_reader(o.tdms, ch, o.fd)
        except FastPathError:
            continue  # refused: npTDMS is used, see test_tdmsfile
        check_reader(reader, o.ref_data(ch), rng, msg=ch.path)


def test_segment_without_raw_flag_is_refused(write_tdms, opened, rng):
    """Data in a segment without kTocRawData: npTDMS counts it, fast path must not guess."""
    b = TdmsBuilder()
    p = obj_path("G", "x")
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 64, rng))])
    b.segment([Obj(p, index="same")], data={p: tb.random_values("f8", 64, rng)}, raw=False)
    b.raw_segment({p: tb.random_values("f8", 64, rng)})
    o = opened(write_tdms(b))
    ch = o.channel(p)
    try:
        reader = make_fast_reader(o.tdms, ch, o.fd)
    except FastPathError:
        return
    tb.assert_same_values(reader.read(0, reader.length), o.ref_data(ch), p)


# -- verification rejects wrong layouts ---------------------------------------------------

def _corrupt(kind):
    def f(parts):
        out = []
        for q in parts:
            isz = q.dtype.itemsize
            if kind == "offset_item":
                q = replace(q, offset=q.offset + isz)
            elif kind == "offset_byte":
                q = replace(q, offset=q.offset + 1)
            elif kind == "offset_back":
                q = replace(q, offset=q.offset - isz)
            elif kind == "byte_order":
                q = replace(q, dtype=q.dtype.newbyteorder("S"))
            elif kind == "chunk_stride":
                q = replace(q, chunk_stride=q.chunk_stride + isz)
            elif kind == "npc":
                q = replace(q, npc=max(1, q.npc - 1))
            elif kind == "item_stride":
                q = replace(q, item_stride=q.item_stride + isz)
            elif kind == "start":
                q = replace(q, start=q.start + (1 if q.start else 0))
            out.append(q)
        return out
    return f


CORRUPTIONS = ["offset_item", "offset_byte", "offset_back", "byte_order", "chunk_stride", "npc",
               "item_stride", "start"]


@pytest.mark.parametrize("api_check", [True, False], ids=["api", "probes_only"])
@pytest.mark.parametrize("kind", CORRUPTIONS)
@pytest.mark.parametrize("name,channel", [("chunks", "small_a"), ("chunks", "small_b"),
                                          ("interleaved", "c_i2"), ("mixed", "m_f8"), ("raw_only", "a")])
def test_verification_rejects_corrupted_layout(kind, name, channel, api_check, scenario, opened, monkeypatch):
    """probes_only: the npTDMS end-to-end check is off (large files); the probes alone must catch it."""
    if not api_check:
        monkeypatch.setattr(fastread, "_API_CHECK_MAX_BYTES", -1)
    path, builder, notes = scenario(name)
    o = opened(path)
    ch = o.channel(obj_path("Group", channel))
    parts, base = build_parts(o.tdms, ch)
    bad = _corrupt(kind)(parts)
    if bad == parts:
        pytest.skip("corruption does not change this layout")
    # A corruption that happens to give the same values is not a real corruption.
    try:
        probe = fastread.FastChannelReader(o.fd, bad, len(ch), base)
        same = probe.read(0, len(ch)).tobytes() == o.ref_data(ch).tobytes()
    except Exception:
        same = False
    if same:
        pytest.skip("corruption gives identical values")
    monkeypatch.setattr(fastread, "build_parts", lambda f, c: (bad, base))
    with pytest.raises(FastPathError):
        make_fast_reader(o.tdms, ch, o.fd)


def test_verification_checks_every_segment_kind(write_tdms, opened, rng, monkeypatch):
    """A layout error in one rare segment kind must be caught.

    30 contiguous segments and one interleaved segment (number 7). A
    wrong offset only in the interleaved part must fail verification.
    """
    b = TdmsBuilder()
    p, q = obj_path("G", "x"), obj_path("G", "y")
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 2000, rng)),
                                          Obj(q, tb.random_values("f8", 2000, rng))])
    for k in range(1, 30):
        b.raw_segment({p: tb.random_values("f8", 2000, rng), q: tb.random_values("f8", 2000, rng)},
                      interleaved=(k == 7))
    o = opened(write_tdms(b))
    ch = o.channel(p)
    parts, base = build_parts(o.tdms, ch)
    assert sum(q_.item_stride != 8 for q_ in parts) == 1
    bad = [replace(q_, offset=q_.offset + 8) if q_.item_stride != 8 else q_ for q_ in parts]
    probe = fastread.FastChannelReader(o.fd, bad, len(ch), base)
    assert probe.read(0, len(ch)).tobytes() != o.ref_data(ch).tobytes()
    monkeypatch.setattr(fastread, "build_parts", lambda f, c: (bad, base))
    with pytest.raises(FastPathError):
        make_fast_reader(o.tdms, ch, o.fd)


def test_probe_picks_cover_ends_and_every_kind():
    sig = np.array([0, 0, 1, 0, 2, 2, 0, 0, 0, 3])
    picks = fastread._pick(sig)
    assert {0, 9, 2, 4, 5, 8}.issubset(picks)  # ends, first and last of kinds 1, 2, 0
    assert all(0 <= p < sig.size for p in picks)
    many = np.arange(10_000) % 1000  # 1000 kinds: bounded number of picks
    assert len(fastread._pick(many)) <= 2 * fastread._MAX_KINDS + fastread._SPREAD_RUNS + 2


def test_file_shrinking_after_open_gives_clean_error(scenario, opened, tmp_path):
    path, builder, notes = scenario("contiguous")
    o = opened(path)
    ch = o.channel(obj_path("Group", "f64"))
    reader = make_fast_reader(o.tdms, ch, o.fd)
    parts, _ = build_parts(o.tdms, ch)
    os.truncate(path, parts[-1].offset + 8)
    with pytest.raises(FastPathError):
        reader.read(reader.length - 10, reader.length)
    # A new reader on the shrunken file must not pass verification.
    with pytest.raises(Exception):
        make_fast_reader(o.tdms, ch, o.fd)


def test_truncated_file_channel_with_empty_last_chunk(write_tdms, opened, rng):
    """Channel b has no values in the truncated last chunk. Layout is right; the reader must be built."""
    b = TdmsBuilder()
    pa, pb = obj_path("G", "a"), obj_path("G", "b")
    last = b.segment(tb.header_objects(["G"]) + [Obj(pa, tb.random_values("f8", 2400, rng)),
                                                 Obj(pb, tb.random_values("f8", 2400, rng))], chunks=4)
    o = opened(write_tdms(b, cut=len(last.raw) - (3 * 600 * 16 + 8 * 100)))
    ch = o.channel(pb)
    parts, _ = build_parts(o.tdms, ch)
    assert sum(q.count for q in parts) == 1800
    reader = make_fast_reader(o.tdms, ch, o.fd)
    check_reader(reader, o.ref_data(ch), rng, msg=pb)


def test_empty_channel_reader(write_tdms, opened):
    b = TdmsBuilder()
    b.segment(tb.header_objects(["G"]) + [Obj(obj_path("G", "e"), np.empty(0, np.float64))])
    o = opened(write_tdms(b))
    ch = o.channel(obj_path("G", "e"))
    reader = make_fast_reader(o.tdms, ch, o.fd)
    assert reader.length == 0
    assert reader.read(0, 10).shape == (0,)


# -- one-pass layout, merged runs, families ---------------------------------------------------

def _all_channels(o):
    return [c for g in o.tdms.groups() for c in g.channels()]


def test_regular_segments_merge_into_one_part(write_tdms, opened, rng):
    """200 equal raw segments: one part per channel (plus the first segment)."""
    b = TdmsBuilder()
    pa, pb = obj_path("G", "a"), obj_path("G", "b")
    b.segment(tb.header_objects(["G"]) + [Obj(pa, tb.random_values("f8", 100, rng)),
                                          Obj(pb, tb.random_values("i2", 100, rng))])
    for _ in range(200):
        b.raw_segment({pa: tb.random_values("f8", 100, rng), pb: tb.random_values("i2", 100, rng)})
    o = opened(write_tdms(b))
    for p in (pa, pb):
        ch = o.channel(p)
        reader = make_fast_reader(o.tdms, ch, o.fd)
        assert reader.n_parts <= 2
        parts, _ = build_parts(o.tdms, ch)
        assert parts[-1].count >= 200 * 100 - 100 and parts[-1].npc == 100
        check_reader(reader, o.ref_data(ch), rng, [q.start for q in parts], p)


def test_irregular_segments_split_runs(write_tdms, opened, rng):
    """A segment of other size breaks a run; all values stay right."""
    b = TdmsBuilder()
    p = obj_path("G", "x")
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 64, rng))])
    for k in range(30):
        n = 64 if k != 12 else 80
        b.segment([Obj(p, tb.random_values("f8", n, rng))], new_obj_list=False)
    for _ in range(5):
        b.raw_segment({p: tb.random_values("f8", 64, rng)})
    o = opened(write_tdms(b))
    ch = o.channel(p)
    reader = make_fast_reader(o.tdms, ch, o.fd)
    assert 2 < reader.n_parts < 10
    parts, _ = build_parts(o.tdms, ch)
    check_reader(reader, o.ref_data(ch), rng, [q.start for q in parts], p)


@pytest.mark.parametrize("name", FAST_SCENARIOS)
def test_read_many_matches_single_reads(name, scenario, opened, rng):
    path, builder, notes = scenario(name)
    o = opened(path)
    chans = _all_channels(o)
    made = fastread.make_fast_readers(o.tdms, chans, o.fd)
    fam: dict = {}
    for ch, r in zip(chans, made):
        if notes["fast"].get(ch.path):
            assert isinstance(r, fastread.FastChannelReader), (ch.path, r)
        if isinstance(r, fastread.FastChannelReader):
            fam.setdefault(r.family, []).append((ch, r))
    assert fam
    for members in fam.values():
        n = members[0][1].length
        assert all(r.length == n for _c, r in members)
        refs = [o.ref_data(c) for c, _r in members]
        for a, b in tb.windows(n, rng, count=25):
            outs = fastread.read_many([r for _c, r in members], a, b)
            lo, hi = tb.clamp_window(n, a, b)
            for (c, _r), got, ref in zip(members, outs, refs):
                tb.assert_same_values(got, ref[lo:hi], f"{name} {c.path} [{a}, {b})")


@pytest.mark.parametrize("block", [8, 100, 1 << 20])
def test_read_many_small_blocks_and_gaps(block, scenario, opened, rng, monkeypatch):
    monkeypatch.setattr(fastread, "_BLOCK_BYTES", block)
    files = [(name, opened(scenario(name)[0])) for name in ("chunks", "interleaved", "mixed", "growing")]
    for gap in (0, 1 << 40):
        monkeypatch.setattr(fastread, "_GAP_READ_MIN", gap)
        for name, o in files:
            chans = _all_channels(o)
            made = fastread.make_fast_readers(o.tdms, chans, o.fd)
            for ch, r in zip(chans, made):
                if isinstance(r, fastread.FastChannelReader):
                    check_reader(r, o.ref_data(ch), rng, msg=f"{name} {ch.path} gap={gap}")
            readers = [r for r in made if isinstance(r, fastread.FastChannelReader)]
            n = min(r.length for r in readers)
            for r, got in zip(readers, fastread.read_many(readers, 0, n)):
                assert got.tobytes() == r.read(0, n).tobytes()


def test_verification_cost_is_bounded_for_large_chunks(write_tdms, opened, rng, monkeypatch):
    """One chunk of 2 x 1M values: npTDMS decodes only probe values, not 16 MB."""
    b = TdmsBuilder()
    pa, pb = obj_path("G", "a"), obj_path("G", "b")
    b.segment(tb.header_objects(["G"]) + [Obj(pa, tb.random_values("f8", 1 << 20, rng)),
                                          Obj(pb, tb.random_values("f8", 1 << 20, rng))])
    o = opened(write_tdms(b))
    got_bytes = [0]
    orig = fastread._LimitedFile.readinto

    def counting(self, buf):
        n = orig(self, buf)
        got_bytes[0] += n
        return n

    monkeypatch.setattr(fastread._LimitedFile, "readinto", counting)

    def no_api(*_a):
        raise AssertionError("end-to-end read must be skipped for large chunks")

    made = fastread.make_fast_readers(o.tdms, _all_channels(o), o.fd, no_api)
    assert all(isinstance(r, fastread.FastChannelReader) for r in made)
    assert 0 < got_bytes[0] <= 64 * 1024
    assert made[0].family == made[1].family


def test_limited_file_stops_after_limit(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(bytes(range(256)) * 4)
    with open(p, "rb", buffering=0) as fh:
        lf = fastread._LimitedFile(fh, 10)
        lf.seek(5)
        buf = bytearray(64)
        assert lf.readinto(buf) == 10 and bytes(buf[:10]) == bytes(range(5, 15))
        assert lf.readinto(buf) == 0
        lf.seek(0)  # a seek gives a new budget
        assert lf.read() == bytes(range(10))


def _fuzz_builder(rng):
    """Random file: channels come and go, chunk counts, interleaved, big endian, raw-only runs."""
    b = TdmsBuilder()
    names = [f"c{i}" for i in range(int(rng.integers(1, 6)))]
    kinds = ("f8", "f4", "i2", "i4", "i8", "u1", "u4", "?")  # no complex: npTDMS cannot read it interleaved
    dt = {n: kinds[int(rng.integers(0, len(kinds)))] for n in names}
    path = {n: obj_path("G", n) for n in names}
    cur: dict = {}
    for s in range(int(rng.integers(1, 40))):
        chunks = int(rng.integers(1, 4)) if rng.random() < 0.3 else 1
        opts = dict(chunks=chunks, interleaved=bool(rng.random() < 0.15), big_endian=bool(rng.random() < 0.1))
        mode = "new" if s == 0 else ("raw", "raw", "raw", "same", "new", "subset")[int(rng.integers(0, 6))]
        if mode in ("raw", "same") and cur:
            vals = {path[n]: tb.random_values(dt[n], k * chunks, rng) for n, k in cur.items()}
            if mode == "raw":
                b.raw_segment(vals, **opts)
            else:
                b.segment([Obj(p, v, index="same") for p, v in vals.items()], new_obj_list=False, **opts)
            continue
        npc = int(rng.integers(1, 300))
        present = names if mode == "new" else [n for n in names if rng.random() < 0.6] or names[:1]
        cur = {n: npc for n in present}
        objs = [Obj(path[n], tb.random_values(dt[n], npc * chunks, rng)) for n in present]
        b.segment((tb.header_objects(["G"]) if s == 0 else []) + objs, **opts)
    return b


@pytest.mark.parametrize("seed", range(40))
def test_fuzz_layouts_against_nptdms(seed, write_tdms, opened, monkeypatch):
    rng = np.random.default_rng(1000 + seed)
    if seed % 2:
        monkeypatch.setattr(fastread, "_GAP_READ_MIN", int(rng.integers(0, 4096)))
        monkeypatch.setattr(fastread, "_BLOCK_BYTES", int(rng.integers(64, 8192)))
    b = _fuzz_builder(rng)
    cut = int(rng.integers(1, 200)) if seed % 5 == 4 else 0
    o = opened(write_tdms(b, cut=cut))
    chans = [c for c in _all_channels(o) if len(c)]
    made = fastread.make_fast_readers(o.tdms, chans, o.fd)
    fam: dict = {}
    for ch, r in zip(chans, made):
        if not cut:
            assert isinstance(r, fastread.FastChannelReader), (seed, ch.path, r)
        if isinstance(r, fastread.FastChannelReader):
            check_reader(r, o.ref_data(ch), rng, msg=f"seed {seed} {ch.path}")
            fam.setdefault(r.family, []).append((ch, r))
    for members in fam.values():
        n = members[0][1].length
        for a, bb in tb.windows(n, rng, count=10):
            outs = fastread.read_many([r for _c, r in members], a, bb)
            lo, hi = tb.clamp_window(n, a, bb)
            for (c, _r), got in zip(members, outs):
                tb.assert_same_values(got, o.ref_data(c)[lo:hi], f"seed {seed} {c.path} [{a}, {bb})")


# -- bulk mode (file read in order) ----------------------------------------------------------

@pytest.mark.parametrize("block", [64, 4096, 1 << 20])
@pytest.mark.parametrize("name", FAST_SCENARIOS)
def test_read_many_bulk_matches_single_reads(name, block, scenario, opened, rng, monkeypatch):
    monkeypatch.setattr(fastread, "_BLOCK_BYTES", block)
    path, builder, notes = scenario(name)
    o = opened(path)
    chans = _all_channels(o)
    made = fastread.make_fast_readers(o.tdms, chans, o.fd)
    fam: dict = {}
    for ch, r in zip(chans, made):
        if isinstance(r, fastread.FastChannelReader):
            fam.setdefault(r.family, []).append((ch, r))
    assert fam
    for members in fam.values():
        n = members[0][1].length
        refs = [o.ref_data(c) for c, _r in members]
        for a, b in tb.windows(n, rng, count=15):
            outs = fastread.read_many([r for _c, r in members], a, b, bulk=True)
            lo, hi = tb.clamp_window(n, a, b)
            for (c, _r), got, ref in zip(members, outs, refs):
                tb.assert_same_values(got, ref[lo:hi], f"{name} {c.path} bulk [{a}, {b})")


def test_bulk_reads_large_pieces_chunk_by_chunk(write_tdms, opened, rng, monkeypatch):
    """Bulk: one read per chunk span (or several chunks per read), not one read per piece."""
    b = TdmsBuilder()
    pa, pb, pc = (obj_path("G", n) for n in "abc")
    n = 8192  # 64 KiB per piece: above _PIECE_READ_MIN
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", n, rng)) for p in (pa, pb, pc)])
    for _ in range(9):
        b.raw_segment({p: tb.random_values("f8", n, rng) for p in (pa, pb, pc)})
    o = opened(write_tdms(b))
    chans = _all_channels(o)
    readers = fastread.make_fast_readers(o.tdms, chans, o.fd)
    assert all(isinstance(r, fastread.FastChannelReader) and not r.fragmented for r in readers)
    assert readers[0].max_chunk_stride >= 3 * n * 8
    reads = []
    orig = fastread._readinto_at

    def counting(fd, mv, offset):
        reads.append(len(mv))
        return orig(fd, mv, offset)

    monkeypatch.setattr(fastread, "_readinto_at", counting)
    length = readers[0].length
    fastread.read_many(readers, 0, length)
    per_piece = len(reads)
    reads.clear()
    outs = fastread.read_many(readers, 0, length, bulk=True)
    assert len(reads) < per_piece and max(reads) >= 3 * n * 8
    for ch, got in zip(chans, outs):
        tb.assert_same_values(got, o.ref_data(ch), ch.path)

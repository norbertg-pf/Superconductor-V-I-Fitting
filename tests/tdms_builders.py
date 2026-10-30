"""Synthetic TDMS files for tests.

TdmsBuilder writes TDMS segments byte by byte: lead-in, metadata
(object list, raw data index, properties) and raw data. It keeps the
same object-list state as a TDMS reader, so a test can build:

    - several segments, several chunks per segment
    - contiguous, interleaved and big-endian data
    - raw-only segments (no kTocMetaData, metadata of the previous
      segment is reused)
    - channels that start in a later segment
    - all numeric types, strings, timestamps, empty channels
    - DAQmx raw data
    - truncated files and .tdms_index files (also stale ones)

`TdmsBuilder.expected(path)` returns all values written for a channel.
Scenario functions at the end of this module build ready-made files
that other test modules can reuse.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field, replace

import numpy as np

# -- format constants ----------------------------------------------------------

kTocMetaData = 1 << 1
kTocNewObjList = 1 << 2
kTocRawData = 1 << 3
kTocInterleavedData = 1 << 5
kTocBigEndian = 1 << 6
kTocDAQmxRawData = 1 << 7

VERSION = 4713
LEAD_IN_SIZE = 28

# TDMS data type codes.
T_VOID = 0
T_I8, T_I16, T_I32, T_I64 = 1, 2, 3, 4
T_U8, T_U16, T_U32, T_U64 = 5, 6, 7, 8
T_F32, T_F64 = 9, 10
T_STRING = 0x20
T_BOOL = 0x21
T_TIME = 0x44
T_C64 = 0x08000C
T_C128 = 0x10000D
T_DAQMX = 0xFFFFFFFF

RAW_INDEX_NONE = 0xFFFFFFFF
RAW_INDEX_SAME = 0x00000000
DAQMX_FORMAT_CHANGING = 0x1269
DAQMX_DIGITAL_LINE = 0x126A
INCOMPLETE_OFFSET = 0xFFFFFFFFFFFFFFFF

CODE_DTYPE = {
    T_I8: np.dtype("i1"), T_I16: np.dtype("i2"), T_I32: np.dtype("i4"), T_I64: np.dtype("i8"),
    T_U8: np.dtype("u1"), T_U16: np.dtype("u2"), T_U32: np.dtype("u4"), T_U64: np.dtype("u8"),
    T_F32: np.dtype("f4"), T_F64: np.dtype("f8"), T_BOOL: np.dtype("?"),
    T_C64: np.dtype("c8"), T_C128: np.dtype("c16"),
}
DTYPE_CODE = {v: k for k, v in CODE_DTYPE.items()}
CODE_SIZE = {**{k: v.itemsize for k, v in CODE_DTYPE.items()}, T_TIME: 16}

# All plain numeric dtypes (fast path candidates).
NUMERIC_DTYPES = ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?", "c8", "c16")

# DAQmx scaler type codes (differ from TDMS codes).
DAQMX_DTYPE = {0: "u1", 1: "i1", 2: "u2", 3: "i2", 4: "u4", 5: "i4", 6: "u8", 7: "i8", 8: "f4", 9: "f8"}

TDMS_EPOCH = np.datetime64("1904-01-01T00:00:00", "us")


# -- paths and values ----------------------------------------------------------

def obj_path(group: str | None = None, channel: str | None = None) -> str:
    """TDMS object path: "/", "/'g'" or "/'g'/'c'" (quotes doubled)."""
    parts = [p for p in (group, channel) if p is not None]
    return "/" + "/".join("'" + p.replace("'", "''") + "'" for p in parts)


def random_values(dtype, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random values of a numpy dtype, including extreme values."""
    dt = np.dtype(dtype)
    if dt.kind == "b":
        return rng.integers(0, 2, n).astype(bool)
    if dt.kind in "iu":
        info = np.iinfo(dt)
        a = rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)
        if n >= 2:
            a[0], a[-1] = info.min, info.max
        return a
    if dt.kind == "f":
        a = rng.standard_normal(n).astype(dt) * dt.type(1000)
        if n >= 6:
            a[1], a[2], a[3], a[4] = np.nan, np.inf, -np.inf, -0.0
            a[5] = np.finfo(dt).tiny
        return a
    if dt.kind == "c":
        ft = np.float32 if dt.itemsize == 8 else np.float64
        a = np.empty(n, dtype=dt)
        a.real = random_values(ft, n, rng)
        a.imag = random_values(ft, n, rng)[::-1]
        return a
    raise TypeError(f"no random values for {dt}")


def random_times(n: int, rng: np.random.Generator, start="2026-07-28T12:05:36") -> np.ndarray:
    """Increasing datetime64[us] values with random microsecond steps."""
    steps = rng.integers(1, 2_000_000, n)
    return np.datetime64(start, "us") + np.cumsum(steps).astype("timedelta64[us]")


def random_strings(n: int, rng: np.random.Generator) -> list[str]:
    """Strings of varying length, with non-ASCII text and empty strings."""
    words = ["", "a", "Temp", "Grüße", "µV", "日本", "x" * 17, "comma,quote\"", "line\nbreak"]
    return [words[int(i)] + str(k) * int(k % 3 == 0) for k, i in enumerate(rng.integers(0, len(words), n))]


def timestamp_parts(t) -> tuple[int, int]:
    """(seconds since 1904, 2^-64 fractions) of a datetime64.

    The fraction is set to the middle of the microsecond, so that
    npTDMS reads back exactly the same microsecond.
    """
    us = int((np.datetime64(t, "us") - TDMS_EPOCH) // np.timedelta64(1, "us"))
    sec, rem = divmod(us, 10**6)
    return sec, ((2 * rem + 1) << 63) // 10**6


# -- properties ----------------------------------------------------------------

@dataclass(frozen=True)
class Prop:
    """Property value with an explicit TDMS type code."""

    code: int
    value: object


@dataclass(frozen=True)
class RawTime:
    """Timestamp property or value given as raw (seconds, fractions)."""

    seconds: int
    fractions: int


def _prop_code(value) -> tuple[int, object]:
    if isinstance(value, Prop):
        return value.code, value.value
    if isinstance(value, RawTime):
        return T_TIME, value
    if isinstance(value, (bool, np.bool_)):
        return T_BOOL, bool(value)
    if isinstance(value, np.generic) and value.dtype in DTYPE_CODE:
        return DTYPE_CODE[value.dtype], value
    if isinstance(value, int):
        return (T_I32 if -2**31 <= value < 2**31 else T_I64), value
    if isinstance(value, float):
        return T_F64, value
    if isinstance(value, str):
        return T_STRING, value
    if isinstance(value, np.datetime64):
        return T_TIME, value
    raise TypeError(f"no TDMS type for property value {value!r}")


def _string(s: str, e: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack(e + "I", len(b)) + b


def _timestamp(value, e: str) -> bytes:
    if isinstance(value, RawTime):
        sec, frac = value.seconds, value.fractions
    else:
        sec, frac = timestamp_parts(value)
    if e == "<":
        return struct.pack("<Qq", frac, sec)
    return struct.pack(">qQ", sec, frac)


def encode_property(name: str, value, e: str = "<") -> bytes:
    """Bytes of one property: name, type code, value."""
    code, v = _prop_code(value)
    out = _string(name, e) + struct.pack(e + "I", code)
    if code == T_STRING:
        return out + _string(str(v), e)
    if code == T_TIME:
        return out + _timestamp(v, e)
    if code == T_BOOL:
        return out + struct.pack("B", 1 if v else 0)
    return out + np.asarray(v, dtype=CODE_DTYPE[code].newbyteorder(e)).tobytes()


def encode_properties(props: dict, e: str = "<") -> bytes:
    return struct.pack(e + "I", len(props)) + b"".join(encode_property(k, v, e) for k, v in props.items())


def wf_props(increment: float, start_offset: float = 0.0, start_time=None, samples: int | None = None) -> dict:
    """NI waveform properties."""
    p = {"wf_increment": float(increment), "wf_start_offset": float(start_offset)}
    if start_time is not None:
        p["wf_start_time"] = start_time if isinstance(start_time, RawTime) else np.datetime64(start_time, "us")
    if samples is not None:
        p["wf_samples"] = Prop(T_I32, samples)
    return p


def linear_scale_props(slope: float, intercept: float, index: int = 0, n_scales: int | None = None,
                       input_source: int = 0xFFFFFFFF) -> dict:
    """NI linear scaling properties (NI_Scale[i]_Linear_*).

    input_source: 0xFFFFFFFF is the raw data, else the index of another scale.
    """
    return {
        "NI_Number_Of_Scales": Prop(T_U32, index + 1 if n_scales is None else n_scales),
        f"NI_Scale[{index}]_Scale_Type": "Linear",
        f"NI_Scale[{index}]_Linear_Slope": float(slope),
        f"NI_Scale[{index}]_Linear_Y_Intercept": float(intercept),
        f"NI_Scale[{index}]_Linear_Input_Source": Prop(T_U32, input_source),
    }


# -- raw values ----------------------------------------------------------------

def encode_values(code: int, values, e: str = "<") -> bytes:
    """Raw data bytes of values of TDMS type `code`, one chunk."""
    if code == T_STRING:
        enc = [str(s).encode("utf-8") for s in values]
        ends = np.cumsum([len(b) for b in enc], dtype=np.int64)
        return ends.astype(e + "u4").tobytes() + b"".join(enc)
    if code == T_TIME:
        vals = list(values)
        if e == "<":
            arr = np.empty(len(vals), dtype=[("f", "<u8"), ("s", "<i8")])
        else:
            arr = np.empty(len(vals), dtype=[("s", ">i8"), ("f", ">u8")])
        for i, v in enumerate(vals):
            sec, frac = (v.seconds, v.fractions) if isinstance(v, RawTime) else timestamp_parts(v)
            arr[i] = (frac, sec) if e == "<" else (sec, frac)
        return arr.tobytes()
    if code == T_VOID:
        return b""
    dt = CODE_DTYPE[code]
    return np.asarray(values, dtype=dt).astype(dt.newbyteorder(e)).tobytes()


def _infer_code(values) -> int:
    if isinstance(values, np.ndarray):
        if values.dtype.kind == "M":
            return T_TIME
        if values.dtype.kind in "OUS":
            return T_STRING
        return DTYPE_CODE[values.dtype.newbyteorder("=")]
    vals = list(values)
    if vals and isinstance(vals[0], str):
        return T_STRING
    if vals and isinstance(vals[0], (np.datetime64, RawTime)):
        return T_TIME
    return DTYPE_CODE[np.asarray(vals).dtype]


# -- segment objects -----------------------------------------------------------

@dataclass(frozen=True)
class Daqmx:
    """DAQmx raw data index of one channel (one scaler, one buffer)."""

    daqmx_type: int  # DAQmx scaler type code (see DAQMX_DTYPE)
    byte_offset: int  # offset of this scaler inside a raw row
    width: int  # raw row width in bytes (same for all channels)
    data_type: int = T_DAQMX  # channel type in the index
    scale_id: int = 0
    digital_line: bool = False


@dataclass
class Obj:
    """One object entry in the metadata of a segment.

    index: "auto" (new index if values are given, else no data),
    "new", "same" (0x00000000, reuse previous index) or "none"
    (0xFFFFFFFF, no data in this segment).
    values: all values of the object in this segment (all chunks).
    """

    path: str
    values: object = None
    type_code: int | None = None
    index: str = "auto"
    props: dict = field(default_factory=dict)
    npc: int | None = None  # values per chunk (default: len(values) // chunks)
    daqmx: Daqmx | None = None


@dataclass
class _State:
    path: str
    has_data: bool = False
    code: int | None = None
    npc: int = 0
    data_size: int = 0  # bytes per chunk
    daqmx: Daqmx | None = None


@dataclass
class SegmentInfo:
    """Bytes and layout of one written segment."""

    toc: int
    lead_in: bytes
    meta: bytes
    raw: bytes
    position: int
    data_objects: list


class TdmsBuilder:
    """Build a TDMS file segment by segment."""

    def __init__(self, version: int = VERSION):
        self.version = version
        self.segments: list[SegmentInfo] = []
        self._list: list[_State] | None = None
        self._last: dict[str, _State] = {}
        self._values: dict[str, list] = {}
        self.properties: dict[str, dict] = {}
        self._size = 0

    # -- building -------------------------------------------------------------

    def segment(self, objects=(), *, chunks: int = 1, interleaved: bool = False, big_endian: bool = False,
                new_obj_list: bool = True, meta: bool = True, raw: bool | None = None, data: dict | None = None,
                toc: int | None = None, extra: bytes = b"", incomplete: bool = False,
                version: int | None = None, next_offset: int | None = None) -> SegmentInfo:
        """Append one segment.

        objects: Obj entries written in the metadata (ignored if meta is False).
        data: values per path for data objects that have no values in
            `objects` (index "same", or raw-only segments).
        raw: set kTocRawData (default: True if there is data).
        toc: full ToC mask override.
        extra: bytes added after the raw data (counted in the segment size).
        incomplete: write 0xFFFFFFFFFFFFFFFF as next segment offset.
        next_offset: override the next segment offset field.
        """
        e = ">" if big_endian else "<"
        data = dict(data or {})
        if not meta:
            if self._list is None:
                raise ValueError("raw-only segment needs a previous segment")
            lst = self._list
            meta_bytes = b""
        else:
            lst = [] if (new_obj_list or self._list is None) else list(self._list)
            parts = [struct.pack(e + "I", len(objects))]
            for o in objects:
                st, index_bytes = self._object_state(o, lst, chunks, e)
                parts.append(_string(o.path, e) + index_bytes + encode_properties(o.props, e))
                pos = next((i for i, s in enumerate(lst) if s.path == o.path), None)
                if pos is None:
                    lst.append(st)
                else:
                    lst[pos] = st
                self._last[o.path] = st
                if o.values is not None:
                    data[o.path] = o.values
                if o.props:
                    self.properties.setdefault(o.path, {}).update(o.props)
            meta_bytes = b"".join(parts)

        data_objs = [s for s in lst if s.has_data]
        vals = {}
        for s in data_objs:
            if s.path not in data and s.npc == 0:
                data[s.path] = []
            if s.path not in data:
                raise ValueError(f"no values for data object {s.path}")
            v = data[s.path]
            v = list(v) if s.code in (T_STRING,) or (s.code == T_TIME and not isinstance(v, np.ndarray)) else v
            if len(v) != s.npc * chunks:
                raise ValueError(f"{s.path}: {len(v)} values, expected {s.npc} x {chunks}")
            vals[s.path] = v
        raw_bytes = self._raw_data(data_objs, vals, chunks, interleaved, e) + extra

        has_raw = bool(data_objs) if raw is None else raw
        daqmx = any(s.daqmx is not None for s in data_objs)
        if toc is None:
            toc = 0
            if meta:
                toc |= kTocMetaData
                if new_obj_list:
                    toc |= kTocNewObjList
            if has_raw:
                toc |= kTocRawData
            if interleaved:
                toc |= kTocInterleavedData
            if big_endian:
                toc |= kTocBigEndian
            if daqmx:
                toc |= kTocDAQmxRawData
        if next_offset is None:
            next_offset = INCOMPLETE_OFFSET if incomplete else len(meta_bytes) + len(raw_bytes)
        lead_in = (b"TDSm" + struct.pack("<i", toc)
                   + struct.pack(e + "iQQ", self.version if version is None else version,
                                 next_offset, len(meta_bytes)))
        info = SegmentInfo(toc, lead_in, meta_bytes, raw_bytes, self._size, [s.path for s in data_objs])
        self.segments.append(info)
        self._size += len(lead_in) + len(meta_bytes) + len(raw_bytes)
        self._list = lst
        for s in data_objs:
            self._values.setdefault(s.path, []).append(vals[s.path])
        return info

    def raw_segment(self, data: dict, **kw) -> SegmentInfo:
        """Segment without metadata (toc has no kTocMetaData)."""
        return self.segment((), meta=False, data=data, **kw)

    def _object_state(self, o: Obj, lst: list[_State], chunks: int, e: str) -> tuple[_State, bytes]:
        prev = next((s for s in lst if s.path == o.path), None) or self._last.get(o.path)
        kind = o.index
        if kind == "auto":
            kind = "new" if (o.values is not None or o.daqmx is not None or o.type_code is not None) else "none"
        if kind == "none":
            st = replace(prev, has_data=False) if prev else _State(o.path)
            return st, struct.pack(e + "I", RAW_INDEX_NONE)
        if kind == "same":
            if prev is None or prev.code is None:
                raise ValueError(f"{o.path}: no previous index to reuse")
            return replace(prev, has_data=True), struct.pack(e + "I", RAW_INDEX_SAME)
        if kind != "new":
            raise ValueError(f"unknown index kind {kind!r}")
        values = [] if o.values is None else o.values
        n = len(values)
        npc = o.npc if o.npc is not None else n // max(1, chunks)
        if o.daqmx is not None:
            d = o.daqmx
            head = DAQMX_DIGITAL_LINE if d.digital_line else DAQMX_FORMAT_CHANGING
            body = struct.pack(e + "IIQI", d.data_type, 1, npc, 1)
            if d.digital_line:
                body += struct.pack(e + "IIIBI", d.daqmx_type, 0, d.byte_offset * 8, 0, d.scale_id)
            else:
                body += struct.pack(e + "IIIII", d.daqmx_type, 0, d.byte_offset, 0, d.scale_id)
            body += struct.pack(e + "II", 1, d.width)
            st = _State(o.path, True, d.data_type, npc, npc * d.width, d)
            return st, struct.pack(e + "I", head) + body
        code = o.type_code if o.type_code is not None else _infer_code(values)
        if code == T_STRING:
            first = list(values)[:npc]
            size = len(encode_values(T_STRING, first, e))
            body = struct.pack(e + "IIQQ", code, 1, npc, size)
            st = _State(o.path, True, code, npc, size)
            return st, struct.pack(e + "I", len(body) + 4) + body  # length includes this field
        body = struct.pack(e + "IIQ", code, 1, npc)
        size = npc * CODE_SIZE.get(code, 0)
        st = _State(o.path, True, code, npc, size)
        return st, struct.pack(e + "I", len(body) + 4) + body  # length includes this field

    def _raw_data(self, objs: list[_State], vals: dict, chunks: int, interleaved: bool, e: str) -> bytes:
        if not objs:
            return b""
        if any(s.daqmx is not None for s in objs):
            width = objs[0].daqmx.width
            rows = objs[0].npc * chunks
            buf = np.zeros((rows, width), dtype=np.uint8)
            for s in objs:
                d = s.daqmx
                dt = np.dtype(DAQMX_DTYPE[d.daqmx_type]).newbyteorder(e)
                b = np.frombuffer(np.asarray(vals[s.path]).astype(dt).tobytes(), np.uint8)
                buf[:, d.byte_offset:d.byte_offset + dt.itemsize] = b.reshape(rows, dt.itemsize)
            return buf.tobytes()
        if interleaved:
            if len({s.npc for s in objs}) != 1:
                raise ValueError("interleaved objects need the same value count")
            rows = objs[0].npc * chunks
            cols = []
            for s in objs:
                b = np.frombuffer(encode_values(s.code, vals[s.path], e), np.uint8)
                cols.append(b.reshape(rows, CODE_SIZE[s.code]))
            return np.hstack(cols).tobytes() if rows else b""
        out = []
        for c in range(chunks):
            for s in objs:
                piece = vals[s.path][c * s.npc:(c + 1) * s.npc]
                b = encode_values(s.code, piece, e)
                if s.code == T_STRING and len(b) != s.data_size:
                    raise ValueError("string chunks must have the same byte size")
                out.append(b)
        return b"".join(out)

    # -- output ---------------------------------------------------------------

    def data_bytes(self) -> bytes:
        return b"".join(s.lead_in + s.meta + s.raw for s in self.segments)

    def index_bytes(self, n_segments: int | None = None) -> bytes:
        """The .tdms_index content for the first n segments (default: all)."""
        segs = self.segments if n_segments is None else self.segments[:n_segments]
        return b"".join(b"TDSh" + s.lead_in[4:] + s.meta for s in segs)

    def write(self, path, index: bool | int = False, cut: int = 0) -> str:
        """Write the data file (without the last `cut` bytes).

        index: True writes a .tdms_index for all segments; an int writes
        one for the first `index` segments only (a stale index).
        """
        path = os.fspath(path)
        data = self.data_bytes()
        if cut:
            data = data[:-cut]
        with open(path, "wb") as fh:
            fh.write(data)
        if index is not False and index is not None:
            n = None if index is True else int(index)
            with open(path + "_index", "wb") as fh:
                fh.write(self.index_bytes(n))
        return path

    @property
    def size(self) -> int:
        return self._size

    def expected(self, path: str):
        """All values written for a channel path (array, or list for strings)."""
        parts = self._values.get(path, [])
        if not parts:
            return np.empty(0)
        if isinstance(parts[0], list):
            return [v for p in parts for v in p]
        return np.concatenate([np.asarray(p) for p in parts])

    def data_paths(self) -> list[str]:
        return list(self._values)


def header_objects(groups=(), file_props: dict | None = None, group_props: dict | None = None) -> list[Obj]:
    """Root and group objects (no data) for the first segment."""
    objs = [Obj("/", props=dict(file_props or {}))]
    for g in groups:
        objs.append(Obj(obj_path(g), props=dict((group_props or {}).get(g, {}))))
    return objs


def write_nptdms(path, segments, version: int = 4712, index_file: bool = False) -> str:
    """Write a file with nptdms.TdmsWriter.

    segments: list of segments, each a list of nptdms objects
    (RootObject, GroupObject, ChannelObject).
    """
    from nptdms import TdmsWriter

    path = os.fspath(path)
    with TdmsWriter(path, version=version, index_file=index_file) as w:
        for seg in segments:
            w.write_segment(seg)
    return path


# -- comparison helpers ----------------------------------------------------------

def windows(n: int, rng: np.random.Generator, count: int = 40, joints=()) -> list[tuple[int, int]]:
    """Sample windows [a, b) for cross checks: edges, joints and random ones.

    Some windows go outside [0, n) on purpose.
    """
    out = {(0, 0), (0, 1), (0, n), (max(0, n - 1), n), (n, n), (n, n + 5), (-3, 2), (-5, -1),
           (max(0, n - 3), n + 10), (n // 2, n // 2 + 1), (n // 3, 2 * n // 3)}
    for j in joints:
        for d in (1, 2, 7):
            out.add((max(0, j - d), min(n, j + d)))
        out.add((j, j + 1))
        out.add((max(0, j - 1), j))
    for _ in range(count):
        a = int(rng.integers(0, max(1, n)))
        b = int(rng.integers(a, n + 1)) if rng.random() < 0.7 else min(n, a + int(rng.integers(1, 64)))
        out.add((a, b))
    return sorted(out)


def clamp_window(n: int, a: int, b: int) -> tuple[int, int]:
    """TdmsSource.read semantics: clamp [a, b) to [0, n)."""
    a = max(0, a)
    b = min(n, b)
    return a, max(a, b)


def same_values(got, ref) -> bool:
    """True if two value arrays are identical (dtype, shape and bytes)."""
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        return False
    if ref.dtype.kind == "O" or got.dtype.kind == "O":
        return list(got) == list(ref)
    return got.dtype == ref.dtype and got.tobytes() == ref.tobytes()


def assert_same_values(got, ref, msg: str = "") -> None:
    """Assert same shape, dtype and bytes (object arrays: same items)."""
    got_a = np.asarray(got)
    ref_a = np.asarray(ref)
    assert got_a.shape == ref_a.shape, f"{msg}: shape {got_a.shape} != {ref_a.shape}"
    if ref_a.dtype.kind == "O" or got_a.dtype.kind == "O":
        assert list(got_a) == list(ref_a), f"{msg}: values differ"
        return
    assert got_a.dtype == ref_a.dtype, f"{msg}: dtype {got_a.dtype} != {ref_a.dtype}"
    if got_a.tobytes() != ref_a.tobytes():
        diff = [i for i in range(got_a.size) if got_a[i:i + 1].tobytes() != ref_a[i:i + 1].tobytes()][:5]
        raise AssertionError(f"{msg}: bytes differ at {diff} got={got_a[diff]} ref={ref_a[diff]}")


def exact_ns(v):
    """Reference: npTDMS raw timestamp(s) -> datetime64[ns], nearest ns, Python ints.

    Independent of tdmsviewer.tdmsfile.tdms_time_to_ns. Other values unchanged.
    """
    from nptdms.timestamp import TdmsTimestamp

    def one(sec, frac):
        return (int(sec) - 2_082_844_800) * 10**9 + (int(frac) * 10**9 + 2**63) // 2**64

    if isinstance(v, TdmsTimestamp):
        return np.datetime64(one(v.seconds, v.second_fractions), "ns")
    if isinstance(v, np.ndarray) and v.dtype.names and "second_fractions" in v.dtype.names:
        ns = [one(s_, f_) for s_, f_ in zip(v["seconds"].tolist(), v["second_fractions"].tolist())]
        return np.array(ns, dtype=np.int64).astype("datetime64[ns]")
    return v


class _RefChannel:
    """npTDMS channel read with raw timestamps; values/properties with exact ns times."""

    def __init__(self, ch):
        self._ch = ch

    def __getitem__(self, idx):
        return exact_ns(self._ch[idx])

    def __len__(self):
        return len(self._ch)

    @property
    def properties(self):
        return {k: exact_ns(v) for k, v in self._ch.properties.items()}

    @property
    def dtype(self):
        d = self._ch.dtype
        return np.dtype("datetime64[ns]") if d.kind == "M" else d

    def __getattr__(self, name):
        return getattr(self._ch, name)


class _RefGroup:
    def __init__(self, g):
        self._g = g

    def __getitem__(self, name):
        return _RefChannel(self._g[name])

    def channels(self):
        return [_RefChannel(c) for c in self._g.channels()]

    @property
    def properties(self):
        return {k: exact_ns(v) for k, v in self._g.properties.items()}

    def __getattr__(self, name):
        return getattr(self._g, name)


class RefFile:
    """TdmsFile wrapper: timestamps as exact datetime64[ns] (like the viewer shows)."""

    def __init__(self, f):
        self._f = f

    def __getitem__(self, name):
        return _RefGroup(self._f[name])

    def groups(self):
        return [_RefGroup(g) for g in self._f.groups()]

    @property
    def properties(self):
        return {k: exact_ns(v) for k, v in self._f.properties.items()}

    def __getattr__(self, name):
        return getattr(self._f, name)


def nptdms_full(path, use_index: bool = True):
    """TdmsFile.read of a file (all data in memory), timestamps exact to 1 ns.

    use_index=False reads the data file alone (ignores .tdms_index).
    """
    from nptdms import TdmsFile

    if use_index:
        return RefFile(TdmsFile.read(os.fspath(path), raw_timestamps=True))
    with open(os.fspath(path), "rb") as fh:
        return RefFile(TdmsFile.read(fh, raw_timestamps=True))


def nptdms_channel_data(tdms, group: str, name: str):
    """Channel values as numpy array from a loaded TdmsFile."""
    return np.asarray(tdms[group][name][:])


# -- scenarios -------------------------------------------------------------------
# Each scenario returns (builder, notes).
#   notes["fast"]: channel path -> expected fast path flag (missing paths are not checked)
#   notes["exact"]: False if builder.expected() does not hold for the file
#   notes["prefix"]: True if npTDMS values are a prefix of builder.expected() (truncated files)

G = "Group"


def _p(name: str, group: str = G) -> str:
    return obj_path(group, name)


def scenario_contiguous(rng, n_segments: int = 5, npc: int = 700) -> tuple[TdmsBuilder, dict]:
    """Several contiguous segments, float64/int32/uint8 channels, metadata repeated."""
    b = TdmsBuilder()
    names = {"f64": "f8", "i32": "i4", "u8": "u1"}
    for k in range(n_segments):
        objs = header_objects([G], {"title": "contiguous"}) if k == 0 else []
        for name, dt in names.items():
            objs.append(Obj(_p(name), random_values(dt, npc + 13 * k, rng),
                            props={"unit_string": "V"} if k == 0 else {}))
        b.segment(objs)
    return b, {"fast": {_p(n): True for n in names}}


def scenario_chunks(rng) -> tuple[TdmsBuilder, dict]:
    """Segments with several chunks: small pieces and large (>= 32 KiB) pieces."""
    b = TdmsBuilder()
    small, large = 50, 5000
    objs = header_objects([G]) + [
        Obj(_p("small_a"), random_values("f8", small * 7, rng)),
        Obj(_p("small_b"), random_values("i2", small * 7, rng)),
    ]
    b.segment(objs, chunks=7)
    b.segment([Obj(_p("small_a"), index="same"), Obj(_p("small_b"), index="same")], chunks=3,
              data={_p("small_a"): random_values("f8", small * 3, rng),
                    _p("small_b"): random_values("i2", small * 3, rng)})
    g2 = "Big"
    objs = header_objects([g2]) + [
        Obj(obj_path(g2, "large_a"), random_values("f8", large * 4, rng)),
        Obj(obj_path(g2, "large_b"), random_values("f4", large * 4, rng)),
    ]
    b.segment(objs, chunks=4)
    fast = {p: True for p in (_p("small_a"), _p("small_b"), obj_path(g2, "large_a"), obj_path(g2, "large_b"))}
    return b, {"fast": fast}


def scenario_single_channel_chunks(rng) -> tuple[TdmsBuilder, dict]:
    """One channel, many chunks (chunk stride == piece size)."""
    b = TdmsBuilder()
    b.segment(header_objects([G]) + [Obj(_p("only"), random_values("f8", 128 * 9, rng))], chunks=9)
    b.raw_segment({_p("only"): random_values("f8", 128 * 5, rng)}, chunks=5)
    return b, {"fast": {_p("only"): True}}


def scenario_interleaved(rng, big_endian: bool = False) -> tuple[TdmsBuilder, dict]:
    """Interleaved segments with mixed item sizes."""
    b = TdmsBuilder()
    # No complex types: npTDMS cannot read interleaved complex data.
    dts = {"c_f8": "f8", "c_i2": "i2", "c_u1": "u1", "c_f4": "f4", "c_i8": "i8", "c_bool": "?"}
    n = 333
    objs = header_objects([G]) + [Obj(_p(k), random_values(v, n * 3, rng)) for k, v in dts.items()]
    b.segment(objs, chunks=3, interleaved=True, big_endian=big_endian)
    b.raw_segment({_p(k): random_values(v, n * 2, rng) for k, v in dts.items()}, chunks=2,
                  interleaved=True, big_endian=big_endian)
    b.segment([Obj(_p(k), index="same") for k in dts], new_obj_list=False, interleaved=True,
              big_endian=big_endian, data={_p(k): random_values(v, n, rng) for k, v in dts.items()})
    return b, {"fast": {_p(k): True for k in dts}}


def scenario_big_endian(rng) -> tuple[TdmsBuilder, dict]:
    """Big-endian contiguous segments with all numeric types."""
    b = TdmsBuilder()
    n = 257
    objs = header_objects([G]) + [
        Obj(_p("be_" + dt), random_values(dt, n * 2, rng), props={"dtype": dt}) for dt in NUMERIC_DTYPES
    ]
    b.segment(objs, chunks=2, big_endian=True)
    b.raw_segment({_p("be_" + dt): random_values(dt, n, rng) for dt in NUMERIC_DTYPES}, big_endian=True)
    return b, {"fast": {_p("be_" + dt): True for dt in NUMERIC_DTYPES}}


def scenario_all_dtypes(rng) -> tuple[TdmsBuilder, dict]:
    """All numeric types, strings, timestamps and empty channels.

    Strings are in their own segments: fastread refuses segments with
    unsized (string) data, see scenario_strings_mixed.
    No Void channel here: npTDMS TdmsFile.read fails on Void raw data.
    """
    b = TdmsBuilder()
    n = 300
    objs = header_objects([G, "Other"])
    for dt in NUMERIC_DTYPES:
        objs.append(Obj(_p("t_" + dt), random_values(dt, n, rng)))
    objs.append(Obj(_p("time"), random_times(n, rng)))
    objs.append(Obj(_p("empty_i32"), np.empty(0, np.int32), type_code=T_I32))
    objs.append(Obj(_p("no_data"), props={"note": "never has data"}))
    b.segment(objs)
    b.segment([Obj(_p("text"), random_strings(n, rng))])
    objs = [Obj(_p("t_" + dt), index="same") for dt in NUMERIC_DTYPES]
    objs.append(Obj(_p("time"), index="same"))
    data = {_p("t_" + dt): random_values(dt, n, rng) for dt in NUMERIC_DTYPES}
    data[_p("time")] = random_times(n, rng, start="2026-07-29T00:00:00")
    b.segment(objs, data=data)
    b.segment([Obj(_p("text"), random_strings(n // 2, rng))])
    fast = {_p("t_" + dt): True for dt in NUMERIC_DTYPES}
    fast.update({_p("text"): False, _p("time"): False, _p("empty_i32"): False, _p("no_data"): False})
    return b, {"fast": fast}


def scenario_strings_mixed(rng) -> tuple[TdmsBuilder, dict]:
    """Numeric and string channels in the same segments (fast flag not checked)."""
    b = TdmsBuilder()
    n = 200
    objs = header_objects([G]) + [Obj(_p("num_before"), random_values("f8", n, rng)),
                                  Obj(_p("words"), random_strings(n, rng)),
                                  Obj(_p("num_after"), random_values("i2", n, rng))]
    b.segment(objs)
    b.segment([Obj(_p("num_before"), random_values("f8", n, rng)), Obj(_p("words"), random_strings(n, rng)),
               Obj(_p("num_after"), random_values("i2", n, rng))])
    return b, {"fast": {_p("words"): False}}


def scenario_raw_only(rng) -> tuple[TdmsBuilder, dict]:
    """Metadata once, then raw-only segments (no kTocMetaData)."""
    b = TdmsBuilder()
    n = 400
    b.segment(header_objects([G]) + [Obj(_p("a"), random_values("f8", n, rng)),
                                     Obj(_p("b"), random_values("u4", n, rng))])
    for k in range(4):
        b.raw_segment({_p("a"): random_values("f8", n, rng), _p("b"): random_values("u4", n, rng)})
    return b, {"fast": {_p("a"): True, _p("b"): True}}


def scenario_growing(rng) -> tuple[TdmsBuilder, dict]:
    """Channels added in later segments, channels pausing (index none/same)."""
    b = TdmsBuilder()
    b.segment(header_objects([G]) + [Obj(_p("first"), random_values("f8", 100, rng))])
    b.segment([Obj(_p("second"), random_values("i4", 100, rng))], new_obj_list=False,
              data={_p("first"): random_values("f8", 100, rng)})
    b.segment([Obj(_p("first"), index="none"), Obj(_p("third"), random_values("f4", 60, rng))],
              new_obj_list=False, data={_p("second"): random_values("i4", 100, rng)})
    b.segment([Obj(_p("first"), index="same")], new_obj_list=False,
              data={_p("first"): random_values("f8", 100, rng), _p("second"): random_values("i4", 100, rng),
                    _p("third"): random_values("f4", 60, rng)})
    # New object list: only "third", with a new index (new chunk size).
    b.segment([Obj(_p("third"), random_values("f4", 77, rng))])
    b.segment([Obj(_p("first"), index="same"), Obj(_p("second"), index="same")],
              data={_p("first"): random_values("f8", 100, rng), _p("second"): random_values("i4", 100, rng)})
    return b, {"fast": {_p("first"): True, _p("second"): True, _p("third"): True}}


def scenario_truncated(rng, mode: str = "mid_chunk") -> tuple[TdmsBuilder, int, dict]:
    """File cut inside the raw data of the last segment.

    Returns (builder, cut_bytes, notes). mode: "mid_chunk" (inside the
    first object of the last chunk), "mid_object" (inside the second
    object), "chunk_edge" (exactly after a chunk), "interleaved"
    (cut inside a row), "flag" (next offset 0xFF..FF and cut).
    """
    b = TdmsBuilder()
    npc = 90
    names = {"ta": "f8", "tb": "i4", "tc": "i2"}
    b.segment(header_objects([G]) + [Obj(_p(k), random_values(v, npc, rng)) for k, v in names.items()])
    inter = mode == "interleaved"
    chunks = 4
    last = b.segment([Obj(_p(k), random_values(v, npc * chunks, rng)) for k, v in names.items()],
                     chunks=chunks, interleaved=inter, incomplete=(mode == "flag"))
    chunk_bytes = npc * (8 + 4 + 2)
    if mode == "mid_chunk":
        keep = 3 * chunk_bytes + 8 * 37 + 3
    elif mode == "mid_object":
        keep = 3 * chunk_bytes + 8 * npc + 4 * 11 + 1
    elif mode == "chunk_edge":
        keep = 2 * chunk_bytes
    elif mode == "interleaved":
        keep = 2 * chunk_bytes + 14 * 17 + 5
    elif mode == "flag":
        keep = 3 * chunk_bytes + 8 * 50
    else:
        raise ValueError(mode)
    cut = len(last.raw) - keep
    return b, cut, {"fast": {_p(k): True for k in names}}


def scenario_scaled(rng) -> tuple[TdmsBuilder, dict]:
    """Linear scaling on channel, group and file level; a 'scaled' status channel."""
    b = TdmsBuilder()
    n = 500
    objs = [Obj("/"), Obj(obj_path(G)), Obj(obj_path("GroupScaled"), props=linear_scale_props(0.5, 1.0))]
    objs.append(Obj(_p("scaled_i16"), random_values("i2", n, rng), props=linear_scale_props(0.001, -3.0)))
    objs.append(Obj(_p("scaled_f64"), random_values("f8", n, rng), props=linear_scale_props(2.0, 0.25)))
    objs.append(Obj(_p("already_scaled"), random_values("f8", n, rng),
                    props={**linear_scale_props(2.0, 0.25), "NI_Scaling_Status": "scaled"}))
    objs.append(Obj(_p("plain"), random_values("f8", n, rng)))
    objs.append(Obj(obj_path("GroupScaled", "by_group"), random_values("i4", n, rng)))
    b.segment(objs)
    fast = {_p("scaled_i16"): False, _p("scaled_f64"): False, _p("already_scaled"): True,
            _p("plain"): True, obj_path("GroupScaled", "by_group"): False}
    return b, {"fast": fast}


def scenario_daqmx(rng) -> tuple[TdmsBuilder, dict]:
    """DAQmx raw data: a DaqMxRawData channel with scaling and an i16 channel."""
    b = TdmsBuilder()
    n = 256
    width = 6
    # Scale 0 is the DAQmx scaler (no type property), scale 1 is linear on top of it.
    raw_props = linear_scale_props(0.01, 0.5, index=1, input_source=0)
    objs = header_objects([G]) + [
        Obj(_p("dq_raw"), random_values("i2", n * 2, rng), daqmx=Daqmx(3, 0, width), props=raw_props),
        Obj(_p("dq_i32"), random_values("i4", n * 2, rng), daqmx=Daqmx(5, 2, width, data_type=T_I32)),
    ]
    b.segment(objs, chunks=2)
    b.raw_segment({_p("dq_raw"): random_values("i2", n, rng), _p("dq_i32"): random_values("i4", n, rng)})
    return b, {"fast": {_p("dq_raw"): False, _p("dq_i32"): False}}


def scenario_waveform(rng) -> tuple[TdmsBuilder, dict]:
    """Waveform channels with wf_* properties, units and a relative-time channel."""
    b = TdmsBuilder()
    n = 1000
    objs = header_objects(["Wave"], {"name": "wave test", "author": "tests"})
    objs.append(Obj(obj_path("Wave", "sine"), np.sin(np.arange(n) / 50.0),
                    props={**wf_props(0.001, 0.5, "2026-07-28T12:05:36.250000"), "unit_string": "V"}))
    objs.append(Obj(obj_path("Wave", "later"), np.cos(np.arange(n) / 30.0),
                    props={**wf_props(0.002, 0.0, "2026-07-28T12:05:40.000000"), "Unit": "A"}))
    objs.append(Obj(obj_path("Wave", "relative"), np.arange(n, dtype=np.float64),
                    props=wf_props(0.01, 1.5, RawTime(0, 0))))
    b.segment(objs)
    return b, {"fast": {obj_path("Wave", "sine"): True, obj_path("Wave", "later"): True,
                        obj_path("Wave", "relative"): True}}


def scenario_mixed(rng) -> tuple[TdmsBuilder, dict]:
    """Same channels in contiguous, interleaved, big-endian and raw-only segments."""
    b = TdmsBuilder()
    dts = {"m_f8": "f8", "m_i4": "i4", "m_u2": "u2", "m_c8": "c8"}

    def vals(n):
        return {_p(k): random_values(v, n, rng) for k, v in dts.items()}

    b.segment(header_objects([G]) + [Obj(_p(k), random_values(v, 120, rng)) for k, v in dts.items()])
    b.raw_segment(vals(120 * 3), chunks=3)
    b.raw_segment(vals(120), big_endian=True)
    b.raw_segment(vals(120 * 2), chunks=2)
    # New index (other chunk size), same object list.
    b.segment([Obj(_p(k), random_values(v, 50 * 4, rng)) for k, v in dts.items()], chunks=4,
              new_obj_list=False)
    b.segment([Obj(_p(k), index="same") for k in dts], data=vals(50), big_endian=True)
    # Interleaved segments without the complex channel (npTDMS cannot read it).
    b.segment([Obj(_p("m_c8"), index="none")], new_obj_list=False, data=vals(50 * 2), chunks=2,
              interleaved=True)
    b.raw_segment({k: v for k, v in vals(50).items() if not k.endswith("m_c8'")}, interleaved=True,
                  big_endian=True)
    b.segment([Obj(_p("m_c8"), index="same")], new_obj_list=False, data=vals(50))
    return b, {"fast": {_p(k): True for k in dts}}


def scenario_extra_bytes(rng) -> tuple[TdmsBuilder, dict]:
    """A middle segment whose data size is not a multiple of the chunk size.

    npTDMS reads the extra bytes as a short final chunk, so
    builder.expected() does not hold for this file (notes["exact"] is False).
    """
    b = TdmsBuilder()
    objs = header_objects([G]) + [Obj(_p("xa"), random_values("f8", 100, rng)),
                                  Obj(_p("xb"), random_values("i4", 100, rng))]
    b.segment(objs, chunks=1, extra=rng.bytes(600))
    b.raw_segment({_p("xa"): random_values("f8", 200, rng), _p("xb"): random_values("i4", 200, rng)}, chunks=2)
    return b, {"fast": {_p("xa"): True, _p("xb"): True}, "exact": False}


SCENARIOS = {
    "contiguous": scenario_contiguous,
    "mixed": scenario_mixed,
    "extra_bytes": scenario_extra_bytes,
    "chunks": scenario_chunks,
    "single_channel_chunks": scenario_single_channel_chunks,
    "interleaved": scenario_interleaved,
    "interleaved_be": lambda rng: scenario_interleaved(rng, big_endian=True),
    "big_endian": scenario_big_endian,
    "all_dtypes": scenario_all_dtypes,
    "strings_mixed": scenario_strings_mixed,
    "raw_only": scenario_raw_only,
    "growing": scenario_growing,
    "scaled": scenario_scaled,
    "daqmx": scenario_daqmx,
    "waveform": scenario_waveform,
}

TRUNCATED_MODES = ("mid_chunk", "mid_object", "chunk_edge", "interleaved", "flag")


def build_scenario(name: str, directory, seed: int = 1234, index: bool | int = False) -> tuple[str, TdmsBuilder, dict]:
    """Write a named scenario (or "truncated_<mode>") to directory. Returns (path, builder, notes)."""
    rng = np.random.default_rng(seed)
    path = os.path.join(os.fspath(directory), f"{name}.tdms")
    if name.startswith("truncated_"):
        b, cut, notes = scenario_truncated(rng, name[len("truncated_"):])
        b.write(path, index=index, cut=cut)
        # npTDMS values are a prefix of builder.expected().
        notes = dict(notes, cut=cut, exact=False, prefix=True)
        return path, b, notes
    b, notes = SCENARIOS[name](rng)
    b.write(path, index=index)
    return path, b, notes


ALL_SCENARIOS = tuple(SCENARIOS) + tuple("truncated_" + m for m in TRUNCATED_MODES)

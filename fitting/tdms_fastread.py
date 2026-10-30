"""Direct reader for fixed-size TDMS channel data (fast path).

Summary
    Reads channel values 3 to 30 times faster than npTDMS. It uses the
    segment table that npTDMS already parsed and reads bytes with
    positional reads (os.preadv; Windows: locked seek + read). Each reader
    is verified against npTDMS before use.

    Ported from TDMSViewer (tdmsviewer/fastread.py). Keep the two copies
    the same: fixes go to both. Additions here: FastChannelReader.bind()
    (tdms_source opens the file for each read and binds the descriptor,
    so no file handle stays open between reads), max_chunk_stride, and
    the bulk mode of read_many() (file read in order when most channels
    of a family are wanted).

Rules
    - Only plain, fixed-size numeric data. No DAQmx raw data, no
      scaling, no strings, no timestamps. The caller uses npTDMS for
      all other channels.
    - No memory map. A file that becomes shorter causes a clean
      error, not a crash (SIGBUS).
    - Thread-safe: preadv does not use the shared file position.
      Windows has no preadv: seek and read run as one locked pair.

Layout (one pass, O(segments))
    Consecutive segments with the same objects, one chunk each and a
    constant distance merge into one run. A run gives one part per
    channel. A part is a regular 2-level layout: chunks of values.

Families
    Channels with equal part boundaries form a family. read_many()
    reads a family in one pass: bytes that are near each other are
    read once for all channels, not once per channel.

Verification (bounded cost)
    npTDMS decodes the first values of selected chunks: first and last
    run of each layout kind, spread runs and truncated chunks. A file
    wrapper stops npTDMS after these values, so a 1 GB chunk costs some
    hundred bytes. Small files also get an end-to-end check with
    channel.read_data().
"""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass

import numpy as np

try:  # Private npTDMS names. Missing names disable the fast path.
    from nptdms import types as _nt
    from nptdms.common import toc_properties as _toc
    from nptdms.scaling import get_scaling as _get_scaling
except Exception:  # pragma: no cover - depends on npTDMS version
    _nt = _toc = _get_scaling = None

# Reads above this size are split into blocks (bytes).
_BLOCK_BYTES = 16 << 20
# A chunk piece of at least this size is read directly into the output.
_PIECE_READ_MIN = 32 << 10
# Gaps of at least this size are skipped with a new read (one read costs
# about 1.2 us, the same as copying 8 KB).
_GAP_READ_MIN = 8 << 10
# Verification: values per probe, layout kinds and spread runs per channel.
_PROBE_VALUES = 64
_MAX_KINDS = 128
_SPREAD_RUNS = 16
# End-to-end check with channel.read_data() only below these costs.
_API_CHECK_MAX_WORK = 1 << 16  # segments x channels (npTDMS index build)
_API_CHECK_MAX_BYTES = 4 << 20  # bytes npTDMS reads per window


class FastPathError(RuntimeError):
    """The fast path cannot read this channel."""


@dataclass(slots=True, frozen=True)
class _Part:
    """One run of channel values with a regular 2-level layout.

    Value v (0 <= v < count) is at file byte
        offset + (v // npc) * chunk_stride + (v % npc) * item_stride
    chunk_stride 0 means one chunk (npc == count).
    """

    start: int
    count: int
    offset: int
    npc: int
    chunk_stride: int
    item_stride: int
    dtype: np.dtype
    rel: int = 0  # offset inside the segment data (diagnostics only)


class _Grid:
    """Part boundaries of a family: rows (start, count, npc, chunk_stride)."""

    __slots__ = ("tab", "starts")

    def __init__(self, tab: np.ndarray):
        self.tab = np.ascontiguousarray(tab, dtype=np.int64).reshape(-1, 4)
        self.starts = np.ascontiguousarray(self.tab[:, 0])


class _Layout:
    """Parts of one channel: shared grid, own offsets and value forms.

    forms[form[k]] = (item_stride, dtype with byte order) of part k.
    """

    __slots__ = ("grid", "off", "form", "forms")

    def __init__(self, grid: _Grid, off: np.ndarray, form: np.ndarray, forms: list):
        self.grid = grid
        self.off = np.ascontiguousarray(off, dtype=np.int64)
        self.form = np.ascontiguousarray(form, dtype=np.uint8)
        self.forms = forms

    @classmethod
    def from_parts(cls, parts: list[_Part]) -> _Layout:
        forms: list = []
        index: dict = {}
        rows, off, form = [], [], []
        for p in parts:
            key = (p.item_stride, p.dtype.str)
            f = index.get(key)
            if f is None:
                f = index[key] = len(forms)
                forms.append((p.item_stride, p.dtype))
            npc = p.npc if p.chunk_stride else p.count
            rows.append((p.start, p.count, npc, p.chunk_stride))
            off.append(p.offset)
            form.append(f)
        return cls(_Grid(np.array(rows, dtype=np.int64)), np.array(off, np.int64),
                   np.array(form, np.uint8), forms)

    def parts(self) -> list[_Part]:
        out = []
        for (s, c, n, cs), o, f in zip(self.grid.tab.tolist(), self.off.tolist(), self.form.tolist()):
            ist, dt = self.forms[f]
            out.append(_Part(s, c, o, n, cs, ist, dt))
        return out


class FastChannelReader:
    """Read values of one channel directly from the file."""

    __slots__ = ("_fd", "_lay", "length", "dtype", "family", "fragmented")

    def __init__(self, fd: int, parts, length: int, dtype: np.dtype):
        self._fd = fd
        self._lay = parts if isinstance(parts, _Layout) else _Layout.from_parts(parts)
        self.length = length
        self.dtype = np.dtype(dtype).newbyteorder("=")
        self.family = None  # readers with equal family value can use read_many()
        t = self._lay.grid.tab
        if t.shape[0]:
            ok = (t[0, 0] == 0 and int(t[-1, 0] + t[-1, 1]) == length and bool((t[:, 1:3] > 0).all())
                  and bool((t[1:, 0] == t[:-1, 0] + t[:-1, 1]).all()) and bool((t[:, 3] >= 0).all()))
        else:
            ok = length == 0
        if not ok:
            raise FastPathError("parts do not cover the channel exactly")
        # Fragmented: most values are in small or interleaved pieces. Then one
        # read of this channel costs about as much as read_many() of its family.
        small = np.zeros(t.shape[0], dtype=bool)
        for f, (ist, dt) in enumerate(self._lay.forms):
            m = self._lay.form == f
            if ist != dt.itemsize:
                small |= m
            else:
                small |= m & (t[:, 3] > 0) & (t[:, 2] * dt.itemsize < _PIECE_READ_MIN)
        self.fragmented = 2 * int(t[small, 1].sum()) > length

    @property
    def n_parts(self) -> int:
        return int(self._lay.off.size)

    @property
    def fd(self) -> int:
        """File descriptor used by the next read (-1: none bound, reads fail)."""
        return self._fd

    @property
    def max_chunk_stride(self) -> int:
        """Largest chunk stride of the parts in bytes (0: every part is one chunk)."""
        t = self._lay.grid.tab
        return int(t[:, 3].max()) if t.shape[0] else 0

    def bind(self, fd: int) -> None:
        """Use file descriptor fd for the next reads.

        The layout stays valid for the same file content. Bind -1 to make
        every read fail (EBADF) instead of reading a reused descriptor.
        """
        self._fd = fd

    def read(self, start: int, stop: int) -> np.ndarray:
        """Return values [start, stop) in native byte order."""
        start = max(0, start)
        stop = min(self.length, stop)
        out = np.empty(max(0, stop - start), dtype=self.dtype)
        if stop > start:
            self.read_into(out, start)
        return out

    def read_into(self, out: np.ndarray, start: int) -> np.ndarray:
        """Fill out (1-D) with values from start."""
        stop = start + out.size
        if start < 0 or stop > self.length:
            raise IndexError("read outside the channel")
        if out.size:
            _read_family(self._fd, self._lay.grid, [(out, self._lay)], start, stop)
        return out


def read_many(readers: list, start: int, stop: int, outs: list | None = None, bulk: bool = False) -> list:
    """Read values [start, stop) of many channels (clamped per channel).

    Channels of one family are read in one pass, so spread values are
    read once for all channels. Returns the output arrays.

    bulk: also large pieces (>= _PIECE_READ_MIN) of multi-chunk parts are
    read chunk by chunk in file order, with one read per chunk span. Use
    it when most channels of a family are wanted: the file is then read
    once in order (good on a cold cache and on disks that seek slowly).
    The scratch buffer grows to one chunk span: keep chunk strides at or
    below _BLOCK_BYTES.
    """
    a = max(0, start)
    if outs is None:
        outs = [np.empty(max(0, min(r.length, stop) - a), dtype=r.dtype) for r in readers]
    groups: dict = {}
    for i, r in enumerate(readers):
        groups.setdefault(id(r._lay.grid), []).append(i)
    for idx in groups.values():
        first = readers[idx[0]]
        b = min(first.length, stop)
        for i in idx:
            if outs[i].shape != (max(0, b - a),):
                raise ValueError("output size differs from the read range")
        if b > a:
            _read_family(first._fd, first._lay.grid, [(outs[i], readers[i]._lay) for i in idx], a, b, bulk)
    return outs


# -- positional reads ----------------------------------------------------------------
# POSIX: pread/preadv (no shared file position, one system call).
# Windows has neither: one lock keeps seek and read together. Reads from several
# threads then run one after the other (Windows does the same for one handle).
# The fd must be opened with O_BINARY on Windows (text mode changes bytes).

_seek_lock = threading.Lock()
_seek_files: dict[int, io.FileIO] = {}  # fd -> unowned FileIO (no copy on read)


def _seek_readinto(fd: int, mv: memoryview, offset: int) -> int:
    with _seek_lock:
        f = _seek_files.get(fd)
        if f is None:
            f = _seek_files[fd] = io.FileIO(fd, "rb", closefd=False)
        f.seek(offset)
        return f.readinto(mv)


def _seek_pread(fd: int, nbytes: int, offset: int) -> bytes:
    buf = bytearray(nbytes)
    return bytes(buf[:_seek_readinto(fd, memoryview(buf), offset)])


# pread(fd, nbytes, offset) -> bytes, short at the end of the file.
if hasattr(os, "preadv"):
    def _readinto_at(fd: int, mv: memoryview, offset: int) -> int:
        return os.preadv(fd, [mv], offset)

    pread = os.pread
else:  # pragma: no cover - Windows (tests force this path on Linux)
    _readinto_at = _seek_readinto
    pread = _seek_pread


# -- read engine ---------------------------------------------------------------------

def _pread_into(fd: int, buf: np.ndarray, offset: int) -> None:
    mv = memoryview(buf).cast("B")
    done = 0
    size = mv.nbytes
    while done < size:
        n = _readinto_at(fd, mv[done:], offset + done)
        if n <= 0:
            raise FastPathError("File is shorter than its index (file changed on disk?)")
        done += n


def _raw_buffer(scratch: list, nbytes: int) -> np.ndarray:
    """First nbytes of the reused scratch buffer (grows when needed)."""
    buf = scratch[0]
    if buf is None or buf.size < nbytes:
        buf = scratch[0] = np.empty(max(nbytes, 1 << 16), dtype=np.uint8)
    return buf[:nbytes]


def _raw(fd: int, scratch: list, pos: int, nbytes: int) -> np.ndarray:
    """Read nbytes at pos into the reused scratch buffer."""
    raw = _raw_buffer(scratch, nbytes)
    _pread_into(fd, raw, pos)
    return raw


def _direct(fd: int, dst: np.ndarray, pos: int, dt: np.dtype, scratch: list) -> None:
    """Contiguous values at pos into dst (zero copy if the dtype matches)."""
    if dst.dtype == dt and dst.flags.c_contiguous:
        _pread_into(fd, dst, pos)
    else:
        dst[:] = np.ndarray(dst.size, dt, _raw(fd, scratch, pos, dst.size * dt.itemsize))


def _read_family(fd: int, grid: _Grid, members: list, start: int, stop: int, bulk: bool = False) -> None:
    """Values [start, stop) of each member (out, layout) into out[0:stop - start]."""
    k = int(np.searchsorted(grid.starts, start, side="right")) - 1
    tab = grid.tab
    scratch = [None]
    w = 0
    n = stop - start
    while w < n:
        s, cnt, npc, cs = tab[k].tolist()
        lo = start + w - s
        hi = min(stop - s, cnt)
        m = hi - lo
        items = []
        for out, lay in members:
            ist, dt = lay.forms[lay.form[k]]
            items.append((out[w:w + m], int(lay.off[k]), ist, dt))
        _read_part(fd, npc if cs else cnt, cs, lo, hi, items, scratch, bulk)
        w += m
        k += 1


def _read_part(fd: int, npc: int, cs: int, lo: int, hi: int, items: list, scratch: list,
               bulk: bool = False) -> None:
    """Values [lo, hi) of one part for each item (dst, offset, item_stride, dtype).

    bulk: large pieces of a multi-chunk part are not read one by one; they
    join the chunk-by-chunk reads below (see read_many).
    """
    c0 = lo // npc
    c1 = (hi - 1) // npc
    single = len(items) == 1
    rest = []
    for it in items:
        dst, off, ist, dt = it
        isz = dt.itemsize
        if ist == isz:
            if c0 == c1:
                if single or (hi - lo) * isz >= _PIECE_READ_MIN:
                    _direct(fd, dst, off + c0 * cs + (lo - c0 * npc) * isz, dt, scratch)
                    continue
            elif npc * isz >= _PIECE_READ_MIN and not bulk:
                for c in range(c0, c1 + 1):
                    a = max(lo, c * npc)
                    b = min(hi, c * npc + npc)
                    _direct(fd, dst[a - lo:b - lo], off + c * cs + (a - c * npc) * isz, dt, scratch)
                continue
        rest.append(it)
    if not rest:
        return
    # Group the remaining pieces by position: near pieces share one read.
    r0, r1 = (lo - c0 * npc, hi - c0 * npc) if c0 == c1 else (0, npc)
    spans = sorted(((off + r0 * ist, off + (r1 - 1) * ist + dt.itemsize, i)
                    for i, (_d, off, ist, dt) in enumerate(rest)))
    cluster: list = []
    end = 0
    for a, b, i in spans:
        if cluster and a - end >= _GAP_READ_MIN:
            _read_cluster(fd, npc, cs, lo, hi, cluster, scratch)
            cluster = []
        end = b if not cluster else max(end, b)
        cluster.append(rest[i])
    _read_cluster(fd, npc, cs, lo, hi, cluster, scratch)


def _read_cluster(fd: int, npc: int, cs: int, lo: int, hi: int, mem: list, scratch: list) -> None:
    """Pieces that are near each other: read their common byte range once."""
    base = min(it[1] for it in mem)
    rel = [(dst, off - base, ist, dt) for dst, off, ist, dt in mem]
    c0 = lo // npc
    c1 = (hi - 1) // npc
    if c0 == c1:
        _read_rows(fd, base + c0 * cs, lo - c0 * npc, hi - c0 * npc, rel, 0, scratch)
        return
    span = max(r + (npc - 1) * ist + dt.itemsize for _d, r, ist, dt in rel)
    if cs - span < _GAP_READ_MIN and 2 * span <= _BLOCK_BYTES:
        # Dense chunks: several chunks per read, strided 2-D copy.
        per = (_BLOCK_BYTES - span) // cs + 1
        c = c0
        while c <= c1:
            ce = min(c1, c + per - 1)
            nb = ce - c + 1
            raw = _raw(fd, scratch, base + c * cs, (nb - 1) * cs + span)
            g0 = max(lo, c * npc)
            g1 = min(hi, (ce + 1) * npc)
            for dst, r, ist, dt in rel:
                view = np.ndarray((nb, npc), dt, raw, r, (cs, ist))
                _copy_2d(dst[g0 - lo:g1 - lo], view, g0 - c * npc, g1 - c * npc)
            c = ce + 1
        return
    # Sparse or large chunks: one read per chunk. Partial end chunks first.
    f0 = c0 + (lo > c0 * npc)
    f1 = c1 - (hi < c1 * npc + npc)
    for c in [c for c in (c0, c1) if not f0 <= c <= f1]:
        a = max(lo, c * npc)
        b = min(hi, c * npc + npc)
        _read_rows(fd, base + c * cs, a - c * npc, b - c * npc, rel, a - lo, scratch)
    if f1 < f0:
        return
    if 2 * span > _BLOCK_BYTES:
        for c in range(f0, f1 + 1):
            _read_rows(fd, base + c * cs, 0, npc, rel, c * npc - lo, scratch)
        return
    # Full chunks: fixed views on one buffer, one read and M copies per chunk.
    if len(rel) == 1 and rel[0][2] == rel[0][3].itemsize and rel[0][0].dtype == rel[0][3]:
        dst, r, _ist, _dt = rel[0]
        for c in range(f0, f1 + 1):
            w = c * npc - lo
            _pread_into(fd, dst[w:w + npc], base + c * cs + r)
        return
    raw = _raw_buffer(scratch, span)
    views = [(dst, np.ndarray((npc,), dt, raw, r, (ist,))) for dst, r, ist, dt in rel]
    for c in range(f0, f1 + 1):
        _pread_into(fd, raw, base + c * cs)
        w = c * npc - lo
        for dst, view in views:
            dst[w:w + npc] = view


def _read_rows(fd: int, pos: int, i0: int, i1: int, rel: list, w: int, scratch: list) -> None:
    """Rows [i0, i1) of one chunk at pos; each member at pos + rel + row * item_stride."""
    if all(ist == dt.itemsize for _d, _r, ist, dt in rel):
        step = i1 - i0  # contiguous pieces: blocks do not make the range shorter
    else:
        step = max(1, _BLOCK_BYTES // max(ist for _d, _r, ist, _t in rel))
    j = i0
    while j < i1:
        j1 = min(i1, j + step)
        a = min(r + j * ist for _d, r, ist, _t in rel)
        b = max(r + (j1 - 1) * ist + dt.itemsize for _d, r, ist, dt in rel)
        raw = _raw(fd, scratch, pos + a, b - a)
        for dst, r, ist, dt in rel:
            dst[w + j - i0:w + j1 - i0] = np.ndarray((j1 - j,), dt, raw, r + j * ist - a, (ist,))
        j = j1


def _copy_2d(dst: np.ndarray, view: np.ndarray, f0: int, f1: int) -> None:
    """dst[:] = view.reshape(-1)[f0:f1] without a temporary copy."""
    npc = view.shape[1]
    r0, i0 = divmod(f0, npc)
    r1, i1 = divmod(f1, npc)
    if r0 == r1:
        dst[:] = view[r0, i0:i1]
        return
    p = 0
    if i0:
        p = npc - i0
        dst[:p] = view[r0, i0:]
        r0 += 1
    if r1 > r0:
        q = p + (r1 - r0) * npc
        dst[p:q].reshape(r1 - r0, npc)[...] = view[r0:r1]
        p = q
    if i1:
        dst[p:p + i1] = view[r1, :i1]


# -- npTDMS helpers ---------------------------------------------------------------------

def _segment_ends(channel):
    """Value count at the end of each npTDMS segment of a channel, or None."""
    try:
        rd = channel._reader
        path = channel.path
        if path not in rd._segment_channel_offsets:
            rd._build_index(path)
        _first, offs = rd._segment_channel_offsets[path]
        return np.unique(np.asarray(offs, dtype=np.int64))
    except Exception:
        return None


def nptdms_read(channel, start: int, count: int):
    """channel.read_data(start, count) with work-arounds for two npTDMS 1.11 bugs.

    1. A window that starts before a segment without this channel and ends
       in a later multi-chunk segment over-reads ("could not broadcast").
    2. A channel with no values in the cut last chunk of a truncated file
       fails the same way.
    On error, read segment by segment; read a failing piece to the end.
    """
    try:
        return channel.read_data(start, count)
    except ValueError:
        pass
    stop = min(len(channel), start + count)
    ends = _segment_ends(channel)
    if ends is None or stop <= start:
        return channel.read_data(start)[:count]
    pieces = []
    a = start
    while a < stop:
        k = int(np.searchsorted(ends, a, side="right"))
        b = int(min(stop, ends[k])) if k < ends.size else stop
        try:
            pieces.append(np.asarray(channel.read_data(a, b - a)))
        except ValueError:
            pieces.append(np.asarray(channel.read_data(a))[: b - a])
        a = b
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def _channel_type(tdms_file, channel):
    """(npTDMS data type, numpy dtype) of a plain channel; FastPathError otherwise."""
    if _nt is None:
        raise FastPathError("npTDMS internals not available")
    dt = channel.data_type
    if dt is None or dt in (_nt.String, _nt.TimeStamp) or getattr(dt, "size", None) is None:
        raise FastPathError("unsized or special type")
    if dt is getattr(_nt, "DaqMxRawData", None) or getattr(dt, "nptype", None) is None:
        raise FastPathError("no numpy type")
    group = tdms_file[channel.group_name]
    if _get_scaling(channel.properties, group.properties, tdms_file.properties) is not None:
        raise FastPathError("channel has scaling")
    base_dtype = np.dtype(dt.nptype)
    if channel.dtype != base_dtype.newbyteorder("="):
        raise FastPathError("scaled dtype differs")
    if base_dtype.itemsize != dt.size:
        raise FastPathError("item size mismatch")
    return dt, base_dtype


# -- layout scan (one pass over all segments) ---------------------------------------------

class _SegLayout:
    """Value layout of one segment kind.

    entries[path] = (rel, npc, item_stride, npTDMS type, byte order, unit):
    value v of the channel in a chunk is at rel + v * item_stride from the
    chunk start; unit is the number of values per chunk (npc can be larger
    for truncated interleaved segments).
    """

    __slots__ = ("entries", "chunk_size", "interleaved", "unsized", "paths", "bad")

    def __init__(self, entries=None, chunk_size=0, interleaved=False, unsized=False, paths=(), bad=None):
        self.entries = entries or {}
        self.chunk_size = chunk_size
        self.interleaved = interleaved
        self.unsized = unsized
        self.paths = paths
        self.bad = bad


class _Scan:
    """Runs of all segments. Run r: data_position D, chunk_stride cs,
    units (chunks), layout id lid, probes [(segment, chunk, k)]."""

    __slots__ = ("segs", "layouts", "bad", "D", "cs", "units", "lid", "probes")


def _segment_layout(seg, objs, toc: int) -> _SegLayout:
    data = [o for o in objs if o.has_data]
    paths = tuple(o.path for o in data)
    try:
        if seg._have_daqmx_objects():
            return _SegLayout(paths=paths, bad="DAQmx segment")
        interleaved = bool(toc & _toc["kTocInterleavedData"]) and seg._have_interleaved_data()
    except Exception as exc:
        return _SegLayout(paths=paths, bad=f"segment not readable ({exc})")
    order = ">" if toc & _toc["kTocBigEndian"] else "<"
    unsized = any(o.data_type is None or getattr(o.data_type, "size", None) is None for o in data)
    entries = {}
    if interleaved:
        if unsized or len({o.number_values for o in data}) != 1:
            return _SegLayout(paths=paths, bad="interleaved segment with unsized data or unequal counts")
        row = sum(o.data_type.size for o in data)
        col = 0
        for o in data:
            if o.number_values:
                entries[o.path] = (col, o.number_values, row, o.data_type, order, o.number_values)
            col += o.data_type.size
        chunk_size = row * data[0].number_values
    else:
        rel = 0
        for o in data:
            size = getattr(o.data_type, "size", None)
            if size is not None and o.number_values:
                entries[o.path] = (rel, o.number_values, size, o.data_type, order, o.number_values)
            rel += o.data_size
        chunk_size = rel
    if chunk_size != seg._get_chunk_size():
        return _SegLayout(paths=paths, bad="chunk size differs from npTDMS")
    return _SegLayout(entries, chunk_size, interleaved, unsized, paths)


def _scan(tdms_file) -> _Scan:
    """One pass over the npTDMS segment table (cost O(segments))."""
    if _nt is None:
        raise FastPathError("npTDMS internals not available")
    segs = tdms_file._reader._segments
    if segs is None:
        raise FastPathError("no segment table")
    raw_flag = _toc["kTocRawData"]
    kind_bits = _toc["kTocInterleavedData"] | _toc["kTocBigEndian"]
    layouts: list[_SegLayout] = []
    by_ids: dict = {}
    by_content: dict = {}
    bad: dict = {}
    D, CS, U, L, P = [], [], [], [], []

    def add(lid, d, cs, units, probes):
        D.append(d)
        CS.append(cs)
        U.append(units)
        L.append(lid)
        P.append(probes)

    def new_layout(lay):
        layouts.append(lay)
        return len(layouts) - 1

    cur = None  # merged run: [lid, D0, stride, units, last D, seg first, seg second, seg last]

    def flush():
        lid, d0, stride, units, _last, sf, s2, sl = cur
        if units == 1:
            add(lid, d0, layouts[lid].chunk_size, 1, ((sf, 0, 0),))
        else:
            add(lid, d0, stride, units, ((sf, 0, 0), (s2, 0, 1), (sl, 0, units - 1)))

    prev_objs = prev_toc = None
    lid = -1
    for si, seg in enumerate(segs):
        toc = seg.toc_mask
        objs = seg.ordered_objects
        if not objs or not toc & raw_flag or seg.num_chunks <= 0:
            continue
        if objs is not prev_objs or toc != prev_toc:
            ids = (toc & kind_bits, tuple(map(id, objs)))
            lid = by_ids.get(ids)
            if lid is None:
                content = (toc & kind_bits, tuple((o.path, o.has_data, o.number_values, o.data_size,
                                                   o.data_type, type(o)) for o in objs))
                lid = by_content.get(content)
                if lid is None:
                    lid = by_content[content] = new_layout(_segment_layout(seg, objs, toc))
                by_ids[ids] = lid
            prev_objs, prev_toc = objs, toc
        lay = layouts[lid]
        if lay.bad:
            for p in lay.paths:
                bad.setdefault(p, lay.bad)
            if cur:
                flush()
                cur = None
            continue
        d = seg.data_position
        nch = seg.num_chunks
        ovr = seg.final_chunk_lengths_override
        if ovr is None and nch == 1:
            if cur and cur[0] == lid:
                step = d - cur[4]
                if cur[3] == 1 and step >= lay.chunk_size > 0:
                    cur[2] = step
                    cur[6] = si
                if step == cur[2]:
                    cur[3] += 1
                    cur[4] = d
                    cur[7] = si
                    continue
            if cur:
                flush()
            cur = [lid, d, None, 1, d, si, -1, si]
            continue
        if cur:
            flush()
            cur = None
        if ovr is None:
            add(lid, d, lay.chunk_size, nch, tuple(dict.fromkeys(((si, 0, 0), (si, 1, 1), (si, nch - 1, nch - 1)))))
            continue
        # Truncated segment: the final chunk has fewer values.
        if lay.unsized:
            for p in lay.paths:
                bad.setdefault(p, "truncated segment with unsized data")
            continue
        n_full = nch - 1
        if lay.interleaved:
            ent = {}
            for p, (col, npc, row, t, order, unit) in lay.entries.items():
                count = npc * n_full + ovr.get(p, 0)
                if count:
                    ent[p] = (col, count, row, t, order, unit)
            probes = ((si, 0, 0), (si, n_full, n_full)) if n_full else ((si, 0, 0),)
            add(new_layout(_SegLayout(ent)), d, 0, 1, probes)
            continue
        if n_full:
            add(lid, d, lay.chunk_size, n_full,
                tuple(dict.fromkeys(((si, 0, 0), (si, 1, 1), (si, n_full - 1, n_full - 1)))))
        ent = {}
        rel = 0
        for o in (o for o in objs if o.has_data):
            nv = ovr.get(o.path, 0)
            e = lay.entries.get(o.path)
            if nv and e is not None:
                ent[o.path] = (rel, nv, e[2], e[3], e[4], nv)
            rel += o.data_type.size * nv
        add(new_layout(_SegLayout(ent)), d + n_full * lay.chunk_size, 0, 1, ((si, n_full, 0),))
    if cur:
        flush()
    scan = _Scan()
    scan.segs = segs
    scan.layouts = layouts
    scan.bad = bad
    scan.D = np.array(D, dtype=np.int64)
    scan.cs = np.array(CS, dtype=np.int64)
    scan.units = np.array(U, dtype=np.int64)
    scan.lid = np.array(L, dtype=np.int64)
    scan.probes = P
    return scan


def _channel_layout(scan: _Scan, path: str, dt, base: np.dtype):
    """Part table of one channel from the runs (vectorized over runs).

    Returns (tab, offsets, form, forms, info); info holds the true
    layout for verification.
    """
    if path in scan.bad:
        raise FastPathError(scan.bad[path])
    nl = len(scan.layouts)
    rel = np.full(nl, -1, dtype=np.int64)
    npc = np.zeros(nl, dtype=np.int64)
    unit = np.zeros(nl, dtype=np.int64)
    ist = np.zeros(nl, dtype=np.int64)
    form = np.zeros(nl, dtype=np.uint8)
    forms: list = []
    index: dict = {}
    for j, lay in enumerate(scan.layouts):
        e = lay.entries.get(path)
        if e is None:
            continue
        r, n, s, t, order, u = e
        if t is not dt:
            raise FastPathError("data type changes between segments")
        f = index.get((s, order))
        if f is None:
            if len(forms) >= 255:
                raise FastPathError("too many value forms")
            f = index[(s, order)] = len(forms)
            forms.append((s, base.newbyteorder(order)))
        rel[j], npc[j], unit[j], ist[j], form[j] = r, n, u, s, f
    sel = np.flatnonzero(rel[scan.lid] >= 0)
    ls = scan.lid[sel]
    units = scan.units[sel]
    counts = npc[ls] * units
    starts = np.cumsum(counts) - counts
    offs = scan.D[sel] + rel[ls]
    npcs = npc[ls]
    css = scan.cs[sel].copy()
    ists = ist[ls]
    flat = (units == 1) | (css == npcs * ists)  # chunks back to back: one strided run
    npcs = np.where(flat, counts, npcs)
    css[flat] = 0
    tab = np.stack([starts, counts, npcs, css], axis=1) if sel.size else np.zeros((0, 4), np.int64)
    info = (sel, unit[ls], ists, ls * 2 + (units > 1), starts, counts)
    return tab, offs, form[ls], forms, info


def build_parts(tdms_file, channel) -> tuple[list[_Part], np.dtype]:
    """Return the value layout of a channel from the npTDMS segment table.

    Raises FastPathError if the channel is not plain fixed-size data.
    """
    dt, base = _channel_type(tdms_file, channel)
    tab, offs, form, forms, _info = _channel_layout(_scan(tdms_file), channel.path, dt, base)
    total = int(tab[:, 1].sum())
    if total != len(channel):
        raise FastPathError(f"value count {total} differs from channel length {len(channel)}")
    return _Layout(_Grid(tab), offs, form, forms).parts(), base


# -- verification --------------------------------------------------------------------------

class _LimitedFile:
    """File wrapper for npTDMS: after each seek, reads stop after `limit` bytes.

    npTDMS reads values with readinto() until it returns 0 and accepts a
    short result. So it decodes only the first values of a chunk, at the
    offsets that npTDMS itself computes.
    """

    __slots__ = ("_fh", "_limit", "_left")

    def __init__(self, fh, limit: int):
        self._fh = fh
        self._limit = limit
        self._left = limit

    def seek(self, pos, whence=0):
        self._left = self._limit
        return self._fh.seek(pos, whence)

    def tell(self):
        return self._fh.tell()

    def readinto(self, b):
        mv = memoryview(b).cast("B")
        n = min(len(mv), self._left)
        if n <= 0:
            return 0
        got = self._fh.readinto(mv[:n]) or 0
        self._left -= got
        return got

    def read(self, n=-1):
        if n is None or n < 0 or n > self._left:
            n = self._left
        data = self._fh.read(n) if n > 0 else b""
        self._left -= len(data)
        return data


def _probe(fh, seg, path: str, chunk: int, nvals: int, item_stride: int) -> np.ndarray:
    """npTDMS values of `path` at the start of one chunk (at most nvals)."""
    gen = seg.read_raw_data_for_channel(_LimitedFile(fh, nvals * item_stride), path, chunk, 1)
    try:
        data = next(gen).data
    finally:
        gen.close()
    ref = np.asarray(data if data is not None else [])
    return ref.astype(ref.dtype.newbyteorder("="), copy=False)


def _pick(sig: np.ndarray) -> list[int]:
    """Parts to probe: ends, spread parts, first and last of each layout kind."""
    n = sig.size
    picks = {0, n - 1, *np.linspace(0, n - 1, min(n, _SPREAD_RUNS)).astype(np.int64).tolist()}
    _u, first = np.unique(sig, return_index=True)
    _u, last = np.unique(sig[::-1], return_index=True)
    last = n - 1 - last
    keep = np.argsort(first, kind="stable")[:_MAX_KINDS]
    picks.update(first[keep].tolist())
    picks.update(last[keep].tolist())
    return sorted(picks)


def _same(ref: np.ndarray, got: np.ndarray, at: int) -> None:
    if ref.dtype != got.dtype or ref.shape != got.shape or ref.tobytes() != got.tobytes():
        raise FastPathError(f"verification failed at value {at}")


def _verify(reader, channel, scan: _Scan, info, fh, ref_read, api_ok: bool) -> None:
    """Compare the reader with npTDMS at the probe points of the true layout."""
    sel, unit, ists, sig, starts, counts = info
    if sel.size == 0:
        return
    path = channel.path
    for p in _pick(sig):
        u, s, v0, cnt = int(unit[p]), int(ists[p]), int(starts[p]), int(counts[p])
        for si, chunk, k in scan.probes[int(sel[p])]:
            nv = min(_PROBE_VALUES, u, cnt - k * u)
            if nv <= 0:
                continue
            v = v0 + k * u
            _same(_probe(fh, scan.segs[si], path, chunk, nv, s), reader.read(v, v + nv), v)
    cost = max(int(unit[0] * ists[0]), int(unit[-1] * ists[-1]))
    if api_ok and ref_read is not None and cost <= _API_CHECK_MAX_BYTES:
        n = reader.length
        w = min(n, 256)
        for a in sorted({0, n - w}):
            _same(np.asarray(ref_read(channel, a, w)), reader.read(a, a + w), a)


def _verify_file(fd: int):
    return open(os.dup(fd), "rb", buffering=0)


def make_fast_readers(tdms_file, channels: list, fd: int, ref_read=nptdms_read) -> list:
    """Build and verify fast readers for many channels with one layout scan.

    Returns one item per channel: a FastChannelReader, or the exception
    that tells why npTDMS must be used. ref_read(channel, start, count)
    gives npTDMS reference values for the end-to-end check.
    """
    out: list = [None] * len(channels)
    todo = []
    for i, ch in enumerate(channels):
        try:
            todo.append((i, ch) + _channel_type(tdms_file, ch))
        except Exception as exc:
            out[i] = exc
    if not todo:
        return out
    try:
        scan = _scan(tdms_file)
    except Exception as exc:
        for i, *_rest in todo:
            out[i] = FastPathError(f"layout scan failed: {exc!r}")
        return out
    grids: dict = {}
    built = []
    for i, ch, dt, base in todo:
        try:
            tab, offs, form, forms, info = _channel_layout(scan, ch.path, dt, base)
            total = int(tab[:, 1].sum())
            if total != len(ch):
                raise FastPathError(f"value count {total} differs from channel length {len(ch)}")
            key = tab.tobytes()
            grid = grids.get(key)
            if grid is None:
                grid = grids[key] = _Grid(tab)
            built.append((i, ch, FastChannelReader(fd, _Layout(grid, offs, form, forms), len(ch), base), info))
        except Exception as exc:
            out[i] = exc
    api_ok = len(scan.segs) * len(built) <= _API_CHECK_MAX_WORK
    families: dict = {}
    with _verify_file(fd) as fh:
        for i, ch, reader, info in built:
            try:
                _verify(reader, ch, scan, info, fh, ref_read, api_ok)
            except Exception as exc:
                out[i] = exc if isinstance(exc, FastPathError) else FastPathError(f"verification error: {exc!r}")
                continue
            reader.family = families.setdefault(id(reader._lay.grid), len(families))
            out[i] = reader
    return out


def make_fast_reader(tdms_file, channel, fd: int, ref_read=None) -> FastChannelReader:
    """Build and verify a fast reader for one channel. Raises on any doubt.

    ref_read(channel, start, count) gives the npTDMS reference values.
    """
    parts, base_dtype = build_parts(tdms_file, channel)
    reader = FastChannelReader(fd, parts, len(channel), base_dtype)
    if reader.length == 0:
        return reader
    dt, base = _channel_type(tdms_file, channel)
    scan = _scan(tdms_file)
    info = _channel_layout(scan, channel.path, dt, base)[4]
    with _verify_file(fd) as fh:
        _verify(reader, channel, scan, info, fh, nptdms_read if ref_read is None else ref_read,
                len(scan.segs) <= _API_CHECK_MAX_WORK)
    return reader

"""Read a TDMS recording: metadata at open, channel values on demand.

Summary
    TdmsRecording opens a file with npTDMS (metadata only) and builds one
    verified direct reader per numeric channel (tdms_fastread). Values are
    read from the file when a channel is first requested. No file handle
    stays open between reads, so the file can be rewritten at any time
    (the fit report does this). Ported from TDMSViewer
    (tdmsviewer/tdmsfile.py and engine.py), reduced to what fitting needs.

Rules
    - Values are float64 arrays. Float, integer and boolean channels are
      numeric. Strings, timestamps and complex values are listed with
      numeric=False and cannot be read.
    - A channel the direct reader cannot verify is read with npTDMS
      (scaled channels, DAQmx raw data, unusual layouts). Same values,
      slower. Interleaved and DAQmx files are read in one pass for all
      such channels.
    - Each read checks size, modification time and inode of the file. A
      changed file is opened again before the read.
    - Damaged files open: bytes after the last valid segment are ignored,
      a .tdms_index that does not match the data file is ignored. Each
      case gives one warning in TdmsRecording.warnings.
    - npTDMS objects are not thread-safe: use one TdmsRecording from one
      thread. Direct reads (pread) are thread-safe.
"""

from __future__ import annotations

import io
import logging
import os
import struct
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass

import numpy as np
from nptdms import TdmsFile

from . import tdms_fastread as fastread
from .tdms_fastread import FastChannelReader, nptdms_read

_log = logging.getLogger(__name__)

# TDMS segment lead-in.
_LEAD_IN = 28
_INCOMPLETE = 0xFFFFFFFFFFFFFFFF  # next segment offset of an unfinished segment
_TOC_BIG_ENDIAN = 1 << 6
_TOC_KNOWN = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6) | (1 << 7)
_O_READ = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)

# numpy dtype kinds that read() converts to float64.
NUMERIC_KINDS = "fiub"


@dataclass(eq=False)
class RecordingChannel:
    """Metadata of one channel. Plain data, safe to keep after the file is closed."""

    group: str
    name: str
    path: str
    length: int
    dtype: np.dtype
    properties: dict
    numeric: bool  # float, integer or boolean values: read() gives float64
    fast: bool = False  # True: verified direct reader; False: npTDMS

    @property
    def label(self) -> str:
        return f"{self.group}/{self.name}"


# -- file layout helpers (damaged files) -------------------------------------------

class _PrefixFile(io.RawIOBase):
    """Read-only view of the first `limit` bytes of a file.

    npTDMS gets the data size with seek(0, SEEK_END) and tell(), so it
    sees `limit` bytes. There is no fileno(): code that asks the OS for
    the size gets an error, not the full size.
    """

    def __init__(self, path: str, limit: int):
        super().__init__()
        self._f = open(path, "rb", buffering=0)
        self._limit = limit

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_END:
            return self._f.seek(self._limit + offset)
        return self._f.seek(offset, whence)

    def tell(self) -> int:
        return self._f.tell()

    def readinto(self, b) -> int:
        mv = memoryview(b).cast("B")
        n = min(mv.nbytes, self._limit - self._f.tell())
        if n <= 0:
            return 0
        return self._f.readinto(mv[:n])

    def close(self) -> None:
        try:
            if not self.closed:
                self._f.close()
        finally:
            super().close()


def _valid_prefix(fd: int, size: int) -> int:
    """End offset of the valid TDMS segments at the start of a file.

    Walks the 28-byte lead-ins from offset 0. Stops at a tag that is not
    'TDSm' or at a lead-in that is not possible (unknown ToC bits,
    metadata larger than the segment). A last segment with an unknown
    size (0xFFFFFFFFFFFFFFFF) or a size past the end of the file goes to
    the end of the file.
    """
    pos = 0
    while pos + _LEAD_IN <= size:
        lead = fastread.pread(fd, _LEAD_IN, pos)
        if len(lead) < _LEAD_IN or lead[:4] != b"TDSm":
            break
        toc = struct.unpack_from("<l", lead, 4)[0]
        if toc & ~_TOC_KNOWN:
            break
        order = ">" if toc & _TOC_BIG_ENDIAN else "<"
        _ver, nxt, raw = struct.unpack_from(order + "lQQ", lead, 8)
        if nxt == _INCOMPLETE:
            return size
        if raw > nxt:
            break
        end = pos + _LEAD_IN + nxt
        if end >= size:
            return size
        pos = end
    return pos


def _parsed_prefix(path: str, limit: int) -> int:
    """End of the segments that npTDMS can parse in the first `limit` bytes.

    npTDMS keeps the segments before the one that fails. Returns `limit`
    if all segments can be parsed, 0 if none.
    """
    from nptdms.reader import TdmsReader

    fh = io.BufferedReader(_PrefixFile(path, limit), buffer_size=1 << 16)
    try:
        reader = TdmsReader(fh)
        try:
            reader.read_metadata()
            return limit
        except Exception:
            segs = reader._segments or []
            return segs[-1].next_segment_pos if segs else 0
    except Exception:
        return 0
    finally:
        fh.close()


def _segment_matches(seg, fd: int, ifd: int, off: int, head: int, end: int) -> bool:
    """True if lead-in and metadata of one segment are the same in data and index file."""
    data = fastread.pread(fd, head, seg.position)
    idx = fastread.pread(ifd, head, off)
    if len(data) != head or len(idx) != head or head < _LEAD_IN:
        return False
    if data[:4] != b"TDSm" or idx[:4] != b"TDSh" or data[4:] != idx[4:]:
        return False
    toc = struct.unpack_from("<l", data, 4)[0]
    if toc != seg.toc_mask:
        return False
    order = ">" if toc & _TOC_BIG_ENDIAN else "<"
    _ver, nxt, raw = struct.unpack_from(order + "lQQ", data, 8)
    if _LEAD_IN + raw != head:
        return False
    if nxt != _INCOMPLETE and min(end, seg.position + _LEAD_IN + nxt) != seg.next_segment_pos:
        return False
    return True


def _index_matches(tdms, path: str, size: int, samples: int = 256) -> bool:
    """True if the .tdms_index describes this data file.

    The index repeats the lead-in (tag TDSh) and the metadata of each
    segment. For the first, the last and up to `samples` spread segments,
    these bytes must be the same in the data file, and the lead-in must
    agree with the segment table. Bytes after the last indexed segment
    must not be a segment (a stale index).
    """
    try:
        segs = tdms._reader._segments
    except Exception:
        return True
    if not segs:
        return size == 0
    end = segs[-1].next_segment_pos
    if end > size:
        return False
    n = len(segs)
    picks = {0, n - 1, *range(0, n, max(1, n // samples))}
    try:
        with open(path, "rb", buffering=0) as fh, open(path + "_index", "rb", buffering=0) as ih:
            fd, ifd = fh.fileno(), ih.fileno()
            if end < size and (fastread.pread(fd, 4, end) == b"TDSm" or _valid_prefix(fd, size) != end):
                return False
            off = 0  # position of the segment in the index file
            for i, seg in enumerate(segs):
                head = seg.data_position - seg.position  # lead-in + metadata
                if i in picks and not _segment_matches(seg, fd, ifd, off, head, end):
                    return False
                off += head
    except (OSError, struct.error, AttributeError):
        return False
    return True


# -- open ----------------------------------------------------------------------------

def _try_limited(path: str, limit: int):
    """TdmsFile of the first `limit` bytes, with the handle to close; None if this fails."""
    fh = io.BufferedReader(_PrefixFile(path, limit), buffer_size=1 << 16)
    try:
        return TdmsFile.open(fh), fh
    except Exception as exc:
        _log.debug("open of the first %d bytes failed: %s", limit, exc)
        fh.close()
        return None


def _open_limited(path: str, size: int, error: Exception) -> tuple:
    """Open the valid start of a file with bytes after its last segment.

    1. Walk the lead-ins (_valid_prefix) and open the part before the
       first bad lead-in.
    2. If that fails too (a lead-in that looks correct, with bad
       metadata), open the segments that npTDMS can parse.
    Returns (TdmsFile, handle, tail note). Raises `error` if this does not help.
    """
    try:
        with open(path, "rb", buffering=0) as fh:
            limit = _valid_prefix(fh.fileno(), size)
    except OSError:
        limit = 0
    if 0 < limit < size:
        opened = _try_limited(path, limit)
        if opened is not None:
            return opened[0], opened[1], None
    if limit > 0:
        good = _parsed_prefix(path, limit)
        if 0 < good < limit:
            opened = _try_limited(path, good)
            if opened is not None:
                note = (f"{size - good} bytes after offset {good} cannot be read "
                        f"({type(error).__name__}: {error}) and were ignored.")
                return opened[0], opened[1], note
    raise error


def _open_without_index(path: str, size: int) -> tuple:
    """Open the data file alone (a file object makes npTDMS skip the index)."""
    fh = open(path, "rb")
    try:
        return TdmsFile.open(fh), fh, None
    except Exception as exc:
        fh.close()
        error = exc
    return _open_limited(path, size, error)


def _open_nptdms(path: str, size: int) -> tuple:
    """Open a file with npTDMS, metadata only.

    Returns (TdmsFile, file object to close or None, warnings). Raises the
    npTDMS error if the file cannot be opened at all.
    """
    warnings: list[str] = []
    fh = None
    tail_note = None
    has_index = os.path.isfile(path + "_index")
    try:
        tdms = TdmsFile.open(path)
    except Exception as exc:
        if has_index:
            # A damaged .tdms_index must not block a good data file.
            tdms, fh, tail_note = _open_without_index(path, size)
            warnings.append(f"The .tdms_index file cannot be read ({type(exc).__name__}: {exc}). "
                            "It was ignored.")
        else:
            tdms, fh, tail_note = _open_limited(path, size, exc)
    else:
        if getattr(tdms._reader, "_index_file_path", None) and not _index_matches(tdms, path, size):
            # A stale or foreign .tdms_index gives wrong data. Read the data file alone.
            tdms.close()
            tdms, fh, tail_note = _open_without_index(path, size)
            warnings.append("The .tdms_index file does not match the data file. It was ignored.")
    try:
        segs = tdms._reader._segments
        used = segs[-1].next_segment_pos if segs else 0
    except Exception:
        used = size
    if used < size:
        warnings.append(tail_note or f"{size - used} bytes after offset {used} are not TDMS data and were ignored.")
    try:
        if tdms.file_status.incomplete_final_segment:
            warnings.append("The last segment of the file is incomplete (the write was not finished).")
    except Exception:
        pass
    return tdms, fh, warnings


def _mixed_layout(tdms) -> bool:
    """True if segments are interleaved or DAQmx (per-channel npTDMS reads are costly)."""
    try:
        from nptdms.common import toc_properties as toc

        for seg in tdms._reader._segments:
            if seg.toc_mask & toc["kTocInterleavedData"]:
                return True
            if seg.ordered_objects and seg._have_daqmx_objects():
                return True
    except Exception:
        return False
    return False


# -- npTDMS reads (fallback) ---------------------------------------------------------

def _gap_cuts(channel) -> np.ndarray:
    """Channel offsets where a read must restart (npTDMS 1.11 bug).

    read_raw_data_for_channel loses count of the segments when a segment
    without values of this channel is inside the window. A read that does
    not cross such a gap is correct.
    """
    try:
        rd = channel._reader
        path = channel.path
        if path not in rd._segment_channel_offsets:
            rd._build_index(path)
        _first, offs = rd._segment_channel_offsets[path]
        offs = np.asarray(offs, dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    if offs.size < 2:
        return np.empty(0, dtype=np.int64)
    gaps = np.flatnonzero(np.diff(offs) == 0) + 1
    return np.unique(offs[gaps])


def _nptdms_values(channel, stop: int) -> np.ndarray:
    """Values [0, stop) from npTDMS; one read per run without gaps (see _gap_cuts)."""
    cuts = _gap_cuts(channel)
    bounds = [int(v) for v in cuts[(cuts > 0) & (cuts < stop)]] + [stop]
    parts = []
    a = 0
    for b in bounds:
        parts.append(np.asarray(nptdms_read(channel, a, b - a)))
        a = b
    out = parts[0] if len(parts) == 1 else np.concatenate(parts)
    if out.shape[0] > stop:
        raise IOError(f"{channel.path}: npTDMS gave {out.shape[0]} values for a window of {stop}")
    return out


def _as_f64(a: np.ndarray) -> np.ndarray:
    return a if a.dtype == np.float64 else a.astype(np.float64)


# -- recording -------------------------------------------------------------------------

class TdmsRecording:
    """One TDMS file: metadata at open, channel values on demand.

    Attributes
        path, size: the file.
        properties: root properties. group_properties: name -> properties.
        channels: all channels in file order (numeric or not).
        warnings: problems found at open, one text each.
        n_segments: segments in the file.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.warnings: list[str] = []
        self._stamp = None
        self._open()

    # -- open ---------------------------------------------------------------------

    def _file_stamp(self) -> tuple:
        st = os.stat(self.path)
        return st.st_size, st.st_mtime_ns, st.st_ino

    def _open(self) -> None:
        stamp = self._file_stamp()  # before the parse: a change during open is seen by the next read
        self.size = stamp[0]
        self.properties: dict = {}
        self.group_properties: dict[str, dict] = {}
        self.channels: list[RecordingChannel] = []
        self.n_segments = 0
        self._by_path: dict[str, RecordingChannel] = {}
        self._readers: dict[str, FastChannelReader] = {}
        self._family_size: dict[int, int] = {}  # family id -> channels with a direct reader
        self._slow_cache: dict[str, np.ndarray] = {}
        self._mixed = False
        tdms, fh, warnings = _open_nptdms(self.path, self.size)
        try:
            objects = self._read_metadata(tdms)
            self._make_readers(tdms, objects)
            self._mixed = _mixed_layout(tdms)
            try:
                self.n_segments = len(tdms._reader._segments)
            except Exception:
                self.n_segments = 0
        finally:
            tdms.close()
            if fh is not None:
                fh.close()
        for w in warnings:
            if w not in self.warnings:
                self.warnings.append(w)
        self._stamp = stamp

    def _read_metadata(self, tdms) -> list:
        """Fill properties and channels. Returns the npTDMS channel object of each channel."""
        self.properties = dict(tdms.properties)
        objects = []
        for g in tdms.groups():
            self.group_properties[g.name] = dict(g.properties)
            for c in g.channels():
                try:
                    dtype = np.dtype(c.dtype)
                except Exception:
                    dtype = np.dtype("V8")
                ch = RecordingChannel(g.name, c.name, c.path, len(c), dtype, dict(c.properties),
                                      numeric=dtype.kind in NUMERIC_KINDS)
                self.channels.append(ch)
                self._by_path.setdefault(ch.path, ch)
                objects.append(c)
        return objects

    def _make_readers(self, tdms, objects: list) -> None:
        """Verified direct readers of all numeric channels (one layout scan)."""
        want = [(ch, obj) for ch, obj in zip(self.channels, objects) if ch.numeric and ch.length]
        if not want:
            return
        try:
            fd = os.open(self.path, _O_READ)
        except OSError as exc:
            _log.debug("direct reader off: %s", exc)
            return
        try:
            made = fastread.make_fast_readers(tdms, [obj for _ch, obj in want], fd, nptdms_read)
        except Exception as exc:  # any doubt -> npTDMS
            _log.debug("direct reader off: %s", exc)
            made = [exc] * len(want)
        finally:
            os.close(fd)
        for (ch, _obj), r in zip(want, made):
            if isinstance(r, FastChannelReader):
                r.bind(-1)
                self._readers[ch.path] = r
                self._family_size[r.family] = self._family_size.get(r.family, 0) + 1
                ch.fast = True
            else:
                _log.debug("direct reader off for %s: %s", ch.path, r)

    def _reopen(self) -> None:
        """The file changed on disk: parse and verify it again."""
        _log.info("%s changed on disk: opened again", self.path)
        self._open()

    # -- metadata -----------------------------------------------------------------

    def find(self, group: str, name: str) -> RecordingChannel | None:
        """The channel of a group, or None."""
        for ch in self.channels:
            if ch.group == group and ch.name == name:
                return ch
        return None

    # -- values -------------------------------------------------------------------

    def channel(self, path: str) -> RecordingChannel:
        """The numeric channel at path; KeyError or ValueError otherwise."""
        ch = self._by_path.get(path)
        if ch is None:
            raise KeyError(path)
        if not ch.numeric:
            raise ValueError(f"{ch.label}: values are not numeric ({ch.dtype})")
        return ch

    def _fresh(self) -> None:
        """Parse the file again if it changed on disk since the last parse."""
        if self._file_stamp() != self._stamp:
            self._reopen()

    def read(self, paths) -> dict[str, np.ndarray]:
        """float64 values of whole channels, by path: {path: values}.

        Channels of one family (same layout) are read in one pass over the
        file. Raises KeyError for an unknown path, ValueError for a channel
        that is not numeric, OSError or FastPathError on file problems.
        """
        paths = list(dict.fromkeys(paths))
        if not paths:
            return {}
        self._fresh()
        out: dict[str, np.ndarray] = {}
        fast: list[str] = []
        slow: list[str] = []
        for p in paths:
            ch = self.channel(p)
            if ch.length == 0:
                out[p] = np.empty(0, dtype=np.float64)
            elif p in self._readers:
                fast.append(p)
            elif p in self._slow_cache:
                out[p] = self._slow_cache[p]
            else:
                slow.append(p)
        if fast:
            out.update(self._read_direct(fast))
        if slow:
            out.update(self._read_nptdms(slow))
        return out

    def read_one(self, path: str) -> np.ndarray:
        """float64 values of one channel."""
        return self.read([path])[path]

    def read_range(self, path: str, start: int, stop: int) -> np.ndarray:
        """float64 values [start, stop) of one channel, clamped to the channel."""
        ch = self.channel(path)
        start = max(0, start)
        stop = min(ch.length, stop)
        if stop <= start:
            return np.empty(0, dtype=np.float64)
        self._fresh()
        reader = self._readers.get(path)
        if reader is None:
            return self.read_one(path)[start:stop]
        fd = os.open(self.path, _O_READ)
        try:
            reader.bind(fd)
            return _as_f64(reader.read(start, stop))
        finally:
            reader.bind(-1)
            os.close(fd)

    def iter_blocks(self, path: str, block: int) -> Iterator[np.ndarray]:
        """Yield the float64 values of one channel in order, `block` values per array.

        The last array is shorter. A channel with a direct reader is read
        block by block (RAM use: one block); other channels are read whole
        first. The file stays open while the generator runs.
        """
        ch = self.channel(path)
        n = ch.length
        block = max(1, int(block))
        if n == 0:
            return
        self._fresh()
        reader = self._readers.get(path)
        if reader is None:
            whole = self.read_one(path)
            for i in range(0, n, block):
                yield whole[i:i + block]
            return
        fd = os.open(self.path, _O_READ)
        try:
            reader.bind(fd)
            for i in range(0, n, block):
                out = np.empty(min(block, n - i), dtype=reader.dtype)
                reader.read_into(out, i)
                yield _as_f64(out)
        finally:
            reader.bind(-1)
            os.close(fd)

    def _read_direct(self, paths: list[str]) -> dict[str, np.ndarray]:
        """Direct reads, one pass per family.

        Fragmented channels (small pieces) are always read together. Large
        pieces are read channel by channel, except when at least half of
        the channels of a family are wanted: then the file is read in
        order, chunk by chunk (read_many bulk mode), which is faster on a
        cold cache and on disks that seek slowly.
        """
        readers = [self._readers[p] for p in paths]
        outs: list = [None] * len(readers)
        families: dict[int, list[int]] = {}
        for i, r in enumerate(readers):
            families.setdefault(r.family, []).append(i)
        fd = os.open(self.path, _O_READ)
        try:
            for r in readers:
                r.bind(fd)
            for fam, idx in families.items():
                group = [readers[i] for i in idx]
                first = group[0]
                bulk = (not first.fragmented and 2 * len(group) >= self._family_size[fam]
                        and 0 < first.max_chunk_stride <= fastread._BLOCK_BYTES)
                bufs = [np.empty(r.length, dtype=r.dtype) for r in group]
                fastread.read_many(group, 0, first.length, bufs, bulk=bulk)
                for i, b in zip(idx, bufs):
                    outs[i] = b
        finally:
            for r in readers:
                r.bind(-1)
            os.close(fd)
        return {p: _as_f64(b) for p, b in zip(paths, outs)}

    def _read_nptdms(self, paths: list[str]) -> dict[str, np.ndarray]:
        """npTDMS reads of channels without a direct reader (one open per call)."""
        tdms, fh, _warnings = _open_nptdms(self.path, self.size)
        try:
            if self._mixed:
                self._chunk_pass(tdms)
                return {p: self._slow_cache[p] for p in paths}
            out = {}
            for p in paths:
                ch = self._by_path[p]
                out[p] = _as_f64(np.asarray(_nptdms_values(tdms[ch.group][ch.name], ch.length)))
            return out
        finally:
            tdms.close()
            if fh is not None:
                fh.close()

    def _chunk_pass(self, tdms) -> None:
        """One pass over the file: all numeric channels without a direct reader go to _slow_cache."""
        slow = [ch for ch in self.channels if ch.numeric and ch.length and ch.path not in self._readers]
        arrays = {ch.path: np.empty(ch.length, dtype=np.float64) for ch in slow}
        filled = dict.fromkeys(arrays, 0)
        by_key = {(ch.group, ch.name): ch.path for ch in slow}
        for chunk in tdms.data_chunks():
            for g in chunk.groups():
                for cc in g.channels():
                    p = by_key.get((g.name, cc.name))
                    if p is None or not len(cc):
                        continue
                    a = np.asarray(cc[:])
                    arr = arrays[p]
                    off = int(cc.offset)
                    if off + a.size > arr.size:
                        raise IOError(f"{p}: npTDMS gave more values than the channel length {arr.size}")
                    arr[off:off + a.size] = a
                    filled[p] = off + a.size
        for p, n in filled.items():
            if n != arrays[p].size:
                raise IOError(f"{p}: short read, {n} of {arrays[p].size} values")
        self._slow_cache.update(arrays)


# -- lazy cache --------------------------------------------------------------------------

class LazyChannelCache(MutableMapping):
    """Mapping name -> float64 array that loads a channel on first access.

    Behaves like the dict it replaces: ``cache[name]``, ``cache.get(name)``,
    ``name in cache``, ``len``, iteration and ``items()`` see every channel,
    loaded or not. Values can be replaced (``cache[name] = array``) and
    new names added; these never touch the file.

    loader(names) returns {name: array} for channels not yet in memory.
    get() returns the default when a load fails (the error is logged);
    ``cache[name]`` raises it. on_change(name) is called when a value is
    replaced or removed (not when a channel is loaded from the file).
    """

    def __init__(self, names, loader: Callable[[list[str]], dict[str, np.ndarray]],
                 on_change: Callable[[str], None] | None = None):
        self._loader = loader
        self._on_change = on_change  # called with the name when a value is replaced or removed
        self._order: list[str] = list(dict.fromkeys(names))  # file order, then added names
        self._pending: set[str] = set(self._order)
        self._data: dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self._data[name]
        except KeyError:
            if name not in self._pending:
                raise
        self.load([name])
        return self._data[name]

    def __setitem__(self, name: str, value) -> None:
        if name not in self._data and name not in self._pending:
            self._order.append(name)
        self._pending.discard(name)
        self._data[name] = value
        if self._on_change is not None:
            self._on_change(name)

    def __delitem__(self, name: str) -> None:
        if name not in self._data and name not in self._pending:
            raise KeyError(name)
        self._data.pop(name, None)
        self._pending.discard(name)
        self._order.remove(name)
        if self._on_change is not None:
            self._on_change(name)

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._order))

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, name) -> bool:
        return name in self._data or name in self._pending

    def get(self, name, default=None):
        """Value of a channel, or default if it is unknown or cannot be loaded."""
        try:
            return self[name]
        except KeyError:
            return default
        except Exception:
            _log.exception("channel %r could not be loaded", name)
            return default

    def loaded(self, name: str) -> bool:
        return name in self._data

    def load(self, names) -> None:
        """Load the given channels that are not in memory yet (one loader call)."""
        todo = [n for n in dict.fromkeys(names) if n in self._pending]
        if not todo:
            return
        got = self._loader(todo)
        for n in todo:
            self._data[n] = got[n]
            self._pending.discard(n)

    def load_all(self) -> None:
        self.load(list(self._pending))

    def values(self):
        self.load_all()
        return super().values()

    def items(self):
        self.load_all()
        return super().items()

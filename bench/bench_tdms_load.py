#!/usr/bin/env python3
"""Benchmark: open a TDMS recording in the Data Fitting controller.

Writes a synthetic recording like DAQUniversal does (Time group, one
device group with N float64 channels, one segment per second of data),
then measures:
    - old: npTDMS TdmsFile.read + float64 copy of every channel (the
      loader before fitting.tdms_source)
    - load: DataFittingController.load_recording (metadata, verified
      readers, Time channel)
    - preview: load + the X and Y channel (what the tab shows after Load)
    - all: load + every channel
Warm page cache, and cold (posix_fadvise DONTNEED) when --cold is given.
Output is markdown on stdout, progress on stderr.

Example:
    QT_QPA_PLATFORM=offscreen python3 bench/bench_tdms_load.py --rate 10000 --cold
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import nptdms  # noqa: E402
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter  # noqa: E402

from fitting.tab import DataFittingController  # noqa: E402

perf = time.perf_counter


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def write_recording(path: str, nch: int, n: int, rate: float) -> int:
    """nch float64 channels + Time, n samples, one segment per second. Returns segments."""
    per_seg = max(1, int(rate))
    nseg = -(-n // per_seg)
    t = np.arange(n, dtype=np.float64) / rate
    rng = np.random.default_rng(1)
    with TdmsWriter(path) as w:
        for k in range(nseg):
            a, b = k * per_seg, min(n, (k + 1) * per_seg)
            objs = []
            if k == 0:
                objs += [RootObject({"Sample_Rate_Hz": rate, "DAQ-U_Version": "bench"}),
                         GroupObject("Time"), GroupObject("cDAQ1Mod1")]
            objs.append(ChannelObject("Time", "Time", t[a:b], {"Unit": "s"} if k == 0 else None))
            block = rng.standard_normal((nch, b - a))
            for c in range(nch):
                props = {"Scale_Factor": 1.0, "Offset": 0.0, "Unit": "V"} if k == 0 else None
                objs.append(ChannelObject("cDAQ1Mod1", f"AI{c}", block[c], props))
            w.write_segment(objs)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return nseg


def evict(path: str) -> None:
    """Drop the file from the page cache (best effort)."""
    if not hasattr(os, "posix_fadvise"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def old_loader(path: str) -> dict:
    """The loader before fitting.tdms_source: everything in RAM with npTDMS."""
    out = {}
    with TdmsFile.read(path) as f:
        out["Time"] = np.asarray(f["Time"]["Time"][:], dtype=float)
        for g in f.groups():
            for ch in g.channels():
                if ch.name.lower() != "time":
                    out[ch.name] = np.asarray(ch[:], dtype=float)
    return out


def timed(fn, repeat: int, cold_path: str | None = None) -> list[float]:
    out = []
    for _ in range(repeat):
        gc.collect()
        if cold_path:
            evict(cold_path)
        t0 = perf()
        fn()
        out.append(perf() - t0)
    return out


def best(times: list[float]) -> float:
    """Best of N: the machine adds stalls of about 1.5 s to some runs (page zeroing), not the code."""
    return float(np.min(times))


def fmt(times: list[float]) -> str:
    return f"{best(times) * 1e3:.0f} ms" if times else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--samples", type=int, default=2_400_000, help="samples per channel")
    ap.add_argument("--rate", type=float, default=10_000.0, help="samples per second (one segment per second)")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--cold", action="store_true", help="also measure with a cold page cache")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="bench_tdms_", dir=args.workdir)
    path = os.path.join(workdir, "bench.tdms")
    log(f"writing {args.channels} x {args.samples} float64 + Time at {args.rate:g} S/s ...")
    nseg = write_recording(path, args.channels, args.samples, args.rate)
    size_mb = os.path.getsize(path) / 1e6
    log(f"{size_mb:.0f} MB, {nseg} segments")

    def load_only():
        c = DataFittingController(None)
        ok, msg = c.load_recording(path)
        assert ok, msg
        return c

    def load_preview():
        c = load_only()
        c.preload(("AI0", "AI1"))
        assert c.channel_cache.loaded("AI0") and c.channel_cache.loaded("AI1")

    def load_all():
        c = load_only()
        c.channel_cache.load_all()

    rows = []
    modes = [("warm", None)] + ([("cold", path)] if args.cold else [])
    for label, cold in modes:
        log(f"{label}: old loader")
        t_old = timed(lambda: old_loader(path), args.repeat, cold)
        log(f"{label}: load_recording")
        t_load = timed(load_only, args.repeat, cold)
        log(f"{label}: load + preview channels")
        t_prev = timed(load_preview, args.repeat, cold)
        log(f"{label}: load + all channels")
        t_all = timed(load_all, args.repeat, cold)
        rows.append((label, t_old, t_load, t_prev, t_all))

    c = load_only()
    fast = sum(ch.fast for ch in c.recording.channels)
    print(f"File: {args.channels} channels x {args.samples} samples float64 + Time, "
          f"{size_mb:.0f} MB, {nseg} segments ({args.rate:g} S/s, 1 segment/s), "
          f"{fast}/{len(c.recording.channels)} channels on the direct reader")
    print(f"npTDMS {nptdms.__version__}, numpy {np.__version__}, Python {platform.python_version()}, "
          f"{os.cpu_count()} CPUs, best of {args.repeat}")
    print()
    print("| Page cache | Old: TdmsFile.read, all channels | Load (metadata + Time) | Load + X, Y (preview) | Load + all channels | Gain, preview | Gain, all |")
    print("|---|---|---|---|---|---|---|")
    for label, t_old, t_load, t_prev, t_all in rows:
        g_prev = best(t_old) / best(t_prev)
        g_all = best(t_old) / best(t_all)
        print(f"| {label} | {fmt(t_old)} | {fmt(t_load)} | {fmt(t_prev)} | {fmt(t_all)} | {g_prev:.1f}x | {g_all:.1f}x |")
    os.remove(path)
    os.rmdir(workdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

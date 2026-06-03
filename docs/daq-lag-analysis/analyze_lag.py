#!/usr/bin/env python3
"""Trace data-acquisition lag in DAQ-Universal runtime telemetry.

DAQ-Universal writes two JSON-Lines streams per measurement session:

* ``*runtime_telemetry_*.jsonl`` - one performance snapshot per ~second,
  plus a ``# session=...`` header line with the run configuration.
* ``*raw_data_*.jsonl``          - the structured event log (one event per
  line), including the app's own ``Slow GUI frame`` warnings.

This script reads a directory of those files and reconstructs *where* the
lag comes from. The headline result for the 2026-06-03 dumps was a fixed
~5 s periodic task that blocks the Qt GUI thread for ~2.3 s (median), with
~98 % of that time spent *blocked* rather than painting - which both
freezes the UI and stalls the acquisition->display pipeline.

Usage
-----
    python analyze_lag.py /path/to/telemetry_dir [--plot out.png]

The directory is scanned for ``*runtime_telemetry*.jsonl`` and
``*raw_data*.jsonl`` files. ``--plot`` requires matplotlib + numpy.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import Counter
from datetime import datetime

SLOW_RE = re.compile(
    r"Slow GUI frame ([\d.]+) ms.*?queue=([\d.]+).*?latest=([\d.]+)"
    r".*?prepApply=([\d.]+).*?setData=([\d.]+).*?markers=([\d.]+).*?yRange=([\d.]+)"
)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _pct(values, p):
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return st.mean(vals) if vals else 0.0


def _max(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return max(vals) if vals else 0.0


def load_telemetry(path):
    """Return (session_config_dict, [snapshot_dicts])."""
    cfg, rows = {}, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# session="):
                cfg = json.loads(line.split("# session=", 1)[1])
            elif line.startswith("#"):
                continue
            else:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return cfg, rows


def load_events(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def parse_slow_frames(events):
    """Extract (timestamp, total_ms, accounted_ms) for every Slow GUI frame."""
    slow = []
    for ev in events:
        msg = ev.get("message") or ""
        if "Slow GUI frame" not in msg:
            continue
        m = SLOW_RE.search(msg)
        if m:
            total = float(m.group(1))
            accounted = sum(float(m.group(i)) for i in range(2, 8))
            slow.append((_ts(ev["timestamp"]), total, accounted))
    return slow


def analyze_session(cfg, rows, label):
    print(f"\n{'=' * 74}\nSESSION {label}  ({len(rows)} snapshots)\n{'=' * 74}")
    if rows:
        dur = (_ts(rows[-1]["ts"]) - _ts(rows[0]["ts"])).total_seconds()
        print(f"  duration: {dur:.0f}s ({dur / 60:.1f} min)")
    print(
        "  config: rate={eff} Hz  avg={avg}  read={rd} Hz  write={wr} Hz  "
        "channels={ch}  plot_tail={tail:.0f}s".format(
            eff=cfg.get("effective_sample_rate_hz"),
            avg=cfg.get("pipeline_average_samples"),
            rd=cfg.get("read_rate_hz"),
            wr=cfg.get("write_rate_hz"),
            ch=cfg.get("active_channel_count"),
            tail=cfg.get("plot_tail_seconds_effective", 0) or 0,
        )
    )

    # --- lag + pipeline depth ---
    def col(k):
        return [r.get(k) for r in rows]

    print("  lag_s            p50/p95/max = "
          f"{_pct(col('lag_s'), 50):.2f} / {_pct(col('lag_s'), 95):.2f} / {_max(col('lag_s')):.2f}")
    print("  lag_ms_p95       p50/p95/max = "
          f"{_pct(col('lag_ms_p95'), 50):.0f} / {_pct(col('lag_ms_p95'), 95):.0f} / "
          f"{_pct(col('lag_ms_p95'), 100):.0f}")
    print("  redraw_ms_p95    p50/p95/max = "
          f"{_pct(col('redraw_ms_p95'), 50):.0f} / {_pct(col('redraw_ms_p95'), 95):.0f} / "
          f"{_pct(col('redraw_ms_p95'), 100):.0f}")

    peak = lambda k: max(((r.get("peak_q") or {}).get(k, 0) or 0) for r in rows) if rows else 0
    print(f"  peak queue depth: qd={peak('qd')} tdms={peak('tdms')} gui={peak('gui')} "
          f"process={peak('process')} cache={peak('cache')}")

    # --- is data actually lost? ---
    gd = max((r.get("gui_drops", 0) or 0) for r in rows) if rows else 0
    cd = max((r.get("cache_drops", 0) or 0) for r in rows) if rows else 0
    wh = Counter(r.get("writer_health") for r in rows)
    print(f"  DROPS: gui_drops={gd} cache_drops={cd}   "
          f"(0 = no data loss, lag is recoverable backlog)")
    print(f"  writer_health: {dict(wh)}")


def analyze_freezes(slow, label):
    if not slow:
        print(f"\n[{label}] no 'Slow GUI frame' events found")
        return
    totals = [s[1] for s in slow]
    unacc = [s[1] - s[2] for s in slow]
    gaps = [(slow[i][0] - slow[i - 1][0]).total_seconds() for i in range(1, len(slow))]
    span = (slow[-1][0] - slow[0][0]).total_seconds() or 1
    # wall-clock phase alignment (peaked => fixed-interval timer)
    phase = Counter(round(s[0].timestamp() % 5) for s in slow)

    print(f"\n{'-' * 74}\nFREEZE ANALYSIS  [{label}]  ({len(slow)} slow frames)\n{'-' * 74}")
    print(f"  total frame time  p50={_pct(totals, 50):.0f}  p95={_pct(totals, 95):.0f}  "
          f"max={max(totals):.0f} ms")
    print(f"  UNACCOUNTED time  p50={_pct(unacc, 50):.0f}  mean={_mean(unacc):.0f} ms  "
          f"=> {sum(unacc) / sum(totals) * 100:.1f}% of freeze time is NOT plot rendering")
    print(f"  gap between freezes  p50={_pct(gaps, 50):.1f}s  "
          f"(histogram: {dict(sorted(Counter(round(g) for g in gaps if g < 12).items()))})")
    print(f"  epoch-mod-5s phase  {dict(sorted(phase.items()))}  "
          f"(single peak => fixed 5 s timer)")
    print(f"  GUI frozen ~{_mean(totals) / 1000 * len(slow) / span * 100:.0f}% of wall-clock time")


def make_plot(sessions, slow_by_label, out_path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # pick the longest session that has slow-frame data for the zoom/cadence panels
    label = max(slow_by_label, key=lambda k: len(slow_by_label[k])) if slow_by_label else None
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "DAQ-Universal data-acquisition lag trace - "
        "root cause: a ~5 s periodic task blocks the GUI thread for ~2.3 s",
        fontsize=14, fontweight="bold",
    )

    if label:
        slow = slow_by_label[label]
        t0 = slow[0][0]
        sx = np.array([(s[0] - t0).total_seconds() for s in slow])
        sy = np.array([s[1] / 1000 for s in slow])

        ax = fig.add_subplot(2, 2, 1)
        sel = (sx >= 300) & (sx <= 420)
        ax.bar(sx[sel], sy[sel], width=0.6, color="steelblue", alpha=.6,
               label="GUI freeze duration (s)")
        ax.set_title("A.  120 s window: a freeze every ~5 s")
        ax.set_xlabel("time into session (s)"); ax.set_ylabel("freeze (s)")
        ax.legend(fontsize=8); ax.grid(alpha=.3)

        ax = fig.add_subplot(2, 2, 2)
        gaps = [(slow[i][0] - slow[i - 1][0]).total_seconds()
                for i in range(1, len(slow)) if (slow[i][0] - slow[i - 1][0]).total_seconds() < 12]
        ax.hist(gaps, bins=np.arange(0, 12, 0.25), color="steelblue", edgecolor="k", lw=.3)
        ax.axvline(5.0, color="crimson", ls="--", lw=2, label="5.0 s")
        ax.set_title("B.  Time between freezes - locked to 5.0 s")
        ax.set_xlabel("gap (s)"); ax.set_ylabel("count"); ax.legend(); ax.grid(alpha=.3)

        ax = fig.add_subplot(2, 2, 3)
        acc = np.array([s[2] for s in slow]); tot = np.array([s[1] for s in slow])
        ua = tot - acc
        bars = ax.bar(["measured\nplot work", "UNACCOUNTED\n(thread blocked)"],
                      [acc.mean(), ua.mean()], color=["seagreen", "crimson"], edgecolor="k")
        for b, v in zip(bars, [acc.mean(), ua.mean()]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} ms", ha="center", va="bottom",
                    fontweight="bold")
        ax.set_title(f"C.  Per-freeze budget ({ua.sum() / tot.sum() * 100:.0f}% not rendering)")
        ax.set_ylabel("ms"); ax.grid(alpha=.3, axis="y")

    ax = fig.add_subplot(2, 2, 4)
    import numpy as np
    labels, data = [], []
    for lab, (cfg, rows) in sorted(sessions.items()):
        labels.append(f"{lab}\n{cfg.get('effective_sample_rate_hz')}Hz/avg{cfg.get('pipeline_average_samples')}")
        data.append(np.array([r.get("lag_s") or 0 for r in rows]))
    bp = ax.boxplot(data, patch_artist=True, whis=[5, 95], showfliers=False, tick_labels=labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue"); patch.set_alpha(.6)
    ax.set_title("D.  Acquisition lag by session (sample-rate makes it worse)")
    ax.set_ylabel("lag_s"); ax.grid(alpha=.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120)
    print(f"\nsaved figure -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("telemetry_dir", help="directory containing the *.jsonl dumps")
    ap.add_argument("--plot", metavar="OUT.png", help="also write the trace figure")
    args = ap.parse_args()

    tel_files = sorted(glob.glob(os.path.join(args.telemetry_dir, "*runtime_telemetry*.jsonl")))
    raw_files = sorted(glob.glob(os.path.join(args.telemetry_dir, "*raw_data*.jsonl")))
    if not tel_files:
        ap.error(f"no *runtime_telemetry*.jsonl found in {args.telemetry_dir}")

    def label_of(path):
        m = re.search(r"(\d{6})\.jsonl$", path)
        return m.group(1) if m else os.path.basename(path)

    sessions = {}
    for f in tel_files:
        cfg, rows = load_telemetry(f)
        lab = label_of(f)
        sessions[lab] = (cfg, rows)
        analyze_session(cfg, rows, lab)

    slow_by_label = {}
    for f in raw_files:
        events = load_events(f)
        lab = label_of(f)
        slow = parse_slow_frames(events)
        if slow:
            slow_by_label[lab] = slow
        analyze_freezes(slow, lab)

    if args.plot:
        make_plot(sessions, slow_by_label, args.plot)


if __name__ == "__main__":
    main()

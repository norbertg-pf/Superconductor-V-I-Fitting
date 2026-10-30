# Changes vs `master`

Scope: only the files that affect `python -m fitting` (the `fitting/` package,
plus `pyproject.toml`/`uv.lock` for the new dependency). Data files, ad-hoc
scripts, and session notes from this working copy are intentionally excluded.

## New: no full-resolution arrays in the fit flow, appended fit metadata

Measured on a 1.6 GB DAQUniversal-like recording (16 float64 channels + Time,
12 M samples each at 10 kS/s, auto AVG = 100), real tab code offscreen:

| Step | Before | After |
|---|---|---|
| Load File (metadata, preview, labels, bands) | 31.4 s, 1.35 GB RAM | 1.0 s, 0.19 GB RAM |
| Add to plot + Run Fit + autosave of the metadata | more than 22 min (not finished) | 2.1 s |

The fit results are the same to the last bit. What changed:

- `DataFittingController.transformed(name, scale, offset, avg_window)` gives
  the transformed and block-averaged channel the tab works with. It is
  computed block by block, streamed from the file when the channel is not
  in RAM, so a long recording never needs its full-resolution array. The
  last results are kept (read-only arrays) and returned again: the widget
  signal cascades of a load (31 recomputations before) now cost nothing.
  `_apply_transforms`, `compute_fit` and `_recompute_curve_from_source` use
  it. `time_array` is read on first access only; the rate label uses the
  first 4096 sample times (`sample_interval`).
- `_post_load_setup` refreshes the preview once (`_preview_suspended`), not
  six times, and the auto-resample patch sets the AVG factor before that
  one refresh instead of after a full-rate pass.
- The same-group fit metadata save appends one metadata-only segment (the
  fitted channels declared again with their `fit_*` properties and no data)
  instead of reading and rewriting the whole file. The recorded bytes are
  never touched; a `.tdms_index` next to the file gets the same segment.
  A file with a torn last segment is rewritten as before. The FitResults
  mode appends with the same checks.
- `service.estimate_point_noise`: the rolling median and the rolling median
  absolute deviation use a sorted window that slides over the array (O(n
  log w)) instead of one `np.median` per point (3 per point, O(n w)). Same
  bits as before (tests compare against the old loops); 12 s -> 0.5 s per
  fit at 120 000 points.

## New: fast TDMS loading (port of the TDMSViewer reader)

- `fitting/tdms_fastread.py` (new, ported from `TDMSViewer/tdmsviewer/fastread.py`):
  direct positional reads of channel data from the npTDMS segment table,
  verified against npTDMS before use. Additions: `bind()`, `max_chunk_stride`,
  `read_many(..., bulk=True)` (file read in order when most channels of a
  family are wanted).
- `fitting/tdms_source.py` (new): `TdmsRecording` opens a file for its metadata
  only and reads a channel when it is first used; no file handle stays open
  between reads; a file rewritten on disk (fit report) is parsed again before
  the next read; bytes after the last segment and a stale `.tdms_index` are
  ignored with a warning. `LazyChannelCache` is the dict-like channel store.
- `DataFittingController.load_recording` reads the metadata and the Time channel
  only. `channel_cache` is a `LazyChannelCache` (same dict interface). New:
  `preload(names)` (one pass for several channels), `clear()`, `recording`,
  `load_warnings`. The load message shows the first warning.
- Channels that are not numeric (strings, timestamps) are skipped; before, one
  such channel made the whole load fail.
- `_auto_resample_patch` takes `Sample_Rate_Hz` from the kept root properties
  (no second parse of the file).
- `pyproject.toml`: `nptdms>=1.11,<1.14` (the reader uses npTDMS internals;
  tested with 1.11 and 1.13), `test` extra (`pytest`). New `tests/` (ported
  reader tests, source and controller tests, offscreen GUI test) and
  `bench/bench_tdms_load.py`.

Measured with `bench/bench_tdms_load.py`: 16 float64 channels + Time, 2.4 M
samples each (328 MB), one segment per second, 4-core VM, best of 5.
"Preview" is what `Load File…` shows (Time, X and Y channel).

| Segments | Page cache | Old loader (all channels) | Load (metadata + Time) | Load + X, Y (preview) | Load + all channels |
|---|---|---|---|---|---|
| 240 (10 kS/s) | warm | 211 ms | 25 ms | 43 ms (4.9x) | 171 ms (1.2x) |
| 240 (10 kS/s) | cold | 443 ms | 75 ms | 129 ms (3.4x) | 373 ms (1.2x) |
| 2400 (1 kS/s) | warm | 389 ms | 132 ms | 163 ms (2.4x) | 283 ms (1.4x) |
| 2400 (1 kS/s) | cold | 866 ms | 485 ms | 638 ms (1.4x) | 745 ms (1.2x) |

The gain grows with the channel count and the file size: the old loader read
and kept every channel; the new one reads only the channels in use. The cost
that stays is the npTDMS metadata parse, about 50 ms per 1000 segments.

## New: Plateau R calculation tab

- `fitting/plateau_tab.py` (new file) adds a second tab, "Plateau R
  calculation," wired up alongside the existing Ic-fitting tab in
  `fitting/standalone.py`.
- Independent TDMS loader (same file format as the Ic-fitting tab, separate
  state) plotting voltage/current channels vs time.
- Draggable time windows report point count and average V / I / R over the
  window (`fitting/service.py: window_average_stats`).
- Auto-detection of stable current plateaus (`find_current_plateaus`) using a
  peak-to-peak current-band / min-duration / min-current heuristic.
- Sparse-point Ic/n fit across detected plateaus (`fit_reduced_ic`,
  `ReducedFitResult`) — solves V0, R, Ic, n simultaneously via `curve_fit`
  against a handful of plateau-averaged (I, V) points, distinct from the
  dense-ramp Step 1-5 pipeline used by `run_full_fit`.

## New: global plot font-size setting

- Settings dialog gets a "Font size" spinbox (`data_fit_font_spin`,
  `fitting/tab.py`) applying one shared point size to tick labels, axis
  titles, and legends across every plot in both tabs
  (`apply_plot_font_size` in `fitting/extras.py`).
- Persisted as `FitPreset.plot_font_pt`.

## New dependency

- `nidaqmx>=1.6.0` added to `pyproject.toml` (and `uv.lock` regenerated).

## Behavior/default changes

- `auto_load_after_acquisition` (the "Auto-load fitted recording" checkbox)
  now defaults to **off**. Previously, loading a TDMS with a saved fit would
  auto-replay it alongside a freshly plotted curve, producing two
  visually-identical "Curve label" entries that were easy to mix up. Fits are
  now generated fresh each time by default; the checkbox opts back into
  replay.
- Auto-trim quench margin reduced from 2% to 1% of points before the
  detected drop.
- "Load File..." always starts from a clean slate (clears previously loaded
  books/curves), rather than the old first-load-vs-subsequent-load branching
  that could silently accumulate multiple books.
- Current-channel auto-guess is now case-insensitive with a revised
  candidate list (`current`, `imon`, `dcct`, `_i`, `ai0`, `ic`).
- Added `Voltage_Tab_Distance_cm` to the recognized voltage-tap-distance
  metadata keys.

## Log-log Ic/n fit correctness

- The Ic/n regression window is now built directly from
  `pick_loglog_i_window_from_thresholds` (walking back from the Ec2 crossing
  to the nearest Ec1 point) instead of a plain `Ec1 <= E <= Ec2` threshold
  mask. The old mask could pick up an earlier, disconnected noise-driven
  excursion above Ec1 elsewhere in the ramp and mix unrelated points into the
  regression alongside the true transition segment.

## Multi-curve fit fixes

- Editing Step 1-5 windows now broadcasts to every plotted curve's stored
  profile instead of only the currently-active curve's.
- Step 2 trim (start/quench-tail) is now read per-curve from each curve's own
  stored profile during a multi-curve Run Fit, instead of every curve being
  trimmed with whichever curve happens to be active in the selector.
- Ec1/Ec2 (the IEC criterion window) are now applied uniformly from the
  visible Settings boxes to every curve on Run Fit, since that window is
  meant to be shared rather than per-curve.
- Newly added/updated curves now become the active "Curve label" selection
  (`_select_curve_profile`), so Step 1-5 edits made right after "Add to plot"
  land on the curve just added.
- Fixed a stale-result bug in `_resolve_fit_parent_and_result` where an
  unrelated cached fit could be picked up ahead of the controller's own
  `last_result`.
- The Step-5 (Ec1/Ec2) preview band now updates within the same "Run Fit"
  click instead of lagging one click behind.

## Removed

- The `vc` (criterion voltage) field is no longer captured/restored as part
  of a curve's saved fit-window profile.

## UI/rendering fixes

- Fixed a grid-overlay repaint bug (`_PlotGridOverlay`) where panning/zooming
  could leave stale gridline fragments from earlier zoom states —
  `prepareGeometryChange()` is now called before the view-driven redraw.
- Small content margins added to the main and residual plots so axis border
  lines aren't clipped at certain DPI scalings.

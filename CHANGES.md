# Changes vs `master`

Scope: only the files that affect `python -m fitting` (the `fitting/` package,
plus `pyproject.toml`/`uv.lock` for the new dependency). Data files, ad-hoc
scripts, and session notes from this working copy are intentionally excluded.

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

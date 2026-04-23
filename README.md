# Superconductor V-I fitting (Data Fitting tab)

Self-contained package implementing the Data Fitting tab used in DAQUniversal:
three-step V-I curve analysis (`dI/dt` → linear baseline → power-law `Ic`, `n`),
an interactive Qt UI, an OriginLab-style graph-settings dialog, and an
IEC 61788-compliant report exporter for inter-lab comparison.

## Files

| File | Purpose |
|------|---------|
| `service.py` | Pure math (numpy / scipy). No Qt imports. |
| `extras.py` | Graph-settings dialog, export dialog, presets, widgets. |
| `tab.py` | Qt layout and action functions (uses `service.py` + `extras.py`). |
| `standalone.py` | Minimal `QMainWindow` that wraps the tab for standalone use. |
| `__main__.py` | `python -m fitting` entry point. |

## Install (Windows PowerShell)

```powershell
cd "D:\Superconductor V-I Fitting"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

## Run standalone

```
python -m fitting
```

## Embedding in another Qt app

```python
from types import SimpleNamespace
from PyQt5.QtWidgets import QWidget
from fitting import setup_data_fitting_tab_layout, tab as _tab

host.ui_state = SimpleNamespace()
host.ui_state.data_fitting_tab = QWidget()    # your tab container
host.data_fitting_refresh_preview = lambda *_: _tab.refresh_preview(host)
# ...plus the other `data_fitting_*` methods (see standalone.DataFittingWindow).
setup_data_fitting_tab_layout(host)
```

## Fitting model

IEC 61788 power-law criterion:

* Without sample length: `V = L·dI/dt + R·I + Vc·(I/Ic)^n`
* With sample length `Ls`: `E = (L/Ls)·dI/dt + ρ·I + Ec·(I/Ic)^n`
  (the tab divides Y by `Ls` before fitting).

`Ic` is defined as the current at which the **baseline-subtracted** voltage
(`V − V0 − R·I`) equals the criterion. `n` is the exponent of the power law
and is reported with a ±1σ uncertainty from the fitter covariance matrix.

## IEC 61788 mode (default)

When the *IEC 61788 mode* checkbox is on (the default), the n-value fit window
is set in **multiples of the criterion** on the baseline-subtracted voltage,
per IEC 61788-1/-2/-3/-14/-21:

* Lower edge: `vc_low_mult · Vc` (default `0.1·Vc`)
* Upper edge: `vc_high_mult · Vc` (default `10·Vc`)

The current bounds that correspond to those voltage levels are found by
linear interpolation of the measured trace, so the fit window does not
depend on `Imax` or `Vmax` — two labs with different ramp endpoints but
identical samples now produce the same Ic and n.

When *Log-log n fit* is on (the default), the n-value is obtained from a
linear regression of `log10(V − V0 − R·I)` vs `log10(I)` over that window
(the IEC reference method). Turn it off to get the NLS n instead.

Disabling IEC mode reverts to the legacy "fraction of Imax / fraction of
Vmax" window; it is kept for backwards compatibility but is not IEC-compliant.

## Named IEC 61788 presets

The *IEC 61788 preset* dropdown applies criterion and window settings from:

* IEC 61788-2 (Nb-Ti, `Ec = 0.1 µV/cm`)
* IEC 61788-3 (Bi-2212 / Bi-2223, `Ec = 1 µV/cm`)
* IEC 61788-14 (REBCO coated conductors, `Ec = 1 µV/cm`)
* IEC 61788-19 (Nb3Sn, `Ec = 0.1 µV/cm`)
* IEC 61788-21 (HTS cables, `Ec = 1 µV/cm`)
* Whole magnet / coil (`Vc = 1 mV`, user to override with coil-specific Vc)

Selecting a preset resets the sample length to 1 cm — enter the real
voltage-tap spacing afterwards.

## Reported uncertainties and fit quality

Every successful fit reports, in both the on-screen parameter table and the
JSON fit-report export:

* `Ic ± σ(Ic)` and `n ± σ(n)` from the fitter covariance matrix
* `V0 ± σ(V0)`, `R ± σ(R)`, baseline `R²`, baseline residual RMS (noise floor)
* `χ²` and `χ²_reduced = χ² / dof`
* fit window in both current (A) and baseline-subtracted voltage (V or V/cm)
* number of points in the n-fit window and decades above `Vc`
* ramp rate `dI/dt`, estimated inductance `L`, ramp direction
* a list of guard warnings — noise floor vs the lower Vc edge, decades above
  Vc, too-few points, `n` hitting the solver bound, baseline overlap with the
  transition, ramp not in the expected direction.

## Inter-lab reports

Click **Save report (JSON)…** after a fit to export a self-contained report
with criterion, sample length, channel names, every window, all fit
parameters with ±1σ uncertainties, `χ²_reduced`, guard warnings, software
version, and ramp metadata. This file is intended to travel with the raw
V-I trace for VAMAS / IEC round-robin comparisons.

## Public API

```python
from fitting import (
    FitResult, FitSettings, run_full_fit, robust_view_range,
    setup_data_fitting_tab_layout,
)
```

`run_full_fit(t, x, y, settings)` returns a `FitResult` without any Qt
dependencies, so the math layer can be used in scripts and tests. Set
`settings.iec_mode = True` (the default) for IEC 61788-compliant fitting.

# Superconductor V-I fitting (Data Fitting tab)

Self-contained package implementing the Data Fitting tab used in DAQUniversal:
three-step V-I curve analysis (`dI/dt` → linear baseline → power-law `Ic`, `n`),
an interactive Qt UI, and an OriginLab-style graph-settings dialog.

## Files

| File | Purpose |
|------|---------|
| `service.py` | Pure math (numpy / scipy). No Qt imports. |
| `extras.py` | Graph-settings dialog, export dialog, presets, widgets. |
| `tab.py` | Qt layout and action functions (uses `service.py` + `extras.py`). |
| `standalone.py` | Minimal `QMainWindow` that wraps the tab for standalone use. |
| `__main__.py` | `python -m src.fitting` entry point. |

## Install (Windows PowerShell)

```powershell
cd "D:\Superconductor V-I Fitting"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .```

## Run Standalon

python -m fitting

## Embedding in another Qt app

```python
from types import SimpleNamespace
from PyQt5.QtWidgets import QWidget
from src.fitting import setup_data_fitting_tab_layout, tab as _tab

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

## Cross-lab consistency checklist (recommended)

To compare `Ic` and `n` against other labs, align these settings first:

1. Same criterion definition (`Vc` in V, or `Ec` in µV/cm with same sample length).
2. Same current ramp region for `dI/dt` estimate.
3. Same linear window for `R`/`ρ` baseline.
4. Same power-law window (`I` low limit and high `V` fraction).
5. Same preprocessing (averaging/windowing, scale/offset handling).
6. Report fit quality together with `Ic`/`n`:
   - chi-squared
   - RMSE
   - R²
   - number of samples used in the power-law window

The service now returns these quality metrics in `FitResult` so they can be
logged and compared across laboratories.

## Public API

```python
from src.fitting import (
    FitResult, FitSettings, run_full_fit, robust_view_range,
    setup_data_fitting_tab_layout,
)
```

`run_full_fit(t, x, y, settings)` returns a `FitResult` without any Qt
dependencies, so the math layer can be used in scripts and tests.

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
pip install -e .
```

## Run Standalone

python -m fitting

The app now shows a gray version label at the bottom-right:
`v1.0 (build N)` where `N` follows the git commit count.

## Build one-file Windows EXE (standalone, no install)

```powershell
pip install pyinstaller
pyinstaller --clean --noconfirm --distpath build superconductorfit.spec
```

Output file:

- `build\Superconductor fitting v1.0 build N.exe` (single-file executable)

Copy this `.exe` to any Windows machine and run it directly.

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

## Public API

```python
from src.fitting import (
    FitResult, FitSettings, run_full_fit, robust_view_range,
    setup_data_fitting_tab_layout,
)
```

`run_full_fit(t, x, y, settings)` returns a `FitResult` without any Qt
dependencies, so the math layer can be used in scripts and tests.

# Superconductor V–I Fitting

Self-contained Python/Qt tool that extracts the **critical current `Iₒ`**, the
**n-value** and the **resistive baseline** from a recorded voltage–current
ramp using the IEC 61788 power-law criterion. It bundles a pure-numpy/scipy
math layer, an interactive PyQt5 UI with an OriginLab-style graph-settings
dialog, presets, multi-curve management, and a per-fit TDMS side-car report.

![Anatomy of a V–I curve](docs/v_i_curve_anatomy.png)

---

## Overview

The fitted relation is the IEC 61788 power-law model with an inductive term
and a resistive baseline:

- Without sample length: **`V = V_ofs + L·dI/dt + R·I + V_c·(I/I_c)ⁿ`**
- With sample length `L_s`: **`E = (L/L_s)·dI/dt + ρ·I + E_c·(I/I_c)ⁿ`**
  (the tab divides Y by `L_s` before fitting)

Three additive parts are extracted from the curve:

1. **Offset / inductive** — `V_ofs + L·dI/dt`, estimated from the linear
   ramp region.
2. **Linear baseline** — `R·I` (or `ρ·I`), fit on points well below `I_c`.
3. **Power-law transition** — `V_c·(I/I_c)ⁿ`, extracted from the
   baseline-subtracted residual on the IEC decade window.

---

## The two fitting methods

| | **Log–log linear** (default — IEC 61788) | **Non-linear (Levenberg–Marquardt)** |
|---|---|---|
| **What it does** | Subtracts the linear baseline first (`V′ = V − V_ofs − R·I`), then fits `log V′` vs `log I` with a straight line on the IEC decade window `[E_c1, E_c2]`. Slope = `n`; `I_c` at `V′ = E_c2`. | Fits the full coupled model directly to the raw V–I data inside an outer self-consistency loop on `I_c`. |
| **Pros** | Reference method per IEC 61788 → reproducible across labs. Closed-form linear regression: very fast, no convergence issues. Robust against multiplicative noise on V. | Returns proper σ on every parameter from the covariance matrix. Handles non-negligible inductive / baseline coupling natively. Uses the entire ramp, not only the decade window. |
| **Cons** | Sensitive to errors in the pre-subtracted baseline `(V_ofs, R)`. Only the data inside the decade is used. Needs ≥ a few dozen samples between `E_c1` and `E_c2`. | Slower; can fail to converge if windows or seeds are poor. Baseline and transition compete in the residual. No standardized acceptance criterion. |

> **Recommendation:** start with **log–log linear** because that is the
> IEC 61788 reference method for reporting `I_c` and `n`. Switch to
> **non-linear** only if the log–log fit reports *insufficient n-window
> points* or the ramp is noticeably inductive.

---

## Log–log linear fit (IEC 61788)

![Log–log linear fit](docs/log_log_fit.png)

The same V–I curve in log–log space. The shaded green band is the IEC
decade window `[E_c1, E_c2]`; blue points fall inside the monotonic
transition segment and enter the fit, grey points sit on the noise floor
or outside the window and are excluded.

**Procedure**

1. Fit the **linear baseline** `V_ofs + R·I` on points well below `I_c`.
2. Subtract it: `V′(I) = V − V_ofs − R·I`.
3. On the **IEC decade window** `[E_c1, E_c2]`, fit
   `log₁₀ V′` vs `log₁₀ I` with `numpy.polyfit` (closed form, no
   iteration).
4. The slope is the **n-value**; `I_c` is the current at which
   `V′ = E_c2`.

**IEC 61788 defaults**

- **`E_c2 = 1 µV/cm`** — the `I_c` criterion for HTS at 77 K.
- **`E_c1 = 0.1 µV/cm`** — lower edge of the decade window.
- **≥ 50** samples inside the decade are recommended for a reliable n-value.
- Without a sample length the equivalent **`V_c` (default 1 mV)** is used;
  `E_c1` and `E_c2` scale by the same ratio.

**What you get** — `I_c`, `n`, σ(I_c) and σ(n) from the polyfit
covariance, R² on `log V′` vs the linear model, and the actual count of
samples inside the decade.

---

## How to fit a curve properly

1. **Load a TDMS recording** with *Load TDMS…* and (optionally) *Load
   metadata from TDMS* to pull per-channel scale/offset and voltage-tap
   distance.
2. **Pick channels** — `Time` (s), `Current X` (A), `Voltage Y` (V).
3. **Set scale & offset** for each channel so values are in physical
   units (`shown = raw · scale − offset`).
4. **Provide the sample length `L_s`** if you want E-field results
   (ρ, `E_c`); leave blank for V-based results (R, `V_c`).
5. **Inspect the preview plot** — three coloured bands show the dI/dt,
   linear and power-law fit windows. Drag the handles or edit the
   percentages so each window covers a clean part of the curve.
6. **Choose the criterion** `V_c` / `E_c` (1 µV/cm is the IEC default).
7. **Press *Run Fit*** and read the result block on the right. The
   overlay shows `I_c`, the criterion line and the n-window points
   actually used.

### Choosing fit windows well

**Common to both methods:**

- **dI/dt window** — pick a region where the current ramps linearly
  (typically **40 %–60 %** of the trace). Avoid the switch-on transient
  and the transition region.
- **Linear baseline window** — well below `I_c` (default
  **5 %–30 %** of `I_max`), where V is dominated by the resistive
  baseline and noise.

**Power-law window depends on the method:**

- **Log–log (IEC 61788)** — the effective fit window is the
  **decade `[E_c1, E_c2]`** on the baseline-subtracted residual
  (defaults **0.1 µV/cm – 1 µV/cm**). The numeric *Power low* / *Power V
  frac* fields only act as a sanity envelope on top of the IEC window.
- **Non-linear (LM)** — uses the full **Power-law window** from a low
  fraction of `I_c` up to the largest voltage you trust
  (**≤ 80 %** of `V_max` by default).

### Reading the result

- **`I_c`** — current at which V reaches `V_c` (or `E_c · L_s`).
- **`n`** — sharpness of the transition; larger ⇒ sharper.
- **`R / ρ`** — resistive baseline (joints, leads, contact).
- **`L`** — inductive offset coefficient from the dI/dt step.
- **R²** and **σ(I_c), σ(n)** — goodness of the power-law fit.

> **Watch out**
> - *“Ramp too fast”* — the inductive term swamps the n-value. Lower
>   `dI/dt`.
> - *“Insufficient n-window points”* — the power-law / decade window
>   contains too few samples; widen it or sample faster.
> - If R² < 0.99 or σ(n)/n > 5 %, recheck your windows.

---

## Settings & options

### Fit method

- **Log–log linear** *(default, IEC 61788 reference)* — closed-form,
  no iteration.
- **Non-linear (LM)** — Levenberg–Marquardt inside an outer
  self-consistency loop on `I_c`.

### Parameters that depend on the method

**Log–log linear — *only these inputs are used***

| Field | Meaning | Default |
|---|---|---|
| `E_c1` | Lower edge of the IEC decade window. | 0.1 µV/cm |
| `E_c2` / `V_c` | Upper edge of the decade and the `I_c` criterion. | 1 µV/cm (or 1 mV without `L_s`) |
| `Linear low / high` | Window for fitting `V_ofs + R·I` — the baseline subtracted before the log fit. | 5 %–30 % of `I_max` |
| `dI/dt low / high` | Window for the inductive-ratio diagnostic (does not change the fitted `I_c`). | 40 %–60 % of t |

> *`Max iterations`, `I_c stop tol` and `Chi-sqr tol` are ignored in this
> mode — the polyfit is one-shot.*

**Non-linear (LM) — *iteration knobs apply here***

| Field | Meaning | Default |
|---|---|---|
| `Max iterations` | Outer-loop cap for the self-consistent `I_c` refinement. | 20 |
| `I_c stop tol (%)` | Stop when \|ΔI_c\|/I_c falls below this. | 0.1 % |
| `Chi-sqr tol` | `ftol`/`xtol`/`gtol` passed to the inner LM solver. | 1e-6 |
| `V_c` / `E_c` | Criterion voltage / electric field for `I_c`. | 1 mV / 1 µV/cm |
| `Power low` / `Power V frac` | Lower current bound and upper voltage bound of the power-law fit window. | 0.05·I_max / 0.80·V_max |

> *`E_c1` is irrelevant in this mode (no decade window).*

### Shared diagnostics

- **Zero-I fraction** (default **2 %**) — used by both methods to pin
  the thermal offset away from any switch-on glitch.
- **dI/dt window** — both methods compute the inductive ratio
  (`L·dI/dt` / criterion) here and raise the *“ramp too fast”* warning
  if it dominates.

### Presets & profiles

- **Save preset…** writes every numeric setting, the criterion and the
  chosen fit method to a JSON file.
- **Load preset…** restores them in one click.
- **Per-curve profiles** remember each curve's individual windows so
  you can fit several samples sequentially without re-adjusting.

### Plot & export

- *Show dI/dt / Linear / Power* bands toggle the coloured overlays.
- *Add plot* stores the active curve; *Add corrected curve* stores the
  baseline-subtracted version of the last fit.
- *Plot scale* switches between linear and log–log axes without
  changing the underlying fit.
- *Export…* writes a publication-quality PNG/PDF and a TDMS side-car
  (`*_fit_report.tdms`) with all metadata.

---

## Metadata fields stored in `*_fit_report.tdms`

Every successful fit is written as a channel under the `FitResults`
group of the side-car next to the source recording. The following
properties are attached to each channel and survive round-trips through
LabVIEW, OriginLab and Python consumers.

**Primary parameters**

| Property | Unit | Description |
|---|---|---|
| `Ic` | A | Critical current at the chosen criterion. |
| `n` | — | n-value (transition sharpness). |
| `sigma_Ic` / `sigma_n` | A / — | 1-σ uncertainties from the covariance matrix. |
| `R_squared` | — | Coefficient of determination of the power-law fit. |

**Criterion & n-window**

| Property | Unit | Description |
|---|---|---|
| `criterion_value` | V or V/cm | Applied `V_c` / `E_c`. |
| `criterion_name` / `criterion_unit` | — | Human-readable label and unit. |
| `Ec1`, `Ec2` | V/cm | IEC decade window (only set for log–log fits). |
| `n_window_I_lo_A`, `n_window_I_hi_A` | A | Current bounds of the n-value window actually used. |
| `n_points_used` | — | Samples that entered the power-law fit. |

**Baseline decomposition** — `V_total = V_ofs + L·dI/dt + R·I + V_c·(I/I_c)ⁿ`

| Property | Unit | Description |
|---|---|---|
| `V_ofs` | V | Thermal/instrumental offset. |
| `V0_inductive` | V | Inductive voltage at the dI/dt-window center. |
| `inductance_L_H` | H | Effective lead/sample inductance. |
| `R_or_rho` | Ω or Ω/cm | Resistive baseline; unit follows `R_unit`. |

**Diagnostic flags**

| Property | Type | Meaning |
|---|---|---|
| `ramp_inductive_ratio` | float | Inductive voltage / criterion voltage. |
| `ramp_too_fast` | True/False | Set if the inductive term dominates — lower dI/dt. |
| `insufficient_n_points` | True/False | Set if the power-law window has too few samples. |
| `thermal_offset_applied` | True/False | Set if a non-zero `V_ofs` was subtracted. |
| `uses_sample_length` | True/False | True for E-field fits (with `L_s`), False otherwise. |

> Booleans are stored as the strings `"True"` / `"False"` for round-trip
> safety with LabVIEW and Origin readers.

---

## Files

| File | Purpose |
|---|---|
| `fitting/service.py` | Pure math (numpy / scipy). No Qt imports. |
| `fitting/extras.py` | Graph-settings dialog, export dialog, presets, widgets. |
| `fitting/tab.py` | Qt layout, action functions, in-app help dialog. |
| `fitting/standalone.py` | Minimal `QMainWindow` that wraps the tab for standalone use. |
| `fitting/__main__.py` | `python -m fitting` entry point. |

---

## Install (Windows PowerShell)

```powershell
cd "D:\Superconductor V-I Fitting"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

## Run standalone

```bash
python -m fitting
```

The app shows a gray version label at the bottom-right:
`v1.0 (build N)` where `N` follows the git commit count.

## Build one-file Windows EXE

```powershell
pip install pyinstaller
pyinstaller --clean --noconfirm --distpath build superconductorfit.spec
```

Output: `build\Superconductor fitting v1.0 build N.exe` — copy to any
Windows machine and run directly.

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

## Public API

```python
from fitting import (
    FitResult, FitSettings, run_full_fit, robust_view_range,
    setup_data_fitting_tab_layout,
)
```

`run_full_fit(t, x, y, settings)` returns a `FitResult` without any Qt
dependencies, so the math layer can be used in scripts and tests.

---

## In-app help

Press **? Help** next to *Load preset…* in the running app to open a
non-modal, tabbed window that mirrors this README — Overview, Log–log
fit, How to fit, Settings & options and Metadata fields.

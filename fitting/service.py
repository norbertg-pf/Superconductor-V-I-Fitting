"""Superconductor V-I fitting service (IEC 61788 power-law criterion: y = L*di/dt + R*x + Vc*(x/Ic)**n).

Exports: estimate_di_dt, fit_linear_baseline, fit_power_law, run_full_fit, robust_view_range, FitSettings, FitResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit


DEFAULT_DIDT_LOW_FRAC = 0.40
DEFAULT_DIDT_HIGH_FRAC = 0.60
DEFAULT_LINEAR_LOW_FRAC = 0.05
DEFAULT_LINEAR_HIGH_FRAC = 0.40
DEFAULT_POWER_LOW_FRAC = 0.05
DEFAULT_POWER_V_FRAC = 0.80
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_IC_TOLERANCE = 0.001    # 0.1 %
DEFAULT_CHI_SQR_TOL = 1.0e-9    # OriginLab-style tolerance on the fitter cost function
DEFAULT_VC_VOLTS = 1.0e-3
DEFAULT_EC_V_PER_CM = 1.0e-4


@dataclass
class FitSettings:
    didt_low_frac: float = DEFAULT_DIDT_LOW_FRAC
    didt_high_frac: float = DEFAULT_DIDT_HIGH_FRAC
    linear_low_frac: float = DEFAULT_LINEAR_LOW_FRAC
    linear_high_frac: float = DEFAULT_LINEAR_HIGH_FRAC
    power_low_frac: float = DEFAULT_POWER_LOW_FRAC
    power_v_frac: float = DEFAULT_POWER_V_FRAC
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    ic_tolerance: float = DEFAULT_IC_TOLERANCE
    chi_sqr_tolerance: float = DEFAULT_CHI_SQR_TOL
    criterion_voltage: float = DEFAULT_VC_VOLTS
    sample_length_cm: Optional[float] = None


@dataclass
class FitResult:
    ok: bool
    message: str = ""
    di_dt: float = 0.0
    inductance_L: float = 0.0
    V0: float = 0.0
    R: float = 0.0
    Ic: float = 0.0
    n_value: float = 0.0
    criterion: float = 0.0
    iterations: int = 0
    chi_sqr: float = 0.0
    rmse: float = 0.0
    r_squared: float = 0.0
    n_points_power: int = 0
    ic_history: list[float] = field(default_factory=list)
    linear_fit_window: tuple[float, float] = (0.0, 0.0)
    power_fit_window: tuple[float, float] = (0.0, 0.0)
    uses_sample_length: bool = False
    fit_x: Optional[np.ndarray] = None
    fit_y: Optional[np.ndarray] = None


def robust_view_range(values, low_pct: float = 1.0, high_pct: float = 99.0,
                      margin: float = 0.1) -> tuple[float, float]:
    """Percentile-based axis range that excludes a few outlier spikes."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))
    if hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def _clean_arrays(*arrs):
    arrs = [np.asarray(a, dtype=float) for a in arrs]
    if not arrs:
        return arrs
    length = min(a.size for a in arrs)
    trimmed = [a[:length] for a in arrs]
    mask = np.ones(length, dtype=bool)
    for a in trimmed:
        mask &= np.isfinite(a)
    return [a[mask] for a in trimmed]


def estimate_di_dt(t: np.ndarray, x: np.ndarray, low_frac: float = DEFAULT_DIDT_LOW_FRAC,
                   high_frac: float = DEFAULT_DIDT_HIGH_FRAC) -> float:
    t, x = _clean_arrays(t, x)
    if t.size < 2:
        return 0.0
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return 0.0
    lo = x_min + low_frac * (x_max - x_min)
    hi = x_min + high_frac * (x_max - x_min)
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        return 0.0
    slope, _ = np.polyfit(t[mask], x[mask], 1)
    return float(slope)


def fit_linear_baseline(x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float) -> tuple[float, float]:
    """Fit y = V0 + R*x on [x_lo, x_hi]. Returns (V0, R)."""
    x, y = _clean_arrays(x, y)
    mask = (x >= x_lo) & (x <= x_hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough points in linear baseline window.")
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(intercept), float(slope)


def _power_law_model(x, Ic, n, V0, R, Vc):
    """Model with V0, R, Vc fixed — only Ic and n are free."""
    return V0 + R * x + Vc * np.power(np.clip(x / Ic, 1e-30, None), n)


def fit_power_law(x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float,
                  V0: float, R: float, Vc: float,
                  initial_Ic: Optional[float] = None,
                  initial_n: float = 20.0,
                  chi_sqr_tol: float = DEFAULT_CHI_SQR_TOL) -> tuple[float, float, float]:
    """Fit Ic, n in y = V0 + R*x + Vc*(x/Ic)^n on [x_lo, x_hi] with V0, R, Vc fixed.

    Returns (Ic, n, chi_sqr) where chi_sqr = sum((y - model)**2) over the window.
    """
    x, y = _clean_arrays(x, y)
    mask = (x >= x_lo) & (x <= x_hi) & (x > 0)
    if np.count_nonzero(mask) < 4:
        raise ValueError("Not enough points in power-law window.")
    xm = x[mask]
    ym = y[mask]
    if initial_Ic is None or initial_Ic <= 0:
        residual = ym - V0 - R * xm
        above = np.where(residual >= Vc)[0]
        initial_Ic = float(xm[above[0]]) if above.size else float(np.max(xm))
    p0 = [max(initial_Ic, float(np.max(xm)) * 0.5), max(1.0, initial_n)]
    bounds = ([float(np.min(xm)) * 0.1, 1.0], [float(np.max(xm)) * 10.0, 200.0])

    def model(x_, Ic_, n_):
        return _power_law_model(x_, Ic_, n_, V0, R, Vc)

    popt, _ = curve_fit(
        model, xm, ym, p0=p0, bounds=bounds, maxfev=10000,
        ftol=chi_sqr_tol, xtol=chi_sqr_tol, gtol=chi_sqr_tol,
    )
    Ic = float(popt[0])
    n_val = float(popt[1])
    residuals = ym - model(xm, Ic, n_val)
    chi_sqr = float(np.sum(residuals ** 2))
    return Ic, n_val, chi_sqr


def run_full_fit(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                 settings: Optional[FitSettings] = None) -> FitResult:
    """Step 1 di/dt, step 2 linear baseline -> V0, R, L, step 3 iterative power-law fit (Ic, n)."""
    settings = settings or FitSettings()
    t, x, y = _clean_arrays(t, x, y)
    if x.size < 8:
        return FitResult(ok=False, message="Not enough valid samples to fit.")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return FitResult(ok=False, message="Current range is empty or degenerate.")

    if settings.max_iterations < 1:
        return FitResult(ok=False, message="max_iterations must be >= 1.")
    if settings.criterion_voltage <= 0:
        return FitResult(ok=False, message="Criterion (Vc/Ec) must be > 0.")
    for name, value in (
        ("didt_low_frac", settings.didt_low_frac),
        ("didt_high_frac", settings.didt_high_frac),
        ("linear_low_frac", settings.linear_low_frac),
        ("linear_high_frac", settings.linear_high_frac),
        ("power_low_frac", settings.power_low_frac),
        ("power_v_frac", settings.power_v_frac),
    ):
        if not (0.0 <= float(value) <= 1.0):
            return FitResult(ok=False, message=f"{name} must be in [0, 1].")
    if settings.didt_low_frac >= settings.didt_high_frac:
        return FitResult(ok=False, message="di/dt low fraction must be < high fraction.")
    if settings.linear_low_frac >= settings.linear_high_frac:
        return FitResult(ok=False, message="linear low fraction must be < high fraction.")

    Vc = float(settings.criterion_voltage)
    uses_length = settings.sample_length_cm is not None and settings.sample_length_cm > 0

    di_dt = estimate_di_dt(t, x, settings.didt_low_frac, settings.didt_high_frac)

    lin_lo = x_min + settings.linear_low_frac * (x_max - x_min)
    lin_hi = x_min + settings.linear_high_frac * (x_max - x_min)
    try:
        V0, R = fit_linear_baseline(x, y, lin_lo, lin_hi)
    except ValueError as exc:
        return FitResult(ok=False, message=f"Linear baseline fit failed: {exc}")

    # Step 2 result: V0 is the constant inductive offset (L*di/dt) in Y-units.
    # When the caller has divided Y by sample length, V0 is in V/cm, so the
    # real voltage offset is V0 * Ls_cm and inductance = (V0 * Ls_cm) / di_dt.
    v0_voltage = V0 * float(settings.sample_length_cm) if uses_length else V0
    inductance_L = v0_voltage / di_dt if abs(di_dt) > 1e-30 else 0.0

    y_max = float(np.max(y))
    y_threshold = settings.power_v_frac * y_max
    above = np.where(y >= y_threshold)[0]
    power_hi = float(x[above[0]]) if above.size else x_max
    power_lo = x_min + settings.power_low_frac * (x_max - x_min)

    Ic = float("nan")
    n_value = float("nan")
    chi_sqr = 0.0
    ic_history: list[float] = []
    last_Ic = None
    iterations_used = 0
    for iteration in range(1, max(1, settings.max_iterations) + 1):
        iterations_used = iteration
        try:
            Ic, n_value, chi_sqr = fit_power_law(
                x, y, power_lo, power_hi,
                V0=V0, R=R, Vc=Vc,
                initial_Ic=last_Ic,
                chi_sqr_tol=settings.chi_sqr_tolerance,
            )
        except (ValueError, RuntimeError) as exc:
            return FitResult(ok=False, message=f"Power-law fit failed: {exc}")
        ic_history.append(Ic)
        if last_Ic is not None and last_Ic > 0:
            rel_change = abs(Ic - last_Ic) / last_Ic
            if rel_change < settings.ic_tolerance:
                last_Ic = Ic
                break
        last_Ic = Ic
        # Shrink the upper bound to Ic for the next iteration.
        power_hi = min(power_hi, Ic)
        if power_hi <= power_lo:
            break

    fit_x = np.linspace(power_lo, x_max, 400)
    fit_y = _power_law_model(fit_x, Ic, n_value, V0, R, Vc)

    mask_power = (x >= power_lo) & (x <= power_hi) & (x > 0)
    xm = x[mask_power]
    ym = y[mask_power]
    model_power = _power_law_model(xm, Ic, n_value, V0, R, Vc)
    residuals = ym - model_power
    n_points_power = int(xm.size)
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if n_points_power else 0.0
    if n_points_power >= 2:
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((ym - np.mean(ym)) ** 2))
        r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else 1.0
    else:
        r_squared = 0.0

    return FitResult(
        ok=True,
        message="Fit succeeded.",
        di_dt=di_dt,
        inductance_L=inductance_L,
        V0=V0,
        R=R,
        Ic=Ic,
        n_value=n_value,
        criterion=Vc,
        iterations=iterations_used,
        chi_sqr=chi_sqr,
        rmse=rmse,
        r_squared=r_squared,
        n_points_power=n_points_power,
        ic_history=ic_history,
        linear_fit_window=(lin_lo, lin_hi),
        power_fit_window=(power_lo, power_hi),
        uses_sample_length=uses_length,
        fit_x=fit_x,
        fit_y=fit_y,
    )

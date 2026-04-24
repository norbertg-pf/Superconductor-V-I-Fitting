"""Superconductor V-I fitting service (IEC 61788 power-law criterion: y = L*di/dt + R*x + Vc*(x/Ic)**n).

Exports: estimate_di_dt, fit_linear_baseline, fit_power_law, fit_n_value_log_log,
run_full_fit, robust_view_range, FitSettings, FitResult.
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
DEFAULT_EC_V_PER_CM = 1.0e-4    # 1 uV/cm = 100 uV/m, IEC 61788-3/-21 default for HTS at 77 K
DEFAULT_EC1_V_PER_CM = 1.0e-5   # 0.1 uV/cm, lower end of IEC decade n-value window
DEFAULT_EC2_V_PER_CM = 1.0e-4   # 1 uV/cm, upper end (= the Ic criterion)

# Fit method identifiers.
FIT_METHOD_LOG_LOG = "log_log"          # linear fit of log10(E_sc) vs log10(I), IEC 61788
FIT_METHOD_NONLINEAR = "nonlinear"       # coupled non-linear V = V0 + R*I + Vc*(I/Ic)^n
DEFAULT_FIT_METHOD = FIT_METHOD_LOG_LOG


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
    # IEC 61788 decade n-value window (expressed in the same units as Y).
    # When Y has been divided by the voltage-tap separation, these are electric
    # fields in V/cm; otherwise they are voltages in V.
    fit_method: str = DEFAULT_FIT_METHOD
    ec1: float = DEFAULT_EC1_V_PER_CM
    ec2: float = DEFAULT_EC2_V_PER_CM


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
    ic_history: list[float] = field(default_factory=list)
    linear_fit_window: tuple[float, float] = (0.0, 0.0)
    power_fit_window: tuple[float, float] = (0.0, 0.0)
    uses_sample_length: bool = False
    fit_x: Optional[np.ndarray] = None
    fit_y: Optional[np.ndarray] = None
    # IEC decade n-value extras (populated when fit_method == FIT_METHOD_LOG_LOG).
    fit_method: str = FIT_METHOD_NONLINEAR
    ec1: float = 0.0
    ec2: float = 0.0
    n_window_I: tuple[float, float] = (0.0, 0.0)
    n_points_used: int = 0


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


def fit_n_value_log_log(x: np.ndarray, y: np.ndarray,
                        V0: float, R: float,
                        Ec1: float, Ec2: float) -> tuple[float, float, float, int, tuple[float, float]]:
    """IEC 61788 decade n-value: linear fit of log10(E_sc) vs log10(I).

    E_sc = y - V0 - R*x is the baseline-subtracted signal. Points with
    E_sc in [Ec1, Ec2] and I > 0 are used. The slope of log10(E_sc) vs
    log10(I) is the n-index; Ic is reported at E = Ec2 (the higher
    criterion, which IEC takes as the Ic criterion for HTS).

    Returns (Ic_at_Ec2, n, chi_sqr, n_points, (I_lo, I_hi)).
    """
    x, y = _clean_arrays(x, y)
    if Ec2 <= Ec1 or Ec1 <= 0:
        raise ValueError("Ec1 must be > 0 and strictly less than Ec2.")
    e_sc = y - V0 - R * x
    mask = (x > 0) & (e_sc >= Ec1) & (e_sc <= Ec2) & np.isfinite(e_sc)
    n_pts = int(np.count_nonzero(mask))
    if n_pts < 4:
        raise ValueError(
            f"Only {n_pts} points fall inside the IEC n-value window "
            f"[{Ec1:.3g}, {Ec2:.3g}]. Need at least 4; increase ramp range, "
            "reduce averaging, or widen the decade."
        )
    log_I = np.log10(x[mask])
    log_E = np.log10(e_sc[mask])
    slope, intercept = np.polyfit(log_I, log_E, 1)
    n_val = float(slope)
    # Ic defined at E = Ec2 (= the main IEC criterion; Ec1 is only there to
    # anchor the low end of the n fit). log10(Ec2) = intercept + n*log10(Ic)
    if abs(n_val) < 1e-12:
        raise ValueError("Power-law slope collapsed to zero; cannot solve for Ic.")
    log_Ic = (np.log10(Ec2) - intercept) / n_val
    Ic_at_Ec2 = float(10.0 ** log_Ic)
    model_log_E = intercept + n_val * log_I
    chi_sqr = float(np.sum((log_E - model_log_E) ** 2))
    I_lo = float(np.min(x[mask]))
    I_hi = float(np.max(x[mask]))
    return Ic_at_Ec2, n_val, chi_sqr, n_pts, (I_lo, I_hi)


def run_full_fit(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                 settings: Optional[FitSettings] = None) -> FitResult:
    """Step 1 di/dt, step 2 linear baseline -> V0, R, L, step 3 Ic/n.

    Step 3 uses the IEC 61788 log-log decade method by default
    (``settings.fit_method == FIT_METHOD_LOG_LOG``): n from the slope of
    log10(E_sc) vs log10(I) on [Ec1, Ec2], Ic at E = Ec2. The legacy
    non-linear fit of V = V0 + R*I + Vc*(I/Ic)^n remains available as
    ``FIT_METHOD_NONLINEAR``.
    """
    settings = settings or FitSettings()
    t, x, y = _clean_arrays(t, x, y)
    if x.size < 8:
        return FitResult(ok=False, message="Not enough valid samples to fit.")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return FitResult(ok=False, message="Current range is empty or degenerate.")

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

    method = getattr(settings, "fit_method", DEFAULT_FIT_METHOD)

    if method == FIT_METHOD_LOG_LOG:
        try:
            Ic, n_value, chi_sqr, n_pts, n_window = fit_n_value_log_log(
                x, y, V0=V0, R=R, Ec1=settings.ec1, Ec2=settings.ec2,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return FitResult(ok=False, message=f"Log-log n-value fit failed: {exc}")
        # Rebuild a smooth model curve for plotting. Use Ec2 as the criterion.
        fit_x = np.linspace(max(x_min, 1e-12), x_max, 400)
        fit_y = V0 + R * fit_x + settings.ec2 * np.power(
            np.clip(fit_x / Ic, 1e-30, None), n_value
        )
        return FitResult(
            ok=True,
            message="IEC 61788 log-log n-value fit succeeded.",
            di_dt=di_dt,
            inductance_L=inductance_L,
            V0=V0,
            R=R,
            Ic=Ic,
            n_value=n_value,
            criterion=settings.ec2,
            iterations=1,
            chi_sqr=chi_sqr,
            ic_history=[Ic],
            linear_fit_window=(lin_lo, lin_hi),
            power_fit_window=n_window,
            uses_sample_length=uses_length,
            fit_x=fit_x,
            fit_y=fit_y,
            fit_method=FIT_METHOD_LOG_LOG,
            ec1=settings.ec1,
            ec2=settings.ec2,
            n_window_I=n_window,
            n_points_used=n_pts,
        )

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
        ic_history=ic_history,
        linear_fit_window=(lin_lo, lin_hi),
        power_fit_window=(power_lo, power_hi),
        uses_sample_length=uses_length,
        fit_x=fit_x,
        fit_y=fit_y,
        fit_method=FIT_METHOD_NONLINEAR,
        ec1=0.0,
        ec2=0.0,
        n_window_I=(power_lo, power_hi),
        n_points_used=0,
    )

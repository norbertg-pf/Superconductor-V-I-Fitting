"""Superconductor V-I fitting service — IEC 61788-compliant.

Model (matches the power-law / electric-field criterion definition of Ic in
IEC 61788-1/-2/-3/-14/-21):

    short sample (per-unit-length mode, Y = E):
        E(I) = (L/Ls)·dI/dt + rho·I + Ec · (I/Ic)^n

    whole sample (Y = V):
        V(I) = L·dI/dt + R·I   + Vc · (I/Ic)^n

Ic is defined as the current at which the baseline-subtracted voltage equals
the criterion (Vc = Ec · Ls). n is the exponent of the power law and, per
IEC 61788, is determined in a voltage window centred on the criterion — by
default [0.1·Vc, 10·Vc] on the baseline-subtracted voltage, with the n-value
computed from a log-log linear regression of that window.

Exports: FitSettings, FitResult, estimate_di_dt, fit_linear_baseline,
fit_power_law, run_full_fit, robust_view_range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit


# IEC 61788 n-value windows are expressed in multiples of the criterion.
# 0.1..10 * Vc is the value used in IEC 61788-2 Annex and typical VAMAS
# round-robin reports; a narrower window (e.g. 1..10 * Vc) is sometimes used
# when the baseline voltage noise is larger than 0.1 * Vc.
DEFAULT_VC_LOW_MULT = 0.1
DEFAULT_VC_HIGH_MULT = 10.0

# Legacy manual-mode defaults (used only when iec_mode=False).
DEFAULT_DIDT_LOW_FRAC = 0.40
DEFAULT_DIDT_HIGH_FRAC = 0.60
DEFAULT_LINEAR_LOW_FRAC = 0.05
DEFAULT_LINEAR_HIGH_FRAC = 0.40
DEFAULT_POWER_LOW_FRAC = 0.05
DEFAULT_POWER_V_FRAC = 0.80

DEFAULT_MAX_ITERATIONS = 10
DEFAULT_IC_TOLERANCE = 0.001    # 0.1 %
DEFAULT_CHI_SQR_TOL = 1.0e-9    # scipy ftol/xtol/gtol
DEFAULT_VC_VOLTS = 1.0e-3       # placeholder default; always supply the real Vc.
DEFAULT_EC_V_PER_CM = 1.0e-6    # IEC 61788 HTS criterion: 1 µV/cm.

# Hard bounds for the n exponent during the NLS fit. Real conductors sit
# around 5..80; 200 is a hard safety cap to keep the optimiser stable.
_N_BOUND_LOW = 1.0
_N_BOUND_HIGH = 200.0


@dataclass
class FitSettings:
    # --- IEC-mode knobs (preferred) ---------------------------------------
    iec_mode: bool = True
    vc_low_mult: float = DEFAULT_VC_LOW_MULT     # lower end of n-window as factor of Vc
    vc_high_mult: float = DEFAULT_VC_HIGH_MULT   # upper end of n-window as factor of Vc
    n_log_fit: bool = True                       # log-log linear regression for n (IEC)

    # --- dI/dt window (fraction of Imax). ---------------------------------
    didt_low_frac: float = DEFAULT_DIDT_LOW_FRAC
    didt_high_frac: float = DEFAULT_DIDT_HIGH_FRAC

    # --- Linear baseline window (fraction of Imax). -----------------------
    linear_low_frac: float = DEFAULT_LINEAR_LOW_FRAC
    linear_high_frac: float = DEFAULT_LINEAR_HIGH_FRAC

    # --- Legacy manual power-law window (only used when iec_mode=False). --
    power_low_frac: float = DEFAULT_POWER_LOW_FRAC
    power_v_frac: float = DEFAULT_POWER_V_FRAC

    # --- Optimiser & iteration. -------------------------------------------
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    ic_tolerance: float = DEFAULT_IC_TOLERANCE
    chi_sqr_tolerance: float = DEFAULT_CHI_SQR_TOL

    # --- Criterion and sample length. -------------------------------------
    criterion_voltage: float = DEFAULT_VC_VOLTS   # in V (whole sample) or V/cm (per-unit-length)
    sample_length_cm: Optional[float] = None      # if set, Y is E = V/Ls and criterion is Ec


@dataclass
class FitResult:
    ok: bool
    message: str = ""

    # Step 1 & 2 outputs.
    di_dt: float = 0.0
    inductance_L: float = 0.0
    V0: float = 0.0
    V0_sigma: float = 0.0
    R: float = 0.0
    R_sigma: float = 0.0
    linear_r2: float = 0.0
    baseline_noise_rms: float = 0.0  # RMS of baseline residuals, same units as Y

    # Step 3 outputs (Ic, n + uncertainties).
    Ic: float = 0.0
    Ic_sigma: float = 0.0
    n_value: float = 0.0
    n_sigma: float = 0.0
    n_method: str = "nls"         # "nls" or "log-linear"

    # Fit quality.
    chi_sqr: float = 0.0          # sum of squared residuals in V units
    reduced_chi_sqr: float = 0.0  # chi_sqr / dof
    dof: int = 0
    n_points_fit: int = 0
    decades_above_Vc: float = 0.0  # log10(V_max_fit / Vc) on baseline-subtracted voltage

    # Provenance.
    criterion: float = 0.0
    iterations: int = 0
    ic_history: list[float] = field(default_factory=list)
    linear_fit_window: tuple[float, float] = (0.0, 0.0)
    power_fit_window: tuple[float, float] = (0.0, 0.0)
    power_fit_window_V: tuple[float, float] = (0.0, 0.0)  # baseline-subtracted voltage bounds
    uses_sample_length: bool = False
    ramp_direction: str = "up"     # "up", "down", or "mixed"
    n_data_points: int = 0

    # Diagnostic flags / human messages (separate from hard errors).
    warnings: list[str] = field(default_factory=list)

    # Drawing helpers.
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


def _detect_ramp_direction(t: np.ndarray, x: np.ndarray) -> str:
    """Return 'up', 'down', or 'mixed' based on the net current change."""
    if t.size < 2 or x.size < 2:
        return "up"
    # Use the first-to-last comparison on the rising portion so that flat-tops
    # and ring-down don't invert the answer.
    i_start = float(x[0])
    i_peak_idx = int(np.argmax(np.abs(x)))
    i_peak = float(x[i_peak_idx])
    if i_peak > i_start:
        # Mostly-up sweep (what IEC 61788 assumes).
        if i_peak_idx < t.size - 1 and float(x[-1]) < i_peak * 0.95:
            return "up"  # still up-ramp; ring-down after is ignored downstream
        return "up"
    if i_peak < i_start:
        return "down"
    return "mixed"


def _interp_current_at_voltage(x: np.ndarray, v_net: np.ndarray, target_v: float,
                               fallback: float) -> float:
    """Linearly interpolate the current at which ``v_net`` first crosses ``target_v``.

    ``x`` and ``v_net`` must be parallel arrays ordered as acquired (time-ordered,
    not magnitude-ordered) on the ramp-up portion of the sweep. Returns
    ``fallback`` if no crossing is found within the data.
    """
    if x.size < 2:
        return fallback
    # Find the first index where v_net >= target_v.
    idx = np.where(v_net >= target_v)[0]
    if idx.size == 0:
        return fallback
    j = int(idx[0])
    if j == 0:
        return float(x[0])
    v_lo = float(v_net[j - 1])
    v_hi = float(v_net[j])
    if v_hi == v_lo:
        return float(x[j])
    frac = (target_v - v_lo) / (v_hi - v_lo)
    return float(x[j - 1] + frac * (x[j] - x[j - 1]))


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


def fit_linear_baseline(x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float,
                        ) -> tuple[float, float, float, float, float, float]:
    """Fit y = V0 + R·x on [x_lo, x_hi].

    Returns (V0, R, sigma_V0, sigma_R, r_squared, residual_rms).
    ``residual_rms`` is the RMS of the y − (V0 + R·x) residuals and is the
    noise-floor estimate used elsewhere to judge IEC comparability.
    """
    x, y = _clean_arrays(x, y)
    mask = (x >= x_lo) & (x <= x_hi)
    n = int(np.count_nonzero(mask))
    if n < 2:
        raise ValueError("Not enough points in linear baseline window.")
    xm = x[mask]
    ym = y[mask]
    coeffs, cov = np.polyfit(xm, ym, 1, cov=True)  # cov is residual-scaled
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    sigma_slope = float(np.sqrt(max(cov[0, 0], 0.0)))
    sigma_intercept = float(np.sqrt(max(cov[1, 1], 0.0)))
    y_hat = intercept + slope * xm
    residuals = ym - y_hat
    ss_res = float(np.sum(residuals ** 2))
    y_mean = float(np.mean(ym))
    ss_tot = float(np.sum((ym - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rms = float(np.sqrt(ss_res / max(1, n)))
    return intercept, slope, sigma_intercept, sigma_slope, r2, rms


def _power_law_model(x, Ic, n, V0, R, Vc):
    """Model with V0, R, Vc fixed — only Ic and n are free."""
    return V0 + R * x + Vc * np.power(np.clip(x / Ic, 1e-30, None), n)


def fit_power_law(x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float,
                  V0: float, R: float, Vc: float,
                  initial_Ic: Optional[float] = None,
                  initial_n: float = 20.0,
                  chi_sqr_tol: float = DEFAULT_CHI_SQR_TOL,
                  ) -> tuple[float, float, float, float, float, int]:
    """Nonlinear least squares fit of Ic, n in V = V0 + R·I + Vc·(I/Ic)^n.

    Returns (Ic, n, sigma_Ic, sigma_n, chi_sqr, n_points).
    """
    x, y = _clean_arrays(x, y)
    mask = (x >= x_lo) & (x <= x_hi) & (x > 0)
    n_pts = int(np.count_nonzero(mask))
    if n_pts < 4:
        raise ValueError("Not enough points in power-law window.")
    xm = x[mask]
    ym = y[mask]
    if initial_Ic is None or initial_Ic <= 0:
        residual = ym - V0 - R * xm
        above = np.where(residual >= Vc)[0]
        initial_Ic = float(xm[above[0]]) if above.size else float(np.max(xm))
    p0 = [max(initial_Ic, float(np.max(xm)) * 0.5), max(1.0, initial_n)]
    bounds = (
        [float(np.min(xm)) * 0.1, _N_BOUND_LOW],
        [float(np.max(xm)) * 10.0, _N_BOUND_HIGH],
    )

    def model(x_, Ic_, n_):
        return _power_law_model(x_, Ic_, n_, V0, R, Vc)

    popt, pcov = curve_fit(
        model, xm, ym, p0=p0, bounds=bounds, maxfev=10000,
        ftol=chi_sqr_tol, xtol=chi_sqr_tol, gtol=chi_sqr_tol,
    )
    Ic = float(popt[0])
    n_val = float(popt[1])
    residuals = ym - model(xm, Ic, n_val)
    chi_sqr = float(np.sum(residuals ** 2))
    # curve_fit returns pcov already scaled by the residual variance when
    # absolute_sigma=False (the default), so σ = sqrt(diag(pcov)).
    if pcov is not None and np.all(np.isfinite(pcov)):
        sigma_Ic = float(np.sqrt(max(pcov[0, 0], 0.0)))
        sigma_n = float(np.sqrt(max(pcov[1, 1], 0.0)))
    else:
        sigma_Ic = float("nan")
        sigma_n = float("nan")
    return Ic, n_val, sigma_Ic, sigma_n, chi_sqr, n_pts


def _n_from_log_fit(x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float,
                    V0: float, R: float,
                    ) -> tuple[float, float, float]:
    """IEC-style log-log linear regression of baseline-subtracted V(I).

    Returns (n, sigma_n, log_intercept) where the intercept is
    log10(Vc') with V' = Vc' * (I/1)^n (i.e. a "Vc at I=1A" placeholder).
    Only ``n`` is physically meaningful here; we use ``sigma_n`` to report
    an IEC-style uncertainty on the n-value independently of the NLS fit.
    """
    x, y = _clean_arrays(x, y)
    v_net = y - V0 - R * x
    mask = (x >= x_lo) & (x <= x_hi) & (x > 0) & (v_net > 0)
    n_pts = int(np.count_nonzero(mask))
    if n_pts < 3:
        raise ValueError("Not enough positive baseline-subtracted points for log-linear n fit.")
    xm = x[mask]
    vm = v_net[mask]
    log_I = np.log10(xm)
    log_V = np.log10(vm)
    coeffs, cov = np.polyfit(log_I, log_V, 1, cov=True)
    n_val = float(coeffs[0])
    log_intercept = float(coeffs[1])
    sigma_n = float(np.sqrt(max(cov[0, 0], 0.0)))
    return n_val, sigma_n, log_intercept


def _iec_current_window(x: np.ndarray, v_net: np.ndarray, Vc: float,
                        vc_low_mult: float, vc_high_mult: float,
                        x_min: float, x_max: float) -> tuple[float, float]:
    """Convert the IEC voltage window [vc_low_mult·Vc, vc_high_mult·Vc] into
    a current window via linear interpolation on the acquired data."""
    v_lo = vc_low_mult * Vc
    v_hi = vc_high_mult * Vc
    i_lo = _interp_current_at_voltage(x, v_net, v_lo, fallback=x_min)
    i_hi = _interp_current_at_voltage(x, v_net, v_hi, fallback=x_max)
    # Keep ordering even if the crossing can't be found.
    if i_hi <= i_lo:
        i_hi = min(x_max, max(i_lo * 1.01, i_lo + 1e-9))
    return i_lo, i_hi


def run_full_fit(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                 settings: Optional[FitSettings] = None) -> FitResult:
    """Three-step IEC 61788 fit.

    Step 1 estimates dI/dt on a fraction of the current ramp.
    Step 2 fits the ohmic/inductive baseline V = V0 + R·I on the pre-transition
    region and returns σ(V0), σ(R), R² as well.
    Step 3 finds Ic and n using either (a) a V-I NLS fit of the full power-law
    model with V0 and R fixed, or (b) the IEC log-log linear regression of the
    baseline-subtracted voltage in the Vc-multiple window. The n-value σ comes
    directly from the fitter's covariance matrix.
    """
    settings = settings or FitSettings()
    t, x, y = _clean_arrays(t, x, y)
    warnings: list[str] = []
    if x.size < 8:
        return FitResult(ok=False, message="Not enough valid samples to fit.",
                         warnings=warnings)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return FitResult(ok=False, message="Current range is empty or degenerate.",
                         warnings=warnings)

    Vc = float(settings.criterion_voltage)
    uses_length = settings.sample_length_cm is not None and settings.sample_length_cm > 0
    ramp_direction = _detect_ramp_direction(t, x)
    if ramp_direction != "up":
        warnings.append(
            f"Ramp direction detected as '{ramp_direction}'. IEC 61788 fits assume "
            "an up-ramp; verify the polarity/slice of the data."
        )

    di_dt = estimate_di_dt(t, x, settings.didt_low_frac, settings.didt_high_frac)

    lin_lo = x_min + settings.linear_low_frac * (x_max - x_min)
    lin_hi = x_min + settings.linear_high_frac * (x_max - x_min)
    try:
        V0, R, sigma_V0, sigma_R, linear_r2, noise_rms = fit_linear_baseline(
            x, y, lin_lo, lin_hi,
        )
    except ValueError as exc:
        return FitResult(ok=False, message=f"Linear baseline fit failed: {exc}",
                         warnings=warnings)

    # Inductance L: V0 is in the same units as Y. When Y has been divided by
    # the sample length, V0 is V/cm, so the real voltage offset is V0·Ls and
    # L = (V0·Ls)/dI/dt.
    v0_voltage = V0 * float(settings.sample_length_cm) if uses_length else V0
    inductance_L = v0_voltage / di_dt if abs(di_dt) > 1e-30 else 0.0

    # Pre-compute baseline-subtracted voltage once for the rest of the pipeline.
    v_net = y - V0 - R * x

    # Guard: does the baseline window overlap the transition?
    if np.any(v_net[(x >= lin_lo) & (x <= lin_hi)] >= 0.5 * Vc):
        warnings.append(
            "Linear baseline window reaches ≥0.5·Vc — the fit for R/V0 may be "
            "biased. Narrow the baseline window to well below the transition."
        )

    # Step 3: power-law / n-value fit. ------------------------------------
    if settings.iec_mode:
        power_lo, power_hi = _iec_current_window(
            x, v_net, Vc, settings.vc_low_mult, settings.vc_high_mult, x_min, x_max,
        )
    else:
        y_max = float(np.max(y))
        y_threshold = settings.power_v_frac * y_max
        above = np.where(y >= y_threshold)[0]
        power_hi = float(x[above[0]]) if above.size else x_max
        power_lo = x_min + settings.power_low_frac * (x_max - x_min)

    Ic = float("nan")
    n_value = float("nan")
    sigma_Ic = float("nan")
    sigma_n = float("nan")
    chi_sqr = 0.0
    n_pts_fit = 0
    ic_history: list[float] = []
    last_Ic: Optional[float] = None
    iterations_used = 0
    for iteration in range(1, max(1, settings.max_iterations) + 1):
        iterations_used = iteration
        try:
            Ic, n_value, sigma_Ic, sigma_n, chi_sqr, n_pts_fit = fit_power_law(
                x, y, power_lo, power_hi,
                V0=V0, R=R, Vc=Vc,
                initial_Ic=last_Ic,
                chi_sqr_tol=settings.chi_sqr_tolerance,
            )
        except (ValueError, RuntimeError) as exc:
            return FitResult(ok=False, message=f"Power-law fit failed: {exc}",
                             warnings=warnings)
        ic_history.append(Ic)
        if last_Ic is not None and last_Ic > 0:
            rel_change = abs(Ic - last_Ic) / last_Ic
            if rel_change < settings.ic_tolerance:
                last_Ic = Ic
                break
        last_Ic = Ic
        if settings.iec_mode:
            # Window is anchored on Vc, not on Ic, so it doesn't shrink each
            # pass. We only iterate in case the first Ic guess was far enough
            # from the truth that the NLS had a bad starting point.
            continue
        # Legacy mode: shrink upper bound to Ic (not IEC-compliant, but kept
        # for backwards compatibility with old presets).
        power_hi = min(power_hi, Ic)
        if power_hi <= power_lo:
            break

    # Optional IEC log-log linear fit for n (preferred for inter-lab comparison).
    n_method = "nls"
    if settings.n_log_fit and settings.iec_mode:
        try:
            n_log, sigma_n_log, _ = _n_from_log_fit(x, y, power_lo, power_hi, V0, R)
            n_value = n_log
            sigma_n = sigma_n_log
            n_method = "log-linear"
        except ValueError as exc:
            warnings.append(
                f"Log-linear n-value fit skipped ({exc}); falling back to the NLS n."
            )

    # Diagnostics / quality metrics.
    dof = max(1, n_pts_fit - 2)  # two free params (Ic, n); V0/R fixed
    reduced_chi_sqr = chi_sqr / dof if dof > 0 else float("nan")
    mask_fit = (x >= power_lo) & (x <= power_hi) & (x > 0)
    v_net_fit = v_net[mask_fit]
    v_pos = v_net_fit[v_net_fit > 0]
    if v_pos.size and Vc > 0:
        decades_above = float(np.log10(np.max(v_pos) / Vc))
    else:
        decades_above = 0.0

    # Post-fit sanity warnings — surfaced to the UI and exported reports so a
    # reviewer can tell at a glance whether the result is IEC-comparable.
    if n_pts_fit < 30:
        warnings.append(
            f"Only {n_pts_fit} points in the n-fit window; IEC round-robin "
            "practice recommends at least ~30."
        )
    # Noise-floor guard: if baseline RMS is comparable to the lower end of the
    # Vc window, the log-log fit is dominated by noise.
    if settings.iec_mode and Vc > 0:
        v_lower = settings.vc_low_mult * Vc
        if noise_rms >= 0.3 * v_lower:
            warnings.append(
                f"Baseline noise RMS ({noise_rms:.3g}) is ≥30% of the lower "
                f"window edge ({v_lower:.3g}). Raise the lower Vc multiplier "
                f"(currently {settings.vc_low_mult:g}) or reduce noise — otherwise "
                f"the n-value is noise-biased."
            )
    if decades_above < 0.9 and settings.iec_mode:
        warnings.append(
            f"n-fit window spans only {decades_above:.2f} decades above Vc. "
            "IEC 61788 typically asks for ≥1 decade (Vc..10·Vc) for a robust n; "
            "either the sweep did not reach 10·Vc or the upper multiplier is <10."
        )
    if abs(n_value - _N_BOUND_HIGH) < 1e-6 or abs(n_value - _N_BOUND_LOW) < 1e-6:
        warnings.append(
            f"n-value hit the solver bound ({_N_BOUND_LOW}..{_N_BOUND_HIGH}); "
            "result is likely not physical."
        )
    if iterations_used >= settings.max_iterations and not settings.iec_mode:
        warnings.append(
            f"Legacy iteration reached the cap ({settings.max_iterations}) "
            f"without meeting the Ic tolerance ({settings.ic_tolerance * 100:.3g}%)."
        )

    # Drawing helpers — sample the fitted model over the plotted range.
    fit_x = np.linspace(power_lo, x_max, 400)
    fit_y = _power_law_model(fit_x, Ic, n_value, V0, R, Vc)

    # Baseline-subtracted voltage window (useful for IEC-style reports).
    v_lo_window = Vc * settings.vc_low_mult if settings.iec_mode else float(np.min(v_net_fit) if v_net_fit.size else 0.0)
    v_hi_window = Vc * settings.vc_high_mult if settings.iec_mode else float(np.max(v_net_fit) if v_net_fit.size else 0.0)

    return FitResult(
        ok=True,
        message="Fit succeeded.",
        di_dt=di_dt,
        inductance_L=inductance_L,
        V0=V0,
        V0_sigma=sigma_V0,
        R=R,
        R_sigma=sigma_R,
        linear_r2=linear_r2,
        baseline_noise_rms=noise_rms,
        Ic=Ic,
        Ic_sigma=sigma_Ic,
        n_value=n_value,
        n_sigma=sigma_n,
        n_method=n_method,
        criterion=Vc,
        iterations=iterations_used,
        chi_sqr=chi_sqr,
        reduced_chi_sqr=reduced_chi_sqr,
        dof=dof,
        n_points_fit=n_pts_fit,
        decades_above_Vc=decades_above,
        ic_history=ic_history,
        linear_fit_window=(lin_lo, lin_hi),
        power_fit_window=(power_lo, power_hi),
        power_fit_window_V=(v_lo_window, v_hi_window),
        uses_sample_length=uses_length,
        ramp_direction=ramp_direction,
        n_data_points=int(x.size),
        warnings=warnings,
        fit_x=fit_x,
        fit_y=fit_y,
    )

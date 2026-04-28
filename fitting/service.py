"""Superconductor V-I fitting service (IEC 61788 power-law criterion: y = L*di/dt + R*x + Vc*(x/Ic)**n).

Exports: estimate_di_dt, fit_linear_baseline, fit_power_law, fit_n_value_log_log,
run_full_fit, robust_view_range, FitSettings, FitResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


DEFAULT_DIDT_LOW_FRAC = 0.40
DEFAULT_DIDT_HIGH_FRAC = 0.60
DEFAULT_LINEAR_LOW_FRAC = 0.05
DEFAULT_LINEAR_HIGH_FRAC = 0.30
DEFAULT_POWER_LOW_FRAC = 0.05
DEFAULT_POWER_V_FRAC = 0.80
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_IC_TOLERANCE = 0.001    # 0.1 %
DEFAULT_CHI_SQR_TOL = 1.0e-9    # OriginLab-style tolerance on the fitter cost function
DEFAULT_VC_VOLTS = 1.0e-3
DEFAULT_EC_V_PER_CM = 1.0e-6    # 1 uV/cm = 100 uV/m, IEC 61788-3/-21 default for HTS at 77 K
DEFAULT_EC1_V_PER_CM = 1.0e-7   # 0.1 uV/cm, lower end of IEC decade n-value window
DEFAULT_EC2_V_PER_CM = 1.0e-6   # 1 uV/cm, upper end (= the Ic criterion)

# Fraction of Imax below which samples are considered part of the quiescent
# "I = 0" segment used to estimate the thermal offset V_ofs (Step 1).
DEFAULT_ZERO_I_FRAC = 0.02      # 2 %
# Post-fit warning thresholds.
RAMP_INDUCTIVE_WARN_RATIO = 0.10   # |L·dI/dt| / (Ec·L_v) above this → quasi-static assumption violated
MIN_N_WINDOW_POINTS = 50           # IEC decade fit becomes noisy with too few samples

# Fit method identifiers.
FIT_METHOD_LOG_LOG = "log_log"          # linear fit of log10(E_sc) vs log10(I), IEC 61788
FIT_METHOD_NONLINEAR = "nonlinear"       # coupled non-linear V = V0 + R*I + Vc*(I/Ic)^n
DEFAULT_FIT_METHOD = FIT_METHOD_LOG_LOG

# Point-weighting modes for Step-4 fitting.
WEIGHT_MODE_EQUAL = "equal"
WEIGHT_MODE_WEIGHTED = "weighted"
WEIGHT_MODE_ROBUST = "robust"
DEFAULT_WEIGHT_MODE = WEIGHT_MODE_EQUAL

# Step-3 baseline fitting mode identifiers.
BASELINE_MODE_OLS = "ols"
BASELINE_MODE_HUBER = "huber"
BASELINE_MODE_THEIL_SEN = "theil_sen"
# Default to Huber: robust against outliers while remaining smooth/stable
# for high-rate DAQ traces.
DEFAULT_BASELINE_MODE = BASELINE_MODE_HUBER


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
    # Step 1 — thermal offset subtraction.
    subtract_thermal_offset: bool = True
    zero_i_frac: float = DEFAULT_ZERO_I_FRAC
    weight_mode: str = DEFAULT_WEIGHT_MODE
    baseline_mode: str = DEFAULT_BASELINE_MODE


@dataclass
class FitResult:
    ok: bool
    message: str = ""
    di_dt: float = 0.0
    inductance_L: float = 0.0
    V_ofs: float = 0.0
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
    fit_method: str = DEFAULT_FIT_METHOD
    ec1: float = 0.0
    ec2: float = 0.0
    n_window_I: tuple[float, float] = (0.0, 0.0)
    n_points_used: int = 0
    # Parameter uncertainties (standard errors) and goodness of fit.
    sigma_Ic: float = 0.0
    sigma_n: float = 0.0
    r_squared: float = 0.0
    # Ramp-rate diagnostic: |L·dI/dt| / (Ec·L_v) — ratio of inductive
    # voltage drop to the Ic criterion voltage. IEC expects the
    # measurement to be effectively quasi-static; ratios above ~0.1
    # indicate the ramp is too fast.
    ramp_inductive_ratio: float = 0.0
    ramp_too_fast: bool = False
    insufficient_n_points: bool = False
    thermal_offset_applied: bool = False
    weighting_mode: str = DEFAULT_WEIGHT_MODE
    baseline_mode: str = DEFAULT_BASELINE_MODE


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


def adaptive_smooth_for_ec_window(y: np.ndarray, ec1: float, ec2: float) -> np.ndarray:
    """Adaptive smoothing shared by Step-4 UI and IEC log-log fit windowing.

    Goal: suppress high-frequency noise so Ec1/Ec2 crossing detection is stable.
    """
    arr = np.asarray(y, dtype=float)
    if arr.size < 7:
        return arr
    diffs = np.diff(arr)
    if diffs.size < 5:
        return arr
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    sigma_hf = 1.4826 * mad / np.sqrt(2.0)

    ec1_abs = max(abs(float(ec1)), 1e-30)
    # Keep this target aligned with the Step-4 UI guidance curve behavior.
    #
    # In IEC log-log mode, Ec1 can be very small (for example when using
    # µV/cm units). If we only scale by Ec1, target_sigma becomes so tiny that
    # the computed window can become unrealistically large and the helper curve
    # looks distorted. Anchor the target to a tiny fraction of the measured
    # signal span as well, so smoothing remains physically meaningful.
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo = float(np.percentile(finite, 5.0))
        hi = float(np.percentile(finite, 95.0))
        span = max(0.0, hi - lo)
    else:
        span = 0.0
    target_sigma = max(0.005 * ec1_abs, 1e-4 * span)
    if sigma_hf <= target_sigma:
        return arr

    # Savitzky-Golay preserves local curve shape (knee position/slope) much
    # better than a plain moving average, which is important for IEC windows.
    # We still size the window from the measured noise ratio.
    win = int(np.ceil((sigma_hf / max(target_sigma, 1e-30)) ** 2))
    # Keep helper curves stable on any dataset and avoid flattening transitions.
    n = int(arr.size)
    max_win_by_len = max(7, n // 20)
    if max_win_by_len % 2 == 0:
        max_win_by_len -= 1
    max_win = min(201, max_win_by_len if max_win_by_len >= 3 else 3)
    win = max(3, min(win, max_win))
    if win % 2 == 0:
        win += 1
    poly = 2 if win >= 5 else 1
    sm = savgol_filter(arr, window_length=win, polyorder=poly, mode="interp")

    # Optional second pass if still above target noise.
    diffs_sm = np.diff(sm)
    if diffs_sm.size >= 5:
        mad_sm = float(np.median(np.abs(diffs_sm - np.median(diffs_sm))))
        sigma_sm = 1.4826 * mad_sm / np.sqrt(2.0)
        if sigma_sm > target_sigma and win < max_win:
            win2 = min(max_win, max(win + 2, int(np.ceil(win * (sigma_sm / target_sigma) ** 2))))
            if win2 % 2 == 0:
                win2 += 1
            sm = savgol_filter(arr, window_length=win2, polyorder=poly, mode="interp")

    return sm


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


def estimate_thermal_offset(x: np.ndarray, y: np.ndarray,
                            zero_i_frac: float = DEFAULT_ZERO_I_FRAC) -> tuple[float, int]:
    """Estimate V_ofs (thermal offset) from the quiescent I = 0 segment.

    Points with |I| ≤ zero_i_frac · max|I| are treated as the I = 0 baseline.
    V_ofs is their median Y value (median is robust to the occasional outlier
    and to any remaining slow drift). Returns (V_ofs, n_points).

    If no points lie below the threshold (e.g. the recording starts after the
    ramp begins), V_ofs is returned as 0.0 with n_points = 0 so callers can
    skip the subtraction.
    """
    x, y = _clean_arrays(x, y)
    if x.size == 0:
        return 0.0, 0
    x_abs_max = float(np.max(np.abs(x)))
    if x_abs_max <= 0:
        return 0.0, 0
    threshold = max(zero_i_frac, 0.0) * x_abs_max
    mask = np.abs(x) <= threshold
    n = int(np.count_nonzero(mask))
    if n == 0:
        return 0.0, 0
    return float(np.median(y[mask])), n


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


def _theil_sen_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust line fit by median pairwise slope (Theil-Sen)."""
    n = int(x.size)
    if n < 2:
        raise ValueError("Not enough points for Theil-Sen baseline fit.")
    # Keep runtime bounded on very dense traces.
    if n > 200:
        idx = np.linspace(0, n - 1, 200, dtype=int)
        x = x[idx]
        y = y[idx]
        n = int(x.size)
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        valid = np.abs(dx) > 1e-12
        if np.any(valid):
            slopes.extend(((y[i + 1:][valid] - y[i]) / dx[valid]).tolist())
    if not slopes:
        raise ValueError("Degenerate current values in baseline window.")
    slope = float(np.median(np.asarray(slopes, dtype=float)))
    intercept = float(np.median(y - slope * x))
    return intercept, slope


def _huber_line(x: np.ndarray, y: np.ndarray, iterations: int = 12) -> tuple[float, float]:
    """Robust line fit by Huber IRLS."""
    slope, intercept = np.polyfit(x, y, 1)
    for _ in range(max(1, iterations)):
        residual = y - (intercept + slope * x)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        scale = max(1e-12, 1.4826 * mad)
        z = residual / scale
        w = _huber_weights(z)
        # np.polyfit weights apply to unsquared residuals; sqrt(w) gives WLS.
        slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(np.clip(w, 1e-12, None)))
    return float(intercept), float(slope)


def fit_linear_baseline(x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float,
                        mode: str = DEFAULT_BASELINE_MODE) -> tuple[float, float]:
    """Fit y = V0 + R*x on [x_lo, x_hi]. Returns (V0, R)."""
    x, y = _clean_arrays(x, y)
    mask = (x >= x_lo) & (x <= x_hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough points in linear baseline window.")
    xm = x[mask]
    ym = y[mask]
    fit_mode = str(mode or DEFAULT_BASELINE_MODE).strip().lower()
    if fit_mode == BASELINE_MODE_OLS:
        slope, intercept = np.polyfit(xm, ym, 1)
        return float(intercept), float(slope)
    if fit_mode == BASELINE_MODE_HUBER:
        return _huber_line(xm, ym)
    if fit_mode == BASELINE_MODE_THEIL_SEN:
        return _theil_sen_line(xm, ym)
    raise ValueError(f"Unknown baseline mode: {mode}")


def _power_law_model(x, Ic, n, V0, R, Vc):
    """Model with V0, R, Vc fixed — only Ic and n are free."""
    return V0 + R * x + Vc * np.power(np.clip(x / Ic, 1e-30, None), n)


def _rolling_median(values: np.ndarray, win: int) -> np.ndarray:
    """Small helper for robust local trend estimation without SciPy filters."""
    n = int(values.size)
    if n == 0:
        return values.copy()
    if win <= 1:
        return values.copy()
    if win % 2 == 0:
        win += 1
    half = win // 2
    out = np.empty_like(values, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def estimate_point_noise(x: np.ndarray, signal: np.ndarray,
                         zero_i_frac: float = DEFAULT_ZERO_I_FRAC) -> np.ndarray:
    """Estimate per-point sigma using local residuals + pre-ramp baseline.

    - Local term: rolling-median residual MAD.
    - Baseline floor: sigma from low-current (pre-ramp) segment.
    """
    x, signal = _clean_arrays(x, signal)
    n = signal.size
    if n == 0:
        return np.asarray([], dtype=float)
    if n < 5:
        return np.full(n, max(float(np.std(signal)), 1e-12), dtype=float)

    x_span = max(1e-30, float(np.max(x) - np.min(x)))
    win = max(7, int(np.ceil(0.03 * n)))
    if win % 2 == 0:
        win += 1
    trend = _rolling_median(signal, win)
    resid = signal - trend
    local_sigma = np.empty(n, dtype=float)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        r = resid[lo:hi]
        mad = float(np.median(np.abs(r - np.median(r))))
        local_sigma[i] = max(1e-12, 1.4826 * mad)

    # Pre-ramp baseline floor from low-current points.
    x_abs_max = float(np.max(np.abs(x)))
    thr = max(0.0, zero_i_frac) * x_abs_max
    base_mask = np.abs(x) <= thr
    if np.count_nonzero(base_mask) >= 4:
        base = signal[base_mask]
        base_mad = float(np.median(np.abs(base - np.median(base))))
        baseline_sigma = max(1e-12, 1.4826 * base_mad)
    else:
        baseline_sigma = max(1e-12, float(np.percentile(local_sigma, 25)))

    # Relax baseline floor slightly at high current where dynamic noise grows.
    xr = (x - float(np.min(x))) / x_span
    floor = baseline_sigma * (1.0 + 0.5 * np.clip(xr, 0.0, 1.0))
    return np.maximum(local_sigma, floor)


def _huber_weights(residuals: np.ndarray, c: float = 1.345) -> np.ndarray:
    a = np.abs(residuals)
    w = np.ones_like(a, dtype=float)
    mask = a > c
    w[mask] = c / np.maximum(a[mask], 1e-12)
    return w


def fit_power_law(x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float,
                  V0: float, R: float, Vc: float,
                  initial_Ic: Optional[float] = None,
                  initial_n: float = 20.0,
                  chi_sqr_tol: float = DEFAULT_CHI_SQR_TOL,
                  point_sigma: Optional[np.ndarray] = None,
                  weight_mode: str = DEFAULT_WEIGHT_MODE,
                  ) -> tuple[float, float, float, float, float, float]:
    """Fit Ic, n in y = V0 + R*x + Vc*(x/Ic)^n on [x_lo, x_hi] with V0, R, Vc fixed.

    Returns (Ic, n, chi_sqr, sigma_Ic, sigma_n, r_squared).
    Uncertainties come from the scaled covariance reported by ``curve_fit``;
    R² is 1 − SS_res/SS_tot over the fit window.
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

    sigma_fit = None
    if point_sigma is not None:
        sig_all = np.asarray(point_sigma, dtype=float)
        sig_all = sig_all[:x.size]
        sig_m = sig_all[mask]
        if sig_m.size == xm.size:
            sigma_fit = np.clip(sig_m, 1e-12, None)
    robust_mode = (weight_mode == WEIGHT_MODE_ROBUST)
    # In robust mode run an IRLS loop by inflating sigma for outliers.
    robust_w = np.ones_like(xm, dtype=float)
    for _ in range(5 if robust_mode else 1):
        sigma_eff = None
        if sigma_fit is not None:
            sigma_eff = sigma_fit / np.sqrt(np.clip(robust_w, 1e-6, None))
        popt, pcov = curve_fit(
            model, xm, ym, p0=p0, bounds=bounds, maxfev=10000,
            sigma=sigma_eff, absolute_sigma=False,
            ftol=chi_sqr_tol, xtol=chi_sqr_tol, gtol=chi_sqr_tol,
        )
        p0 = [float(popt[0]), float(popt[1])]
        if not robust_mode:
            break
        pred = model(xm, p0[0], p0[1])
        resid = ym - pred
        if sigma_fit is not None:
            resid = resid / np.clip(sigma_fit, 1e-12, None)
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale <= 1e-12:
            break
        robust_w = _huber_weights(resid / scale)
    Ic = float(popt[0])
    n_val = float(popt[1])
    model_y = model(xm, Ic, n_val)
    residuals = ym - model_y
    chi_sqr = float(np.sum(residuals ** 2))
    # Standard errors from the covariance diagonal (curve_fit has already
    # scaled it by the residual variance unless absolute_sigma=True).
    try:
        sigma_Ic = float(np.sqrt(max(0.0, pcov[0, 0])))
        sigma_n = float(np.sqrt(max(0.0, pcov[1, 1])))
        if not np.isfinite(sigma_Ic):
            sigma_Ic = 0.0
        if not np.isfinite(sigma_n):
            sigma_n = 0.0
    except Exception:
        sigma_Ic = 0.0
        sigma_n = 0.0
    ss_tot = float(np.sum((ym - np.mean(ym)) ** 2))
    r_squared = float(1.0 - chi_sqr / ss_tot) if ss_tot > 0 else 0.0
    return Ic, n_val, chi_sqr, sigma_Ic, sigma_n, r_squared


def fit_n_value_log_log(x: np.ndarray, y: np.ndarray,
                        V0: float, R: float,
                        Ec1: float, Ec2: float,
                        criterion_E: Optional[float] = None,
                        point_sigma: Optional[np.ndarray] = None,
                        weight_mode: str = DEFAULT_WEIGHT_MODE,
                        ) -> tuple[float, float, float, int, tuple[float, float],
                                   float, float, float]:
    """IEC 61788 decade n-value: linear fit of log10(E_sc) vs log10(I).

    E_sc = y - V0 - R*x is the baseline-subtracted signal. To keep Step-4
    Low(X)/High(X), the plotted helper curve, and the actual log-log fit
    fully aligned, point selection for the fit is based on that same
    corrected+smoothed curve.

    The slope of log10(E_sc) vs log10(I) on this segment is the n-index;
    Ic is reported at E = Ec2 (the IEC criterion for HTS at 77 K).

    How the I-window is computed for this log-log power-law fit:
      1) Sort by current and compute E_sc = y - (V0 + R*I).
      2) Build E_sc_smooth with adaptive_smooth_for_ec_window(...).
      3) Ignore the first 20 % of points (ramp-start guard).
      4) Keep points where E_sc_smooth is inside [Ec1, Ec2].
      5) Fit log10(E_sc_smooth) vs log10(I).
      6) Report I-window from the same threshold crossings used by Step-4 UI:
         first I where E_sc_smooth >= Ec1 and first I where E_sc_smooth >= Ec2.

    Returns (Ic_at_Ec2, n, chi_sqr, n_points, (I_lo, I_hi),
             sigma_Ic, sigma_n, r_squared).
    Standard errors are derived from the log-space polyfit covariance;
    R² is computed on log10(E_sc) vs the linear model.
    """
    x, y = _clean_arrays(x, y)
    if Ec2 <= Ec1 or Ec1 <= 0:
        raise ValueError("Ec1 must be > 0 and strictly less than Ec2.")
    # Sort by current so "transition segment" is a contiguous index range.
    order = np.argsort(x)
    xs = x[order]
    e_sc = y[order] - V0 - R * xs
    pos = xs > 0
    if not np.any(pos):
        raise ValueError("No points with I > 0; cannot fit on a log axis.")
    xs = xs[pos]
    e_sc = e_sc[pos]
    e_sc_bounds = adaptive_smooth_for_ec_window(e_sc, Ec1, Ec2)
    idx_guard_lo = int(np.floor(0.20 * xs.size))
    in_guard = np.arange(xs.size) >= idx_guard_lo
    above_Ec2 = np.where((e_sc_bounds >= Ec2) & in_guard)[0]
    if above_Ec2.size == 0:
        raise ValueError(
            f"Data never reaches Ec2 = {Ec2:.3g} on the corrected+smoothed curve; "
            "ramp further or lower Ec2."
        )
    mask = (
        (e_sc_bounds >= Ec1)
        & (e_sc_bounds <= Ec2)
        & in_guard
        & np.isfinite(e_sc_bounds)
        & np.isfinite(xs)
    )
    n_pts = int(np.count_nonzero(mask))
    if n_pts < 4:
        raise ValueError(
            f"Only {n_pts} points fall inside the IEC n-value window "
            f"[{Ec1:.3g}, {Ec2:.3g}] on the corrected+smoothed curve. "
            "Slow the ramp, reduce averaging, or widen the decade."
        )
    log_I = np.log10(xs[mask])
    e_fit = np.clip(e_sc_bounds[mask], 1e-30, None)
    log_E = np.log10(e_fit)
    w = np.ones_like(log_E, dtype=float)
    if point_sigma is not None and weight_mode in (WEIGHT_MODE_WEIGHTED, WEIGHT_MODE_ROBUST):
        sig_all = np.asarray(point_sigma, dtype=float)
        sig_all = sig_all[:x.size]
        sig_sorted = sig_all[order][pos]
        sig_seg = np.clip(sig_sorted[mask], 1e-12, None)
        sigma_log = np.clip(sig_seg / (e_fit * np.log(10.0)), 1e-12, None)
        w = 1.0 / (sigma_log ** 2)

    robust_mode = (weight_mode == WEIGHT_MODE_ROBUST)
    coeffs = np.polyfit(log_I, log_E, 1, w=np.sqrt(w))
    for _ in range(5 if robust_mode else 1):
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        if not robust_mode:
            break
        res = log_E - (intercept + slope * log_I)
        scale = 1.4826 * np.median(np.abs(res - np.median(res)))
        if scale <= 1e-12:
            break
        w_rob = _huber_weights(res / scale)
        coeffs = np.polyfit(log_I, log_E, 1, w=np.sqrt(w * w_rob))
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    try:
        _, cov = np.polyfit(log_I, log_E, 1, w=np.sqrt(w), cov=True)
        sigma_slope = float(np.sqrt(max(0.0, cov[0, 0])))
        sigma_intercept = float(np.sqrt(max(0.0, cov[1, 1])))
        cov_slope_intercept = float(cov[0, 1])
    except (ValueError, np.linalg.LinAlgError):
        sigma_slope = sigma_intercept = cov_slope_intercept = 0.0
    n_val = slope
    if abs(n_val) < 1e-12:
        raise ValueError("Power-law slope collapsed to zero; cannot solve for Ic.")
    crit_E = float(Ec2 if (criterion_E is None or criterion_E <= 0) else criterion_E)
    log_crit = float(np.log10(crit_E))
    log_Ic = (log_crit - intercept) / n_val
    Ic_at_crit = float(10.0 ** log_Ic)
    model_log_E = intercept + n_val * log_I
    chi_sqr = float(np.sum((log_E - model_log_E) ** 2))
    ss_tot = float(np.sum((log_E - np.mean(log_E)) ** 2))
    r_squared = float(1.0 - chi_sqr / ss_tot) if ss_tot > 0 else 0.0
    # Uncertainty in log10(Ic) from propagation through
    # log_Ic = (log_crit - intercept) / slope.
    d_by_intercept = -1.0 / n_val
    d_by_slope = -(log_crit - intercept) / (n_val ** 2)
    var_log_Ic = (
        d_by_intercept ** 2 * sigma_intercept ** 2
        + d_by_slope ** 2 * sigma_slope ** 2
        + 2.0 * d_by_intercept * d_by_slope * cov_slope_intercept
    )
    sigma_log_Ic = float(np.sqrt(max(0.0, var_log_Ic)))
    # σ(Ic) ≈ Ic · ln(10) · σ(log10 Ic) for small relative error.
    sigma_Ic = float(Ic_at_crit * np.log(10.0) * sigma_log_Ic)
    sigma_n = float(sigma_slope)
    # Report I-window with the same rule used by Step-4 Low(X)/High(X):
    # first threshold crossings on the corrected+smoothed reference trace.
    # Ignore first 20 % of points when picking Ec1/Ec2 crossings for the
    # reported Low(X)/High(X) window edges (ramp-start region is often noisy).
    idx_lo_all = np.where((e_sc_bounds >= Ec1) & in_guard)[0]
    idx_hi_all = np.where((e_sc_bounds >= Ec2) & in_guard)[0]
    I_lo = float(xs[idx_lo_all[0]]) if idx_lo_all.size else float(xs[-1])
    I_hi = float(xs[idx_hi_all[0]]) if idx_hi_all.size else float(xs[-1])
    if I_hi <= I_lo:
        I_hi = I_lo + max(1e-12, 0.01 * (float(np.max(xs)) - float(np.min(xs))))
    return (Ic_at_crit, n_val, chi_sqr, n_pts, (I_lo, I_hi),
            sigma_Ic, sigma_n, r_squared)


def _ramp_ratio(V0: float, criterion: float) -> float:
    """|L·dI/dt| expressed as a fraction of the Ic criterion voltage.

    After the thermal offset has been removed, V0 (the intercept from
    the linear baseline fit) is exactly L·dI/dt in the same Y-units as
    the criterion (V or V/cm), so the ratio is |V0| / criterion.
    """
    if criterion is None or criterion == 0 or not np.isfinite(criterion):
        return 0.0
    return float(abs(V0) / abs(criterion))


def run_full_fit(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                 settings: Optional[FitSettings] = None) -> FitResult:
    """Step 1 V_ofs, Step 2 di/dt, Step 3 baseline → V0, R, L, Step 4 Ic/n.

    Step 1 (optional): estimate V_ofs from the I = 0 segment and subtract
    it from y so the downstream baseline fit isolates the inductive term
    (V0 = L·dI/dt) cleanly from the thermal offset.
    Step 2 estimates dI/dt from the linear ramp.
    Step 3 fits y - V_ofs = V0 + R·I on the low-current baseline window.
    Step 4 fits Ic and n; default is the IEC 61788 log-log decade method
    (``settings.fit_method == FIT_METHOD_LOG_LOG``). The legacy coupled
    non-linear fit of V = V0 + R·I + Vc·(I/Ic)^n remains available as
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

    # Step 1: subtract the thermal offset measured on the quiescent I = 0
    # segment. Without this, the baseline fit lumps V_ofs into V0 and the
    # inductive-ratio diagnostic (|L·dI/dt| / (Ec·L_v)) is wrong.
    V_ofs = 0.0
    thermal_applied = False
    if getattr(settings, "subtract_thermal_offset", True):
        V_ofs, n_zero = estimate_thermal_offset(x, y, settings.zero_i_frac)
        if n_zero > 0:
            y = y - V_ofs
            thermal_applied = True
        else:
            V_ofs = 0.0

    # Step 2: di/dt on the linear-ramp window.
    di_dt = estimate_di_dt(t, x, settings.didt_low_frac, settings.didt_high_frac)

    # Step 3: linear baseline → V0 (= L·dI/dt in Y-units after Step 1) and R.
    lin_lo = x_min + settings.linear_low_frac * (x_max - x_min)
    lin_hi = x_min + settings.linear_high_frac * (x_max - x_min)
    baseline_mode = str(getattr(settings, "baseline_mode", DEFAULT_BASELINE_MODE) or DEFAULT_BASELINE_MODE)
    try:
        V0, R = fit_linear_baseline(x, y, lin_lo, lin_hi, mode=baseline_mode)
    except ValueError as exc:
        return FitResult(ok=False, message=f"Linear baseline fit failed: {exc}")

    # V0 is in Y-units. With sample-length normalisation, Y is in V/cm, so
    # the full inductive voltage is V0 * L_v and L = (V0 * L_v) / di_dt.
    v0_voltage = V0 * float(settings.sample_length_cm) if uses_length else V0
    inductance_L = v0_voltage / di_dt if abs(di_dt) > 1e-30 else 0.0

    method = getattr(settings, "fit_method", DEFAULT_FIT_METHOD)
    weight_mode = getattr(settings, "weight_mode", DEFAULT_WEIGHT_MODE)
    point_sigma = None
    if weight_mode in (WEIGHT_MODE_WEIGHTED, WEIGHT_MODE_ROBUST):
        point_sigma = estimate_point_noise(x, y - V0 - R * x, settings.zero_i_frac)

    if method == FIT_METHOD_LOG_LOG:
        # Use the user-entered Ec/Vc as the criterion for Ic if provided;
        # fall back to Ec2 (the legacy default) when criterion_voltage is
        # not set or non-positive.
        crit_for_ic = Vc if (Vc is not None and Vc > 0) else settings.ec2
        try:
            (Ic, n_value, chi_sqr, n_pts, n_window,
             sigma_Ic, sigma_n, r_squared) = fit_n_value_log_log(
                x, y, V0=V0, R=R, Ec1=settings.ec1, Ec2=settings.ec2,
                criterion_E=crit_for_ic,
                point_sigma=point_sigma,
                weight_mode=weight_mode,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return FitResult(ok=False, message=f"Log-log n-value fit failed: {exc}")
        # Rebuild a smooth model curve for plotting using the user's criterion.
        fit_x = np.linspace(max(x_min, 1e-12), x_max, 400)
        fit_y = V0 + R * fit_x + crit_for_ic * np.power(
            np.clip(fit_x / Ic, 1e-30, None), n_value
        )
        # Add the thermal offset back so the model curve aligns with the
        # raw (unshifted) data the user still sees on screen.
        if thermal_applied:
            fit_y = fit_y + V_ofs
        ratio = _ramp_ratio(V0, crit_for_ic)
        return FitResult(
            ok=True,
            message="IEC 61788 log-log n-value fit succeeded.",
            di_dt=di_dt,
            inductance_L=inductance_L,
            V_ofs=V_ofs,
            V0=V0,
            R=R,
            Ic=Ic,
            n_value=n_value,
            criterion=crit_for_ic,
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
            sigma_Ic=sigma_Ic,
            sigma_n=sigma_n,
            r_squared=r_squared,
            ramp_inductive_ratio=ratio,
            ramp_too_fast=ratio > RAMP_INDUCTIVE_WARN_RATIO,
            insufficient_n_points=n_pts < MIN_N_WINDOW_POINTS,
            thermal_offset_applied=thermal_applied,
            weighting_mode=weight_mode,
            baseline_mode=baseline_mode,
        )

    y_max = float(np.max(y))
    y_threshold = settings.power_v_frac * y_max
    above = np.where(y >= y_threshold)[0]
    power_hi = float(x[above[0]]) if above.size else x_max
    power_lo = x_min + settings.power_low_frac * (x_max - x_min)

    Ic = float("nan")
    n_value = float("nan")
    chi_sqr = 0.0
    sigma_Ic = 0.0
    sigma_n = 0.0
    r_squared = 0.0
    ic_history: list[float] = []
    last_Ic = None
    iterations_used = 0
    for iteration in range(1, max(1, settings.max_iterations) + 1):
        iterations_used = iteration
        try:
            Ic, n_value, chi_sqr, sigma_Ic, sigma_n, r_squared = fit_power_law(
                x, y, power_lo, power_hi,
                V0=V0, R=R, Vc=Vc,
                initial_Ic=last_Ic,
                chi_sqr_tol=settings.chi_sqr_tolerance,
                point_sigma=point_sigma,
                weight_mode=weight_mode,
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
    if thermal_applied:
        fit_y = fit_y + V_ofs

    ratio = _ramp_ratio(V0, Vc)

    return FitResult(
        ok=True,
        message="Fit succeeded.",
        di_dt=di_dt,
        inductance_L=inductance_L,
        V_ofs=V_ofs,
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
        sigma_Ic=sigma_Ic,
        sigma_n=sigma_n,
        r_squared=r_squared,
        ramp_inductive_ratio=ratio,
        ramp_too_fast=ratio > RAMP_INDUCTIVE_WARN_RATIO,
        insufficient_n_points=False,
        thermal_offset_applied=thermal_applied,
        weighting_mode=weight_mode,
        baseline_mode=baseline_mode,
    )

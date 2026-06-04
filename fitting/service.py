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
DEFAULT_LINEAR_LOW_FRAC = 0.15
DEFAULT_LINEAR_HIGH_FRAC = 0.50
DEFAULT_POWER_LOW_FRAC = 0.05
DEFAULT_POWER_V_FRAC = 0.80
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_IC_TOLERANCE = 0.001    # 0.1 %
DEFAULT_CHI_SQR_TOL = 1.0e-14    # OriginLab-style tolerance on the fitter cost function
DEFAULT_VC_VOLTS = 1.0e-3
DEFAULT_EC_V_PER_CM = 1.0e-6    # 1 uV/cm = 100 uV/m, IEC 61788-3/-21 default for HTS at 77 K
DEFAULT_EC1_V_PER_CM = 1.0e-7   # 0.1 uV/cm, lower end of IEC decade n-value window
DEFAULT_EC2_V_PER_CM = 1.0e-6   # 1 uV/cm, upper end (= the Ic criterion)
DEFAULT_EC_WINDOW_GUARD_FRAC = 0.50

# Auto-adjust Ec1/Ec2 defaults. When enabled, the IEC decade window is allowed
# to slide both upward and downward as a unit (both ends scaled by the same
# factor k, so Ec2/Ec1 stays at the IEC 10:1 ratio) to escape a drifting
# baseline at one end or shot noise at the other. The user-supplied min/max
# caps below define how far each end is allowed to move; the search reduces
# the joint constraint to a 1D interval on Ec2.
DEFAULT_AUTO_EC_TARGET_R2 = 0.998
DEFAULT_AUTO_EC1_MIN_V_PER_CM = 1.0e-7   # 0.1 µV/cm = the IEC default Ec1
DEFAULT_AUTO_EC2_MIN_V_PER_CM = 1.0e-6   # 1.0 µV/cm = the IEC default Ec2
DEFAULT_AUTO_EC1_MAX_V_PER_CM = 1.0e-6   # 1.0 µV/cm = 10× the IEC default Ec1
DEFAULT_AUTO_EC2_MAX_V_PER_CM = 5.0e-6   # 5.0 µV/cm = 5× the IEC default Ec2
AUTO_EC_MAX_ITERATIONS = 8
AUTO_EC_REL_TOL = 0.05  # stop refinement when bracket width ≤ 5 %
AUTO_EC_PROBES_PER_SIDE = 5  # log-spaced probes per scaling direction
IEC_DECADE_RATIO = 10.0  # IEC 61788 fixes Ec2/Ec1 = 10 (one decade)

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

# Point-weighting modes for Step-5 fitting.
WEIGHT_MODE_EQUAL = "equal"
WEIGHT_MODE_WEIGHTED = "weighted"
WEIGHT_MODE_ROBUST = "robust"
DEFAULT_WEIGHT_MODE = WEIGHT_MODE_WEIGHTED
# Transition-region emphasis applied on top of inverse-variance weights in
# weighted / robust mode. The weight of a point grows linearly in log-E from
# 1 (at Ec1) to (1 + TRANSITION_WEIGHT_GAIN) (at Ec2), so the cleaner upper
# part of the n-value window dominates the fit and the reported R².
TRANSITION_WEIGHT_GAIN = 1.0

# Step-4 baseline fitting mode identifiers.
BASELINE_MODE_OLS = "ols"
BASELINE_MODE_HUBER = "huber"
BASELINE_MODE_THEIL_SEN = "theil_sen"
# Default to Huber: robust against outliers while remaining smooth/stable
# for high-rate DAQ traces.
DEFAULT_BASELINE_MODE = BASELINE_MODE_HUBER
# Step-3 di/dt slope estimator mode identifiers.
DIDT_MODE_OLS = BASELINE_MODE_OLS
DIDT_MODE_HUBER = BASELINE_MODE_HUBER
DIDT_MODE_THEIL_SEN = BASELINE_MODE_THEIL_SEN
DEFAULT_DIDT_MODE = DIDT_MODE_HUBER


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
    # Auto-adjust the IEC decade window when R² of the log-log fit falls below
    # ``auto_ec_target_r2``. When ``auto_ec_lock_iec_ratio`` is True (default,
    # IEC 61788-compliant) both ends are scaled by the same factor so the
    # decade ratio Ec2/Ec1 = 10 is preserved. When False, Ec1 and Ec2 are
    # searched independently within their caps (legacy behaviour, useful when
    # the data does not support a full decade window). The window may slide
    # either upward (capped by ``auto_ec1_max``/``auto_ec2_max``) or downward
    # (floored by ``auto_ec1_min``/``auto_ec2_min``). ``None`` on a bound
    # disables the search in that direction.
    auto_ec_adjust: bool = False
    auto_ec_lock_iec_ratio: bool = True
    auto_ec1_min: Optional[float] = None
    auto_ec2_min: Optional[float] = None
    auto_ec1_max: Optional[float] = None
    auto_ec2_max: Optional[float] = None
    auto_ec_target_r2: float = DEFAULT_AUTO_EC_TARGET_R2
    # Step 1 — thermal offset subtraction.
    subtract_thermal_offset: bool = True
    zero_i_frac: float = DEFAULT_ZERO_I_FRAC
    weight_mode: str = DEFAULT_WEIGHT_MODE
    baseline_mode: str = DEFAULT_BASELINE_MODE
    didt_mode: str = DEFAULT_DIDT_MODE
    # Optional reference range used to interpret didt/linear/power
    # ``*_low_frac`` and ``*_high_frac`` as fractions of the *untrimmed*
    # current sweep instead of the trimmed array's extents. The Data Fitting
    # tab anchors percentages to the untrimmed range so the displayed
    # Low(X)/High(X) editors stay stable when the Step 2 trim changes; this
    # field lets the fitter use the same anchor so the actual fit window
    # matches what the user sees on screen.
    pct_x_min: Optional[float] = None
    pct_x_max: Optional[float] = None


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
    # Populated when the auto-adjust feature shifted the decade window. The
    # ``ec1``/``ec2`` fields above always reflect the values *actually used*
    # for the fit; ``ec1_initial``/``ec2_initial`` keep the user's original
    # entry so the report is honest about what was changed.
    ec1_auto_adjusted: bool = False
    ec1_initial: float = 0.0
    ec2_initial: float = 0.0
    auto_ec_iterations: int = 0
    auto_ec_target_r2: float = 0.0
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
    # Captured at fit time so the metadata writer can convert R (Ω/cm) and
    # V_ofs (V/cm) back to total-tape units (Ω, V) for the saved properties.
    sample_length_cm: Optional[float] = None
    # Block-average factor applied upstream of the fit (1 = no averaging).
    avg_window: int = 1


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
        lo = lo + 1.0
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def adaptive_smooth_for_ec_window(y: np.ndarray, ec1: float, ec2: float) -> np.ndarray:
    """Adaptive smoothing shared by Step-5 UI and IEC log-log fit windowing.

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
    # Keep this target aligned with the Step-5 UI guidance curve behavior.
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
    # Also constrain peak-to-peak noise of the smoothed helper curve so that
    # Step-4 crossings remain stable across very different sample rates
    # (for example 10 S/s vs 10 kS/s datasets).
    target_p2p = max(0.9 * ec1_abs, 1e-3 * span)
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

    # Iteratively increase smoothing until both:
    #   1) high-frequency sigma is below target_sigma, and
    #   2) residual peak-to-peak noise is below target_p2p (≈ Ec1).
    # This keeps the curve shape while adapting to widely different sample
    # rates and point densities.
    max_iter = 6
    for _ in range(max_iter):
        diffs_sm = np.diff(sm)
        if diffs_sm.size >= 5:
            mad_sm = float(np.median(np.abs(diffs_sm - np.median(diffs_sm))))
            sigma_sm = 1.4826 * mad_sm / np.sqrt(2.0)
        else:
            sigma_sm = 0.0

        resid = arr - sm
        finite_resid = resid[np.isfinite(resid)]
        if finite_resid.size >= 8:
            p2p_noise = float(np.percentile(finite_resid, 95.0) - np.percentile(finite_resid, 5.0))
        elif finite_resid.size:
            p2p_noise = float(np.max(finite_resid) - np.min(finite_resid))
        else:
            p2p_noise = 0.0

        if sigma_sm <= target_sigma and p2p_noise <= target_p2p:
            break
        if win >= max_win:
            break

        grow_sigma = (sigma_sm / max(target_sigma, 1e-30)) ** 2
        grow_p2p = (p2p_noise / max(target_p2p, 1e-30)) ** 1.5
        grow = max(1.2, grow_sigma, grow_p2p)
        win_next = int(np.ceil(win * min(grow, 2.0))) + 2
        win = min(max_win, max(win + 2, win_next))
        if win % 2 == 0:
            win += 1
        poly = 2 if win >= 5 else 1
        sm = savgol_filter(arr, window_length=win, polyorder=poly, mode="interp")

    return sm


def pick_loglog_i_window_from_thresholds(
    x_sorted: np.ndarray,
    e_sc_smoothed: np.ndarray,
    *,
    ec1: float,
    ec2: float,
    guard_fraction: float = DEFAULT_EC_WINDOW_GUARD_FRAC,
) -> tuple[float, float]:
    """Pick (Low(X), High(X)) from corrected+smoothed IEC thresholds."""
    xs = np.asarray(x_sorted, dtype=float)
    ys = np.asarray(e_sc_smoothed, dtype=float)
    n = int(min(xs.size, ys.size))
    if n == 0:
        return 0.0, 0.0
    xs = xs[:n]
    ys = ys[:n]
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return 0.0, 0.0
    xs = xs[finite]
    ys = ys[finite]
    if xs.size == 0:
        return 0.0, 0.0

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    span = max(0.0, x_max - x_min)
    x_guard_lo = x_min + float(np.clip(guard_fraction, 0.0, 0.95)) * span
    in_guard = xs >= x_guard_lo

    idx_hi_all = np.where((ys >= ec2) & in_guard)[0]
    idx_hi = int(idx_hi_all[0]) if idx_hi_all.size else int(xs.size - 1)

    # Low(X): walk backwards from High(X) while current decreases and stop at
    # the first point that reaches Ec1 (or below).
    idx_lo = idx_hi
    for j in range(idx_hi, -1, -1):
        if ys[j] <= ec1:
            idx_lo = j
            break

    i_lo = float(xs[idx_lo])
    i_hi = float(xs[idx_hi])
    if i_hi <= i_lo:
        i_hi = i_lo + max(1e-12, 0.01 * (span if span > 0 else 1.0))
    return i_lo, i_hi


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
                   high_frac: float = DEFAULT_DIDT_HIGH_FRAC,
                   mode: str = DEFAULT_DIDT_MODE,
                   x_lo: Optional[float] = None,
                   x_hi: Optional[float] = None) -> float:
    t, x = _clean_arrays(t, x)
    if t.size < 2:
        return 0.0
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return 0.0
    if x_lo is not None and x_hi is not None:
        lo = float(x_lo)
        hi = float(x_hi)
    else:
        lo = x_min + low_frac * (x_max - x_min)
        hi = x_min + high_frac * (x_max - x_min)
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        return 0.0
    tm = t[mask]
    xm = x[mask]
    fit_mode = str(mode or DEFAULT_DIDT_MODE).strip().lower()
    if fit_mode == DIDT_MODE_OLS:
        slope, _ = np.polyfit(tm, xm, 1)
        return float(slope)
    if fit_mode == DIDT_MODE_HUBER:
        _, slope = _huber_line(tm, xm)
        return float(slope)
    if fit_mode == DIDT_MODE_THEIL_SEN:
        _, slope = _theil_sen_line(tm, xm)
        return float(slope)
    raise ValueError(f"Unknown di/dt mode: {mode}")


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

    E_sc = y - V0 - R*x is the baseline-subtracted signal. To keep Step-5
    Low(X)/High(X), the plotted helper curve, and the actual log-log fit
    fully aligned, point selection for the fit is based on that same
    corrected+smoothed curve.

    The slope of log10(E_sc) vs log10(I) on this segment is the n-index;
    Ic is reported at E = Ec2 (the IEC criterion for HTS at 77 K).

    How the I-window is computed for this log-log power-law fit:
      1) Sort by current and compute E_sc = y - (V0 + R*I).
      2) Build E_sc_smooth with adaptive_smooth_for_ec_window(...).
      3) Ignore the first 50 % of current span (ramp-start guard).
      4) Keep points where E_sc_smooth is inside [Ec1, Ec2].
      5) Fit log10(E_sc_smooth) vs log10(I).
      6) Report I-window from the same threshold crossings used by Step-5 UI:
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
    x_guard_lo = float(np.min(xs)) + DEFAULT_EC_WINDOW_GUARD_FRAC * (float(np.max(xs)) - float(np.min(xs)))
    in_guard = xs >= x_guard_lo
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
    if weight_mode in (WEIGHT_MODE_WEIGHTED, WEIGHT_MODE_ROBUST):
        if point_sigma is not None:
            sig_all = np.asarray(point_sigma, dtype=float)
            sig_all = sig_all[:x.size]
            sig_sorted = sig_all[order][pos]
            sig_seg = np.clip(sig_sorted[mask], 1e-12, None)
            sigma_log = np.clip(sig_seg / (e_fit * np.log(10.0)), 1e-12, None)
            w = 1.0 / (sigma_log ** 2)
        # Transition emphasis: ramp the weight linearly in log-E from 1 at
        # Ec1 to (1 + TRANSITION_WEIGHT_GAIN) at Ec2. The upper part of the
        # decade is where the power-law signal is strongest and least
        # contaminated by baseline drift, so weighting it more heavily lets
        # the slope (and the reported R²) track the cleaner transition
        # rather than the noise floor near Ec1.
        log_ec1 = float(np.log10(Ec1))
        log_ec2 = float(np.log10(Ec2))
        span = max(log_ec2 - log_ec1, 1e-12)
        position = np.clip((log_E - log_ec1) / span, 0.0, 1.0)
        w = w * (1.0 + TRANSITION_WEIGHT_GAIN * position)

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
    residuals = log_E - model_log_E
    chi_sqr = float(np.sum(residuals ** 2))
    # Weighted R² when inverse-variance weights are in play: down-weights
    # noisy Ec1-end points whose σ_log dwarfs the signal, so the metric
    # tracks fit quality across the cleaner transition instead of being
    # dragged down by the baseline-dominated low end. Reduces to the standard
    # unweighted R² when w ≡ 1 (equal-weight mode).
    w_sum = float(np.sum(w))
    if w_sum > 0:
        log_E_mean_w = float(np.sum(w * log_E) / w_sum)
        ss_res_w = float(np.sum(w * residuals ** 2))
        ss_tot_w = float(np.sum(w * (log_E - log_E_mean_w) ** 2))
        r_squared = float(1.0 - ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0.0
    else:
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
    I_lo, I_hi = pick_loglog_i_window_from_thresholds(
        xs, e_sc_bounds, ec1=Ec1, ec2=Ec2, guard_fraction=DEFAULT_EC_WINDOW_GUARD_FRAC,
    )
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


def auto_adjust_loglog_window(
    x: np.ndarray, y: np.ndarray, *,
    V0: float, R: float,
    ec1: float, ec2: float,
    criterion_E: float,
    point_sigma: Optional[np.ndarray],
    weight_mode: str,
    ec1_min: Optional[float] = None,
    ec2_min: Optional[float] = None,
    ec1_max: Optional[float],
    ec2_max: Optional[float],
    target_r2: float,
    lock_iec_ratio: bool = True,
    max_iterations: int = AUTO_EC_MAX_ITERATIONS,
    rel_tol: float = AUTO_EC_REL_TOL,
) -> tuple[tuple, float, float, int, bool]:
    """Auto-adjust the log-log decade window to maximize R².

    Two search modes are available, selected by ``lock_iec_ratio``:

    * ``True`` (default, IEC 61788-compliant): the pair slides as a unit
      with ``Ec2/Ec1 = 10`` held fixed. See
      :func:`_auto_adjust_loglog_window_iec_ratio`.
    * ``False`` (legacy): ``Ec1`` and ``Ec2`` are searched independently
      within their caps with only a 3:1 minimum ratio. See
      :func:`_auto_adjust_loglog_window_independent`.

    Returns ``(loglog_result, ec1_used, ec2_used, n_evals, target_met)``.
    """
    if lock_iec_ratio:
        return _auto_adjust_loglog_window_iec_ratio(
            x, y, V0=V0, R=R, ec1=ec1, ec2=ec2,
            criterion_E=criterion_E, point_sigma=point_sigma,
            weight_mode=weight_mode,
            ec1_min=ec1_min, ec2_min=ec2_min,
            ec1_max=ec1_max, ec2_max=ec2_max,
            target_r2=target_r2,
            max_iterations=max_iterations, rel_tol=rel_tol,
        )
    return _auto_adjust_loglog_window_independent(
        x, y, V0=V0, R=R, ec1=ec1, ec2=ec2,
        criterion_E=criterion_E, point_sigma=point_sigma,
        weight_mode=weight_mode,
        ec1_min=ec1_min, ec2_min=ec2_min,
        ec1_max=ec1_max, ec2_max=ec2_max,
        target_r2=target_r2,
        max_iterations=max_iterations, rel_tol=rel_tol,
    )


def _auto_adjust_loglog_window_iec_ratio(
    x: np.ndarray, y: np.ndarray, *,
    V0: float, R: float,
    ec1: float, ec2: float,
    criterion_E: float,
    point_sigma: Optional[np.ndarray],
    weight_mode: str,
    ec1_min: Optional[float],
    ec2_min: Optional[float],
    ec1_max: Optional[float],
    ec2_max: Optional[float],
    target_r2: float,
    max_iterations: int,
    rel_tol: float,
) -> tuple[tuple, float, float, int, bool]:
    """Slide the IEC decade window to maximize R² while preserving Ec2/Ec1 = 10.

    The decade ratio mandated by IEC 61788 (``Ec2 = 10·Ec1``) is held fixed
    so the auto-adjusted window remains a true decade fit and the reported
    Ic / n stay comparable to a standard IEC measurement. Only the absolute
    position of the pair is varied: a single scale factor moves both ends
    together. Ic is reported at ``Ec2`` regardless of where the window
    lands, so the criterion definition shifts with the window — this is
    intentional and reflected in the saved ``Ec1``, ``Ec2`` properties.

    Bounds: the search is carried out in ``Ec2`` space, with the feasible
    range derived from the intersection of the two user-supplied caps:

        Ec2 ∈ [max(ec2_min, R·ec1_min), min(ec2_max, R·ec1_max)]

    where ``R = IEC_DECADE_RATIO``. ``Ec1`` is then ``Ec2/R``. If the caps
    are incompatible the function falls back to evaluating the user's
    starting pair.
    """
    GRID_N = 7
    R_RATIO = float(IEC_DECADE_RATIO)
    phi = (1.0 + 5.0 ** 0.5) / 2.0

    def _eval(e1: float, e2: float):
        return fit_n_value_log_log(
            x, y, V0=V0, R=R,
            Ec1=float(e1), Ec2=float(e2),
            criterion_E=criterion_E,
            point_sigma=point_sigma,
            weight_mode=weight_mode,
        )

    # Translate the per-end caps into a single feasible interval on Ec2.
    # When a cap is missing, the only constraint on that side is the
    # starting pair itself, so the search collapses gracefully.
    e1_lo = float(ec1_min) if (ec1_min is not None and ec1_min > 0) else float(ec1) / R_RATIO
    e1_hi = float(ec1_max) if (ec1_max is not None and ec1_max > 0) else float(ec1) / R_RATIO
    e2_lo_cap = float(ec2_min) if (ec2_min is not None and ec2_min > 0) else float(ec2)
    e2_hi_cap = float(ec2_max) if (ec2_max is not None and ec2_max > 0) else float(ec2)
    if e1_hi < e1_lo:
        e1_hi = e1_lo
    if e2_hi_cap < e2_lo_cap:
        e2_hi_cap = e2_lo_cap

    e2_lo = max(e2_lo_cap, R_RATIO * max(e1_lo, 1.0e-30))
    e2_hi = min(e2_hi_cap, R_RATIO * max(e1_hi, 1.0e-30))

    n_evals = 0

    def _try_e2(e2: float) -> Optional[tuple]:
        nonlocal n_evals
        e1 = e2 / R_RATIO
        if not (e2 > 0 and e1 > 0):
            return None
        try:
            r = _eval(e1, e2)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
        n_evals += 1
        return r

    # Anchor on the user's Ec2 (snapped onto the IEC ratio: Ec1 := Ec2/R).
    # If the starting pair already meets the target there's no need for a
    # search.
    start_e2 = float(ec2)
    base = _try_e2(start_e2)
    samples: list[tuple[float, tuple]] = []
    if base is not None:
        samples.append((start_e2, base))
        if base[7] >= target_r2:
            return base, start_e2 / R_RATIO, start_e2, n_evals, True

    if not (e2_hi > e2_lo):
        # Caps inconsistent with the 10:1 ratio — return the snapped pair.
        if base is None:
            base = _eval(start_e2 / R_RATIO, start_e2)
            n_evals += 1
        return base, start_e2 / R_RATIO, start_e2, n_evals, base[7] >= target_r2

    # Coarse 1D log-spaced sweep over the feasible Ec2 interval.
    if e2_hi > e2_lo * 1.001:
        e2_grid = np.geomspace(e2_lo, e2_hi, GRID_N)
    else:
        e2_grid = np.array([0.5 * (e2_lo + e2_hi)])
    for e2 in e2_grid:
        r = _try_e2(float(e2))
        if r is not None:
            samples.append((float(e2), r))

    if not samples:
        base = _eval(start_e2 / R_RATIO, start_e2)
        n_evals += 1
        return base, start_e2 / R_RATIO, start_e2, n_evals, base[7] >= target_r2

    best_e2, best_result = max(samples, key=lambda s: s[1][7])

    # Golden-section refinement in log(Ec2) inside the feasible interval.
    if e2_hi > e2_lo * 1.001:
        a = float(np.log(e2_lo))
        b = float(np.log(e2_hi))
        for _ in range(min(max(max_iterations, 1), 8)):
            c = b - (b - a) / phi
            d = a + (b - a) / phi
            ec_c = float(np.exp(c))
            ec_d = float(np.exp(d))
            rc = _try_e2(ec_c)
            rd = _try_e2(ec_d)
            if rc is None or rd is None:
                break
            if rc[7] > rd[7]:
                b = d
                if rc[7] > best_result[7]:
                    best_e2, best_result = ec_c, rc
            else:
                a = c
                if rd[7] > best_result[7]:
                    best_e2, best_result = ec_d, rd
            if (b - a) < rel_tol:
                break

    return (best_result, float(best_e2 / R_RATIO), float(best_e2),
            n_evals, best_result[7] >= target_r2)


def _auto_adjust_loglog_window_independent(
    x: np.ndarray, y: np.ndarray, *,
    V0: float, R: float,
    ec1: float, ec2: float,
    criterion_E: float,
    point_sigma: Optional[np.ndarray],
    weight_mode: str,
    ec1_min: Optional[float],
    ec2_min: Optional[float],
    ec1_max: Optional[float],
    ec2_max: Optional[float],
    target_r2: float,
    max_iterations: int,
    rel_tol: float,
) -> tuple[tuple, float, float, int, bool]:
    """Independently optimise Ec1 and Ec2 to maximize R² (legacy mode).

    Ec1 and Ec2 are searched within their separate caps with only a 3:1
    minimum ratio enforced. The resulting window may deviate from the IEC
    61788 decade definition, so Ic / n are no longer directly comparable
    to a standard IEC measurement — useful when the data does not support
    a full decade window (high noise floor, short voltage taps).

    Strategy: a coarse 2D log-spaced grid over the
    ``[ec1_min, ec1_max] × [ec2_min, ec2_max]`` rectangle, followed by
    alternating golden-section refinement on each axis (coordinate
    descent) around the grid winner.
    """
    GRID_N = 5
    MIN_RATIO = 10  # Ec2 ≥ MIN_RATIO * Ec1 to keep the n-fit well-conditioned
    phi = (1.0 + 5.0 ** 0.5) / 2.0

    def _eval(e1: float, e2: float):
        return fit_n_value_log_log(
            x, y, V0=V0, R=R,
            Ec1=float(e1), Ec2=float(e2),
            criterion_E=criterion_E,
            point_sigma=point_sigma,
            weight_mode=weight_mode,
        )

    e1_lo = float(ec1_min) if (ec1_min is not None and ec1_min > 0) else float(ec1)
    e1_hi = float(ec1_max) if (ec1_max is not None and ec1_max > 0) else float(ec1)
    e2_lo = float(ec2_min) if (ec2_min is not None and ec2_min > 0) else float(ec2)
    e2_hi = float(ec2_max) if (ec2_max is not None and ec2_max > 0) else float(ec2)
    if e1_hi < e1_lo:
        e1_hi = e1_lo
    if e2_hi < e2_lo:
        e2_hi = e2_lo
    e1_lo = max(e1_lo, 1.0e-30)
    e2_lo = max(e2_lo, 1.0e-30)

    n_evals = 0

    def _try(e1: float, e2: float) -> Optional[tuple]:
        nonlocal n_evals
        if e2 <= e1 * MIN_RATIO:
            return None
        try:
            r = _eval(e1, e2)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
        n_evals += 1
        return r

    base = _try(float(ec1), float(ec2))
    samples: list[tuple[float, float, tuple]] = []
    if base is not None:
        samples.append((float(ec1), float(ec2), base))
        if base[7] >= target_r2:
            return base, float(ec1), float(ec2), n_evals, True

    e1_grid = (np.geomspace(e1_lo, e1_hi, GRID_N)
               if e1_hi > e1_lo * 1.001 else np.array([e1_lo]))
    e2_grid = (np.geomspace(e2_lo, e2_hi, GRID_N)
               if e2_hi > e2_lo * 1.001 else np.array([e2_lo]))
    for e1 in e1_grid:
        for e2 in e2_grid:
            r = _try(float(e1), float(e2))
            if r is not None:
                samples.append((float(e1), float(e2), r))

    if not samples:
        base = _eval(float(ec1), float(ec2))
        n_evals += 1
        return base, float(ec1), float(ec2), n_evals, base[7] >= target_r2

    best_e1, best_e2, best_result = max(samples, key=lambda s: s[2][7])

    def _gss_axis(fixed_other: float, lo: float, hi: float, vary_ec1: bool,
                  current_best: tuple[float, tuple]) -> tuple[float, tuple]:
        nonlocal n_evals
        if hi <= lo * 1.001:
            return current_best
        a = float(np.log(lo))
        b = float(np.log(hi))
        best_val, best_r = current_best
        for _ in range(min(max(max_iterations, 1), 5)):
            c = b - (b - a) / phi
            d = a + (b - a) / phi
            ec_c = float(np.exp(c))
            ec_d = float(np.exp(d))
            if vary_ec1:
                pair_c = (ec_c, fixed_other)
                pair_d = (ec_d, fixed_other)
                feasible_c = fixed_other > ec_c * MIN_RATIO
                feasible_d = fixed_other > ec_d * MIN_RATIO
            else:
                pair_c = (fixed_other, ec_c)
                pair_d = (fixed_other, ec_d)
                feasible_c = ec_c > fixed_other * MIN_RATIO
                feasible_d = ec_d > fixed_other * MIN_RATIO
            if not (feasible_c and feasible_d):
                if vary_ec1:
                    if not feasible_d:
                        b = d
                    if not feasible_c:
                        a = c
                else:
                    if not feasible_c:
                        a = c
                    if not feasible_d:
                        b = d
                if (b - a) < rel_tol:
                    break
                continue
            try:
                rc = _eval(*pair_c); n_evals += 1
                rd = _eval(*pair_d); n_evals += 1
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                break
            if rc[7] > rd[7]:
                b = d
                if rc[7] > best_r[7]:
                    best_val, best_r = ec_c, rc
            else:
                a = c
                if rd[7] > best_r[7]:
                    best_val, best_r = ec_d, rd
            if (b - a) < rel_tol:
                break
        return best_val, best_r

    for _ in range(2):
        prev_r2 = best_result[7]
        new_e1, new_result = _gss_axis(best_e2, e1_lo, e1_hi, True,
                                       (best_e1, best_result))
        if new_result[7] > best_result[7]:
            best_e1, best_result = new_e1, new_result
        new_e2, new_result = _gss_axis(best_e1, e2_lo, e2_hi, False,
                                       (best_e2, best_result))
        if new_result[7] > best_result[7]:
            best_e2, best_result = new_e2, new_result
        if best_result[7] - prev_r2 < 1.0e-6:
            break

    return best_result, float(best_e1), float(best_e2), n_evals, best_result[7] >= target_r2


def run_full_fit(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                 settings: Optional[FitSettings] = None) -> FitResult:
    """Step 1 V_ofs, Step 3 di/dt, Step 4 baseline → V0, R, L, Step 5 Ic/n.

    Step 1 (optional): estimate V_ofs from the I = 0 segment and subtract
    it from y so the downstream baseline fit isolates the inductive term
    (V0 = L·dI/dt) cleanly from the thermal offset.
    Step 3 estimates dI/dt from the linear ramp.
    Step 4 fits y - V_ofs = V0 + R·I on the low-current baseline window.
    Step 5 fits Ic and n; default is the IEC 61788 log-log decade method
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

    # Anchor the *_low_frac/*_high_frac editors to the untrimmed sweep when
    # the caller supplied an override (Data Fitting tab + auto-fit do this).
    # Falling back to the input array's extents preserves the legacy contract
    # for unit tests and standalone callers that pass fractions of the data
    # they hand in.
    pct_x_min = float(settings.pct_x_min) if settings.pct_x_min is not None else x_min
    pct_x_max = float(settings.pct_x_max) if settings.pct_x_max is not None else x_max
    if pct_x_max <= pct_x_min:
        pct_x_min, pct_x_max = x_min, x_max
    pct_span = pct_x_max - pct_x_min

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

    # Step 3: di/dt on the linear-ramp window.
    di_dt_mode = str(getattr(settings, "didt_mode", DEFAULT_DIDT_MODE) or DEFAULT_DIDT_MODE)
    didt_lo = pct_x_min + settings.didt_low_frac * pct_span
    didt_hi = pct_x_min + settings.didt_high_frac * pct_span
    try:
        di_dt = estimate_di_dt(
            t, x, settings.didt_low_frac, settings.didt_high_frac,
            mode=di_dt_mode, x_lo=didt_lo, x_hi=didt_hi,
        )
    except ValueError as exc:
        return FitResult(ok=False, message=f"di/dt slope fit failed: {exc}")

    # Step 4: linear baseline → V0 (= L·dI/dt in Y-units after Step 1) and R.
    lin_lo = pct_x_min + settings.linear_low_frac * pct_span
    lin_hi = pct_x_min + settings.linear_high_frac * pct_span
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
        ec1_used = float(settings.ec1)
        ec2_used = float(settings.ec2)
        ec1_initial = ec1_used
        ec2_initial = ec2_used
        auto_iters = 0
        auto_target_met = True
        auto_adjusted = False
        if getattr(settings, "auto_ec_adjust", False):
            try:
                (auto_result, ec1_used, ec2_used, auto_iters, auto_target_met) = (
                    auto_adjust_loglog_window(
                        x, y, V0=V0, R=R,
                        ec1=ec1_initial, ec2=ec2_initial,
                        criterion_E=crit_for_ic,
                        point_sigma=point_sigma,
                        weight_mode=weight_mode,
                        ec1_min=settings.auto_ec1_min,
                        ec2_min=settings.auto_ec2_min,
                        ec1_max=settings.auto_ec1_max,
                        ec2_max=settings.auto_ec2_max,
                        target_r2=settings.auto_ec_target_r2,
                        lock_iec_ratio=getattr(
                            settings, "auto_ec_lock_iec_ratio", True),
                    )
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                return FitResult(ok=False, message=f"Log-log n-value fit failed: {exc}")
            (Ic, n_value, chi_sqr, n_pts, n_window,
             sigma_Ic, sigma_n, r_squared) = auto_result
            ec1_used = float(ec1_used)
            ec2_used = float(ec2_used)
            auto_adjusted = (
                ec1_initial > 0 and abs(ec1_used / ec1_initial - 1.0) > 1.0e-6
            ) or (
                ec2_initial > 0 and abs(ec2_used / ec2_initial - 1.0) > 1.0e-6
            )
        else:
            try:
                (Ic, n_value, chi_sqr, n_pts, n_window,
                 sigma_Ic, sigma_n, r_squared) = fit_n_value_log_log(
                    x, y, V0=V0, R=R, Ec1=ec1_used, Ec2=ec2_used,
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
        if auto_adjusted and not auto_target_met:
            message = (
                f"IEC 61788 log-log fit succeeded; auto-Ec reached its limit "
                f"without hitting target R² = {settings.auto_ec_target_r2:.4g}."
            )
        elif auto_adjusted:
            message = (
                f"IEC 61788 log-log fit succeeded with auto-adjusted decade "
                f"window (R² target = {settings.auto_ec_target_r2:.4g})."
            )
        else:
            message = "IEC 61788 log-log n-value fit succeeded."
        return FitResult(
            ok=True,
            message=message,
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
            ec1=ec1_used,
            ec2=ec2_used,
            ec1_auto_adjusted=auto_adjusted,
            ec1_initial=ec1_initial,
            ec2_initial=ec2_initial,
            auto_ec_iterations=auto_iters,
            auto_ec_target_r2=float(settings.auto_ec_target_r2),
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
            sample_length_cm=settings.sample_length_cm if uses_length else None,
        )

    y_max = float(np.max(y))
    y_threshold = settings.power_v_frac * y_max
    above = np.where(y >= y_threshold)[0]
    power_hi = float(x[above[0]]) if above.size else x_max
    power_lo = pct_x_min + settings.power_low_frac * pct_span

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
        sample_length_cm=settings.sample_length_cm if uses_length else None,
    )

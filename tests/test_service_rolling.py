"""The sliding-window median and MAD give the same bits as the per-window formulas."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fitting import service


def reference_rolling_median(values: np.ndarray, win: int) -> np.ndarray:
    """The implementation before the sliding window (one np.median per point)."""
    n = int(values.size)
    if n == 0 or win <= 1:
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


def reference_rolling_mad(values: np.ndarray, win: int) -> np.ndarray:
    if win % 2 == 0:
        win += 1
    half = win // 2
    n = int(values.size)
    out = np.empty(n, dtype=float)
    for i in range(n):
        r = values[max(0, i - half):min(n, i + half + 1)]
        out[i] = float(np.median(np.abs(r - np.median(r))))
    return out


def reference_point_noise(x: np.ndarray, signal: np.ndarray, zero_i_frac: float) -> np.ndarray:
    """estimate_point_noise before the sliding window (loops copied from the old code)."""
    x, signal = service._clean_arrays(x, signal)
    n = signal.size
    if n == 0:
        return np.asarray([], dtype=float)
    if n < 5:
        return np.full(n, max(float(np.std(signal)), 1e-12), dtype=float)
    x_span = max(1e-30, float(np.max(x) - np.min(x)))
    win = max(7, int(np.ceil(0.03 * n)))
    if win % 2 == 0:
        win += 1
    trend = reference_rolling_median(signal, win)
    resid = signal - trend
    local_sigma = np.empty(n, dtype=float)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        r = resid[lo:hi]
        mad = float(np.median(np.abs(r - np.median(r))))
        local_sigma[i] = max(1e-12, 1.4826 * mad)
    x_abs_max = float(np.max(np.abs(x)))
    thr = max(0.0, zero_i_frac) * x_abs_max
    base_mask = np.abs(x) <= thr
    if np.count_nonzero(base_mask) >= 4:
        base = signal[base_mask]
        base_mad = float(np.median(np.abs(base - np.median(base))))
        baseline_sigma = max(1e-12, 1.4826 * base_mad)
    else:
        baseline_sigma = max(1e-12, float(np.percentile(local_sigma, 25)))
    xr = (x - float(np.min(x))) / x_span
    floor = baseline_sigma * (1.0 + 0.5 * np.clip(xr, 0.0, 1.0))
    return np.maximum(local_sigma, floor)


def sample(rng, n: int, kind: int) -> np.ndarray:
    if kind == 0:
        return rng.standard_normal(n) * 1e-6
    if kind == 1:
        return rng.integers(-3, 4, n).astype(float)  # many equal values
    return np.round(rng.standard_normal(n), 1) + np.linspace(0, 5, n)


CASES = [(0, 7), (1, 7), (2, 7), (5, 7), (6, 7), (7, 7), (8, 3), (9, 2), (50, 9), (300, 31), (1000, 31),
         (4000, 121), (4000, 4000), (4000, 5001)]


@pytest.mark.parametrize("n,win", CASES)
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_rolling_median_and_mad_bit_for_bit(n, win, kind):
    rng = np.random.default_rng(n * 7 + win + kind)
    v = sample(rng, n, kind)
    med = service._rolling_median(v, win)
    assert med.dtype == np.float64 and med.shape == (n,)
    assert med.tobytes() == reference_rolling_median(v, win).tobytes()
    mad = service._rolling_mad(v, win)
    assert mad.dtype == np.float64 and mad.shape == (n,)
    assert mad.tobytes() == reference_rolling_mad(v, win).tobytes()


def test_rolling_with_nan_keeps_numpy_behaviour():
    v = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # numpy may warn on NaN medians
        ref_med = reference_rolling_median(v, 3)
        ref_mad = reference_rolling_mad(v, 3)
        got_med = service._rolling_median(v, 3)
        got_mad = service._rolling_mad(v, 3)
    assert np.array_equal(got_med, ref_med, equal_nan=True)
    assert np.array_equal(got_mad, ref_mad, equal_nan=True)
    assert np.isnan(got_med[1:4]).all() and not np.isnan(got_med[5:]).any()


def test_rolling_median_window_one_or_empty():
    v = np.arange(5.0)
    assert np.array_equal(service._rolling_median(v, 1), v)
    assert service._rolling_median(v, 1) is not v
    assert service._rolling_median(np.empty(0), 7).shape == (0,)
    assert service._rolling_mad(np.empty(0), 7).shape == (0,)


@pytest.mark.parametrize("n", [5, 6, 20, 333, 2000, 12_000])
def test_estimate_point_noise_bit_for_bit(n):
    rng = np.random.default_rng(n)
    x = np.linspace(0.0, 100.0, n)
    signal = 1e-7 * x + 1e-3 * (x / 80.0) ** 20 + rng.standard_normal(n) * 2e-7
    signal[: n // 10] = rng.standard_normal(n // 10) * 1e-7  # quiet pre-ramp segment
    for frac in (0.0, 0.05, 0.2):
        got = service.estimate_point_noise(x, signal, frac)
        assert got.tobytes() == reference_point_noise(x, signal, frac).tobytes(), (n, frac)

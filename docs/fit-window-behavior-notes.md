# Fit window behavior notes (engineer-facing)

This note explains why the Step-4 **Low(X)/High(X)** values can differ from the final **Run fit** window.

## Log E vs Log I mode (IEC 61788)

When you type `Ec1` / `Ec2`, the GUI computes `Low(X)` / `High(X)` as a **preview**:

1. Build a corrected signal using the latest baseline:
   - `E_sc = y - (V0 + R*I)`
2. Smooth it for display guidance.
3. Find first current where smoothed `E_sc >= Ec1` -> `Low(X)`.
4. Find first current where smoothed `E_sc >= Ec2` -> `High(X)`.

This is done by `_update_loglog_power_x_from_ec()` in `fitting/tab.py` and it uses
`_ensure_step4_reference_curve()` + `_adaptive_smooth_visual()`.

## Why Run fit can use a different I window in log-log mode

`Run fit` uses a more robust selection in `fit_n_value_log_log()` (`fitting/service.py`):

- Sort by current.
- Compute `E_sc = y - V0 - R*I`.
- Use adaptive smoothing only to detect transition onset robustly.
- Build a monotonic envelope to avoid noisy false crossings.
- Fit only points in `[Ec1, Ec2]` inside the monotonic transition slice.
- Finally report `I(Ec1)` and `I(Ec2)` from the fitted line, not from first threshold hits.

So preview X-window (smoothed crossing) and fit X-window (model-derived IEC window)
are intentionally different.

## Power-law (nonlinear) mode uses different limits by design

In nonlinear mode, the same Step-4 boxes do **not** mean Ec1/Ec2:

- Low value = `% of I range` (`power_low_frac`).
- High value = first current where measured `V` reaches `% of Vmax` (`power_v_frac`).

Then each iteration can shrink upper bound toward estimated `Ic` (`power_hi = min(power_hi, Ic)`).
That is why power-law fitting can use a different I-window than the IEC log-log method.

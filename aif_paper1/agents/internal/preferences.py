"""
Shared preference-shape helpers for ActInf agents.
"""

from __future__ import annotations

import numpy as np


PREFERENCE_MODES = ("monotonic_low", "range_band")


def normalize_preference_mode(mode: str) -> str:
    """Validate and canonicalize the interoceptive preference shape name."""
    out = str(mode).strip().lower()
    if out not in PREFERENCE_MODES:
        raise ValueError(
            f"Unknown intero_preference_mode={mode!r}; expected {PREFERENCE_MODES}"
        )
    return out


def validate_preference_band(band_min: float, band_max: float, *, label: str) -> tuple[float, float]:
    """Validate a normalized homeostatic target band on ``[0, 1]``."""
    lo = float(band_min)
    hi = float(band_max)
    if lo < 0.0 or lo > 1.0:
        raise ValueError(f"{label}_min must be within [0, 1], got {lo}")
    if hi < 0.0 or hi > 1.0:
        raise ValueError(f"{label}_max must be within [0, 1], got {hi}")
    if hi < lo:
        raise ValueError(f"{label}_max must be >= {label}_min (got {hi} < {lo})")
    return lo, hi


def log_pref_monotonic_low(n_levels: int, *, preference_scale: float) -> np.ndarray:
    """Log preferences that monotonically prefer lower observation levels."""
    scores = np.linspace(0.0, -float(preference_scale), int(n_levels), dtype=np.float32)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    return np.log(probs + 1e-16)


def log_pref_range_band(
    n_levels: int,
    *,
    preference_scale: float,
    band_min: float,
    band_max: float,
) -> np.ndarray:
    """
    Log preferences for staying inside a normalized homeostatic band.

    Values inside the band tie for best score. Values outside the band are
    penalized quadratically so the loss increases smoothly with distance from
    the acceptable range.
    """
    n = int(max(1, int(n_levels)))
    if n <= 1:
        return np.log(np.ones((n,), dtype=np.float32))

    idx = np.arange(n, dtype=np.float32)
    x = idx / float(max(1, n - 1))
    d_lo = np.maximum(0.0, float(band_min) - x)
    d_hi = np.maximum(0.0, x - float(band_max))
    # Quadratic distance outside band to make penalties smooth near boundaries.
    dist = np.square(d_lo + d_hi)
    scores = -float(preference_scale) * dist
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    return np.log(probs + 1e-16)


def homeostatic_need_signal(
    need_level: float,
    *,
    max_level: float,
    mode: str,
    band_min: float,
    band_max: float,
) -> float:
    """
    Convert a raw need level into a scalar urgency signal for resource shaping.

    In ``monotonic_low`` mode the raw deficit is used directly. In
    ``range_band`` mode the signal measures how far the normalized value lies
    outside the acceptable band, then rescales that distance back into the
    original level units.
    """
    raw = float(max(0.0, float(need_level)))
    if str(mode) != "range_band":
        return raw
    lvl_max = max(1e-8, float(max_level))
    x = min(1.0, max(0.0, raw / lvl_max))
    d_lo = max(0.0, float(band_min) - x)
    d_hi = max(0.0, x - float(band_max))
    return float((d_lo + d_hi) * lvl_max)

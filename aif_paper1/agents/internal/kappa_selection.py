"""
Shared selector utilities for externally controlled interoceptive focus (kappa).
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

from aif_paper1.affectworld.helpers import Need, Resource
from aif_paper1.affectworld.intero_precision import channel_precisions_jax, channel_precisions_np


KAPPA_SELECTOR_MODES: tuple[str, ...] = (
    "belief_argmax",
    "belief_argmin",
    "random",
)



def normalize_kappa_selector_mode(mode: str) -> str:
    """Validate and canonicalize the external-κ selection rule name."""
    out = str(mode).strip().lower()
    if out not in KAPPA_SELECTOR_MODES:
        raise ValueError(
            f"Unknown kappa_selector_mode={mode!r}; expected {KAPPA_SELECTOR_MODES}"
        )
    return out


def _normalized_expectation_and_std_np(q: np.ndarray, *, max_level: float, deficit: bool) -> tuple[float, float]:
    """
    Compress a categorical Type-I belief into burden mean and uncertainty.

    The state support is mapped onto ``[0, 1]``. For deficit-like variables
    such as energy, hydration, and oxygen, low physiological levels should mean
    high burden, so ``deficit=True`` flips the scale. For sleep debt,
    ``deficit=False`` keeps higher indices as higher burden. ``max_level`` is
    retained for signature compatibility; the normalization is driven by the
    support size of ``q``.
    """
    probs = np.asarray(q, dtype=np.float32).reshape(-1)
    probs = np.clip(probs, 1e-8, None)
    probs = probs / max(1e-8, float(probs.sum()))
    idx = np.arange(probs.shape[0], dtype=np.float32)
    frac = idx / float(max(1, probs.shape[0] - 1))
    val = (1.0 - frac) if bool(deficit) else frac
    mean = float(np.dot(probs, val))
    var = float(np.dot(probs, np.square(val - mean)))
    return mean, float(np.sqrt(max(0.0, var)))


def belief_need_stats_np(
    *,
    q_energy: np.ndarray,
    q_hydration: np.ndarray,
    q_sleep: np.ndarray,
    q_oxygen: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return per-need burden and uncertainty from inferred Type-I beliefs.

    The output order is always ``[hunger, thirst, fatigue, suffocation]`` so it
    can be used directly as a κ score vector.
    """
    hunger_mu, hunger_std = _normalized_expectation_and_std_np(q_energy, max_level=1.0, deficit=True)
    thirst_mu, thirst_std = _normalized_expectation_and_std_np(q_hydration, max_level=1.0, deficit=True)
    fatigue_mu, fatigue_std = _normalized_expectation_and_std_np(q_sleep, max_level=1.0, deficit=False)
    suffoc_mu, suffoc_std = _normalized_expectation_and_std_np(q_oxygen, max_level=1.0, deficit=True)
    return (
        np.asarray([hunger_mu, thirst_mu, fatigue_mu, suffoc_mu], dtype=np.float32),
        np.asarray([hunger_std, thirst_std, fatigue_std, suffoc_std], dtype=np.float32),
    )


def body_need_stats_np(
    *,
    energy: int,
    hydration: int,
    sleep_debt: int,
    oxygen: int,
    max_energy: int,
    max_hydration: int,
    max_sleepdebt: int,
    max_oxygen: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return oracle need burdens from true body counters.

    This is intentionally non-standard and should only be used for oracle
    ablations: unlike the belief-based path, it bypasses inferred state
    uncertainty and therefore reports zero uncertainty for every need.
    """
    scores = np.asarray(
        [
            max(0.0, float(max_energy - int(energy))) / float(max(1, int(max_energy))),
            max(0.0, float(max_hydration - int(hydration))) / float(max(1, int(max_hydration))),
            max(0.0, float(int(sleep_debt))) / float(max(1, int(max_sleepdebt))),
            max(0.0, float(max_oxygen - int(oxygen))) / float(max(1, int(max_oxygen))),
        ],
        dtype=np.float32,
    )
    return scores, np.zeros_like(scores, dtype=np.float32)


def sticky_kappa_choice_np(
    *,
    scores: Sequence[float],
    current_kappa: int,
    step_index: int,
    last_switch_step: int,
    margin: float,
    min_dwell_steps: int,
) -> int:
    """
    Choose κ with hysteresis.

    A challenger must both exceed the current focus by ``margin`` and remain
    better for at least ``min_dwell_steps`` before a switch occurs. This keeps
    external κ selection from oscillating between nearly tied needs.
    """
    vals = np.asarray(scores, dtype=np.float32).reshape(-1)
    if vals.size != 4:
        raise ValueError(f"sticky_kappa_choice_np expected 4 scores, got shape {vals.shape}")
    challenger = int(np.argmax(vals))
    current = int(current_kappa)
    if current < 0 or current >= 4:
        return challenger
    if challenger == current:
        return current
    dwell = int(step_index) - int(last_switch_step)
    if dwell < int(min_dwell_steps):
        return current
    if float(vals[challenger]) >= float(vals[current]) + float(margin):
        return challenger
    return current


def _normalize_prob_vec_np(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Clip and renormalize a vector into a probability simplex."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.full((int(arr.size),), 1.0 / float(max(1, int(arr.size))), dtype=np.float32)
    return arr / total


def _target_resource_idx_for_need(need_idx: int) -> int:
    """Map a need channel onto the resource family that can relieve it."""
    need_i = int(need_idx)
    if need_i == int(Need.HUNGER):
        return int(Resource.FOOD)
    if need_i in {int(Need.THIRST), int(Need.SUFFOCATION)}:
        return int(Resource.WATER)
    if need_i == int(Need.FATIGUE):
        return int(Resource.SLEEP)
    return int(Resource.AIR)


def _position_proximity_np(weights: np.ndarray, *, width: int) -> np.ndarray:
    """Return a smooth proximity field induced by target position weights."""
    mass = _normalize_prob_vec_np(weights)
    n = int(mass.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    xs = np.arange(n, dtype=np.int32) % max(1, int(width))
    ys = np.arange(n, dtype=np.int32) // max(1, int(width))
    dist = np.abs(xs[:, None] - xs[None, :]) + np.abs(ys[:, None] - ys[None, :])
    return np.asarray((mass[None, :] / (1.0 + dist.astype(np.float32))).sum(axis=1), dtype=np.float32)


def belief_need_actionability_np(
    *,
    q_pos: np.ndarray,
    q_ctx: np.ndarray,
    resource_likelihood: np.ndarray | None,
    trans: np.ndarray | None,
    width: int,
) -> np.ndarray:
    """
    Estimate how actionable each need is from current beliefs and resource model.

    The score measures whether one-step control can move the agent closer to a
    resource that could relieve the need. It stays in ``[0, 1]`` and is
    intended as a multiplicative bonus on top of burden, not as a standalone
    need urgency signal.
    """
    if resource_likelihood is None or trans is None:
        return np.zeros((4,), dtype=np.float32)
    A_res = np.asarray(resource_likelihood, dtype=np.float32)
    trans_arr = np.asarray(trans, dtype=np.int32)
    if A_res.ndim != 3 or trans_arr.ndim != 2:
        return np.zeros((4,), dtype=np.float32)
    n_pos = int(A_res.shape[1])
    q_pos_vec = _normalize_prob_vec_np(np.asarray(q_pos, dtype=np.float32).reshape(-1))
    q_ctx_vec = _normalize_prob_vec_np(np.asarray(q_ctx, dtype=np.float32).reshape(-1))
    if int(q_pos_vec.size) != n_pos:
        q_pos_pad = np.zeros((n_pos,), dtype=np.float32)
        q_pos_pad[: min(n_pos, int(q_pos_vec.size))] = q_pos_vec[: min(n_pos, int(q_pos_vec.size))]
        q_pos_vec = _normalize_prob_vec_np(q_pos_pad)
    n_ctx = int(A_res.shape[2])
    if int(q_ctx_vec.size) != n_ctx:
        q_ctx_pad = np.zeros((n_ctx,), dtype=np.float32)
        q_ctx_pad[: min(n_ctx, int(q_ctx_vec.size))] = q_ctx_vec[: min(n_ctx, int(q_ctx_vec.size))]
        q_ctx_vec = _normalize_prob_vec_np(q_ctx_pad)
    move_trans = np.asarray(trans_arr[: min(5, int(trans_arr.shape[0]))], dtype=np.int32)
    if move_trans.size <= 0:
        return np.zeros((4,), dtype=np.float32)
    move_trans = np.clip(move_trans, 0, max(0, n_pos - 1))

    out = np.zeros((4,), dtype=np.float32)
    baseline = 1.0 / float(max(1, n_pos))
    for need_idx in range(4):
        target_res = _target_resource_idx_for_need(need_idx)
        if target_res < 0 or target_res >= int(A_res.shape[0]):
            continue
        mass = np.tensordot(A_res[target_res], q_ctx_vec, axes=(1, 0))
        if mass.shape != (n_pos,):
            continue
        signal_raw = np.clip(np.asarray(mass, dtype=np.float32) - baseline, 0.0, None)
        if float(np.max(signal_raw)) <= 0.0:
            continue
        signal = _normalize_prob_vec_np(signal_raw)
        proximity = _position_proximity_np(signal, width=int(width))
        start_proximity = float(np.dot(q_pos_vec, proximity))
        next_proximity = np.sum(q_pos_vec[None, :] * proximity[move_trans], axis=1)
        next_signal = np.sum(q_pos_vec[None, :] * signal[move_trans], axis=1)
        raw = float(np.max(np.maximum(next_proximity - start_proximity, 0.0) + next_signal))
        out[need_idx] = float(np.clip(raw, 0.0, 1.0))
    return out


def argmax_kappa_choice_np(scores: Sequence[float]) -> int:
    """Choose the highest-scoring need with no dwell or switching penalty."""
    vals = np.asarray(scores, dtype=np.float32).reshape(-1)
    if vals.size != 4:
        raise ValueError(f"argmax_kappa_choice_np expected 4 scores, got shape {vals.shape}")
    return int(np.argmax(vals))


def argmin_kappa_choice_np(scores: Sequence[float]) -> int:
    """Choose the lowest-scoring need — the anti-attention ablation.

    This is a directional control used in Paper 1 to bound the downside of the
    selector family: attending to the need the agent believes is *least*
    burdened should perform worse than both baseline and belief_argmax.
    """
    vals = np.asarray(scores, dtype=np.float32).reshape(-1)
    if vals.size != 4:
        raise ValueError(f"argmin_kappa_choice_np expected 4 scores, got shape {vals.shape}")
    return int(np.argmin(vals))


def focused_intero_templates_np(
    *,
    templates: Sequence[np.ndarray],
    attended: int | None,
    base_precision: float,
    att_precision: float,
    budget_mode: str = "none",
    budget_bg_floor: float = 0.05,
    budget_total: float | None = None,
) -> list[np.ndarray]:
    """
    Rebuild interoceptive likelihood templates for a chosen κ.

    Each returned matrix has the same deterministic target levels as the input
    template, and each modality uses the same per-channel precision allocation
    rule as the body sampler. In budgeted modes this means external-κ runtime
    likelihoods conserve the same total interoceptive budget used to generate
    observations.
    """
    out: list[np.ndarray] = []
    att_idx = -1 if attended is None else int(attended)
    p_vec = channel_precisions_np(
        attended=int(att_idx),
        n_channels=int(len(templates)),
        base_precision=float(base_precision),
        att_precision=float(att_precision),
        mode=str(budget_mode),
        bg_floor=float(budget_bg_floor),
        total_budget=budget_total,
    )
    for need_idx, templ in enumerate(templates):
        arr = np.asarray(templ, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected intero template rank-2, got shape {arr.shape}")
        n_obs, _ = arr.shape
        p = float(np.asarray(p_vec, dtype=np.float32)[need_idx])
        p = float(np.clip(p, 0.0, 1.0))
        off = 0.0 if n_obs <= 1 else (1.0 - p) / float(n_obs - 1)
        targets = np.asarray(np.argmax(arr, axis=0), dtype=np.int32)
        cols = []
        for tgt in targets.tolist():
            col = np.full((n_obs,), off, dtype=np.float32)
            col[int(tgt)] = p
            cols.append(col)
        out.append(np.stack(cols, axis=1))
    return out


def _normalized_expectation_and_std_jax(q: jnp.ndarray, *, deficit: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX equivalent of :func:`_normalized_expectation_and_std_np`."""
    probs = jnp.asarray(q, dtype=jnp.float32).reshape((-1,))
    probs = jnp.maximum(probs, 1e-8)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
    idx = jnp.arange(probs.shape[0], dtype=jnp.float32)
    frac = idx / jnp.asarray(float(max(1, int(probs.shape[0]) - 1)), dtype=jnp.float32)
    val = 1.0 - frac if bool(deficit) else frac
    mean = jnp.sum(probs * val)
    var = jnp.sum(probs * jnp.square(val - mean))
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


def _normalize_prob_vec_jax(values: jnp.ndarray) -> jnp.ndarray:
    """JAX equivalent of :func:`_normalize_prob_vec_np`."""
    arr = jnp.asarray(values, dtype=jnp.float32).reshape((-1,))
    arr = jnp.clip(arr, 0.0, None)
    n = max(1, int(arr.shape[0]))
    total = jnp.sum(arr)
    uniform = jnp.full((n,), 1.0 / float(n), dtype=jnp.float32)
    return jnp.where(total > 1e-12, arr / total, uniform)


def _position_proximity_jax(weights: jnp.ndarray, *, width: int) -> jnp.ndarray:
    """JAX equivalent of :func:`_position_proximity_np`."""
    mass = _normalize_prob_vec_jax(weights)
    n = max(1, int(mass.shape[0]))
    xs = jnp.arange(n, dtype=jnp.int32) % max(1, int(width))
    ys = jnp.arange(n, dtype=jnp.int32) // max(1, int(width))
    dist = jnp.abs(xs[:, None] - xs[None, :]) + jnp.abs(ys[:, None] - ys[None, :])
    return jnp.asarray((mass[None, :] / (1.0 + dist.astype(jnp.float32))).sum(axis=1), dtype=jnp.float32)


def belief_need_actionability_jax(
    *,
    q_pos: jnp.ndarray,
    q_ctx: jnp.ndarray,
    resource_likelihood: jnp.ndarray | None,
    trans: jnp.ndarray | None,
    width: int,
) -> jnp.ndarray:
    """JAX equivalent of :func:`belief_need_actionability_np`."""
    if resource_likelihood is None or trans is None:
        return jnp.zeros((4,), dtype=jnp.float32)
    A_res = jnp.asarray(resource_likelihood, dtype=jnp.float32)
    trans_arr = jnp.asarray(trans, dtype=jnp.int32)
    if A_res.ndim != 3 or trans_arr.ndim != 2:
        return jnp.zeros((4,), dtype=jnp.float32)
    n_pos = int(A_res.shape[1])
    n_ctx = int(A_res.shape[2])
    q_pos_raw = jnp.asarray(q_pos, dtype=jnp.float32).reshape((-1,))
    q_ctx_raw = jnp.asarray(q_ctx, dtype=jnp.float32).reshape((-1,))
    q_pos_pad = jnp.zeros((n_pos,), dtype=jnp.float32).at[: min(n_pos, int(q_pos_raw.shape[0]))].set(
        q_pos_raw[: min(n_pos, int(q_pos_raw.shape[0]))]
    )
    q_ctx_pad = jnp.zeros((n_ctx,), dtype=jnp.float32).at[: min(n_ctx, int(q_ctx_raw.shape[0]))].set(
        q_ctx_raw[: min(n_ctx, int(q_ctx_raw.shape[0]))]
    )
    q_pos_vec = _normalize_prob_vec_jax(q_pos_pad)
    q_ctx_vec = _normalize_prob_vec_jax(q_ctx_pad)
    move_trans = jnp.clip(trans_arr[: min(5, int(trans_arr.shape[0]))], 0, max(0, n_pos - 1))
    baseline = jnp.asarray(1.0 / float(max(1, n_pos)), dtype=jnp.float32)
    out: list[jnp.ndarray] = []
    for need_idx in range(4):
        target_res = _target_resource_idx_for_need(need_idx)
        if target_res < 0 or target_res >= int(A_res.shape[0]):
            out.append(jnp.asarray(0.0, dtype=jnp.float32))
            continue
        mass = jnp.tensordot(A_res[target_res], q_ctx_vec, axes=(1, 0))
        signal_raw = jnp.clip(jnp.asarray(mass, dtype=jnp.float32) - baseline, 0.0, None)
        signal = _normalize_prob_vec_jax(signal_raw)
        proximity = _position_proximity_jax(signal, width=int(width))
        start_proximity = jnp.sum(q_pos_vec * proximity)
        next_proximity = jnp.sum(q_pos_vec[None, :] * proximity[move_trans], axis=1)
        next_signal = jnp.sum(q_pos_vec[None, :] * signal[move_trans], axis=1)
        raw = jnp.max(jnp.maximum(next_proximity - start_proximity, 0.0) + next_signal)
        out.append(jnp.where(jnp.max(signal_raw) > 0.0, jnp.clip(raw, 0.0, 1.0), jnp.asarray(0.0, dtype=jnp.float32)))
    return jnp.stack(out, axis=0)


def belief_need_stats_jax(
    *,
    q_energy: jnp.ndarray,
    q_hydration: jnp.ndarray,
    q_sleep: jnp.ndarray,
    q_oxygen: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX equivalent of :func:`belief_need_stats_np`."""
    hunger_mu, hunger_std = _normalized_expectation_and_std_jax(q_energy, deficit=True)
    thirst_mu, thirst_std = _normalized_expectation_and_std_jax(q_hydration, deficit=True)
    fatigue_mu, fatigue_std = _normalized_expectation_and_std_jax(q_sleep, deficit=False)
    suffoc_mu, suffoc_std = _normalized_expectation_and_std_jax(q_oxygen, deficit=True)
    return (
        jnp.stack([hunger_mu, thirst_mu, fatigue_mu, suffoc_mu], axis=0),
        jnp.stack([hunger_std, thirst_std, fatigue_std, suffoc_std], axis=0),
    )


def focused_intero_templates_jax(
    *,
    templates: Sequence[jnp.ndarray],
    attended: jnp.ndarray,
    base_precision: float,
    att_precision: float,
    budget_mode_index: int = 0,
    budget_bg_floor: float | jnp.ndarray = 0.05,
    budget_total: jnp.ndarray | None = None,
) -> list[jnp.ndarray]:
    """JAX equivalent of :func:`focused_intero_templates_np`."""
    out: list[jnp.ndarray] = []
    att = jnp.asarray(attended, dtype=jnp.int32)
    n_channels = int(len(templates))
    if budget_total is None:
        total_budget = (
            jnp.asarray(float(att_precision), dtype=jnp.float32)
            + jnp.asarray(float(max(0, n_channels - 1) * float(base_precision)), dtype=jnp.float32)
        )
    else:
        total_budget = jnp.asarray(budget_total, dtype=jnp.float32)
    p_vec = channel_precisions_jax(
        attended=att,
        n_channels=n_channels,
        base_precision=jnp.asarray(float(base_precision), dtype=jnp.float32),
        att_precision=jnp.asarray(float(att_precision), dtype=jnp.float32),
        mode_index=int(budget_mode_index),
        bg_floor=jnp.asarray(budget_bg_floor, dtype=jnp.float32),
        total_budget=total_budget,
    )
    for need_idx, templ in enumerate(templates):
        arr = jnp.asarray(templ, dtype=jnp.float32)
        n_obs = int(arr.shape[0])
        p = jnp.clip(jnp.asarray(p_vec[need_idx], dtype=jnp.float32), 0.0, 1.0)
        p = jnp.clip(p, 0.0, 1.0)
        off = jnp.asarray(0.0 if n_obs <= 1 else 1.0 / float(n_obs - 1), dtype=jnp.float32) * (1.0 - p)
        targets = jnp.argmax(arr, axis=0).astype(jnp.int32)
        oh = jnn.one_hot(targets, n_obs, dtype=jnp.float32).T
        out.append(off + (p - off) * oh)
    return out


def sticky_kappa_choice_jax(
    *,
    scores: jnp.ndarray,
    current_kappa: jnp.ndarray,
    step_index: jnp.ndarray,
    last_switch_step: jnp.ndarray,
    margin: float,
    min_dwell_steps: int,
) -> jnp.ndarray:
    """JAX equivalent of :func:`sticky_kappa_choice_np`."""
    vals = jnp.asarray(scores, dtype=jnp.float32).reshape((-1,))
    challenger = jnp.argmax(vals).astype(jnp.int32)
    current = jnp.asarray(current_kappa, dtype=jnp.int32)
    valid_current = (current >= 0) & (current < 4)
    current_score = jnp.where(valid_current, vals[current], -jnp.inf)
    dwell = jnp.asarray(step_index, dtype=jnp.int32) - jnp.asarray(last_switch_step, dtype=jnp.int32)
    can_switch = (dwell >= int(min_dwell_steps)) & (vals[challenger] >= current_score + float(margin))
    return jnp.where(valid_current & (~can_switch) & (challenger != current), current, challenger)


def default_kappa_state() -> int:
    """Return the neutral κ state used at reset: no attended need."""
    return int(Need.NONE)

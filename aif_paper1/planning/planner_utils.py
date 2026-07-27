"""Shared helpers used across planning backends."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import lax, nn
from jax import random as jr
from jax.scipy import special as jsps

from aif_paper1.affectworld.helpers import Interact, Move
from aif_paper1.planning.policy_space import _normalize_policy_mode


def _allowed_mask(
    *,
    policy_mode: str,
    depth: jnp.ndarray,
    horizon: int,
    move: jnp.ndarray,
    interact: jnp.ndarray,
    need: Optional[jnp.ndarray],
    root_need: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """
    Return the planner's legality mask for one lookahead depth.

    The repo's restricted ``policy_mode`` settings are implemented in two
    different ways:
    - full-policy planners build a smaller policy set up front;
    - one-step search planners keep a reusable action library and apply this
      mask at each depth instead.

    ``navigate`` is the most non-standard case: intermediate steps are forced
    to avoid interaction, and for attentional planners the κ choice made at the
    root is held fixed deeper in the tree.
    """
    mode = _normalize_policy_mode(policy_mode)
    horizon = int(horizon)

    move_rest = int(Move.REST)
    interact_none = int(Interact.NONE)

    mask = jnp.ones_like(move, dtype=bool)

    if mode == "move_or_act":
        mask = (interact == interact_none) | (move == move_rest)
        return mask

    if mode == "competence":
        return jnp.where(depth <= 0, True, move == move_rest)

    if mode == "navigate":
        if horizon > 2:
            inter = (depth > 0) & (depth < (horizon - 1))
            mask = jnp.where(inter, interact == interact_none, True)
        if need is not None and root_need is not None:
            mask = jnp.where(depth <= 0, mask, mask & (need == root_need))
        return mask

    return mask


def _topk_observation_tuples(
    qo: list[jnp.ndarray],
    *,
    grid: jnp.ndarray,
    obs_topk: int,
    obs_top_m: int,
    obs_prob_threshold: float,
    rng_key: Optional[jnp.ndarray] = None,
    stochastic: bool = False,
    randomize_order: bool = False,
    enable_obs_pruning: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return the top joint observation tuples under a factorized ``qo``.

    Search planners treat observation factors as conditionally independent given
    predicted latent state, so they can rank joint observation tuples by taking
    the product of per-modality top entries. By default the ranking is
    deterministic top-k; stochastic modes add Gumbel noise to randomize the
    expansion order without changing the underlying probabilities.
    """
    n_modalities = int(len(qo))
    top_m = int(obs_top_m)
    n_combo = int(grid.shape[0])
    topk = int(min(int(obs_topk), n_combo))

    top_vals = []
    top_idx = []
    for qo_m in qo:
        values, indices = lax.top_k(jnp.asarray(qo_m, dtype=jnp.float32), k=top_m)
        top_vals.append(values)
        top_idx.append(indices.astype(jnp.int32))

    joint = jnp.ones((n_combo,), dtype=jnp.float32)
    for modality_idx in range(n_modalities):
        joint = joint * top_vals[modality_idx][grid[:, modality_idx]]

    rank_scores = joint
    if bool(stochastic or randomize_order):
        if rng_key is None:
            rng_key = jr.PRNGKey(0)
        gumbel = -jnp.log(
            -jnp.log(jr.uniform(rng_key, shape=(n_combo,), minval=1e-6, maxval=1.0 - 1e-6))
        )
        rank_scores = rank_scores + gumbel

    _, combo_idx = lax.top_k(rank_scores, k=topk)
    probs = joint[combo_idx]
    if bool(enable_obs_pruning):
        threshold = jnp.asarray(float(obs_prob_threshold), dtype=jnp.float32)
        probs = jnp.where(probs >= threshold, probs, 0.0)
    prob_sum = probs.sum()

    obs = []
    for modality_idx in range(n_modalities):
        obs.append(top_idx[modality_idx][grid[combo_idx, modality_idx]])
    obs_idx = jnp.stack(obs, axis=-1).astype(jnp.int32)

    return obs_idx, probs.astype(jnp.float32), prob_sum.astype(jnp.float32)


def _outer_belief_mass(qs: list[jnp.ndarray], deps: list[int]) -> jnp.ndarray:
    """Form the outer product over the hidden-state marginals a modality depends on."""
    if len(deps) == 0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    mass = jnp.asarray(qs[int(deps[0])], dtype=jnp.float32)
    for dep in deps[1:]:
        mass = mass[..., None] * jnp.asarray(qs[int(dep)], dtype=jnp.float32)
    return mass


def _normalize_likelihood_columns(alpha: jnp.ndarray) -> jnp.ndarray:
    """Normalize likelihood or count tensors column-wise over observation outcomes."""
    den = jnp.maximum(alpha.sum(axis=0, keepdims=True), 1e-12)
    return alpha / den


def _materialize_likelihoods(
    *,
    base_A: list[jnp.ndarray],
    branch_counts: Optional[list[Optional[jnp.ndarray]]],
) -> list[jnp.ndarray]:
    """
    Convert optional branch-local counts into branch-local likelihood matrices.

    If a branch has no local counts for a modality, the base ``A`` is reused.
    Otherwise the counts are normalized into a proper likelihood matrix for that
    branch only.
    """
    if branch_counts is None:
        return base_A
    out: list[jnp.ndarray] = []
    for modality_idx, A_m in enumerate(base_A):
        counts = branch_counts[modality_idx] if modality_idx < len(branch_counts) else None
        if counts is None:
            out.append(A_m)
        else:
            out.append(_normalize_likelihood_columns(jnp.asarray(counts, dtype=jnp.float32)))
    return out


def _seed_branch_counts_from_likelihoods(
    *,
    base_A: list[jnp.ndarray],
    concentration: float,
) -> list[Optional[jnp.ndarray]]:
    """
    Seed synthetic branch-local counts from a fixed likelihood model.

    Compiled SI/SL cannot pull live Python-side Dirichlet counts mid-search, so
    this helper creates a count tensor whose normalized columns reproduce the
    current likelihoods while still allowing branch-local updates downstream.
    """
    conc = jnp.asarray(float(max(1e-6, concentration)), dtype=jnp.float32)
    out: list[Optional[jnp.ndarray]] = []
    for A_m in base_A:
        A_arr = jnp.asarray(A_m, dtype=jnp.float32)
        if A_arr.ndim < 2:
            out.append(None)
            continue
        A_prob = _normalize_likelihood_columns(jnp.maximum(A_arr, 1e-8))
        out.append(jnp.maximum(conc * A_prob, 1e-8))
    return out


def _expected_model_novelty(
    *,
    branch_counts: Optional[list[Optional[jnp.ndarray]]],
    qs_pred: list[jnp.ndarray],
    A_dependencies: list[list[int]],
) -> jnp.ndarray:
    """
    Estimate epistemic novelty from branch-local likelihood uncertainty.

    The score is the expected information gain about each learnable modality's
    likelihood parameters under the predicted next-state beliefs. SL uses this
    to reward branches that would improve the observation model, not just
    immediate utility or state information gain.
    """
    if branch_counts is None:
        return jnp.asarray(0.0, dtype=jnp.float32)

    novelty = jnp.asarray(0.0, dtype=jnp.float32)
    for modality_idx, alpha_m in enumerate(branch_counts):
        if alpha_m is None:
            continue
        alpha = jnp.maximum(jnp.asarray(alpha_m, dtype=jnp.float32), 1e-8)
        total = jnp.maximum(alpha.sum(axis=0), 1e-8)
        pred = alpha / total[None, ...]
        term = jnp.log(total[None, ...] / alpha)
        term = term + jsps.digamma(alpha + 1.0) - jsps.digamma(total[None, ...] + 1.0)
        kl = (pred * term).sum(axis=0)
        dep_mass = _outer_belief_mass(qs_pred, list(A_dependencies[modality_idx]))
        novelty = novelty + (dep_mass * kl).sum()
    return novelty.astype(jnp.float32)


def _branch_update_model_counts(
    *,
    branch_counts: Optional[list[Optional[jnp.ndarray]]],
    obs_idx: jnp.ndarray,
    qs_post: list[jnp.ndarray],
    A_dependencies: list[list[int]],
    lr: float,
    threshold: float,
) -> Optional[list[Optional[jnp.ndarray]]]:
    """
    Apply a branch-local likelihood-count update for one hypothetical observation.

    This is the mechanism that makes SL different from plain SI: each branch can
    imagine what its own observation would teach the model and carry those local
    counts deeper into the tree.
    """
    if branch_counts is None:
        return None

    out: list[Optional[jnp.ndarray]] = list(branch_counts)
    lr_t = jnp.asarray(float(max(0.0, lr)), dtype=jnp.float32)
    threshold_t = jnp.asarray(float(max(0.0, threshold)), dtype=jnp.float32)
    for modality_idx, alpha_m in enumerate(branch_counts):
        if alpha_m is None:
            continue
        alpha = jnp.asarray(alpha_m, dtype=jnp.float32)
        obs_onehot = nn.one_hot(
            obs_idx[modality_idx].astype(jnp.int32),
            alpha.shape[0],
            dtype=jnp.float32,
        )
        dep_mass = _outer_belief_mass(qs_post, list(A_dependencies[modality_idx]))
        contrib = obs_onehot[(slice(None),) + (None,) * int(dep_mass.ndim)] * dep_mass[None, ...]
        if float(threshold) > 0.0:
            contrib = jnp.where(contrib <= threshold_t, 0.0, contrib)
        out[modality_idx] = jnp.maximum(alpha + lr_t * contrib, 1e-8)
    return out


def _coerce_runtime_model_list(
    arrs: Optional[list[jnp.ndarray]],
    fallback: list[jnp.ndarray],
) -> list[jnp.ndarray]:
    """Validate runtime ``A``/``C`` overrides against a fallback model list."""
    if arrs is None:
        return fallback
    if int(len(arrs)) != int(len(fallback)):
        raise ValueError(
            f"runtime model override length mismatch: got={len(arrs)} expected={len(fallback)}"
        )
    out: list[jnp.ndarray] = []
    for idx, arr in enumerate(arrs):
        base_i = jnp.asarray(fallback[idx], dtype=jnp.float32)
        arr_i = jnp.asarray(arr, dtype=jnp.float32)
        if arr_i.shape != base_i.shape:
            raise ValueError(
                f"runtime model override shape mismatch at index {idx}: "
                f"got={arr_i.shape} expected={base_i.shape}"
            )
        out.append(arr_i)
    return out


def coerce_runtime_arr_batched(
    arr: Any,
    *,
    expected_per_element_shape: tuple[int, ...],
    batch_size: int,
    label: str = "runtime override",
) -> jnp.ndarray:
    """Coerce a runtime override tensor to the ``(B, *per_element_shape)`` shape
    that batched scorers expect.

    Accepts either:
    - **Per-element**: ``arr.shape == expected_per_element_shape`` → broadcast
      to ``(B, *)`` (every batch element shares the same override).
    - **Per-batch**: ``arr.shape == (B, *expected_per_element_shape)`` →
      pass-through (each batch element has its own override).

    Raises ``ValueError`` for any other shape, with a label-prefixed message
    for upstream-friendly diagnostics.

    This consolidates the ad-hoc ``_coerce_A_runtime`` / ``_coerce_C_runtime``
    helpers in ``efe_repo_local.py`` (and the ones the search-planner
    factories will need under their bs > 1 lift) into a single shared
    contract. See ``docs/architecture/benchmarks/search_planners_batching_plan.md``
    Phase 1.1 for the design rationale.
    """
    a = jnp.asarray(arr, dtype=jnp.float32)
    expected_per_element = tuple(int(s) for s in expected_per_element_shape)
    batched_shape = (int(batch_size),) + expected_per_element
    if a.shape == expected_per_element:
        return jnp.broadcast_to(a[None, ...], batched_shape)
    if a.shape == batched_shape:
        return a
    raise ValueError(
        f"{label}: shape {tuple(a.shape)} matches neither per-element "
        f"{expected_per_element} nor per-batch {batched_shape}"
    )


def coerce_runtime_arr_list_batched(
    runtime: Optional[list[Any]],
    *,
    expected_per_element_shapes: tuple[tuple[int, ...], ...],
    batch_size: int,
    label: str = "runtime override",
    fallback: Optional[tuple[jnp.ndarray, ...]] = None,
) -> Optional[tuple[jnp.ndarray, ...]]:
    """List variant of :func:`coerce_runtime_arr_batched`.

    When ``runtime is None``, returns ``fallback`` (the planner factory's
    closure-captured ``(B, *)`` defaults). Otherwise validates list length
    and coerces each entry against its expected per-element shape.
    """
    if runtime is None:
        return fallback
    if int(len(runtime)) != int(len(expected_per_element_shapes)):
        raise ValueError(
            f"{label}: length mismatch — got {len(runtime)}, expected "
            f"{len(expected_per_element_shapes)}"
        )
    return tuple(
        coerce_runtime_arr_batched(
            arr,
            expected_per_element_shape=expected_per_element_shapes[m],
            batch_size=batch_size,
            label=f"{label}[{m}]",
        )
        for m, arr in enumerate(runtime)
    )


def _coerce_runtime_policy_prior(
    arr: Optional[jnp.ndarray],
    fallback: jnp.ndarray,
) -> jnp.ndarray:
    """
    Validate and normalize a runtime policy prior ``E`` override.

    The result is always a flat probability vector, whether the caller supplied
    nothing, a 1-D prior, or a batch-shaped ``(1, n_policies)`` tensor.
    """
    base = jnp.asarray(fallback, dtype=jnp.float32).reshape(-1)
    if arr is None:
        raw = base
    else:
        raw = jnp.asarray(arr, dtype=jnp.float32)
        if raw.ndim == 2:
            if int(raw.shape[0]) != 1:
                raise ValueError(
                    "runtime policy prior batch mismatch: "
                    f"got shape={raw.shape}; expected (1, n_policies)"
                )
            raw = raw[0]
        raw = raw.reshape(-1)
        if raw.shape != base.shape:
            raise ValueError(
                f"runtime policy prior shape mismatch: got={raw.shape} expected={base.shape}"
            )
    raw = jnp.maximum(raw, 1e-8)
    den = jnp.maximum(raw.sum(), 1e-8)
    return raw / den


__all__ = [
    "coerce_runtime_arr_batched",
    "coerce_runtime_arr_list_batched",
    "_allowed_mask",
    "_branch_update_model_counts",
    "_coerce_runtime_model_list",
    "_coerce_runtime_policy_prior",
    "_expected_model_novelty",
    "_materialize_likelihoods",
    "_normalize_likelihood_columns",
    "_outer_belief_mass",
    "_seed_branch_counts_from_likelihoods",
    "_topk_observation_tuples",
]

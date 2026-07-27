"""
policy_lib.py
=============

Direct construction of restricted policy libraries to avoid enumerating the full
cartesian product `|A|^H` only to filter it afterwards.

This is a performance feature and must preserve scientific traceability:
`policy_mode` remains a first-class variant axis.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _cartesian_product_per_step(step_choices: list[np.ndarray]) -> np.ndarray:
    """
    Build all sequences by taking the cartesian product of per-step choices.

    Parameters
    ----------
    step_choices:
        List of arrays, one per timestep t, each shaped (K_t, action_dim).

    Returns
    -------
    policies : np.ndarray, shape (prod_t K_t, T, action_dim)
    """
    if not step_choices:
        raise ValueError("step_choices must be non-empty (policy_len >= 1)")

    T = int(len(step_choices))
    action_dim = int(step_choices[0].shape[-1])
    for t, c in enumerate(step_choices):
        c = np.asarray(c)
        if c.ndim != 2 or c.shape[-1] != action_dim:
            raise ValueError(f"step_choices[{t}] has shape {c.shape}, expected (K, {action_dim})")

    K = [int(c.shape[0]) for c in step_choices]
    n_total = int(np.prod(np.array(K, dtype=np.int64)))
    out = np.empty((n_total, T, action_dim), dtype=np.int32)

    repeat = 1
    for t in reversed(range(T)):
        choices = np.asarray(step_choices[t], dtype=np.int32)
        kt = int(choices.shape[0])

        block = np.repeat(choices, repeats=repeat, axis=0)  # (kt*repeat, action_dim)
        tile = np.tile(block, (n_total // (kt * repeat), 1))  # (n_total, action_dim)
        out[:, t, :] = tile

        repeat *= kt

    return out


def _all_pairs(n0: int, n1: int) -> np.ndarray:
    """Enumerate the full Cartesian product of two discrete action factors."""
    a0 = np.arange(int(n0), dtype=np.int32)
    a1 = np.arange(int(n1), dtype=np.int32)
    g0, g1 = np.meshgrid(a0, a1, indexing="ij")
    return np.stack([g0.reshape(-1), g1.reshape(-1)], axis=-1).astype(np.int32, copy=False)


def _all_triples(n0: int, n1: int, n2: int) -> np.ndarray:
    """Enumerate the full Cartesian product of three discrete action factors."""
    a0 = np.arange(int(n0), dtype=np.int32)
    a1 = np.arange(int(n1), dtype=np.int32)
    a2 = np.arange(int(n2), dtype=np.int32)
    g0, g1, g2 = np.meshgrid(a0, a1, a2, indexing="ij")
    return np.stack([g0.reshape(-1), g1.reshape(-1), g2.reshape(-1)], axis=-1).astype(np.int32, copy=False)


def baseline_policies(*, policy_mode: str, horizon: int, n_move: int = 5, n_interact: int = 5) -> jnp.ndarray:
    """
    Return multi-action policies for the baseline agent: (Move, Interact).

    Output shape: ``(n_policies, horizon, 2)``.

    This constructor is used when a planner expects explicit full-horizon
    policies rather than a one-step action library. The repo keeps ``policy_mode``
    as a first-class experimental axis, so restricted policy sets are built
    directly here instead of enumerating the full Cartesian product and
    filtering it afterwards.
    """
    mode = str(policy_mode).lower()
    H = int(horizon)
    if H <= 0:
        raise ValueError("horizon must be >= 1")

    MOVE_REST = int(n_move - 1)
    INTERACT_NONE = 0

    full = _all_pairs(n_move, n_interact)  # (25, 2)
    move_only = np.stack([np.arange(n_move, dtype=np.int32), np.full(n_move, INTERACT_NONE, dtype=np.int32)], axis=-1)
    act_only = np.stack([np.full(n_interact, MOVE_REST, dtype=np.int32), np.arange(n_interact, dtype=np.int32)], axis=-1)
    move_or_act = np.unique(np.concatenate([move_only, act_only], axis=0), axis=0).astype(np.int32, copy=False)

    if mode in {"full", "", "none"}:
        # Let pymdp enumerate in this case (kept here for completeness / tests).
        step_choices = [full for _ in range(H)]
        pol = _cartesian_product_per_step(step_choices)
        return jnp.asarray(pol, dtype=jnp.int32)

    if mode == "move_or_act":
        step_choices = [move_or_act for _ in range(H)]
        pol = _cartesian_product_per_step(step_choices)
        return jnp.asarray(pol, dtype=jnp.int32)

    if mode == "competence":
        if H == 1:
            return jnp.asarray(full[:, None, :], dtype=jnp.int32)
        future = np.stack([np.full(n_interact, MOVE_REST, dtype=np.int32), np.arange(n_interact, dtype=np.int32)], axis=-1)
        step_choices = [full] + [future for _ in range(H - 1)]
        pol = _cartesian_product_per_step(step_choices)
        return jnp.asarray(pol, dtype=jnp.int32)

    if mode == "navigate":
        if H == 1:
            return jnp.asarray(full[:, None, :], dtype=jnp.int32)
        inter = np.stack(
            [np.arange(n_move, dtype=np.int32), np.full(n_move, INTERACT_NONE, dtype=np.int32)],
            axis=-1,
        )  # (5,2)
        if H == 2:
            step_choices = [full, full]
        else:
            step_choices = [full] + [inter for _ in range(H - 2)] + [full]
        pol = _cartesian_product_per_step(step_choices)
        return jnp.asarray(pol, dtype=jnp.int32)

    raise ValueError(f"Unknown policy_mode={policy_mode!r}")


def attentional_policies(
    *,
    policy_mode: str,
    horizon: int,
    n_move: int = 5,
    n_interact: int = 5,
    n_need: int = 5,
) -> jnp.ndarray:
    """
    Return multi-action policies for the attentional agent: (Move, Interact, Need).

    Output shape: ``(n_policies, horizon, 3)``.

    The attentional version mirrors ``baseline_policies`` but includes the κ
    control factor. In ``navigate`` mode the constructor also holds κ constant
    across the horizon, which is a deliberate tractability approximation used
    throughout the repo.
    """
    mode = str(policy_mode).lower()
    H = int(horizon)
    if H <= 0:
        raise ValueError("horizon must be >= 1")

    MOVE_REST = int(n_move - 1)
    INTERACT_NONE = 0

    full = _all_triples(n_move, n_interact, n_need)  # (125, 3)

    move_only = _all_triples(n_move, 1, n_need)
    move_only[:, 1] = INTERACT_NONE
    act_only = _all_triples(1, n_interact, n_need)
    act_only[:, 0] = MOVE_REST
    move_or_act = np.unique(np.concatenate([move_only, act_only], axis=0), axis=0).astype(np.int32, copy=False)

    if mode in {"full", "", "none"}:
        step_choices = [full for _ in range(H)]
        pol = _cartesian_product_per_step(step_choices)
        return jnp.asarray(pol, dtype=jnp.int32)

    if mode == "move_or_act":
        step_choices = [move_or_act for _ in range(H)]
        pol = _cartesian_product_per_step(step_choices)
        return jnp.asarray(pol, dtype=jnp.int32)

    if mode == "competence":
        if H == 1:
            return jnp.asarray(full[:, None, :], dtype=jnp.int32)
        future = _all_triples(1, n_interact, n_need)
        future[:, 0] = MOVE_REST
        step_choices = [full] + [future for _ in range(H - 1)]
        pol = _cartesian_product_per_step(step_choices)
        return jnp.asarray(pol, dtype=jnp.int32)

    if mode == "navigate":
        # navigate: hold κ constant over the horizon + enforce Interact.NONE on intermediate steps (if any)
        if H == 1:
            return jnp.asarray(full[:, None, :], dtype=jnp.int32)

        full_mi = _all_pairs(n_move, n_interact)  # (25,2)
        inter_mi = np.stack(
            [np.arange(n_move, dtype=np.int32), np.full(n_move, INTERACT_NONE, dtype=np.int32)],
            axis=-1,
        )  # (5,2)

        if H == 2:
            step_choices_mi = [full_mi, full_mi]
        else:
            step_choices_mi = [full_mi] + [inter_mi for _ in range(H - 2)] + [full_mi]

        base_mi = _cartesian_product_per_step(step_choices_mi).astype(np.int32, copy=False)  # (n_base, H, 2)
        n_base = int(base_mi.shape[0])

        mi_tiled = np.tile(base_mi, (n_need, 1, 1))  # (n_need*n_base, H, 2)
        need_vals = np.repeat(np.arange(n_need, dtype=np.int32), repeats=n_base)  # (n_need*n_base,)
        need_seq = np.repeat(need_vals[:, None], repeats=H, axis=1)  # (n_need*n_base, H)

        pol = np.concatenate([mi_tiled, need_seq[:, :, None]], axis=-1)  # (n_need*n_base, H, 3)
        return jnp.asarray(pol, dtype=jnp.int32)

    raise ValueError(f"Unknown policy_mode={policy_mode!r}")

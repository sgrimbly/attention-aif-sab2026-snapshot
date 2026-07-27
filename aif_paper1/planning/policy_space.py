"""Planner naming helpers and action-library builders."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from aif_paper1.affectworld.helpers import Interact, Move


def _normalize_planner_name(name: str) -> str:
    """Canonicalize planner aliases used across configs, scripts, and docs."""
    n = str(name).lower().strip()
    if n in {"efe", "lefe"}:
        return "efe"
    raise ValueError(f"Unknown planner={name!r}; expected efe")


def _normalize_policy_mode(mode: str) -> str:
    """Canonicalize policy-space restriction names."""
    m = str(mode).lower().strip()
    if m in {"", "none"}:
        return "full"
    if m in {"full", "move_or_act", "competence", "navigate"}:
        return m
    raise ValueError(
        f"Unknown policy_mode={mode!r}; expected full|move_or_act|competence|navigate"
    )


def baseline_action_library(*, policy_mode: str, n_move: int, n_interact: int) -> jnp.ndarray:
    """
    Single-step action library for baseline agents (Move, Interact).

    This is the one-step search space used by the repo-local planners. For
    ``move_or_act`` the restriction is applied directly when constructing the
    library. For ``competence`` and ``navigate`` the library remains full and
    the depth-dependent constraints are enforced later by planner masks.
    """
    mode = _normalize_policy_mode(policy_mode)
    n_move = int(n_move)
    n_interact = int(n_interact)
    if n_move <= 0 or n_interact <= 0:
        raise ValueError("n_move and n_interact must be positive")

    a0 = np.arange(n_move, dtype=np.int32)
    a1 = np.arange(n_interact, dtype=np.int32)
    g0, g1 = np.meshgrid(a0, a1, indexing="ij")
    full = np.stack([g0.reshape(-1), g1.reshape(-1)], axis=-1).astype(np.int32, copy=False)

    if mode != "move_or_act":
        return jnp.asarray(full, dtype=jnp.int32)

    move_rest = int(Move.REST)
    interact_none = int(Interact.NONE)

    move_only = np.stack(
        [np.arange(n_move, dtype=np.int32), np.full(n_move, interact_none, dtype=np.int32)],
        axis=-1,
    )
    act_only = np.stack(
        [np.full(n_interact, move_rest, dtype=np.int32), np.arange(n_interact, dtype=np.int32)],
        axis=-1,
    )
    move_or_act = np.unique(np.concatenate([move_only, act_only], axis=0), axis=0).astype(
        np.int32,
        copy=False,
    )
    return jnp.asarray(move_or_act, dtype=jnp.int32)


def attentional_action_library(
    *,
    policy_mode: str,
    n_move: int,
    n_interact: int,
    n_need: int,
) -> jnp.ndarray:
    """
    Single-step action library for attentional agents (Move, Interact, Need).

    As with the baseline library, only ``move_or_act`` changes the concrete
    one-step action set. Other modes preserve the full immediate action library
    and rely on planner-side masks to constrain deeper lookahead steps.
    """
    mode = _normalize_policy_mode(policy_mode)
    n_move = int(n_move)
    n_interact = int(n_interact)
    n_need = int(n_need)
    if n_move <= 0 or n_interact <= 0 or n_need <= 0:
        raise ValueError("n_move, n_interact, n_need must be positive")

    a0 = np.arange(n_move, dtype=np.int32)
    a1 = np.arange(n_interact, dtype=np.int32)
    a2 = np.arange(n_need, dtype=np.int32)
    g0, g1, g2 = np.meshgrid(a0, a1, a2, indexing="ij")
    full = np.stack([g0.reshape(-1), g1.reshape(-1), g2.reshape(-1)], axis=-1).astype(
        np.int32,
        copy=False,
    )

    if mode != "move_or_act":
        return jnp.asarray(full, dtype=jnp.int32)

    move_rest = int(Move.REST)
    interact_none = int(Interact.NONE)

    move_only = np.stack(
        [
            np.repeat(np.arange(n_move, dtype=np.int32), repeats=n_need),
            np.full(n_move * n_need, interact_none, dtype=np.int32),
            np.tile(np.arange(n_need, dtype=np.int32), reps=n_move),
        ],
        axis=-1,
    )
    act_only = np.stack(
        [
            np.full(n_interact * n_need, move_rest, dtype=np.int32),
            np.repeat(np.arange(n_interact, dtype=np.int32), repeats=n_need),
            np.tile(np.arange(n_need, dtype=np.int32), reps=n_interact),
        ],
        axis=-1,
    )
    move_or_act = np.unique(np.concatenate([move_only, act_only], axis=0), axis=0).astype(
        np.int32,
        copy=False,
    )
    return jnp.asarray(move_or_act, dtype=jnp.int32)


__all__ = [
    "_normalize_planner_name",
    "_normalize_policy_mode",
    "attentional_action_library",
    "baseline_action_library",
]

"""
Canonical single-step helpers for AffectWorld rollouts.

These functions centralize the maintained Python step ordering so callers that
need per-step records can share one semantics source instead of reimplementing
the env/body/agent transition logic.
"""

from __future__ import annotations

import numpy as np

from aif_paper1.affectworld.helpers import Need, Resource


def to_attended_idx(kappa: object) -> int | None:
    """Normalize a κ/Need-like value into an attended interoceptive channel index."""

    if kappa is None:
        return None
    if isinstance(kappa, Need):
        return None if kappa == Need.NONE else int(kappa)
    try:
        idx = int(kappa)
    except (TypeError, ValueError):
        return None
    return None if idx < 0 else idx


def capture_initial_step_record(
    env,
    body,
    obs,
    typeII_obs: np.ndarray,
) -> dict[str, object]:
    """Capture the episode-initial state before the first action is selected."""

    pos0, ctx0 = env.state
    tile_res0 = Resource(int(env.res_map[int(pos0), int(ctx0)]))
    levels0 = np.array(
        [
            int(body.energy),
            int(body.hydration),
            int(body.sleep_debt),
            int(body.oxygen),
        ],
        dtype=np.int32,
    )
    return {
        "pos_t": int(pos0),
        "ctx_t": int(ctx0),
        "tile_res_t": int(tile_res0),
        "obs_t": int(obs[0]),
        "obs_res_t": int(obs[1]),
        "season_obs_t": np.array(obs[2], copy=True),
        "typeI_levels_t": np.array(levels0, copy=True),
        "typeII_true_t": np.array(body.get_typeII_levels(), copy=True),
        "typeII_obs_t": np.array(typeII_obs, copy=True),
    }


def run_canonical_step(
    *,
    env,
    body,
    agent,
    obs,
    typeII_obs: np.ndarray,
) -> tuple[object, np.ndarray, dict[str, object]]:
    """
    Execute one maintained Python rollout step and return the next observations
    plus a JSON-ish record of the transition.
    """

    obs_t = obs
    typeII_obs_t = np.array(typeII_obs, copy=True)
    pos_t, ctx_t = env.state
    tile_res_t = Resource(int(env.res_map[int(pos_t), int(ctx_t)]))
    typeI_levels_t = np.array(
        [
            int(body.energy),
            int(body.hydration),
            int(body.sleep_debt),
            int(body.oxygen),
        ],
        dtype=np.int32,
    )
    typeII_true_t = np.array(body.get_typeII_levels(), copy=True)

    action = agent.step(obs_t, typeII_obs_t)
    if isinstance(action, tuple) and len(action) == 3:
        move, interact, kappa = action
    else:
        move, interact = action  # type: ignore[misc]
        kappa = None

    obs_t1, env_done, info = env.step(move, interact)
    pos_next, ctx_next = env.state
    tile_res_next = Resource(int(info.get("resource_after", info.get("resource", -1))))

    if env_done:
        body_done = False
        typeI_levels_t1 = np.array(
            [body.energy, body.hydration, body.sleep_debt, body.oxygen],
            dtype=np.int8,
        )
        typeII_true_t1 = np.array(body.get_typeII_levels(), copy=True)
        typeII_obs_t1 = np.array(typeII_obs_t, copy=True)
        obs_next = obs_t1
    else:
        typeI_levels_t1, _norm, _impair, typeII_true_t1, body_done = body.tick(interact, tile_res_next)
        typeII_obs_t1 = np.array(
            body.sample_interoception(attended=to_attended_idx(kappa)),
            copy=True,
        )
        obs_next = obs_t1

    observe_transition = getattr(agent, "observe_transition", None)
    if callable(observe_transition):
        observe_transition(
            obs_t=obs_t,
            typeII_obs_t=np.array(typeII_obs_t, copy=True),
            action=(
                int(move),
                int(interact),
                None if kappa is None else int(kappa),
            ),
            obs_t1=obs_t1,
            typeII_obs_t1=np.array(typeII_obs_t1, copy=True),
            done=bool(env_done or body_done),
            env_done=bool(env_done),
            body_done=bool(body_done),
            info=dict(info),
        )

    record: dict[str, object] = {
        "pos_t": int(pos_t),
        "ctx_t": int(ctx_t),
        "tile_res_t": int(tile_res_t),
        "pos_next": int(pos_next),
        "ctx_next": int(ctx_next),
        "tile_res_next": int(tile_res_next),
        "obs_t": int(obs_t[0]),
        "obs_res_t": int(obs_t[1]),
        "season_obs_t": np.array(obs_t[2], copy=True),
        "typeI_levels_t": np.array(typeI_levels_t, copy=True),
        "typeII_true_t": np.array(typeII_true_t, copy=True),
        "typeII_obs_t": np.array(typeII_obs_t, copy=True),
        "move": int(move),
        "interact": int(interact),
        "kappa": None if kappa is None else int(kappa),
        "beliefs": agent.latest_beliefs() if hasattr(agent, "latest_beliefs") else {},
        "policy": agent.latest_policy_diagnostics() if hasattr(agent, "latest_policy_diagnostics") else {},
        "typeI_levels_t1": np.array(typeI_levels_t1, copy=True),
        "typeII_true_t1": np.array(typeII_true_t1, copy=True),
        "typeII_obs_t1": np.array(typeII_obs_t1, copy=True),
        "obs_t1": int(obs_t1[0]),
        "obs_res_t1": int(obs_t1[1]),
        "season_obs_t1": np.array(obs_t1[2], copy=True),
        "env_info": dict(info),
        "env_done": bool(env_done),
        "body_done": bool(body_done),
    }
    return obs_next, np.array(typeII_obs_t1, copy=True), record


__all__ = [
    "capture_initial_step_record",
    "run_canonical_step",
    "to_attended_idx",
]

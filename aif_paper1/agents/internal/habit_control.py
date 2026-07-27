"""
Shared helpers for habit/default-policy mechanisms and option-style control.

Design goals:
- keep habit logic reusable across ActInf and RL-style agents
- keep "no habits" as a strict no-op default
- provide both prior-based and option-based habitual pathways
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from aif_paper1.affectworld.helpers import Interact, Move, Need, Resource, need_to_interact


def normalize_habit_mode(mode: str) -> str:
    """Validate and canonicalize the habit-control mode name."""
    m = str(mode).strip().lower()
    if m in {"", "none"}:
        return "none"
    raise ValueError(f"Unknown habit_mode={mode!r}; expected none")


def normalize_proposal_mode(mode: str) -> str:
    """Validate and canonicalize the root-action proposal mode name."""
    m = str(mode).strip().lower()
    if m in {"", "none"}:
        return "none"
    raise ValueError(f"Unknown proposal_mode={mode!r}; expected none")


def target_resource_for_need(need_idx: int) -> Resource:
    """Map a salient need to the resource tile that can relieve it."""
    n = int(need_idx)
    if n == int(Need.HUNGER):
        return Resource.FOOD
    if n in {int(Need.THIRST), int(Need.SUFFOCATION)}:
        return Resource.WATER
    if n == int(Need.FATIGUE):
        return Resource.SLEEP
    return Resource.AIR


def target_interact_for_need(need_idx: int) -> Interact:
    """Map a salient need to the corresponding interaction primitive."""
    n = int(need_idx)
    if n in {int(Need.HUNGER), int(Need.THIRST), int(Need.FATIGUE), int(Need.SUFFOCATION)}:
        return need_to_interact(Need(n))
    return Interact.NONE


def _target_resource_for_interact(interact_idx: int) -> Resource:
    """Map an interaction primitive to the resource it meaningfully uses."""
    interact = int(interact_idx)
    if interact == int(Interact.EAT):
        return Resource.FOOD
    if interact in {int(Interact.DRINK), int(Interact.BREATHE)}:
        return Resource.WATER
    if interact == int(Interact.SLEEP):
        return Resource.SLEEP
    return Resource.AIR


def dominant_need_idx(typeII_obs: Any) -> int:
    """
    Choose the currently most salient need from Type-II observations.

    This is a cheap proxy used by the habit system: whichever interoceptive
    channel reports the largest observed level becomes the candidate need. If
    all reported levels are zero or missing, the function falls back to
    ``Need.NONE``.
    """
    arr = np.asarray(typeII_obs, dtype=np.int32).reshape(-1)
    if arr.size <= 0:
        return int(Need.NONE)
    if arr.size < 4:
        pad = np.zeros((4,), dtype=np.int32)
        pad[: arr.size] = arr
        arr = pad
    arr4 = np.clip(arr[:4], 0, None)
    idx = int(np.argmax(arr4))
    return idx if int(arr4[idx]) > 0 else int(Need.NONE)


def _normalize_prob_vec(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.full_like(arr, 1.0 / float(arr.size))
    return arr / total


def _idx_to_xy(idx: int, width: int) -> tuple[int, int]:
    w = max(1, int(width))
    return int(idx % w), int(idx // w)


def _manhattan(a: int, b: int, width: int) -> int:
    ax, ay = _idx_to_xy(int(a), int(width))
    bx, by = _idx_to_xy(int(b), int(width))
    return abs(ax - bx) + abs(ay - by)


def _resource_map_array(env, resource_map: np.ndarray | None = None) -> np.ndarray | None:
    if resource_map is None:
        arr = np.asarray(env.res_map, dtype=np.int32)
    else:
        arr = np.asarray(resource_map, dtype=np.int32)
    if arr.ndim != 2:
        return None
    return arr


def _resource_target_position_weights(
    env,
    *,
    ctx: int,
    resource: Resource,
    resource_map: np.ndarray | None = None,
    resource_likelihood: np.ndarray | None = None,
    ctx_belief: np.ndarray | None = None,
) -> np.ndarray | None:
    if resource_likelihood is not None:
        arr = np.asarray(resource_likelihood, dtype=np.float32)
        if arr.ndim != 3:
            return None
        n_res, n_pos, n_ctx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        target = int(resource)
        if target < 0 or target >= n_res:
            return None
        if ctx_belief is None:
            ctx_w = np.zeros((n_ctx,), dtype=np.float32)
            ctx_w[int(np.clip(int(ctx), 0, max(0, n_ctx - 1)))] = 1.0
        else:
            raw = np.asarray(ctx_belief, dtype=np.float32).reshape(-1)
            ctx_w = np.zeros((n_ctx,), dtype=np.float32)
            if raw.size > 0:
                ctx_w[: min(n_ctx, int(raw.size))] = raw[: min(n_ctx, int(raw.size))]
            ctx_w = _normalize_prob_vec(ctx_w)
        pos_mass = np.tensordot(arr[target], ctx_w, axes=(1, 0))
        if pos_mass.shape != (n_pos,):
            return None
        return np.clip(np.asarray(pos_mass, dtype=np.float32), 0.0, None)

    res_map = _resource_map_array(env, resource_map)
    if res_map is None or res_map.ndim != 2:
        return None
    n_pos, n_ctx = int(res_map.shape[0]), int(res_map.shape[1])
    c = int(np.clip(int(ctx), 0, max(0, n_ctx - 1)))
    weights = np.zeros((n_pos,), dtype=np.float32)
    weights[res_map[:, c] == int(resource)] = 1.0
    return weights


def nearest_resource_position(
    env,
    *,
    ctx: int,
    resource: Resource,
    start_pos: int,
    resource_map: np.ndarray | None = None,
) -> Optional[int]:
    """
    Return the closest tile containing ``resource`` in the selected context.

    Distance is measured with Manhattan geometry over the grid, which keeps the
    option system intentionally simple and local.
    """
    res_map = _resource_map_array(env, resource_map)
    if res_map is None:
        return None
    if res_map.ndim != 2:
        return None
    n_pos, n_ctx = int(res_map.shape[0]), int(res_map.shape[1])
    c = int(np.clip(int(ctx), 0, max(0, n_ctx - 1)))
    target = int(resource)
    pos = np.flatnonzero(res_map[:, c] == target)
    if pos.size <= 0:
        return None
    w = int(getattr(env, "w", max(1, int(np.sqrt(max(1, n_pos))))))
    d = np.asarray([_manhattan(int(start_pos), int(p), w) for p in pos], dtype=np.int32)
    return int(pos[int(np.argmin(d))])


def _position_proximity(weights: np.ndarray, *, width: int) -> np.ndarray:
    mass = _normalize_prob_vec(weights)
    n = int(mass.size)
    xs = np.arange(n, dtype=np.int32) % max(1, int(width))
    ys = np.arange(n, dtype=np.int32) // max(1, int(width))
    dist = np.abs(xs[:, None] - xs[None, :]) + np.abs(ys[:, None] - ys[None, :])
    return np.asarray((mass[None, :] / (1.0 + dist.astype(np.float32))).sum(axis=1), dtype=np.float32)


def _rollout_positions(env, *, start_pos: int, move_seq: np.ndarray) -> np.ndarray:
    trans = np.asarray(getattr(env, "_trans"), dtype=np.int32)
    if trans.ndim != 2:
        raise ValueError("env._trans must be a 2D transition table")
    moves = np.asarray(move_seq, dtype=np.int32)
    out = np.empty_like(moves, dtype=np.int32)
    cur = np.full((int(moves.shape[0]),), int(start_pos), dtype=np.int32)
    max_move = max(0, int(trans.shape[0]) - 1)
    for t in range(int(moves.shape[1])):
        mv_t = np.clip(moves[:, t], 0, max_move).astype(np.int32, copy=False)
        cur = trans[mv_t, cur]
        out[:, t] = cur
    return out


def greedy_move_towards(env, *, src_pos: int, dst_pos: int) -> Move:
    """
    Pick the one-step move that most reduces Manhattan distance to a target.

    This is not full path planning. It is a cheap directional heuristic used by
    option habits when the full planner is skipped or down-weighted.
    """
    if int(src_pos) == int(dst_pos):
        return Move.REST

    trans = np.asarray(getattr(env, "_trans"), dtype=np.int32)
    if trans.ndim != 2:
        return Move.REST

    w = int(getattr(env, "w", 1))
    best = Move.REST
    best_dist = _manhattan(int(src_pos), int(dst_pos), w)
    for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.REST):
        nxt = int(trans[int(mv), int(src_pos)])
        d = _manhattan(nxt, int(dst_pos), w)
        if d < best_dist:
            best = mv
            best_dist = d
    return best


@dataclass
class OptionState:
    """Persistent state for an in-progress habit option."""

    need_idx: int
    target_pos: int
    target_res: int
    remaining: int
    age: int = 0


def choose_option_action(
    env,
    *,
    pos: int,
    ctx: int,
    tile_res_obs: int | None,
    resource_map: np.ndarray | None,
    suggested_need: int,
    active_option: OptionState | None,
    commit_steps: int,
    max_steps: int,
) -> tuple[int, int, int, OptionState | None, dict[str, Any]]:
    """
    Roll a habit option forward by one environment step.

    In order, this helper:
    1. reuses the current option if it is still committed, otherwise starts a
       new option from the currently salient need;
    2. selects the nearest resource tile that can satisfy that need;
    3. either interacts immediately when already at the target, or emits a
       greedy move toward it;
    4. returns the next option state so commitment can persist across steps.

    The returned ``need_idx`` is the option-aligned need label, which the
    attentional agents can also use as the κ-aligned first-step action.
    """
    info: dict[str, Any] = {"source": "option"}
    pos_i = int(pos)
    ctx_i = int(ctx)
    need_i = int(suggested_need)

    opt = active_option
    if opt is None or int(opt.remaining) <= 0 or int(opt.age) >= int(max_steps):
        target_res = target_resource_for_need(need_i)
        target_pos = nearest_resource_position(
            env,
            ctx=ctx_i,
            resource=target_res,
            start_pos=pos_i,
            resource_map=resource_map,
        )
        if target_pos is None:
            info.update({"option_valid": False, "reason": "no_target"})
            return int(Move.REST), int(Interact.NONE), need_i, None, info
        opt = OptionState(
            need_idx=need_i,
            target_pos=int(target_pos),
            target_res=int(target_res),
            remaining=max(1, int(commit_steps)),
            age=0,
        )

    need_i = int(opt.need_idx)
    target_pos = int(opt.target_pos)
    target_res = Resource(int(opt.target_res))
    interact_t = target_interact_for_need(need_i)

    if tile_res_obs is None:
        res_map = _resource_map_array(env, resource_map)
        if res_map is None:
            tile_res = int(Resource.AIR)
        else:
            ctx_clip = int(np.clip(int(ctx_i), 0, max(0, int(res_map.shape[1]) - 1)))
            pos_clip = int(np.clip(int(pos_i), 0, max(0, int(res_map.shape[0]) - 1)))
            tile_res = int(res_map[pos_clip, ctx_clip])
    else:
        tile_res = int(tile_res_obs)
    at_target = int(pos_i) == int(target_pos) or tile_res == int(target_res)
    if at_target and int(interact_t) != int(Interact.NONE):
        move = Move.REST
        interact = interact_t
        next_opt = None
    else:
        move = greedy_move_towards(env, src_pos=pos_i, dst_pos=target_pos)
        interact = Interact.NONE
        rem = int(opt.remaining) - 1
        age = int(opt.age) + 1
        next_opt = None if rem <= 0 or age >= int(max_steps) else OptionState(
            need_idx=need_i,
            target_pos=target_pos,
            target_res=int(target_res),
            remaining=rem,
            age=age,
        )

    info.update(
        {
            "option_valid": True,
            "need_idx": need_i,
            "target_pos": target_pos,
            "target_res": int(target_res),
            "at_target": bool(at_target),
            "remaining": 0 if next_opt is None else int(next_opt.remaining),
        }
    )
    return int(move), int(interact), need_i, next_opt, info


def decode_policy_controls(
    policies: np.ndarray,
    *,
    has_kappa: bool,
    n_interact: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Decode whole policy sequences into aligned ``(move, interact, need?)`` arrays.

    Returned arrays have shape ``(n_policies, H)``.
    """
    pol = np.asarray(policies, dtype=np.int32)
    if pol.ndim != 3 or int(pol.shape[0]) <= 0:
        raise ValueError(f"Unexpected policies shape={pol.shape}; expected (n_policies, H, n_actions)")

    if int(pol.shape[-1]) in {2, 3}:
        move = pol[:, :, 0]
        interact = pol[:, :, 1]
        need = pol[:, :, 2] if int(pol.shape[-1]) == 3 and bool(has_kappa) else None
        return move.astype(np.int32), interact.astype(np.int32), None if need is None else need.astype(np.int32)

    move = pol[:, :, 0].astype(np.int32)
    joint = pol[:, :, 2].astype(np.int32)
    interact = np.mod(joint, int(max(1, n_interact))).astype(np.int32)
    need = pol[:, :, -1].astype(np.int32) if bool(has_kappa) else None
    return move, interact, need


def decode_first_step_controls(
    policies: np.ndarray,
    *,
    has_kappa: bool,
    n_interact: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Extract the actually executed first-step controls from a policy library.

    Policies may be stored either directly by control factor or in flattened
    hidden-factor form after ``B`` action dependencies are expanded. This helper
    normalizes both encodings to ``(move, interact, need?)`` arrays.
    """
    move, interact, need = decode_policy_controls(
        np.asarray(policies, dtype=np.int32),
        has_kappa=bool(has_kappa),
        n_interact=int(n_interact),
    )
    need_first = None if need is None else need[:, 0]
    return move[:, 0], interact[:, 0], need_first


def qpi_for_action_match(
    policies: np.ndarray,
    *,
    move_idx: int,
    interact_idx: int,
    has_kappa: bool,
    n_interact: int,
    need_idx: int | None = None,
) -> np.ndarray:
    """
    Build a distribution that is uniform over policies matching a first action.

    This is used to convert an option or heuristic recommendation into a
    categorical distribution over the planner's existing policy set. If no
    policy matches exactly, the fallback is uniform over all policies.
    """
    move, interact, need = decode_first_step_controls(
        np.asarray(policies, dtype=np.int32),
        has_kappa=bool(has_kappa),
        n_interact=int(n_interact),
    )
    mask = (move == int(move_idx)) & (interact == int(interact_idx))
    if bool(has_kappa) and need_idx is not None and need is not None:
        mask = mask & (need == int(need_idx))

    p = np.zeros((int(mask.shape[0]),), dtype=np.float32)
    if np.any(mask):
        p[mask] = 1.0
        p /= float(max(1, p.sum()))
        return p
    p[:] = 1.0 / float(max(1, p.shape[0]))
    return p


def habit_prior_from_need(
    policies: np.ndarray,
    *,
    env,
    pos: int,
    ctx: int,
    resource_map: np.ndarray | None,
    resource_likelihood: np.ndarray | None = None,
    ctx_belief: np.ndarray | None = None,
    need_idx: int,
    has_kappa: bool,
    n_interact: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Score policies with a cheap need-conditioned habit heuristic.

    Compared with the planner's expected-free-energy objective, this is a much
    simpler bias: it rewards first-step interactions that match the target need,
    rewards moves that get closer to the nearest relevant resource, and, when κ
    exists, mildly rewards policies whose first-step κ already matches the same
    need.

    Important interpretation: when this distribution is injected directly into
    `E(pi)`, it becomes an additional policy-selection term alongside EFE. The
    cleaner alternative is to use the same heuristic only to propose or prune
    candidate policies, then let plain EFE rank the retained set.
    """
    move_seq, interact_seq, need_seq = decode_policy_controls(
        np.asarray(policies, dtype=np.int32),
        has_kappa=bool(has_kappa),
        n_interact=int(n_interact),
    )
    need_i = int(need_idx)
    target_res = target_resource_for_need(need_i)
    target_interact = target_interact_for_need(need_i)
    target_weights = _resource_target_position_weights(
        env,
        ctx=int(ctx),
        resource=target_res,
        resource_map=resource_map,
        resource_likelihood=resource_likelihood,
        ctx_belief=ctx_belief,
    )

    scores = np.zeros((int(move_seq.shape[0]),), dtype=np.float32)
    wrong_non_none = (interact_seq != int(target_interact)) & (interact_seq != int(Interact.NONE))
    scores -= 0.35 * wrong_non_none.astype(np.float32).sum(axis=1)

    if target_weights is not None and float(np.max(target_weights)) > 0.0:
        mass = _normalize_prob_vec(target_weights)
        baseline = 1.0 / float(max(1, mass.size))
        signal = np.clip(mass - baseline, 0.0, None)
        if float(signal.sum()) > 0.0:
            signal = _normalize_prob_vec(signal)
            pos_seq = _rollout_positions(env, start_pos=int(pos), move_seq=move_seq)
            tile_signal = signal[pos_seq]
            match_signal = tile_signal * (interact_seq == int(target_interact)).astype(np.float32)
            proximity = _position_proximity(signal, width=int(getattr(env, "w", 1)))
            start_proximity = float(proximity[int(np.clip(int(pos), 0, max(0, int(proximity.size) - 1)))])
            best_proximity = np.max(proximity[pos_seq], axis=1)
            scores += 1.25 * np.maximum(best_proximity - start_proximity, 0.0).astype(np.float32)
            scores += 1.50 * np.max(tile_signal, axis=1).astype(np.float32)
            scores += 2.00 * np.max(match_signal, axis=1).astype(np.float32)
            immediate_match = (interact_seq[:, 0] == int(target_interact)).astype(np.float32)
            scores += 0.75 * (tile_signal[:, 0] * immediate_match).astype(np.float32)
            scores -= 0.50 * ((1.0 - tile_signal[:, 0]) * immediate_match).astype(np.float32)

            immediate_non_none = (interact_seq[:, 0] != int(Interact.NONE)).astype(np.float32)
            wrong_target = np.asarray(
                [
                    _target_resource_for_interact(int(interact_t)) != target_res
                    for interact_t in interact_seq[:, 0].tolist()
                ],
                dtype=np.float32,
            )
            scores -= 0.40 * ((1.0 - tile_signal[:, 0]) * immediate_non_none * wrong_target).astype(np.float32)

    if bool(has_kappa) and need_seq is not None:
        need0 = need_seq[:, 0]
        scores += np.where(need0 == need_i, 0.8, 0.0).astype(np.float32)
        scores -= np.where((need0 != need_i) & (need0 != int(Need.NONE)), 0.2, 0.0).astype(np.float32)

    tau = float(max(1e-3, temperature))
    logits = scores / tau
    logits -= float(np.max(logits))
    p = np.exp(logits, dtype=np.float64)
    p = p / max(1e-16, float(np.sum(p)))
    return np.asarray(p, dtype=np.float32)


def habit_root_action_proposal_from_need(
    action_multi: np.ndarray,
    *,
    env,
    pos: int,
    ctx: int,
    resource_map: np.ndarray | None,
    resource_likelihood: np.ndarray | None = None,
    ctx_belief: np.ndarray | None = None,
    need_idx: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Score one-step root actions with the same need-conditioned heuristic family.

    This is the search-planner analogue of `habit_prior_from_need`: instead of
    proposing a subset of full policies, it proposes a subset of root actions
    before beam search evaluates retained candidates with its unchanged search
    objective.
    """
    action_arr = np.asarray(action_multi, dtype=np.int32)
    if action_arr.ndim != 2 or int(action_arr.shape[0]) <= 0 or int(action_arr.shape[1]) < 2:
        raise ValueError(
            f"Unexpected action library shape={action_arr.shape}; expected (n_actions, >=2)."
        )

    move0 = action_arr[:, 0].astype(np.int32, copy=False)
    interact0 = action_arr[:, 1].astype(np.int32, copy=False)
    need_i = int(need_idx)
    target_res = target_resource_for_need(need_i)
    target_interact = target_interact_for_need(need_i)
    target_weights = _resource_target_position_weights(
        env,
        ctx=int(ctx),
        resource=target_res,
        resource_map=resource_map,
        resource_likelihood=resource_likelihood,
        ctx_belief=ctx_belief,
    )

    scores = np.zeros((int(action_arr.shape[0]),), dtype=np.float32)
    wrong_non_none = (interact0 != int(target_interact)) & (interact0 != int(Interact.NONE))
    scores -= 0.35 * wrong_non_none.astype(np.float32)

    if target_weights is not None and float(np.max(target_weights)) > 0.0:
        mass = _normalize_prob_vec(target_weights)
        baseline = 1.0 / float(max(1, mass.size))
        signal = np.clip(mass - baseline, 0.0, None)
        if float(signal.sum()) > 0.0:
            signal = _normalize_prob_vec(signal)
            trans = np.asarray(getattr(env, "_trans"), dtype=np.int32)
            if trans.ndim != 2:
                raise ValueError("env._trans must be a 2D transition table")
            pos_clip = int(np.clip(int(pos), 0, max(0, int(trans.shape[1]) - 1)))
            move_clip = np.clip(move0, 0, max(0, int(trans.shape[0]) - 1))
            next_pos = trans[move_clip, pos_clip]
            tile_signal = signal[next_pos]
            match_signal = tile_signal * (interact0 == int(target_interact)).astype(np.float32)
            proximity = _position_proximity(signal, width=int(getattr(env, "w", 1)))
            start_proximity = float(
                proximity[int(np.clip(int(pos_clip), 0, max(0, int(proximity.size) - 1)))]
            )
            next_proximity = proximity[next_pos]
            scores += 1.25 * np.maximum(next_proximity - start_proximity, 0.0).astype(np.float32)
            scores += 1.50 * tile_signal.astype(np.float32)
            scores += 2.00 * match_signal.astype(np.float32)
            target_interact_mask = (interact0 == int(target_interact)).astype(np.float32)
            scores += 0.75 * (tile_signal * target_interact_mask).astype(np.float32)
            scores -= 0.50 * ((1.0 - tile_signal) * target_interact_mask).astype(np.float32)

            non_none_mask = (interact0 != int(Interact.NONE)).astype(np.float32)
            target_res_for_interact = np.asarray(
                [_target_resource_for_interact(int(interact_idx)) for interact_idx in interact0.tolist()],
                dtype=np.int32,
            )
            wrong_target = (target_res_for_interact != int(target_res)).astype(np.float32)
            scores -= 0.40 * ((1.0 - tile_signal) * non_none_mask * wrong_target).astype(np.float32)

    tau = float(max(1e-3, temperature))
    logits = scores / tau
    logits -= float(np.max(logits))
    p = np.exp(logits, dtype=np.float64)
    p = p / max(1e-16, float(np.sum(p)))
    return np.asarray(p, dtype=np.float32)


def habit_action_proposal_keep_indices(
    proposal: np.ndarray,
    *,
    action_multi: np.ndarray,
    mass_threshold: float = 0.90,
    min_keep: int = 4,
    max_keep: int = 12,
    keep_per_move: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Retain a diverse high-mass subset of root actions from a proposal.

    Diversity is enforced over the root move primitive rather than exact
    `(Move, Interact)` pairs, which keeps at least one candidate per movement
    direction when the proposal is sharply peaked.
    """
    probs = _normalize_prob_vec(np.asarray(proposal, dtype=np.float32).reshape(-1))
    action_arr = np.asarray(action_multi, dtype=np.int32)
    if action_arr.ndim != 2 or int(action_arr.shape[0]) != int(probs.size):
        raise ValueError(
            f"Action proposal/library size mismatch: proposal={probs.shape} action_multi={action_arr.shape}"
        )
    n_actions = int(probs.size)
    if n_actions <= 0:
        return np.zeros((0,), dtype=np.int32), {
            "applied": False,
            "reason": "empty_action_library",
            "kept": 0,
            "available": 0,
        }

    min_keep_i = int(max(1, min_keep))
    max_keep_i = int(max_keep)
    max_keep_eff = n_actions if max_keep_i <= 0 else int(max(min_keep_i, min(max_keep_i, n_actions)))
    keep_per_move_i = int(max(0, keep_per_move))
    target_mass = float(np.clip(float(mass_threshold), 0.0, 1.0))

    uniform_p = 1.0 / float(n_actions)
    entropy = float(-(probs * np.log(probs + 1e-16)).sum())
    max_entropy = float(np.log(float(n_actions))) if n_actions > 1 else 0.0
    entropy_ratio = 1.0 if max_entropy <= 1e-12 else float(entropy / max_entropy)
    top_prob = float(np.max(probs))
    near_uniform = bool(
        (n_actions <= max_keep_eff)
        or (entropy_ratio >= 0.995 and top_prob <= 1.25 * uniform_p)
    )
    if near_uniform:
        keep_idx = np.arange(n_actions, dtype=np.int32)
        return keep_idx, {
            "applied": False,
            "reason": "proposal_near_uniform",
            "kept": int(keep_idx.size),
            "available": int(n_actions),
            "retained_mass": 1.0,
            "target_mass": float(target_mass),
            "entropy_ratio": float(entropy_ratio),
            "top_prob": float(top_prob),
            "uniform_prob": float(uniform_p),
        }

    order = np.argsort(-probs).astype(np.int32, copy=False)
    keep: list[int] = []
    selected: set[int] = set()
    move_counts: dict[int, int] = {}
    if keep_per_move_i > 0:
        move0 = action_arr[:, 0].astype(np.int32, copy=False)
        for idx in order.tolist():
            mv = int(move0[idx])
            if int(move_counts.get(mv, 0)) >= keep_per_move_i:
                continue
            keep.append(int(idx))
            selected.add(int(idx))
            move_counts[mv] = int(move_counts.get(mv, 0)) + 1

    cum_mass = float(sum(float(probs[idx]) for idx in keep))
    for idx in order.tolist():
        if len(keep) >= max_keep_eff:
            break
        if idx in selected:
            continue
        keep.append(int(idx))
        selected.add(int(idx))
        cum_mass += float(probs[idx])
        if len(keep) >= min_keep_i and cum_mass >= target_mass:
            break

    if not keep:
        keep = [int(order[0])]
        cum_mass = float(probs[int(order[0])])

    keep_idx = np.asarray(sorted(set(keep)), dtype=np.int32)
    retained_mass = float(probs[keep_idx].sum()) if keep_idx.size > 0 else 0.0
    return keep_idx, {
        "applied": bool(keep_idx.size < n_actions),
        "reason": "pruned" if keep_idx.size < n_actions else "all_kept",
        "kept": int(keep_idx.size),
        "available": int(n_actions),
        "retained_mass": float(retained_mass),
        "target_mass": float(target_mass),
        "entropy_ratio": float(entropy_ratio),
        "top_prob": float(top_prob),
        "uniform_prob": float(uniform_p),
        "keep_per_move": int(keep_per_move_i),
    }


def habit_proposal_keep_indices(
    proposal: np.ndarray,
    *,
    policies: np.ndarray,
    has_kappa: bool,
    n_interact: int,
    mass_threshold: float = 0.95,
    min_keep: int = 64,
    max_keep: int = 512,
    keep_per_first_action: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Retain a diverse high-mass subset of policies from a habit proposal.

    This helper is the cleaner "proposal/prune" alternative to injecting the
    heuristic directly into the policy posterior. The same need-conditioned
    heuristic can propose candidate policies, while the retained subset is still
    ranked later by the planner's unchanged EFE objective.
    """
    probs = _normalize_prob_vec(np.asarray(proposal, dtype=np.float32).reshape(-1))
    n_policies = int(probs.size)
    if n_policies <= 0:
        return np.zeros((0,), dtype=np.int32), {
            "applied": False,
            "reason": "empty_policy_library",
            "kept": 0,
            "available": 0,
        }

    min_keep_i = int(max(1, min_keep))
    max_keep_i = int(max_keep)
    max_keep_eff = n_policies if max_keep_i <= 0 else int(max(min_keep_i, min(max_keep_i, n_policies)))
    keep_per_action_i = int(max(0, keep_per_first_action))
    target_mass = float(np.clip(float(mass_threshold), 0.0, 1.0))

    uniform_p = 1.0 / float(n_policies)
    entropy = float(-(probs * np.log(probs + 1e-16)).sum())
    max_entropy = float(np.log(float(n_policies))) if n_policies > 1 else 0.0
    entropy_ratio = 1.0 if max_entropy <= 1e-12 else float(entropy / max_entropy)
    top_prob = float(np.max(probs))
    near_uniform = bool(
        (n_policies <= max_keep_eff)
        or (entropy_ratio >= 0.995 and top_prob <= 1.25 * uniform_p)
    )
    if near_uniform:
        keep_idx = np.arange(n_policies, dtype=np.int32)
        return keep_idx, {
            "applied": False,
            "reason": "proposal_near_uniform",
            "kept": int(keep_idx.size),
            "available": int(n_policies),
            "retained_mass": 1.0,
            "target_mass": float(target_mass),
            "entropy_ratio": float(entropy_ratio),
            "top_prob": float(top_prob),
            "uniform_prob": float(uniform_p),
        }

    order = np.argsort(-probs).astype(np.int32, copy=False)
    keep: list[int] = []
    selected: set[int] = set()
    first_step_counts: dict[tuple[int, int, int | None], int] = {}
    if keep_per_action_i > 0:
        move0, interact0, need0 = decode_first_step_controls(
            np.asarray(policies, dtype=np.int32),
            has_kappa=bool(has_kappa),
            n_interact=int(n_interact),
        )
        for idx in order.tolist():
            sig = (
                int(move0[idx]),
                int(interact0[idx]),
                None if need0 is None else int(need0[idx]),
            )
            if int(first_step_counts.get(sig, 0)) >= keep_per_action_i:
                continue
            keep.append(int(idx))
            selected.add(int(idx))
            first_step_counts[sig] = int(first_step_counts.get(sig, 0)) + 1

    retained_mass = float(sum(float(probs[idx]) for idx in keep))
    for idx in order.tolist():
        idx_i = int(idx)
        if idx_i in selected:
            continue
        keep.append(idx_i)
        selected.add(idx_i)
        retained_mass += float(probs[idx_i])
        enough_count = len(keep) >= min_keep_i
        enough_mass = retained_mass >= target_mass
        capped = len(keep) >= max_keep_eff
        if capped or (enough_count and enough_mass):
            break

    keep = keep[:max_keep_eff]
    keep_idx = np.asarray(keep, dtype=np.int32)
    retained_mass = float(probs[keep_idx].sum()) if keep_idx.size > 0 else 0.0
    return keep_idx, {
        "applied": bool(keep_idx.size < n_policies),
        "reason": "proposal_subset",
        "kept": int(keep_idx.size),
        "available": int(n_policies),
        "retained_mass": float(retained_mass),
        "target_mass": float(target_mass),
        "entropy_ratio": float(entropy_ratio),
        "top_prob": float(top_prob),
        "uniform_prob": float(uniform_p),
        "keep_per_first_action": int(keep_per_action_i),
        "min_keep": int(min_keep_i),
        "max_keep": int(max_keep_eff),
    }


def update_computation_budget(
    budget: float,
    *,
    budget_max: float,
    recovery: float,
    step_cost: float,
    switch_cost: float = 0.0,
    switched: bool = False,
) -> float:
    """
    Update the bounded computation budget used for planner/habit arbitration.

    Each step subtracts the chosen control pathway's cost, optionally charges an
    extra switch penalty when the source changes, adds passive recovery, then
    clamps the result into ``[0, budget_max]``.
    """
    bmax = float(max(1e-6, budget_max))
    b = float(np.clip(float(budget), 0.0, bmax))
    c = float(max(0.0, step_cost))
    if bool(switched):
        c += float(max(0.0, switch_cost))
    b = b - c + float(max(0.0, recovery))
    return float(np.clip(b, 0.0, bmax))

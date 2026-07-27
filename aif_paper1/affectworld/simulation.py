"""
simulation.py
=============

Canonical simulation loop to keep timing/causal ordering consistent across
scripts and experiments.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np

from aif_paper1.affectworld.canonical_step import capture_initial_step_record, run_canonical_step
from aif_paper1.affectworld.helpers import Resource
from aif_paper1.affectworld.logging_utils import EpisodeLog

StepCallback = Callable[
    [
        int,  # t
        dict[str, object],  # step record (small, JSON-ish)
    ],
    None,
]


class Simulation:
    """Reference Python rollout loop that preserves the repo's canonical step ordering."""

    def __init__(self, env, body, agent, *, max_steps: int = 200):
        """Store the environment, body, and agent objects used for episodic rollouts."""

        self.env = env
        self.body = body
        self.agent = agent
        self.max_steps = int(max_steps)

    def run_episode(
        self,
        *,
        start_pos: Optional[int] = None,
        start_ctx: Optional[int] = None,
        body_init: Optional[Mapping[str, int]] = None,
        on_step: Optional[StepCallback] = None,
    ) -> EpisodeLog:
        """
        Run a single episode using the canonical ordering:
          1) observe extero + intero_t
          2) agent selects (move, interact [, kappa])
          3) env.step(move) -> destination tile + extero_{t+1}
          4) body.tick(interact, tile_at_destination_pos)
          5) sample intero_{t+1} using kappa_t (if any)
        """
        obs = self.env.reset(pos=start_pos, ctx=None if start_ctx is None else int(start_ctx))
        self.body.reset()
        if body_init is not None and hasattr(self.body, "apply_initial_state_dict"):
            self.body.apply_initial_state_dict(body_init)
        self.agent.reset()

        typeII_obs = self.body.sample_interoception(attended=None)

        if on_step is not None:
            on_step(-1, capture_initial_step_record(self.env, self.body, obs, np.asarray(typeII_obs)))

        pos_hist = []
        ctx_hist = []
        tile_res_hist = []
        pos_next_hist = []
        ctx_next_hist = []
        tile_res_next_hist = []
        env_ctx_changed_hist = []
        env_tile_mutated_hist = []
        env_mutated_pos_hist = []
        env_mutated_from_hist = []
        env_mutated_to_hist = []
        env_underwater_hist = []
        env_predator_hist = []
        obs_pos_hist = []
        obs_res_hist = []
        season_obs_hist = []
        move_hist = []
        interact_hist = []
        kappa_hist = []
        typeI_hist = []
        typeII_true_hist = []
        typeII_obs_hist = []

        terminated_by = "timeout"

        for _t in range(self.max_steps):
            pos_t, ctx_t = self.env.state
            tile_res_t = Resource(int(self.env.res_map[int(pos_t), int(ctx_t)]))
            typeII_obs_t = np.array(typeII_obs, copy=True)

            # Log decision-time quantities.
            pos_hist.append(int(pos_t))
            ctx_hist.append(int(ctx_t))
            tile_res_hist.append(int(tile_res_t))
            obs_pos_hist.append(int(obs[0]))
            obs_res_hist.append(int(obs[1]))
            season_obs_hist.append(np.array(obs[2], copy=True))
            typeII_obs_hist.append(np.array(typeII_obs_t, copy=True))

            obs, typeII_obs, step_rec = run_canonical_step(
                env=self.env,
                body=self.body,
                agent=self.agent,
                obs=obs,
                typeII_obs=typeII_obs_t,
            )
            move = int(step_rec["move"])
            interact = int(step_rec["interact"])
            kappa = step_rec.get("kappa")
            pos_next = int(step_rec["pos_next"])
            ctx_next = int(step_rec["ctx_next"])
            tile_res_next = Resource(int(step_rec["tile_res_next"]))
            obs_t1 = [step_rec["obs_t1"], step_rec["obs_res_t1"], step_rec["season_obs_t1"]]
            _info = dict(step_rec["env_info"])
            env_done = bool(step_rec["env_done"])
            body_done = bool(step_rec["body_done"])
            levels = np.asarray(step_rec["typeI_levels_t1"], dtype=np.int8)
            typeII_true = np.asarray(step_rec["typeII_true_t1"], dtype=np.int8)

            move_hist.append(int(move))
            interact_hist.append(int(interact))
            kappa_hist.append(-1 if kappa is None else int(kappa))

            pos_next_hist.append(int(pos_next))
            ctx_next_hist.append(int(ctx_next))
            tile_res_next_hist.append(int(tile_res_next))
            env_ctx_changed_hist.append(bool(_info.get("ctx_changed", False)))
            env_tile_mutated_hist.append(bool(_info.get("tile_mutated", False)))
            env_mutated_pos_hist.append(-1 if _info.get("mutated_pos") is None else int(_info.get("mutated_pos")))
            env_mutated_from_hist.append(-1 if _info.get("mutated_from") is None else int(_info.get("mutated_from")))
            env_mutated_to_hist.append(-1 if _info.get("mutated_to") is None else int(_info.get("mutated_to")))
            env_underwater_hist.append(int(_info.get("underwater", 0)))
            env_predator_hist.append(bool(_info.get("predator", False)))

            typeI_hist.append(np.array(levels, copy=True))
            typeII_true_hist.append(np.array(typeII_true, copy=True))

            if on_step is not None:
                on_step(int(_t), step_rec)

            if body_done:
                terminated_by = "body"
                break
            if env_done:
                break

        steps = len(pos_hist)

        kappa_arr = np.array(kappa_hist, dtype=np.int32)
        kappa_out: Optional[np.ndarray]
        if np.all(kappa_arr == -1):
            kappa_out = None
        else:
            kappa_out = kappa_arr

        return EpisodeLog(
            steps=steps,
            terminated_by=terminated_by,
            pos=np.array(pos_hist, dtype=np.int32),
            ctx=np.array(ctx_hist, dtype=np.int32),
            tile_res=np.array(tile_res_hist, dtype=np.int8),
            pos_next=np.array(pos_next_hist, dtype=np.int32),
            ctx_next=np.array(ctx_next_hist, dtype=np.int32),
            tile_res_next=np.array(tile_res_next_hist, dtype=np.int8),
            env_ctx_changed=np.array(env_ctx_changed_hist, dtype=bool),
            env_tile_mutated=np.array(env_tile_mutated_hist, dtype=bool),
            env_mutated_pos=np.array(env_mutated_pos_hist, dtype=np.int32),
            env_mutated_from=np.array(env_mutated_from_hist, dtype=np.int32),
            env_mutated_to=np.array(env_mutated_to_hist, dtype=np.int32),
            env_underwater=np.array(env_underwater_hist, dtype=np.int32),
            env_predator=np.array(env_predator_hist, dtype=bool),
            obs_pos=np.array(obs_pos_hist, dtype=np.int32),
            obs_res=np.array(obs_res_hist, dtype=np.int8),
            season_obs=np.array(season_obs_hist, dtype=np.int8),
            move=np.array(move_hist, dtype=np.int8),
            interact=np.array(interact_hist, dtype=np.int8),
            kappa=kappa_out,
            typeI_levels=np.array(typeI_hist, dtype=np.int8),
            typeII_true=np.array(typeII_true_hist, dtype=np.int8),
            typeII_obs=np.array(typeII_obs_hist, dtype=np.int8),
        )

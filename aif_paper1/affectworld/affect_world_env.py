"""
AffectWorld Environment
=======================
Contextual grid-world for Solmsian Active-Inference experiments.

Current observation format
--------------------------
obs = [pos_idx, res_idx, season_vec, ...optional_resource_field_levels]

* pos_idx    : int   – proprioceptive index (0 … width·height−1)
* res_idx    : int   – categorical observation in {AIR,WATER,FOOD,SLEEP,POISON}
* season_vec : (5,)  – one-hot [NoContext, Spring, Summer, Autumn, Winter]
* optional resource-field levels: one discrete level per configured channel

Season dynamics
---------------
`gaussian` mode models season tenure directly. Each season has an expected
length of `year_length / 4` steps. The probability of *staying* in the current
season starts high just after a transition and decays with time spent in that
season:

    stay(age) = p_max · exp(− phase² / (2 σ²) )

`phase = age / season_len`, clipped at `1.0`, and `age` resets to zero when the
context advances.

Other mechanics
---------------
* Lake tiles (water); drowning is handled by `BodyProcess` oxygen dynamics
* Predators in NE corner (toggleable)
* Hill cue for the current context
* Occasional food→poison mutation (toggleable)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from aif_paper1.affectworld.context_layout import build_default_context_resource_map
from aif_paper1.affectworld.helpers import Interact, Move, Resource, Season


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gaussian_stay_prob(
    season_age: int,
    year_len: int,
    p_max: float = 0.995,
    sigma_phase: float = 0.17,
) -> float:
    """
    Compute probability that the context/season stays the same given the
    number of steps already spent in the current season.

    This means:
      * Right after a season transition the context is highly stable.
      * As season tenure approaches the expected season length, advancing
        becomes increasingly likely.
    """
    season_len = year_len // 4
    if season_len <= 0:
        # Degenerate case: treat as always stay.
        return 1.0
    phase = min(max(float(season_age), 0.0) / float(season_len), 1.0)
    stay = p_max * np.exp(-(phase**2) / (2 * sigma_phase**2))
    return float(np.clip(stay, 0.0, 1.0))


def c2i(x: int, y: int, w: int) -> int:
    """Convert grid coordinates (x, y) to a flat index (row-major order)."""
    return y * w + x


def i2c(idx: int, w: int) -> Tuple[int, int]:
    """Convert a flat position index back to grid coordinates (x, y)."""
    return idx % w, idx // w


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class AffectWorldEnv:
    """
    Contextual grid-world with configurable season dynamics.

    Hidden state:
        - pos_idx ∈ {0,…,w*h-1}
        - ctx_idx ∈ {0,…,3}   (enumerating Season values in helpers.Season)

    Resource map:
        self.res_map[pos_idx, ctx_idx] -> Resource enum value

    Episode state:
        self.state = [pos_idx, ctx_idx]
        self.t     = current time-step
        self.ctx_age = steps spent in the current context
        self.under = consecutive steps spent underwater
    """

    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        year_length: int = 100,
        lake_frac: float = 0.10,
        seed: Optional[int] = None,
        *,
        season_mode: str = "frozen",
        season_stay_prob: float = 0.95,
        predator_enabled: bool = False,
        predator_prob: float = 0.10,
        mutation_enabled: bool = False,
        mutation_prob: float = 0.05,
        resource_field_enabled: bool = False,
        resource_field_resources: Sequence[str] = ("FOOD", "WATER", "SLEEP"),
        resource_field_include_predator: bool = False,
        resource_field_mode: str = "sum_exp",
        resource_field_tau: float = 2.0,
        resource_field_radius: int = 2,
        resource_field_bins: int = 6,
        resource_field_obs_true_prob: float = 0.95,
    ):
        """Construct the grid layout, stochastic hazard settings, and observation channels."""

        self.w, self.h = width, height
        self.year_len = year_length
        self.lake_frac = float(lake_frac)

        # Number of contextual seasons (should match helpers.Season length).
        self.n_ctx = 4

        # PRNG for all environment-level stochasticity.
        self.rng = np.random.default_rng(seed)

        season_mode = str(season_mode).lower()
        if season_mode not in {"frozen", "markov", "gaussian"}:
            raise ValueError(
                f"Unknown season_mode={season_mode!r}; expected frozen|markov|gaussian"
            )
        self.season_mode = season_mode
        self.season_stay_prob = float(season_stay_prob)

        self.predator_enabled = bool(predator_enabled)
        self.predator_prob = float(predator_prob)
        self.mutation_enabled = bool(mutation_enabled)
        self.mutation_prob = float(mutation_prob)

        # Optional local resource-availability field channels.
        self.resource_field_enabled = bool(resource_field_enabled)
        field_mode = str(resource_field_mode).lower().strip()
        if field_mode not in {"sum_exp", "density_radius"}:
            raise ValueError(
                f"Unknown resource_field_mode={resource_field_mode!r}; "
                "expected sum_exp|density_radius"
            )
        self.resource_field_mode = field_mode
        self.resource_field_tau = float(max(1e-6, float(resource_field_tau)))
        self.resource_field_radius = int(max(0, int(resource_field_radius)))
        self.resource_field_bins = int(resource_field_bins)
        if self.resource_field_bins < 2:
            raise ValueError("resource_field_bins must be >= 2")
        self.resource_field_obs_true_prob = float(
            np.clip(float(resource_field_obs_true_prob), 0.0, 1.0)
        )
        self.resource_field_channels = self._normalize_resource_field_channels(
            resource_field_resources,
            include_predator=bool(resource_field_include_predator),
        )

        # Fixed landmarks: hill cell and predator region (NE corner).
        self.hill_idx = c2i(width // 2, height // 2, width)
        self.pred_slice = (slice(width - 2, width), slice(0, 2))

        # Precompute deterministic moves for each action and cell.
        self._trans = self._precompute_moves()

        # Context-specific resource map: M[pos_idx, ctx_idx] -> Resource
        self.res_map = self._init_resources(lake_frac)

        # Episode state [position_index, context_index]
        self.state = np.zeros(2, np.int64)
        self.t = 0
        self.ctx_age = 0
        self.done = False
        self.under = 0

    @staticmethod
    def _normalize_resource_field_channels(
        resources: Sequence[str],
        *,
        include_predator: bool,
    ) -> Tuple[str, ...]:
        """Validate and deduplicate configured resource-field channels while preserving order."""

        allowed = {"AIR", "WATER", "FOOD", "SLEEP", "POISON"}
        out: list[str] = []
        seen: set[str] = set()
        for raw in resources:
            name = str(raw).strip().upper()
            if not name:
                continue
            if name not in allowed:
                raise ValueError(
                    f"Unknown resource field channel {raw!r}; expected one of {sorted(allowed)}"
                )
            if name in seen:
                continue
            out.append(name)
            seen.add(name)
        if include_predator and "PREDATOR" not in seen:
            out.append("PREDATOR")
        return tuple(out)

    def _predator_positions(self) -> np.ndarray:
        """Return flat grid indices that belong to the predator hazard region."""

        xs = np.arange(self.pred_slice[0].start, self.pred_slice[0].stop, dtype=np.int32)
        ys = np.arange(self.pred_slice[1].start, self.pred_slice[1].stop, dtype=np.int32)
        if xs.size <= 0 or ys.size <= 0:
            return np.empty((0,), dtype=np.int32)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        return (gy.reshape(-1) * int(self.w) + gx.reshape(-1)).astype(np.int32, copy=False)

    def _field_source_positions(self, channel: str, ctx: int) -> np.ndarray:
        """Return source tiles that contribute signal to one resource-field channel."""

        if channel == "PREDATOR":
            if not self.predator_enabled:
                return np.empty((0,), dtype=np.int32)
            return self._predator_positions()
        res = int(Resource[str(channel)])
        pos = np.flatnonzero(self.res_map[:, int(ctx)] == res)
        return np.asarray(pos, dtype=np.int32)

    def _resource_field_level(self, *, pos: int, ctx: int, channel: str) -> int:
        """Score local availability around `(pos, ctx)` and quantize it into one field level."""

        src = self._field_source_positions(channel, int(ctx))
        if src.size <= 0:
            return 0

        px, py = i2c(int(pos), int(self.w))
        sx = src % int(self.w)
        sy = src // int(self.w)
        dist = np.abs(sx - int(px)) + np.abs(sy - int(py))

        if self.resource_field_mode == "sum_exp":
            score = float(np.exp(-dist.astype(np.float32) / float(self.resource_field_tau)).sum())
            value = 1.0 - float(np.exp(-score))
        else:
            radius = int(self.resource_field_radius)
            local_count = int(np.sum(dist <= radius))
            # Manhattan-ball capacity at radius r = 1 + 2r(r+1).
            cap = int(1 + 2 * radius * (radius + 1))
            value = float(np.clip(local_count / float(max(1, cap)), 0.0, 1.0))

        level = int(np.rint(value * float(self.resource_field_bins - 1)))
        return int(np.clip(level, 0, self.resource_field_bins - 1))

    def _sample_field_obs(self, true_level: int) -> int:
        """Sample a noisy discrete field observation from the latent local-availability level."""

        level = int(np.clip(int(true_level), 0, self.resource_field_bins - 1))
        if self.resource_field_bins <= 1:
            return 0
        if self.rng.random() < self.resource_field_obs_true_prob:
            return level
        candidates = np.delete(np.arange(self.resource_field_bins, dtype=np.int32), level)
        return int(self.rng.choice(candidates))

    def get_resource_field_channels(self) -> Tuple[str, ...]:
        """Return the resource-field channel order appended to observations."""

        return tuple(self.resource_field_channels)

    # ------------- movement helpers -------------
    def _precompute_moves(self) -> np.ndarray:
        """
        Precompute a deterministic transition table for moves.

        _trans[move, idx] gives the new position index if we take `move`
        from position `idx`. Motion is clamped at grid boundaries.
        """
        tbl = np.empty((len(Move), self.w * self.h), np.int64)
        for idx in range(self.w * self.h):
            x, y = i2c(idx, self.w)
            tbl[Move.UP, idx]    = c2i(x, max(0, y - 1), self.w)
            tbl[Move.DOWN, idx]  = c2i(x, min(self.h - 1, y + 1), self.w)
            tbl[Move.LEFT, idx]  = c2i(max(0, x - 1), y, self.w)
            tbl[Move.RIGHT, idx] = c2i(min(self.w - 1, x + 1), y, self.w)
            tbl[Move.REST, idx]  = idx
        return tbl

    def _init_resources(self, lake_frac: float) -> np.ndarray:
        """
        Initialise a context-specific resource map of shape (n_cells, n_ctx).

        Layout (per context) is deliberately structured so context changes the
        opportunity map:
          - Food, Water, and Sleep rotate across sectors between contexts.
          - The hill remains AIR in every context.
          - The season cue therefore has real navigation value.
        """
        return build_default_context_resource_map(
            width=int(self.w),
            height=int(self.h),
            hill_idx=int(self.hill_idx),
            n_ctx=int(self.n_ctx),
            lake_frac=float(lake_frac),
        )

    # ------------- API -------------
    def reset(self, pos: Optional[int] = None, ctx: Optional[int] = None):
        """
        Reset the environment to a fresh episode.

        Parameters
        ----------
        pos : Optional[int]
            If given, start position index. Otherwise sampled uniformly.
        ctx : Optional[int]
            If given, start context index. Otherwise sampled uniformly.

        Returns
        -------
        obs : [pos_idx, res_obs, season_vec, ...]
            Initial observation with optional resource-field levels appended.
        """
        self.state[0] = (
            pos if pos is not None else self.rng.integers(self.w * self.h)
        )
        self.state[1] = (
            ctx if ctx is not None else self.rng.integers(self.n_ctx)
        )
        self.t = 0
        self.ctx_age = 0
        self.done = False
        self.under = 0
        return self._observe()

    def step(self, move: Move, interact: Optional[Interact] = None):
        """
        Execute one time-step given an external move + interact action.

        Logic:
          1. Update position deterministically using precomputed table.
          2. Possibly update context (season) using the configured stay process.
          3. Apply tile effects (food→poison, predator, diagnostics counters).
          4. Mark episode as done if predated.
          5. Return new observation, done flag, and info dict.

        Notes
        -----
        - `interact` is currently unused by the environment. Interactions are applied
          in the BodyProcess using the resource at the *destination* tile after the Move
          (see `simulation.py:Simulation.run_episode` for the canonical ordering).
        - "Drowning" is implemented in the body via oxygen dynamics. The environment
          exposes an `underwater` counter for diagnostics only.
        """
        if self.done:
            raise RuntimeError("Episode finished – call reset().")

        pos, ctx = self.state
        prev_ctx = int(ctx)

        # 1. deterministic position update
        pos = int(self._trans[move, pos])

        # 2. season transition
        if self.season_mode == "gaussian":
            if self.rng.random() > gaussian_stay_prob(self.ctx_age, self.year_len):
                ctx = (ctx + 1) % self.n_ctx
        elif self.season_mode == "markov":
            if self.rng.random() > self.season_stay_prob:
                ctx = (ctx + 1) % self.n_ctx
        elif self.season_mode == "frozen":
            pass
        else:
            raise RuntimeError(f"Unhandled season_mode={self.season_mode!r}")
        ctx_changed = (int(ctx) != prev_ctx)
        if ctx_changed:
            self.ctx_age = 0
        else:
            self.ctx_age += 1

        # 3. tile effects
        res_before = Resource(int(self.res_map[pos, ctx]))

        # food→poison mutation with small probability
        tile_mutated = False
        mutated_pos = None
        mutated_from = None
        mutated_to = None
        if (
            self.mutation_enabled
            and res_before == Resource.FOOD
            and self.rng.random() < self.mutation_prob
        ):
            self.res_map[pos, ctx] = Resource.POISON
            tile_mutated = True
            mutated_pos = int(pos)
            mutated_from = int(Resource.FOOD)
            mutated_to = int(Resource.POISON)

        res_after = Resource(int(self.res_map[pos, ctx]))

        # Underwater counter is exposed for diagnostics only; drowning is handled
        # by the body's oxygen dynamics (see `body.py`).
        if res_after == Resource.WATER:
            self.under += 1
        else:
            self.under = 0

        # predator region: any step into NE corner may be fatal
        x, y = i2c(pos, self.w)
        predator = False
        if self.predator_enabled:
            predator = (
                x in range(self.pred_slice[0].start, self.pred_slice[0].stop)
                and y in range(self.pred_slice[1].start, self.pred_slice[1].stop)
                and self.rng.random() < self.predator_prob
            )

        self.done = predator
        self.state[:] = (pos, ctx)
        self.t += 1

        info: Dict[str, object] = {
            # Convenience: "resource" is the post-effect resource.
            "resource": int(res_after),
            "resource_before": int(res_before),
            "resource_after": int(res_after),
            "ctx": int(ctx),
            "ctx_changed": bool(ctx_changed),
            "tile_mutated": bool(tile_mutated),
            "mutated_pos": mutated_pos,
            "mutated_from": mutated_from,
            "mutated_to": mutated_to,
            "underwater": self.under,
            "predator": predator,
        }
        return self._observe(), self.done, info

    # ------------- observation -------------
    def _observe(self) -> List[np.ndarray]:
        """
        Construct the current observation list:
            [pos_idx, res_idx, season_vec, ...optional_resource_field_levels]

        Notes
        -----
        - res_obs is a potentially noisy copy of the true resource at this
          tile; 95% of the time it equals `res_truth`, otherwise a random
          different resource is reported.
        - season_vec is a (5,) one-hot vector:
            [NoContext, Spring, Summer, Autumn, Winter].
          Only on the hill cell do we emit the true context; elsewhere
          we emit the dummy NoContext code.
        """
        pos, ctx = self.state

        # Resource observation with small noise
        res_truth = self.res_map[pos, ctx]
        if self.rng.random() < 0.95:
            res_obs = int(res_truth)
        else:
            # Sample a different resource label
            candidates = np.delete(np.arange(len(Resource)), res_truth)
            res_obs = int(self.rng.choice(candidates))

        # Season cue: encoded only on the hill
        season_vec = np.zeros(len(Season) + 1, np.int8)   # +1 = NoContext
        if pos == self.hill_idx:
            season_vec[ctx + 1] = 1
        else:
            season_vec[0] = 1

        obs: List[np.ndarray] = [np.int64(pos), np.int64(res_obs), season_vec]
        if self.resource_field_enabled and len(self.resource_field_channels) > 0:
            for channel in self.resource_field_channels:
                lvl_true = self._resource_field_level(
                    pos=int(pos),
                    ctx=int(ctx),
                    channel=channel,
                )
                lvl_obs = self._sample_field_obs(lvl_true)
                obs.append(np.int64(lvl_obs))

        return obs

    def get_grid(self, ctx: Optional[int] = None) -> np.ndarray:
        """Return (h,w) grid of Resource ids for a given context (default=current)."""
        c = int(self.state[1]) if ctx is None else int(ctx)
        return self.res_map[:, c].reshape(self.h, self.w)

    def get_landmarks(self) -> dict:
        """Convenience for renderers/loggers."""
        return {
            "hill_idx": int(self.hill_idx),
            "pred_slice": self.pred_slice,
            "predator_enabled": bool(self.predator_enabled),
            "resource_field_enabled": bool(self.resource_field_enabled),
            "resource_field_channels": tuple(self.resource_field_channels),
            "resource_field_bins": int(self.resource_field_bins),
        }

    # ------------- plotting helpers -------------
    COLOUR = {
        Resource.AIR:    "gainsboro",
        Resource.WATER:  "tab:blue",
        Resource.FOOD:   "tab:green",
        Resource.SLEEP:  "tab:orange",
        Resource.POISON: "tab:red",
    }

    def plot(self, ax: Optional[plt.Axes] = None):
        """Plot a snapshot of the current context's resource map."""
        ax = ax or plt.gca()
        ax.set_aspect("equal")
        ax.axis("off")
        ctx = self.state[1]
        for idx in range(self.w * self.h):
            x, y = i2c(idx, self.w)
            col = self.COLOUR[self.res_map[idx, ctx]]
            ax.scatter(x, y, s=380, color=col, edgecolors="black")
        # landmark overlays
        hx, hy = i2c(self.hill_idx, self.w)
        ax.scatter(hx, hy, s=450, marker="^", color="yellow",
                   edgecolors="black", linewidths=1.2)
        if self.predator_enabled:
            for x in range(self.pred_slice[0].start, self.pred_slice[0].stop):
                for y in range(self.pred_slice[1].start, self.pred_slice[1].stop):
                    ax.scatter(x, y, s=380, marker="s", facecolor="none",
                               edgecolors="red", linewidths=2)
        ax.scatter(*i2c(self.state[0], self.w), s=450, color="black")

    def plot_transition_arrows(self, move: Move,
                               ax: Optional[plt.Axes] = None):
        """Visualise the deterministic movement table for a given move."""
        ax = ax or plt.gca()
        ax.set_aspect("equal")
        ax.axis("off")
        xs, ys, dxs, dys = [], [], [], []
        for idx in range(self.w * self.h):
            x0, y0 = i2c(idx, self.w)
            x1, y1 = i2c(self._trans[move, idx], self.w)
            xs.append(x0); ys.append(y0)
            dxs.append(x1 - x0); dys.append(y1 - y0)
        ax.quiver(xs, ys, dxs, dys, angles="xy",
                  scale_units="xy", scale=1, width=0.004)
        ax.set_title(f"Move: {move.name}")

    @staticmethod
    def plot_season_cycle(year_len: int = 100, steps: int = 400,
                          ax: Optional[plt.Axes] = None):
        """
        Plot the Gaussian stay-probability as a function of season tenure.
        """
        ax = ax or plt.gca()
        t = np.arange(steps)
        stay = [gaussian_stay_prob(int(tt), year_len) for tt in t]
        ax.plot(t, stay)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("season age")
        ax.set_ylabel("p(stay)")
        ax.set_title("Gaussian stay-probability over season tenure")


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = AffectWorldEnv(seed=0)
    print("obs0:", env.reset())

    for _ in range(8):
        obs, done, info = env.step(Move.REST, Interact.BREATHE)
        print(obs, info)
        if done:
            print("terminated:", info)
            break

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 3.5))
    AffectWorldEnv.plot_season_cycle(ax=ax0)
    env.plot(ax1)
    env.plot_transition_arrows(Move.LEFT, ax2)
    plt.tight_layout()
    plt.show()

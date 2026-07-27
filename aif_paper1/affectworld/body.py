"""
body.py – AffectWorld bodily-state process
=========================================

The maintained Paper 1 interpretation is:

- latent nutrient level (internal compatibility field: `energy`)
- latent hydration level (`hydration`)
- latent oxygenation level (`oxygen`)
- a dormant fourth compatibility channel (`sleep_debt`)

Interoceptive observations remain categorical need/proxy channels derived from
those latent bodily counters. The fourth channel stays in the engine for
compatibility, but maintained Paper 1 presets neutralize it.

API
---
`tick()` returns:
    levels_vec   : np.int8[4]   raw Type I integers
    norm_vec     : np.float32[4] normalised 0–1
    impair       : dict of impairment multipliers
    typeII_vec   : np.int8[4]   Type II categorical levels
    done         : bool         True if any Type I has hit a lethal bound

Use `sample_interoception(attended)` to obtain noisy interoceptive observations with
precision control.
"""

from typing import Mapping, Tuple, Optional

from aif_paper1.affectworld.configs import BodyConfig

import numpy as np

from aif_paper1.affectworld.helpers import Interact, Need, Resource
from aif_paper1.affectworld.intero_precision import channel_precisions_np, normalize_budget_mode


class BodyProcess:
    """Stateful numpy implementation of AffectWorld body dynamics and interoceptive sampling."""

    # ---------------- configuration ----------------
    # Internal bodily maxima. Paper 1 surfaces map energy->nutrient and oxygen->oxygenation.
    MAX_ENERGY     = 15
    MAX_HYDRATION  = 15
    MAX_SLEEPDEBT  = 20
    MAX_OXYGEN     = 10

    # Stepwise basal changes
    DEC_ENERGY     = 1
    DEC_HYDRATION  = 1
    INC_SLEEPDEBT  = 1
    DEC_OXY_WATER  = 1          # oxygen fall per step if submerged

    # Refills from resources
    EAT_ENERGY_GAIN    = 3
    SLEEP_DEBT_REDUCE  = 4

    POISON_PENALTY = 4         # energy drop if on poison

    # Interoceptive categorical needs / proxy observations (order matters and should match the agent)
    TYPEII_NAMES = ("HUNGER", "THIRST", "FATIGUE", "SUFFOCATION")
    TYPEII_NEEDS = (Need.HUNGER, Need.THIRST, Need.FATIGUE, Need.SUFFOCATION)
    if TYPEII_NAMES != tuple(n.name for n in TYPEII_NEEDS):
        raise ValueError(
            "BodyProcess TYPEII_NAMES ordering must match helpers.Need ordering "
            f"(expected {tuple(n.name for n in TYPEII_NEEDS)}, got {TYPEII_NAMES})"
        )
    TYPEII_MAX_LEVEL = 5       # 0 = no need, 5 = max deficit

    # Interoceptive precision (for sampling noisy Type II observations)
    # baseline vs attended channel
    INTERO_BASE_PRECISION = 0.65
    INTERO_ATT_PRECISION = 0.90
    INTERO_BUDGET_MODE = "none"
    INTERO_BUDGET_BG_FLOOR = 0.05
    INTERO_BUDGET_TOTAL = -1.0
    AUTO_RESOURCE_INTERACT = False

    # ------------------------------------------------
    def __init__(self, seed: Optional[int] = None, config: Optional[BodyConfig] = None):
        """Initialize bodily dynamics from defaults or a `BodyConfig`, then reset state."""

        if seed is None and config is not None:
            seed = int(config.seed)
        # Default to non-lethal sleep/oxygen (aligned with BodyConfig defaults).
        self._lethal_sleep = False
        self._lethal_oxygen = False
        self._process_noise = 0.0
        self._intero_budget_mode = str(self.INTERO_BUDGET_MODE)
        self._intero_budget_bg_floor = float(self.INTERO_BUDGET_BG_FLOOR)
        self._intero_budget_total = float(self.INTERO_BUDGET_TOTAL)
        self._auto_resource_interact = bool(self.AUTO_RESOURCE_INTERACT)
        if config is not None:
            self.MAX_ENERGY = int(config.max_energy)
            self.MAX_HYDRATION = int(config.max_hydration)
            self.MAX_SLEEPDEBT = int(config.max_sleepdebt)
            self.MAX_OXYGEN = int(config.max_oxygen)

            self.DEC_ENERGY = int(config.dec_energy)
            self.DEC_HYDRATION = int(config.dec_hydration)
            self.INC_SLEEPDEBT = int(config.inc_sleepdebt)
            self.DEC_OXY_WATER = int(config.dec_oxy_water)

            self.INTERO_BASE_PRECISION = float(config.intero_base_precision)
            self.INTERO_ATT_PRECISION = float(config.intero_att_precision)
            self._intero_budget_mode = normalize_budget_mode(str(config.intero_budget_mode))
            self._intero_budget_bg_floor = float(config.intero_budget_bg_floor)
            self._intero_budget_total = float(config.intero_budget_total)
            self._auto_resource_interact = bool(config.auto_resource_interact)

            self._lethal_sleep = bool(config.lethal_sleep)
            self._lethal_oxygen = bool(config.lethal_oxygen)
            self._process_noise = float(config.process_noise)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset latent bodily counters and their derived interoceptive levels."""
        self.energy      = self.MAX_ENERGY
        self.hydration   = self.MAX_HYDRATION
        self.sleep_debt  = 0
        self.oxygen      = self.MAX_OXYGEN
        self._typeII_levels = np.zeros(len(self.TYPEII_NAMES), dtype=np.int8)

    def apply_initial_state(
        self,
        *,
        energy: int | None = None,
        hydration: int | None = None,
        sleep_debt: int | None = None,
        oxygen: int | None = None,
    ) -> None:
        """Apply an explicit post-reset body state before the first intero sample."""
        if energy is not None:
            self.energy = self._clamp(int(energy), self.MAX_ENERGY)
        if hydration is not None:
            self.hydration = self._clamp(int(hydration), self.MAX_HYDRATION)
        if sleep_debt is not None:
            self.sleep_debt = self._clamp(int(sleep_debt), self.MAX_SLEEPDEBT)
        if oxygen is not None:
            self.oxygen = self._clamp(int(oxygen), self.MAX_OXYGEN)
        self._update_typeII_levels()

    def apply_initial_state_dict(self, state: Mapping[str, int] | None) -> None:
        """Convenience wrapper around :meth:`apply_initial_state` for JSON-ish configs."""
        if state is None:
            return
        self.apply_initial_state(
            energy=None if state.get("energy") is None else int(state["energy"]),
            hydration=None if state.get("hydration") is None else int(state["hydration"]),
            sleep_debt=None if state.get("sleep_debt") is None else int(state["sleep_debt"]),
            oxygen=None if state.get("oxygen") is None else int(state["oxygen"]),
        )

    # ------------------------------------------------
    def _clamp(self, val: int, upper: int) -> int:
        """Clamp an integer to [0, upper]."""
        return max(0, min(val, upper))

    def _discretise(self, fraction: float) -> int:
        """
        Map a fraction in [0,1] to a discrete interoceptive level {0…TYPEII_MAX_LEVEL}.

        This is currently linear; we may later replace this with a non-linear
        mapping (e.g. logistic) for Weber–Fechner-like behaviour.
        """
        fraction = np.clip(fraction, 0.0, 1.0)
        return int(round(fraction * self.TYPEII_MAX_LEVEL))

    def _impairment_factor(self,
                           value: int,
                           good_lo: int, good_hi: int,
                           dead_lo: int, dead_hi: int) -> float:
        """
        Returns a factor in [0,1] that equals 1 when `value` lies inside the
        preferred band [good_lo, good_hi] and falls linearly to 0 when it hits a
        lethal bound (≤ dead_lo or ≥ dead_hi).

        This is used to scale movement or cognition in a coarse way.
        """
        if good_lo <= value <= good_hi:
            return 1.0
        if value < good_lo:
            # heading towards starvation/dehydration (too low)
            return max(0.0, (value - dead_lo) / (good_lo - dead_lo + 1e-9))
        # value > good_hi: heading towards max-sleep-debt, hyper-oxygenation, etc.
        return max(0.0, (dead_hi - value) / (dead_hi - good_hi + 1e-9))

    # ------------------------------------------------
    def _update_typeII_levels(self) -> None:
        """
        Derive discrete interoceptive need levels from the current bodily values.

        Current scheme:
          - Nutrient need ~ (max_energy - energy) / max_energy
          - Hydration need ~ (max_hydration - hydration) / max_hydration
          - Reserved compatibility need ~ sleep_debt / max_sleepdebt
          - Oxygenation / CO2 proxy need ~ (max_oxygen - oxygen) / max_oxygen
        All are then discretised via _discretise.
        """
        hunger = self._discretise(
            (self.MAX_ENERGY - self.energy) / self.MAX_ENERGY
        )
        thirst = self._discretise(
            (self.MAX_HYDRATION - self.hydration) / self.MAX_HYDRATION
        )
        fatigue = self._discretise(
            self.sleep_debt / self.MAX_SLEEPDEBT
        )
        suffocation = self._discretise(
            (self.MAX_OXYGEN - self.oxygen) / self.MAX_OXYGEN
        )
        self._typeII_levels[:] = (hunger, thirst, fatigue, suffocation)

    # ------------------------------------------------
    def tick(self,
             interact: Interact | None,
             tile_res: Resource
             ) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, bool]:
        """
        Advance the body state by one time-step.

        Parameters
        ----------
        interact : Interact | None
            The chosen interaction action (EAT, DRINK, SLEEP, BREATHE, REST).
        tile_res : Resource
            The currently occupied resource tile.

        Returns
        -------
        levels : np.ndarray[int8, shape (4,)]
            Raw bodily values [energy, hydration, sleep_debt, oxygen].
        norm : np.ndarray[float32, shape (4,)]
            Normalised bodily values in [0,1].
        impair : dict
            Impairment multipliers for movement ("move_factor")
            and cognition/planning ("cog_factor").
        typeII_vec : np.ndarray[int8, shape (4,)]
            Derived interoceptive categorical need levels.
        dead : bool
            True if any Type I variable has hit a lethal bound.
        """
        # -------- basal change --------
        self.energy     -= self.DEC_ENERGY
        self.hydration  -= self.DEC_HYDRATION
        self.sleep_debt += self.INC_SLEEPDEBT

        if tile_res == Resource.WATER:
            self.oxygen -= self.DEC_OXY_WATER
        else:
            # Gradual oxygen recovery when not underwater.
            self.oxygen = self._clamp(self.oxygen + 1, self.MAX_OXYGEN)

        # -------- interactions --------
        if self._auto_resource_interact and interact in (None, Interact.NONE):
            # why: MATLAB environments reset resource deficits on resource occupancy; this
            # approximates that passive acquisition while retaining explicit actions elsewhere.
            if tile_res == Resource.FOOD:
                interact = Interact.EAT
            elif tile_res == Resource.WATER:
                interact = Interact.DRINK
            elif tile_res == Resource.SLEEP:
                interact = Interact.SLEEP

        # Note: interactions are only effective when tile_res matches.
        if interact == Interact.BREATHE and tile_res == Resource.WATER:
            self.oxygen = self.MAX_OXYGEN

        elif interact == Interact.EAT and tile_res == Resource.FOOD:
            self.energy += self.EAT_ENERGY_GAIN

        elif interact == Interact.DRINK and tile_res == Resource.WATER:
            self.hydration = self.MAX_HYDRATION

        elif interact == Interact.SLEEP and tile_res == Resource.SLEEP:
            self.sleep_debt -= self.SLEEP_DEBT_REDUCE

        # poison hurts energy regardless of interact choice
        if tile_res == Resource.POISON:
            self.energy -= self.POISON_PENALTY

        # -------- clamp to valid range --------
        self.energy     = self._clamp(self.energy,    self.MAX_ENERGY)
        self.hydration  = self._clamp(self.hydration, self.MAX_HYDRATION)
        self.sleep_debt = self._clamp(self.sleep_debt, self.MAX_SLEEPDEBT)
        self.oxygen     = self._clamp(self.oxygen,    self.MAX_OXYGEN)

        # -------- optional local process noise --------
        if self._process_noise > 0.0:
            for attr, upper in (
                ("energy", self.MAX_ENERGY),
                ("hydration", self.MAX_HYDRATION),
                ("sleep_debt", self.MAX_SLEEPDEBT),
                ("oxygen", self.MAX_OXYGEN),
            ):
                if self.rng.random() < self._process_noise:
                    delta = int(self.rng.choice(np.array([-1, 1], dtype=np.int8)))
                    setattr(self, attr, self._clamp(int(getattr(self, attr)) + delta, int(upper)))

        # -------- death test --------
        dead = self.energy == 0 or self.hydration == 0
        if self._lethal_oxygen:
            dead = dead or self.oxygen == 0
        if self._lethal_sleep:
            dead = dead or self.sleep_debt == self.MAX_SLEEPDEBT

        # -------- impairment factors --------
        # Movement impairment is still driven by the dormant compatibility channel.
        move_factor = self._impairment_factor(
            self.sleep_debt,
            good_lo=0,                       good_hi=4,
            dead_lo=0,                       dead_hi=self.MAX_SLEEPDEBT,
        )

        # Cognitive impairment mainly driven by energy.
        cog_factor = self._impairment_factor(
            self.energy,
            good_lo=4,                       good_hi=self.MAX_ENERGY,
            dead_lo=0,                       dead_hi=self.MAX_ENERGY,
        )

        impair = {
            "move_factor": move_factor,   # multiply success prob / add cost
            "cog_factor":  cog_factor,    # scale planning precision
        }

        # -------- pack outputs --------
        levels = np.array(
            [self.energy, self.hydration, self.sleep_debt, self.oxygen],
            dtype=np.int8,
        )

        norm = levels.astype(np.float32) / np.array(
            [
                self.MAX_ENERGY,
                self.MAX_HYDRATION,
                self.MAX_SLEEPDEBT,
                self.MAX_OXYGEN,
            ],
            dtype=np.float32,
        )

        # Update and copy Type II levels
        self._update_typeII_levels()
        typeII_vec = self._typeII_levels.copy()

        return levels, norm, impair, typeII_vec, dead

    # ------------------------------------------------
    def get_typeII_levels(self) -> np.ndarray:
        """Return a copy of the current Type II categorical levels."""
        return self._typeII_levels.copy()

    def get_typeII_preferences(self) -> np.ndarray:
        """
        Preferred Type II levels (default = all zeros, i.e., no deficit).
        This may later be made asymmetric if some needs are more urgent.
        """
        return np.zeros(len(self.TYPEII_NAMES), dtype=np.int8)

    def sample_interoception(self,
                             attended: Optional[int],
                             high_precision: Optional[float] = None,
                             base_precision: Optional[float] = None,
                             noise_scale: float = 1.0,
                             budget_mode: Optional[str] = None,
                             budget_bg_floor: Optional[float] = None,
                             budget_total: Optional[float] = None) -> np.ndarray:
        """
        Sample a noisy observation of Type II levels.

        Parameters
        ----------
        attended : Optional[int]
            Index of the attended need (0..3) or None if no particular
            need is foregrounded. The attended channel will have higher
            precision (less noise).
        high_precision : float
            Baseline probability of reporting the true level for the
            attended channel.
        base_precision : float
            Baseline probability of reporting the true level for
            unattended channels.
        noise_scale : float
            Global multiplier to both precisions; can be used to model
            additional jitter from fatigue, stress, etc.

        Returns
        -------
        obs : np.ndarray[int8, shape (4,)]
            Noisy Type II observations, each in {0,…,TYPEII_MAX_LEVEL}.
        """
        if high_precision is None:
            high_precision = float(self.INTERO_ATT_PRECISION)
        if base_precision is None:
            base_precision = float(self.INTERO_BASE_PRECISION)

        mode = self._intero_budget_mode if budget_mode is None else normalize_budget_mode(str(budget_mode))
        bg_floor = (
            float(self._intero_budget_bg_floor)
            if budget_bg_floor is None
            else float(budget_bg_floor)
        )
        total_budget = (
            None
            if budget_total is None and float(self._intero_budget_total) <= 0.0
            else float(self._intero_budget_total if budget_total is None else budget_total)
        )
        attended_idx = -1 if attended is None else int(attended)
        per_chan = channel_precisions_np(
            attended=attended_idx,
            n_channels=int(len(self.TYPEII_NAMES)),
            base_precision=float(base_precision),
            att_precision=float(high_precision),
            mode=mode,
            bg_floor=float(bg_floor),
            total_budget=total_budget,
        )

        obs = np.empty(len(self.TYPEII_NAMES), dtype=np.int8)
        for idx, true_level in enumerate(self._typeII_levels):
            # Pick precision from the selected allocation mechanism.
            precision = float(per_chan[idx])
            precision = float(np.clip(precision * noise_scale, 0.0, 1.0))

            if self.rng.random() < precision:
                # Report the correct level.
                obs[idx] = true_level
            else:
                # Report an incorrect level (never equals the true level).
                candidates = np.delete(
                    np.arange(self.TYPEII_MAX_LEVEL + 1, dtype=int),
                    int(true_level),
                )
                obs[idx] = int(self.rng.choice(candidates))
        return obs


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple manual sanity checks.
    body = BodyProcess(seed=0)
    print("Initial tick on AIR:")
    levels, norm, impair, typeII_vec, dead = body.tick(None, Resource.AIR)
    print("  levels:", levels, "norm:", norm, "typeII:", typeII_vec, "dead:", dead)

    print("\nSwim test in WATER:")
    body.reset()
    for t in range(6):
        levels, _, _, typeII_vec, dead = body.tick(None, Resource.WATER)
        print(f"t={t} levels={levels} typeII={typeII_vec} dead={dead}")
        if dead:
            break

    print("\nInteroception sample (no attention):")
    print("  true:", body.get_typeII_levels())
    print("  obs :", body.sample_interoception(attended=None))

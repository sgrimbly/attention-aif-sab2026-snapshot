"""
configs.py
==========

Central configuration dataclasses for AffectWorld experiments.

These are intentionally lightweight so they can be imported by env/body/agent
without creating circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Serializable settings for constructing numpy and JAX AffectWorld environments."""

    width: int = 6
    height: int = 6
    year_length: int = 100
    lake_frac: float = 0.10
    season_mode: str = "frozen"  # {"frozen","markov","gaussian"}
    season_stay_prob: float = 0.95
    predator_enabled: bool = False
    predator_prob: float = 0.10
    mutation_enabled: bool = False
    mutation_prob: float = 0.05
    # Optional local resource-availability field observations.
    # When enabled, env observations append one discrete level per selected channel.
    resource_field_enabled: bool = False
    resource_field_resources: tuple[str, ...] = ("FOOD", "WATER", "SLEEP")
    resource_field_include_predator: bool = False
    resource_field_mode: str = "sum_exp"  # {"sum_exp","density_radius"}
    resource_field_tau: float = 2.0
    resource_field_radius: int = 2
    resource_field_bins: int = 6
    resource_field_obs_true_prob: float = 0.95
    seed: int = 0


@dataclass
class BodyConfig:
    """Serializable settings for body dynamics and interoceptive observation noise."""

    # Defaults are tuned for "solvable + fast enough to benchmark" regimes.
    max_energy: int = 15
    max_hydration: int = 15
    max_sleepdebt: int = 20
    max_oxygen: int = 10
    dec_energy: int = 1
    dec_hydration: int = 1
    inc_sleepdebt: int = 1
    dec_oxy_water: int = 1
    lethal_sleep: bool = False
    lethal_oxygen: bool = False
    process_noise: float = 0.0  # local slip probability (optional)
    intero_base_precision: float = 0.65
    intero_att_precision: float = 0.90
    intero_budget_mode: str = "none"  # {"none","soft","hard"}
    intero_budget_bg_floor: float = 0.05
    # <=0 means: derive from legacy total = p_att + (n_need-1)*p_base.
    intero_budget_total: float = -1.0
    # If True, being on a matching resource tile auto-applies the corresponding
    # interaction effect (MATLAB-style passive resource acquisition approximation).
    auto_resource_interact: bool = False
    seed: int = 0


@dataclass
class AgentConfig:
    """Shared lightweight defaults for small agent constructors and demos."""

    planning_horizon: int = 2
    preference_scale: float = 12.0
    season_stay_prob: float = 0.95
    physio_slip_model: float = 0.0
    seed: int = 0


__all__ = [
    "AgentConfig",
    "BodyConfig",
    "EnvConfig",
]

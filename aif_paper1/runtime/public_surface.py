"""Public naming, aliasing, and Paper 1 scope metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PRIMARY_PAPER_AGENTS: tuple[str, ...] = ("baseline", "attn_extk")

PAPER1_RESOURCE_LABELS: dict[str, str] = {
    "AIR": "Oxygen",
    "WATER": "Water",
    "FOOD": "Food",
    "SLEEP": "Dormant Compatibility",
    "POISON": "Poison",
}

PAPER1_NEED_LABELS: dict[str, str] = {
    "HUNGER": "Nutrient Need",
    "THIRST": "Hydration Need",
    "FATIGUE": "Dormant Compatibility Need",
    "SUFFOCATION": "Oxygenation / CO2 Proxy Need",
    "NONE": "No Prioritized Need",
}

PAPER1_BELIEF_LABELS: dict[str, str] = {
    "position_state": "position_state",
    "season_state": "context_state",
    "energy_state": "nutrient_state",
    "hydration_state": "hydration_state",
    "sleep_debt_state": "dormant_rest_state",
    "oxygen_state": "oxygenation_state",
    "kappa_state": "attention_target_state",
}

PAPER1_INTERO_OBS_LABELS: dict[str, str] = {
    "hunger_obs": "nutrient_obs",
    "thirst_obs": "hydration_obs",
    "fatigue_obs": "dormant_rest_obs",
    "suffocation_obs": "oxygenation_proxy_obs",
}

PAPER1_DEATH_CAUSE_LABELS: dict[str, str] = {
    "energy": "nutrient",
    "hydration": "hydration",
    "sleep": "dormant_rest",
    "oxygen": "oxygenation",
    "unknown": "unknown",
}

PAPER1_BENCHMARK_ARG_ALIASES: dict[str, str] = {
    "max_nutrient": "max_energy",
    "max_oxygenation": "max_oxygen",
    "dec_nutrient": "dec_energy",
    "dec_oxygenation_water": "dec_oxy_water",
    "lethal_oxygenation": "lethal_oxygen",
    "intero_range_nutrient_min": "intero_range_hunger_min",
    "intero_range_nutrient_max": "intero_range_hunger_max",
    "intero_range_hydration_min": "intero_range_thirst_min",
    "intero_range_hydration_max": "intero_range_thirst_max",
    "intero_range_rest_compat_min": "intero_range_fatigue_min",
    "intero_range_rest_compat_max": "intero_range_fatigue_max",
    "intero_range_oxygenation_min": "intero_range_suffocation_min",
    "intero_range_oxygenation_max": "intero_range_suffocation_max",
}

PAPER1_PROFILE_DEFAULTS: dict[str, Any] = {
    "n_sleep": 0,
    "max_sleepdebt": 1,
    "inc_sleepdebt": 0,
    "lethal_sleep": False,
    "season_mode": "markov",
    "season_stay_prob": 0.95,
}

PAPER1_CANONICAL_LANE: dict[str, Any] = {
    "planner": "efe",
    "policy_mode": "full",
    "planning_horizon": 3,
    "intero_budget_mode": "soft",
    "intero_budget_bg_floor": 0.05,
    "intero_budget_total": 2.6,
    "season_mode": "markov",
    "season_stay_prob": 0.95,
}

_COMPAT_SLEEP_CHANNEL_DEFAULTS: dict[str, Any] = {
    "n_sleep": 1,
    "max_sleepdebt": 20,
    "inc_sleepdebt": 1,
    "lethal_sleep": False,
}

AGENT_PUBLIC_INFO: dict[str, dict[str, Any]] = {
    "baseline": {
        "display_name": "Standard Active Inference",
        "family_name": "Standard Active Inference",
        "paper_track": "primary",
        "reporting_track": "primary_active",
        "fairness_class": "n/a",
        "observation_access_class": "fair_public_obs",
        "architecture_class": "active_inference",
        "objective_class": "efe",
        "readiness_status": "tracked",
        "legacy": False,
    },
    "attn_extk": {
        "display_name": "Likelihood-Precision Selective Attention",
        "family_name": "Likelihood-Precision Selective Attention",
        "paper_track": "primary",
        "reporting_track": "primary_active",
        "fairness_class": "n/a",
        "observation_access_class": "fair_public_obs",
        "architecture_class": "active_inference",
        "objective_class": "efe",
        "readiness_status": "tracked",
        "legacy": False,
    },
}


def agent_public_info(agent: str) -> dict[str, Any]:
    """Return public metadata for an agent/baseline identifier."""
    key = str(agent).strip().lower()
    info = AGENT_PUBLIC_INFO.get(key)
    if info is not None:
        return {"agent": key, **info}
    return {
        "agent": key,
        "display_name": key.upper(),
        "family_name": "Unknown",
        "paper_track": "secondary",
        "legacy": False,
    }


def is_legacy_agent(agent: str) -> bool:
    """Return whether the named agent belongs to the legacy surface."""
    return bool(agent_public_info(agent)["legacy"])


def paper_track_for_agent(agent: str) -> str:
    """Return whether the agent is on the Paper 1 primary or secondary track."""
    return str(agent_public_info(agent)["paper_track"])


def reporting_track_for_agent(agent: str) -> str:
    """Return the reporting panel/category used for the agent in public outputs."""
    return str(agent_public_info(agent).get("reporting_track", "secondary_active"))


def fairness_class_for_agent(agent: str) -> str:
    """Return the fairness/oracle classification exposed in public reporting."""
    return str(agent_public_info(agent).get("fairness_class", "n/a"))


def observation_access_class_for_agent(agent: str) -> str:
    """Return whether the agent uses public observations or oracle state access."""
    return str(agent_public_info(agent).get("observation_access_class", "n/a"))


def architecture_class_for_agent(agent: str) -> str:
    """Return the architecture comparison class exposed in public reporting."""
    return str(agent_public_info(agent).get("architecture_class", "n/a"))


def objective_class_for_agent(agent: str) -> str:
    """Return the public objective-family label associated with the agent."""
    return str(agent_public_info(agent).get("objective_class", "n/a"))


def readiness_status_for_agent(agent: str) -> str:
    """Return the public readiness label associated with the agent."""
    return str(agent_public_info(agent).get("readiness_status", "tracked"))


def paper1_resource_name(resource: Any) -> str:
    """Map an internal resource enum/name onto its Paper 1 public label."""
    key = str(getattr(resource, "name", resource)).strip().upper()
    return PAPER1_RESOURCE_LABELS.get(key, str(resource))


def paper1_need_name(need: Any) -> str:
    """Map an internal need enum/name onto its Paper 1 public label."""
    key = str(getattr(need, "name", need)).strip().upper()
    return PAPER1_NEED_LABELS.get(key, str(need))


def paper1_belief_label(key: str) -> str:
    """Map an internal belief-factor key onto its Paper 1 report label."""
    return PAPER1_BELIEF_LABELS.get(str(key), str(key))


def paper1_intero_obs_label(key: str) -> str:
    """Map an internal interoceptive observation key onto its Paper 1 label."""
    return PAPER1_INTERO_OBS_LABELS.get(str(key), str(key))


def paper1_death_cause(label: str | None) -> str | None:
    """Map an internal death-cause label onto its Paper 1 public wording."""
    if label is None:
        return None
    return PAPER1_DEATH_CAUSE_LABELS.get(str(label), str(label))


def normalize_paper1_benchmark_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize public Paper 1 config aliases onto internal benchmark keys."""
    if args is None:
        return {}
    normalized = dict(args)
    for alias, canonical in PAPER1_BENCHMARK_ARG_ALIASES.items():
        if alias not in normalized:
            continue
        if canonical not in normalized or normalized.get(canonical) in (None, ""):
            normalized[canonical] = normalized[alias]
        normalized.pop(alias, None)
    planner = normalized.get("planner")
    if planner not in (None, ""):
        from aif_paper1.planning.policy_space import _normalize_planner_name

        normalized["planner"] = _normalize_planner_name(str(planner))
    sleep_defaults = (
        PAPER1_PROFILE_DEFAULTS
        if bool(normalized.get("paper1_profile", False))
        else _COMPAT_SLEEP_CHANNEL_DEFAULTS
    )
    for key, value in sleep_defaults.items():
        if normalized.get(key) in (None, ""):
            normalized[key] = value
    return normalized


def paper1_profile_validation_errors(args: Mapping[str, Any] | None) -> list[str]:
    """Return canonical-lane validation errors when ``paper1_profile`` is enabled."""
    normalized = normalize_paper1_benchmark_args(args)
    if not bool(normalized.get("paper1_profile", False)):
        return []

    errors: list[str] = []
    if str(normalized.get("planner", "")).strip().lower() != str(PAPER1_CANONICAL_LANE["planner"]):
        errors.append("paper1_profile requires planner=efe.")
    if str(normalized.get("policy_mode", "")).strip().lower() != str(PAPER1_CANONICAL_LANE["policy_mode"]):
        errors.append("paper1_profile requires policy_mode=full.")
    try:
        planning_horizon = int(normalized.get("planning_horizon"))
    except Exception:
        planning_horizon = None
    if planning_horizon != int(PAPER1_CANONICAL_LANE["planning_horizon"]):
        errors.append("paper1_profile requires planning_horizon=3.")
    if str(normalized.get("intero_budget_mode", "")).strip().lower() != str(PAPER1_CANONICAL_LANE["intero_budget_mode"]):
        errors.append("paper1_profile requires intero_budget_mode=soft.")
    try:
        intero_budget_bg_floor = float(normalized.get("intero_budget_bg_floor"))
    except Exception:
        intero_budget_bg_floor = None
    if intero_budget_bg_floor is None or abs(intero_budget_bg_floor - float(PAPER1_CANONICAL_LANE["intero_budget_bg_floor"])) > 1e-9:
        errors.append("paper1_profile requires intero_budget_bg_floor=0.05.")
    try:
        intero_budget_total = float(normalized.get("intero_budget_total"))
    except Exception:
        intero_budget_total = None
    if intero_budget_total is None or abs(intero_budget_total - float(PAPER1_CANONICAL_LANE["intero_budget_total"])) > 1e-9:
        errors.append("paper1_profile requires intero_budget_total=2.6.")
    if str(normalized.get("season_mode", "")).strip().lower() != str(PAPER1_CANONICAL_LANE["season_mode"]):
        errors.append("paper1_profile requires season_mode=markov.")
    try:
        season_stay_prob = float(normalized.get("season_stay_prob"))
    except Exception:
        season_stay_prob = None
    if season_stay_prob is None or abs(season_stay_prob - float(PAPER1_CANONICAL_LANE["season_stay_prob"])) > 1e-9:
        errors.append("paper1_profile requires season_stay_prob=0.95.")
    return errors


__all__ = [
    "AGENT_PUBLIC_INFO",
    "PAPER1_BENCHMARK_ARG_ALIASES",
    "PAPER1_BELIEF_LABELS",
    "PAPER1_CANONICAL_LANE",
    "PAPER1_DEATH_CAUSE_LABELS",
    "PAPER1_INTERO_OBS_LABELS",
    "PAPER1_NEED_LABELS",
    "PAPER1_PROFILE_DEFAULTS",
    "PAPER1_RESOURCE_LABELS",
    "PRIMARY_PAPER_AGENTS",
    "agent_public_info",
    "architecture_class_for_agent",
    "fairness_class_for_agent",
    "is_legacy_agent",
    "normalize_paper1_benchmark_args",
    "objective_class_for_agent",
    "observation_access_class_for_agent",
    "paper1_profile_validation_errors",
    "paper1_belief_label",
    "paper1_death_cause",
    "paper1_intero_obs_label",
    "paper1_need_name",
    "paper1_resource_name",
    "paper_track_for_agent",
    "readiness_status_for_agent",
    "reporting_track_for_agent",
]

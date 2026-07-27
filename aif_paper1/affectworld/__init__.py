"""Environment/world core package for AffectWorld."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AffectWorldEnv": ("aif_paper1.affectworld.affect_world_env", "AffectWorldEnv"),
    "BodyConfig": ("aif_paper1.affectworld.configs", "BodyConfig"),
    "BodyProcess": ("aif_paper1.affectworld.body", "BodyProcess"),
    "EnvConfig": ("aif_paper1.affectworld.configs", "EnvConfig"),
    "EpisodeLog": ("aif_paper1.affectworld.logging_utils", "EpisodeLog"),
    "Interact": ("aif_paper1.affectworld.helpers", "Interact"),
    "Move": ("aif_paper1.affectworld.helpers", "Move"),
    "Need": ("aif_paper1.affectworld.helpers", "Need"),
    "Resource": ("aif_paper1.affectworld.helpers", "Resource"),
    "RunMetadata": ("aif_paper1.affectworld.logging_utils", "RunMetadata"),
    "Season": ("aif_paper1.affectworld.helpers", "Season"),
    "SimpleRegime": ("aif_paper1.affectworld.regimes", "SimpleRegime"),
    "Simulation": ("aif_paper1.affectworld.simulation", "Simulation"),
    "StepLogWriter": ("aif_paper1.affectworld.logging_utils", "StepLogWriter"),
    "apply_simple_regime": ("aif_paper1.affectworld.regimes", "apply_simple_regime"),
    "budget_mode_to_index": ("aif_paper1.affectworld.intero_precision", "budget_mode_to_index"),
    "c2i": ("aif_paper1.affectworld.affect_world_env", "c2i"),
    "channel_precisions_jax": ("aif_paper1.affectworld.intero_precision", "channel_precisions_jax"),
    "channel_precisions_np": ("aif_paper1.affectworld.intero_precision", "channel_precisions_np"),
    "gaussian_stay_prob": ("aif_paper1.affectworld.affect_world_env", "gaussian_stay_prob"),
    "i2c": ("aif_paper1.affectworld.affect_world_env", "i2c"),
    "load_simple_regime_bank": ("aif_paper1.affectworld.regimes", "load_simple_regime_bank"),
    "make_env": ("aif_paper1.affectworld.regimes", "make_env"),
    "need_to_interact": ("aif_paper1.affectworld.helpers", "need_to_interact"),
    "normalize_budget_mode": ("aif_paper1.affectworld.intero_precision", "normalize_budget_mode"),
    "save_simple_regime_bank": ("aif_paper1.affectworld.regimes", "save_simple_regime_bank"),
    "simple_regime_diagnostics": ("aif_paper1.affectworld.regimes", "simple_regime_diagnostics"),
    "simple_regime_from_dict": ("aif_paper1.affectworld.regimes", "simple_regime_from_dict"),
    "simple_regime_to_dict": ("aif_paper1.affectworld.regimes", "simple_regime_to_dict"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazily import public AffectWorld symbols on first attribute access."""

    target = _EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports in interactive discovery tools such as `dir()`."""

    return sorted(set(globals().keys()) | set(__all__))

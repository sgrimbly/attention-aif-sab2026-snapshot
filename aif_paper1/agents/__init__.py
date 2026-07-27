"""Agent-family package exposing the maintained public surfaces lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "Agent",
    "ExternalKappaAgent",
]

_EXPORTS = {
    "Agent": ("aif_paper1.agents.standard_active_inference", "Agent"),
    "ExternalKappaAgent": ("aif_paper1.agents.likelihood_precision_selective_attention", "ExternalKappaAgent"),
}


def __getattr__(name: str) -> Any:
    """Import public agent classes on first access to avoid package-import cycles."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from aif_paper1.agents.likelihood_precision_selective_attention import ExternalKappaAgent
    from aif_paper1.agents.standard_active_inference import Agent

"""Maintained helpers for integrating the repo with the local upstream `pymdp`.

The canonical Active Inference dependency for this repo is the editable sibling
checkout in ``../pymdp``. Maintained code should target the current merged
upstream API and avoid relying on `pymdp.legacy.*`.
"""

from __future__ import annotations

import importlib
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import random as jr
import jax.tree_util as jtu

from pymdp.control import Policies
from pymdp.planning.mcts import mcts_policy_search
from pymdp.planning.si import si_policy_search

from aif_paper1.planning.planner_utils import _coerce_runtime_model_list, _coerce_runtime_policy_prior

from typing import Any


_PYMDP_CAPABILITIES: dict[str, bool] = {
    "infer_and_plan": True,
    "Policies.policy_arr": ("policy_arr" in getattr(Policies, "__annotations__", {})) or hasattr(Policies, "policy_arr"),
    "si_policy_search": callable(si_policy_search),
    "mcts_policy_search": callable(mcts_policy_search),
}


def assert_pymdp_capabilities() -> None:
    """Fail fast if the sibling upstream ``pymdp`` checkout lacks maintained APIs."""
    missing = [name for name, ok in _PYMDP_CAPABILITIES.items() if not bool(ok)]
    if missing:
        raise RuntimeError(
            "The sibling pymdp checkout is missing required maintained APIs: "
            + ", ".join(missing)
        )


def _clone_agent_with_runtime_overrides(*_args: Any, **_kwargs: Any):
    """
    Deprecated. Replaced by the in-place ``_agent_runtime_overrides`` context
    manager in ``src/planning/upstream.py``.

    The original implementation rebuilt the pymdp ``Agent`` per call via
    ``equinox.tree_at``, which leaks anonymous mmap regions per step because
    of the upstream ``Policies(static=True)`` design (a JAX array in the
    static partition). With Tier 1 of the routing fix, no production planner
    reaches this helper; the stub raises so any stale call site fails loudly
    rather than silently re-introducing the leak.

    See ``docs/architecture/efe_repo_local_scorer.md`` and
    ``docs/architecture/pymdp_static_array_fork_fix.md``.
    """
    raise RuntimeError(
        "_clone_agent_with_runtime_overrides (pymdp_integration) is deprecated. "
        "All planners now route through `custom_step` and use the repo-local "
        "EFE scorer or the in-place context manager `_agent_runtime_overrides`. "
        "See docs/architecture/efe_repo_local_scorer.md."
    )


def infer_states_compat(
    agent: Any,
    observation: list[Any],
    prior: list[Any],
    *,
    past_actions: Any | None = None,
    qs_hist: Any | None = None,
    valid_steps: Any | None = None,
    mask: Any | None = None,
    preprocess_fn: Any | None = None,
    return_info: bool = False,
):
    """Call ``Agent.infer_states`` using the maintained upstream signature."""

    return agent.infer_states(
        observations=observation,
        empirical_prior=prior,
        past_actions=past_actions,
        qs_hist=qs_hist,
        valid_steps=valid_steps,
        mask=mask,
        preprocess_fn=preprocess_fn,
        return_info=return_info,
    )


def update_empirical_prior_compat(agent: Any, action: Any, qs: list[Any]) -> list[Any]:
    """Return the propagated empirical prior from the current upstream API."""

    return agent.update_empirical_prior(action, qs)


def _resolve_empirical_prior_local(
    agent: Any,
    qs_prev: list[Any],
    action_prev: Any | None,
    empirical_prior: list[Any] | None,
) -> list[Any]:
    """Mirror the maintained upstream empirical-prior resolution without env imports."""

    if empirical_prior is not None:
        return empirical_prior
    if action_prev is None:
        return agent.D
    try:
        if bool(jnp.any(jnp.asarray(action_prev) < 0)):
            return agent.D
    except Exception:
        pass
    return agent.update_empirical_prior(action_prev, qs_prev)


def _update_parameters_online_local(
    agent: Any,
    *,
    qs_prev: list[Any],
    qs: list[Any],
    observation: list[Any],
    action_prev: Any,
) -> Any:
    """
    Best-effort local counterpart of upstream online parameter updates.

    The maintained repo path that uses ``infer_and_plan_compat`` does not rely on
    sequence-window learning. We therefore keep the non-windowed learning update
    local here and avoid importing ``pymdp.envs.rollout`` at module import time.
    """

    if not ((action_prev is not None) and (getattr(agent, "learning_mode", None) == "online")):
        return agent
    if not (bool(getattr(agent, "learn_A", False)) or bool(getattr(agent, "learn_B", False))):
        return agent

    if bool(getattr(agent, "learn_B", False)):
        beliefs_B = jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), qs_prev, qs)
        action_for_learning = jnp.expand_dims(jnp.asarray(action_prev), axis=1)
    else:
        beliefs_B = None
        action_for_learning = action_prev

    return agent.infer_parameters(
        qs,
        observation,
        action_for_learning,
        beliefs_B=beliefs_B,
    )


def infer_and_plan_compat(*_args: Any, **_kwargs: Any):
    """
    Deprecated. The all-in-one upstream ``infer_and_plan`` orchestrator goes
    through pymdp's ``Agent.infer_states`` and ``Agent.infer_policies``
    methods, which are vmapped over the agent itself and trigger the upstream
    ``Policies(static=True)`` JIT-cache poisoning bug. That allocates fresh
    JAX device buffers per step, accumulating ~3960 anonymous mmap regions
    per trial.

    Tier 1 of the routing fix in ``src/planning/routing.py`` already routes
    every planner away from this code path. Production should use the
    repo-local ``custom_step`` runtime backend, which composes
    ``_infer_states_from_arrays`` + the planner's ``_infer_policies`` +
    ``sample_action`` directly without going through agent methods.

    Stubbed to raise so any stale callsite is caught immediately rather than
    silently re-introducing the leak. To restore the upstream fast path,
    apply the upstream patch documented in
    ``docs/architecture/pymdp_static_array_fork_fix.md`` and re-implement
    this helper using the in-place context-manager pattern from
    ``src/planning/upstream.py::_agent_runtime_overrides``.
    """
    raise RuntimeError(
        "infer_and_plan_compat is deprecated; route via "
        "`resolve_actinf_runtime_backend` -> `custom_step`. "
        "See docs/architecture/efe_repo_local_scorer.md."
    )


def rollout_compat(*args: Any, **kwargs: Any):
    """Lazy wrapper over upstream `pymdp.envs.rollout.rollout`."""

    mod = importlib.import_module("pymdp.envs.rollout")
    return mod.rollout(*args, **kwargs)


__all__ = [
    "assert_pymdp_capabilities",
    "infer_states_compat",
    "infer_and_plan_compat",
    "rollout_compat",
    "update_empirical_prior_compat",
]

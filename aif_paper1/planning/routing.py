"""Central planner naming, validation, and backend routing rules."""

from __future__ import annotations

from dataclasses import dataclass

from aif_paper1.planning.policy_space import _normalize_planner_name, _normalize_policy_mode


@dataclass(frozen=True)
class PlannerRoute:
    """
    Static metadata describing one planner family.

    The route tells the agent constructor which implementation owns the search,
    whether the planner expects full policy enumeration or a one-step action
    library, and which kinds of runtime overrides the backend can accept.
    """

    name: str
    planner_backend: str
    policy_space: str
    full_policy_mode_only: bool
    supports_runtime_ace: bool


_ROUTES: dict[str, PlannerRoute] = {
    "efe": PlannerRoute(
        name="efe",
        # ``efe`` runs through the repo-local shape-stable scorer at
        # ``src/planning/efe_repo_local.py`` rather than upstream
        # ``pymdp.Agent.infer_policies``. See
        # ``docs/architecture/efe_repo_local_scorer.md`` for the why.
        planner_backend="repo_local",
        policy_space="full_policy",
        full_policy_mode_only=False,
        supports_runtime_ace=True,
    ),
}


def planner_choices() -> tuple[str, ...]:
    """Return the canonical planner names accepted by the repo."""
    return tuple(_ROUTES.keys())


def get_planner_route(name: str) -> PlannerRoute:
    """Resolve a user-facing planner alias to its canonical route metadata."""
    canonical = _normalize_planner_name(name)
    return _ROUTES[canonical]


def validate_planner_policy_mode(planner: str, policy_mode: str) -> None:
    """
    Enforce planner-specific restrictions on ``policy_mode``.

    Upstream SI and upstream MCTS operate on the agent's full pre-enumerated
    policy library, so they cannot also apply the repo's one-step masking
    regimes such as ``navigate`` or ``competence``.
    """
    route = get_planner_route(planner)
    mode = _normalize_policy_mode(policy_mode)
    if route.full_policy_mode_only and mode != "full":
        raise ValueError(
            f"planner={route.name!r} requires policy_mode='full'; "
            "use efe for restricted planning spaces."
        )


def planner_backend(planner: str) -> str:
    """Return the implementation backend label for a planner."""
    return str(get_planner_route(planner).planner_backend)


def resolve_actinf_runtime_backend(
    *,
    planner: str,  # noqa: ARG001 — kept for signature parity with legacy callers
    engine: str | None = None,
    oracle_state_access: bool = False,  # noqa: ARG001
    external_kappa: bool = False,  # noqa: ARG001
    gate_active: bool = False,  # noqa: ARG001
    habit_mode: str = "none",  # noqa: ARG001
    proposal_mode: str = "none",  # noqa: ARG001
    unknown_model: bool = False,  # noqa: ARG001
    dynamic_resource_preferences: bool = False,  # noqa: ARG001
) -> str:
    """
    Choose which ActInf execution loop should own a decision step.

    Always returns ``custom_step`` (or ``jax_scan_runner`` for the dedicated
    scan engine). The legacy ``pymdp_infer_and_plan`` fast path routed through
    ``infer_and_plan_compat`` -> ``_clone_agent_with_runtime_overrides``,
    which rebuilt the pymdp ``Agent`` every step via ``eqx.tree_at``. The
    upstream ``Policies(static=True)`` design poisons that rebuild's static
    partition with a JAX array, allocating fresh device buffers for
    ``A``/``C``/``E`` each step and accumulating ~3960 anonymous mmap regions
    per trial. ``custom_step`` composes ``_infer_states`` + ``_infer_policies``
    + ``_sample_action`` directly through the repo-local EFE scorer, avoiding
    the rebuild.

    The signature retains the original keyword arguments so older callers do
    not break; they are accepted and ignored.
    """
    if str(engine or "").strip().lower() == "jax_scan":
        return "jax_scan_runner"
    return "custom_step"


__all__ = [
    "PlannerRoute",
    "get_planner_route",
    "planner_backend",
    "planner_choices",
    "resolve_actinf_runtime_backend",
    "validate_planner_policy_mode",
]

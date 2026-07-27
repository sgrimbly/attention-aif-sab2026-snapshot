"""
Repo-local, JIT-stable EFE policy scorer.

The factory ``make_efe_infer_policies`` returns a function with the canonical
``_infer_policies(qs, rng_key, A_runtime, C_runtime, pA_runtime, E_runtime,
root_action_mask_runtime, *, keep_mask)`` signature used by maintained
ActInf agents. Unlike the previous inline path that called
``pymdp_agent.infer_policies(qs)`` after ``equinox.tree_at`` rebuilds, this
factory captures every "config" value as a Python-side closure constant.
Only call-site tensors (``qs``, ``A_runtime``, ``C_runtime``, ``E_runtime``,
``keep_mask``) flow through the JIT boundary, and their shapes are stable
across episodes — so XLA compiles the policy scorer at most once per agent
and reuses it for the full run.

Batching. ``batch_size > 1`` is supported via an outer ``vmap`` over the
batch axis. The closure constants captured from ``pymdp_agent`` are kept as
``(B, *per_element_shape)`` tuples; ``_compute_one`` operates on a single
batch element and ``_compute`` vmaps it over the batch axis. At ``B=1``
(Paper 1's current production configuration) the math is identical to the
pre-batched scorer; at ``B>1`` each element produces an independent
``(num_policies,)`` posterior. See ``docs/architecture/benchmarks/
batching_audit.md`` for the full implementation plan; this file lands
Step 1 of that plan.

See ``docs/architecture/efe_repo_local_scorer.md`` for the rationale, the
considered alternatives, and revert paths.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
from jax import nn, vmap
from pymdp.control import compute_neg_efe_policy_inductive

from aif_paper1.planning.planner_utils import (
    coerce_runtime_arr_list_batched,
)

# floor offset added below the minimum kept ``neg_efe`` so masked-out policies
# stay finite (preserves downstream consumers that compare G values) while the
# softmax effectively zeroes their posterior weight.
_PRUNED_NEG_EFE_FLOOR_OFFSET: float = 1.0e3


def make_efe_infer_policies(
    pymdp_agent: Any,
    *,
    jit: bool = False,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Build a shape-stable EFE ``_infer_policies`` for ``pymdp_agent``.

    The returned callable accepts the same positional arguments as the legacy
    inline EFE wrapper plus a ``keep_mask`` keyword argument. ``keep_mask``,
    when supplied, is a fixed-shape boolean mask over the policy library that
    selects which policies remain in the posterior; pruned entries get a large
    negative penalty before the softmax and zero posterior weight after it.

    Parameters
    ----------
    pymdp_agent
        The constructed pymdp Agent. The factory reads its current
        ``A``/``B``/``C``/``E``/``pA``/``pB``/``I``/``gamma`` tensors and
        configuration flags. After construction the factory does not
        re-read the agent; in-place updates (e.g. learning the resource
        likelihood, dynamic preference rewrites) flow through the runtime
        override path instead.
    jit
        Whether the factory should wrap the returned function in
        ``equinox.filter_jit``. Most callers leave this ``False`` and rely
        on the agent-level ``jit_planner`` switch, which already wraps via
        ``filter_jit``. Set to ``True`` for standalone use (numeric parity
        smokes, ad-hoc scripts).

    Notes
    -----
    Supports ``batch_size >= 1``. Closure constants are kept as
    ``(B, *per_element_shape)`` tuples; the per-element scorer ``_compute_one``
    is vmap'd over the leading batch axis. Output shape is ``(B, num_policies)``
    for both ``q_pi`` and ``G``.
    """

    batch_size = int(getattr(pymdp_agent, "batch_size", 1))

    # Closure constants — keep the leading batch axis. Each tuple element
    # has shape ``(B, *per_element_shape)``; the outer vmap slices axis 0
    # at scoring time to give ``_compute_one`` per-element views.
    A_default = tuple(jnp.asarray(a) for a in pymdp_agent.A)
    B_default = tuple(jnp.asarray(b) for b in pymdp_agent.B)
    C_default = tuple(jnp.asarray(c) for c in pymdp_agent.C)
    E_default = jnp.asarray(pymdp_agent.E)
    pA_default = (
        None
        if pymdp_agent.pA is None
        else tuple(jnp.asarray(p) for p in pymdp_agent.pA)
    )
    pB_default = (
        None
        if pymdp_agent.pB is None
        else tuple(jnp.asarray(p) for p in pymdp_agent.pB)
    )
    I_const = tuple(jnp.asarray(x) for x in pymdp_agent.I)

    policy_matrix = jnp.asarray(pymdp_agent.policies.policy_arr)
    A_deps = tuple(tuple(d) for d in pymdp_agent.A_dependencies)
    B_deps = tuple(tuple(d) for d in pymdp_agent.B_dependencies)
    gamma_vec = jnp.asarray(pymdp_agent.gamma)
    inductive_eps_vec = jnp.asarray(pymdp_agent.inductive_epsilon)

    use_utility = bool(pymdp_agent.use_utility)
    use_states_info_gain = bool(pymdp_agent.use_states_info_gain)
    use_param_info_gain = bool(pymdp_agent.use_param_info_gain)
    use_inductive = bool(pymdp_agent.use_inductive)

    num_policies = int(policy_matrix.shape[0])
    full_mask = jnp.ones((num_policies,), dtype=jnp.bool_)
    floor_offset = jnp.float32(_PRUNED_NEG_EFE_FLOOR_OFFSET)

    # Expected per-batch-element shapes (drop leading batch axis).
    A_expected_shapes = tuple(jnp.asarray(a).shape[1:] for a in pymdp_agent.A)
    C_expected_shapes = tuple(jnp.asarray(c).shape[1:] for c in pymdp_agent.C)
    E_expected_shape = (num_policies,)

    def _coerce_A_runtime(runtime):
        # Delegate to the shared helper so all coercions share a single contract.
        return coerce_runtime_arr_list_batched(
            runtime,
            expected_per_element_shapes=A_expected_shapes,
            batch_size=batch_size,
            label="runtime A override",
            fallback=A_default,
        )

    def _coerce_C_runtime(runtime):
        return coerce_runtime_arr_list_batched(
            runtime,
            expected_per_element_shapes=C_expected_shapes,
            batch_size=batch_size,
            label="runtime C override",
            fallback=C_default,
        )

    def _coerce_E_runtime(runtime):
        if runtime is None:
            return E_default
        e = jnp.asarray(runtime, dtype=jnp.float32).reshape(-1)
        if e.shape != E_expected_shape:
            raise ValueError(
                f"runtime E override shape mismatch: "
                f"got={e.shape} expected={E_expected_shape}"
            )
        e = jnp.clip(e, 1e-8, None)
        e = e / jnp.clip(e.sum(), 1e-8, None)
        return jnp.broadcast_to(e[None, :], (batch_size, num_policies))

    def _score_one_policy(
        qs_latest_b, A_list_b, B_list_b, C_list_b,
        pA_list_b, pB_list_b, I_list_b, ind_eps_b, policy_i,
    ):
        return compute_neg_efe_policy_inductive(
            list(qs_latest_b),
            list(A_list_b),
            list(B_list_b),
            list(C_list_b),
            None if pA_list_b is None else list(pA_list_b),
            None if pB_list_b is None else list(pB_list_b),
            list(A_deps),
            list(B_deps),
            list(I_list_b),
            policy_i,
            inductive_epsilon=ind_eps_b,
            use_utility=use_utility,
            use_states_info_gain=use_states_info_gain,
            use_param_info_gain=use_param_info_gain,
            use_inductive=use_inductive,
        )

    def _compute_one(
        qs_latest_b, A_list_b, B_list_b, C_list_b,
        pA_list_b, pB_list_b, I_list_b,
        E_vec_b, gamma_b, ind_eps_b, mask_arr,
    ):
        score = lambda pi: _score_one_policy(  # noqa: E731
            qs_latest_b, A_list_b, B_list_b, C_list_b,
            pA_list_b, pB_list_b, I_list_b, ind_eps_b, pi,
        )
        neg_efe = vmap(score)(policy_matrix)

        # Pruned entries: subtract a large penalty from the logits, then zero
        # the resulting q_pi mass and renormalize over the kept set.
        penalty = jnp.where(mask_arr, jnp.float32(0.0), floor_offset)
        logits = gamma_b * (neg_efe - penalty) + jnp.log(E_vec_b + 1e-16)
        q_pi = nn.softmax(logits, axis=-1)
        mask_f = mask_arr.astype(jnp.float32)
        q_pi = q_pi * mask_f
        q_pi = q_pi / jnp.clip(q_pi.sum(), 1e-8, None)

        # G values for pruned positions: floor below the minimum kept value so
        # downstream comparisons keep their ordering.
        floor_g = jnp.min(neg_efe) - floor_offset
        G_out = jnp.where(mask_arr, neg_efe, floor_g)

        return q_pi, G_out

    def _compute(
        qs_latest, A_list, B_list, C_list,
        pA_list, pB_list, I_list,
        E_mat, gamma, ind_eps, mask_arr, *, mask_per_b: bool = False,
    ):
        """Outer vmap over the leading batch axis.

        ``mask_arr`` may be either:
        - shape ``(num_policies,)`` shared across the batch (``mask_per_b=False``),
          or
        - shape ``(B, num_policies)`` per-batch-element (``mask_per_b=True``).

        Per-batch masks are required for habit_mode='prior_prune' at bs > 1
        (block E lift), where each batch element computes its own
        proposal-derived keep mask from per-element observations.
        """
        pa_axes = None if pA_list is None else 0
        pb_axes = None if pB_list is None else 0
        mask_axes = 0 if mask_per_b else None
        return vmap(
            _compute_one,
            in_axes=(0, 0, 0, 0, pa_axes, pb_axes, 0, 0, 0, 0, mask_axes),
        )(
            qs_latest, A_list, B_list, C_list,
            pA_list, pB_list, I_list,
            E_mat, gamma, ind_eps, mask_arr,
        )

    def _infer_policies(
        qs,
        rng_key=None,  # noqa: ARG001 — kept for signature parity
        A_runtime=None,
        C_runtime=None,
        pA_runtime=None,
        E_runtime=None,
        root_action_mask_runtime=None,
        *,
        keep_mask=None,
    ):
        # ``pA_runtime`` is silently ignored when ``use_param_info_gain`` is
        # False (the Paper 1 default). This matches the prior inline scorer's
        # ``pA_runtime=None,  # noqa: ARG001`` semantic — agents pass the
        # current Dirichlet posterior during learning, but it never enters the
        # EFE compute when the parameter-information-gain term is off. If a
        # if ``use_param_info_gain=True`` is needed, thread the
        # override through the closure (open work item).
        if use_param_info_gain and pA_runtime is not None:
            raise ValueError(
                "Repo-local EFE scorer does not yet thread pA_runtime through "
                "the JIT boundary when use_param_info_gain=True. Either set "
                "use_param_info_gain=False or extend the scorer to accept a "
                "shape-stable pA_runtime override."
            )
        if root_action_mask_runtime is not None:
            raise ValueError(
                "planner='efe' does not support root-action proposal pruning."
            )

        A_list = _coerce_A_runtime(A_runtime)
        C_list = _coerce_C_runtime(C_runtime)
        E_mat = _coerce_E_runtime(E_runtime)

        # keep_mask may be (num_policies,) shared across batch (legacy
        # contract) or (B, num_policies) per-batch-element (block E lift
        # for habit_mode='prior_prune' at bs > 1). We dispatch on shape
        # so the bs=1 path remains bit-identical.
        mask_per_b = False
        if keep_mask is None:
            mask_arr = full_mask
        else:
            keep_arr = jnp.asarray(keep_mask, dtype=jnp.bool_)
            if keep_arr.ndim == 1:
                mask_arr = keep_arr.reshape(-1)
            elif keep_arr.ndim == 2 and int(keep_arr.shape[0]) == int(batch_size):
                mask_arr = keep_arr
                mask_per_b = True
            else:
                raise ValueError(
                    f"keep_mask shape {keep_arr.shape} does not match "
                    f"either ({num_policies},) [shared] or "
                    f"({batch_size}, {num_policies}) [per-batch]"
                )

        # qs[f] has shape (B, T, num_states_f). Take the last timestep along
        # axis 1 and keep the batch axis so the outer vmap can slice it.
        qs_latest = tuple(q[:, -1] for q in qs)

        return _compute(
            qs_latest, A_list, B_default, C_list,
            pA_default, pB_default, I_const,
            E_mat, gamma_vec, inductive_eps_vec, mask_arr,
            mask_per_b=mask_per_b,
        )

    return eqx.filter_jit(_infer_policies) if jit else _infer_policies


__all__ = ["make_efe_infer_policies"]

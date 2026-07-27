"""Shared Active Inference helpers used by active agent families."""

from __future__ import annotations

import io
import logging
import os
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import vmap
from pymdp import control, inference
from pymdp.agent import Agent as PymdpAgent


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _configure_quiet_jax_logging_from_env() -> None:
    """
    Suppress noisy JAX/XLA bridge logs only when the caller opted into quiet mode.

    Importing the maintained agent packages should not globally change logging
    behavior unless a runtime entrypoint explicitly requested that quieter mode.
    """
    if _env_truthy("AFFECTWORLD_QUIET_JAX_XLA") or _env_truthy("AFFECTWORLD_QUIET_PYMDP"):
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)


_configure_quiet_jax_logging_from_env()


def _policies_array(self, dtype=None):
    arr = np.asarray(self.policy_arr)
    if dtype is None:
        return arr
    return arr.astype(dtype, copy=False)


if getattr(control.Policies, "__array__", None) is None:
    control.Policies.__array__ = _policies_array  # type: ignore[attr-defined]


class _NullWriter(io.TextIOBase):
    def write(self, s):  # type: ignore[override]
        return 0 if s is None else len(s)

    def flush(self) -> None:  # pragma: no cover
        return None


_QUIET_SINK = _NullWriter()


def _policy_array_np(pymdp_agent: PymdpAgent) -> np.ndarray:
    policies = getattr(pymdp_agent, "policies")
    return np.asarray(getattr(policies, "policy_arr", policies))


def _policy_array_jnp(pymdp_agent: PymdpAgent) -> jnp.ndarray:
    policies = getattr(pymdp_agent, "policies")
    return jnp.asarray(getattr(policies, "policy_arr", policies))


def _coerce_runtime_model_list(
    base_model: list[Any],
    runtime_model: list[Any] | None,
    *,
    label: str,
) -> list[Any]:
    """
    Validate an optional runtime model override against the agent's base tensors.

    The maintained agents store `A` and `C` with a leading batch dimension. The
    override path accepts per-modality tensors without that leading batch axis
    and expands them here into the batched shape expected by pymdp.
    """
    if runtime_model is None:
        return base_model

    if int(len(runtime_model)) != int(len(base_model)):
        raise ValueError(
            f"runtime {label} override length mismatch: got={len(runtime_model)} expected={len(base_model)}"
        )

    model_batch: list[Any] = []
    for idx, model_i in enumerate(runtime_model):
        model_arr = jnp.asarray(model_i, dtype=jnp.float32)
        expected = jnp.asarray(base_model[idx][0], dtype=jnp.float32)
        if model_arr.shape != expected.shape:
            raise ValueError(
                f"runtime {label} override shape mismatch at modality {idx}: "
                f"got={model_arr.shape} expected={expected.shape}"
            )
        model_batch.append(model_arr[None, ...])

    return model_batch


def _infer_states_from_arrays(
    *,
    A_base: list[Any],
    B_base: list[Any],
    observation: list[Any],
    prior: list[Any],
    A_runtime: list[Any] | None,
    A_dependencies: list[list[int]],
    B_dependencies: list[list[int]],
    num_iter: int,
    method: str,
    inference_horizon: int | None,
    past_actions: Any | None = None,
    qs_hist: list[Any] | None = None,
    valid_steps: Any | None = None,
    return_info: bool = False,
    preprocess_fn: Any | None = None,
) -> list[Any]:
    """
    Run state inference without closing a JIT trace over ``pymdp.Agent``.

    The upstream Agent stores the policy library in an Equinox static field. In
    long Paper 1 sweeps that static object can make otherwise simple state
    inference traces much larger than necessary. This helper keeps the JIT
    signature to arrays and small Python metadata only, calling the leaf
    function ``pymdp.inference.update_posterior_states`` directly through a
    vmap-over-batch.

    Parameters
    ----------
    A_base, B_base, observation, prior, A_runtime, A_dependencies,
    B_dependencies, num_iter, method, inference_horizon
        See call sites in maintained agents; these mirror the Paper 1 path.
    past_actions
        Optional action history for sequence-window inference (``mmp``/``vmp``
        methods). Default ``None`` matches the prior hard-coded behaviour.
    qs_hist
        Optional posterior history buffer; one-step methods append to it when
        provided. Default ``None`` matches the prior hard-coded behaviour.
    valid_steps
        Optional unpadded-timestep count for fixed-window sequence inputs.
        Default ``None`` matches the prior hard-coded behaviour.
    return_info
        When ``True`` the leaf returns ``(qs, info)`` with canonical VFE
        diagnostics. Default ``False`` matches the prior hard-coded behaviour
        and what production agents expect.
    preprocess_fn
        Optional Agent-style observation transform applied once *before* the
        vmap. Default ``None`` is a no-op. Mirrors the upstream
        ``Agent.preprocess_fn`` slot for parity with the wider
        ``Agent.infer_states`` API surface.

    These additional parameters are pass-throughs to the leaf function and
    cost effectively nothing when defaults are kept (one ``is None`` check
    per call). They exist so a future feature that needs the wider state-
    inference API surface (Paper 2 sequence inference, on-demand VFE
    diagnostics, agent preprocessing) does not require unwinding the
    architectural decision documented in
    ``docs/architecture/efe_repo_local_scorer.md`` and
    ``docs/architecture/benchmarks/pymdp_parity.md``. The ``mask`` keyword
    and ``past_actions``-driven empirical-prior auto-resolution remain
    intentionally absent; see the parity report for the reasoning.
    """

    A_eff = _coerce_runtime_model_list(A_base, A_runtime, label="A")
    if preprocess_fn is not None:
        observation = preprocess_fn(observation)
    infer_states = partial(
        inference.update_posterior_states,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_iter=int(num_iter),
        method=str(method),
        distr_obs=True,
        inference_horizon=inference_horizon,
        return_info=bool(return_info),
    )
    return vmap(infer_states)(
        A_eff,
        B_base,
        observation,
        past_actions,
        prior=prior,
        qs_hist=qs_hist,
        valid_steps=valid_steps,
    )


def _evaluate_efe_policy_subset(
    *,
    infer_policies_fn: Any,
    qs,
    keep_idx: np.ndarray | jnp.ndarray,
    num_policies: int,
    A_runtime: list[Any] | None = None,
    C_runtime: list[Any] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate plain EFE on a retained subset by masking the repo-local scorer.

    The repo-local EFE scorer (``src/planning/efe_repo_local.py``) accepts a
    fixed-shape ``keep_mask`` keyword. Routing the prune-then-EFE path through
    the same compiled artifact eliminates a per-call recompile that the
    earlier subset evaluator triggered (the policy-subset shape varied per
    call). After workstream A this helper is a thin adapter: build a bool
    mask, delegate, return.

    The mask layout is ``mask[i] = True`` for retained indices ``i`` in
    ``keep_idx``. The scorer subtracts a large penalty from masked logits
    before the softmax, zeros their posterior weight after it, and floors
    their ``neg_efe`` below ``min(kept) - 1e3`` for downstream consumers.
    """
    keep_arr = np.asarray(keep_idx, dtype=np.int32).reshape(-1)
    if keep_arr.size <= 0:
        raise ValueError("Policy subset must keep at least one policy.")
    if np.unique(keep_arr).size != keep_arr.size:
        raise ValueError("Policy subset indices must be unique.")
    n_policies = int(num_policies)
    if int(np.min(keep_arr)) < 0 or int(np.max(keep_arr)) >= n_policies:
        raise ValueError(
            f"Policy subset indices out of bounds for library of size {n_policies}."
        )

    mask = np.zeros((n_policies,), dtype=np.bool_)
    mask[keep_arr] = True
    keep_mask = jnp.asarray(mask, dtype=jnp.bool_)

    return infer_policies_fn(
        qs, None, A_runtime, C_runtime, None, None,
        keep_mask=keep_mask,
    )


def _flatten_b_action_dims_for_init(
    B: list[np.ndarray],
    *,
    b_action_dependencies: list[list[int]],
    num_controls_multi: list[int],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """
    Flatten multi-control action dimensions the same way pymdp does internally.

    This lets us provide restricted EFE policy libraries up front without first
    forcing upstream full policy enumeration in `PymdpAgent.__init__`.
    """
    action_maps: list[dict[str, Any]] = []
    B_flat: list[np.ndarray] = []
    for factor_idx, (B_f, action_dependency) in enumerate(zip(B, b_action_dependencies)):
        B_arr = np.asarray(B_f)
        if action_dependency == []:
            B_flat.append(np.expand_dims(B_arr, axis=-1))
            action_maps.append(
                {
                    "multi_dependency": [],
                    "multi_dims": [],
                    "flat_dependency": [factor_idx],
                    "flat_dims": [1],
                }
            )
            continue

        dims = [int(num_controls_multi[d]) for d in action_dependency]
        flat_dim = int(np.prod(np.asarray(dims, dtype=np.int64)))
        target_shape = list(B_arr.shape[: -len(action_dependency)]) + [flat_dim]
        B_flat.append(B_arr.reshape(target_shape))
        action_maps.append(
            {
                "multi_dependency": list(action_dependency),
                "multi_dims": dims,
                "flat_dependency": [factor_idx],
                "flat_dims": [flat_dim],
            }
        )

    return B_flat, action_maps


def _flatten_policies_from_action_maps(
    policies_multi: np.ndarray | jnp.ndarray,
    *,
    action_maps: list[dict[str, Any]],
) -> np.ndarray:
    """
    Convert per-control policies into the flattened per-factor policy encoding
    expected by pymdp when B action dimensions have been flattened.
    """
    policies = np.asarray(policies_multi, dtype=np.int32)
    if policies.ndim < 2:
        raise ValueError(f"Expected policies with rank >= 2, got shape {policies.shape}")

    policies_flat: list[np.ndarray] = []
    for action_map in action_maps:
        deps = list(action_map["multi_dependency"])
        if not deps:
            policies_flat.append(np.zeros_like(policies[..., 0], dtype=np.int32))
            continue

        dims = [int(dim) for dim in action_map["multi_dims"]]
        combos = np.asarray(policies[..., deps], dtype=np.int32)
        flat_idx = np.zeros(combos.shape[:-1], dtype=np.int32)
        product = 1
        for dep_idx in reversed(range(len(dims))):
            flat_idx += combos[..., dep_idx] * product
            product *= dims[dep_idx]
        policies_flat.append(flat_idx)

    return np.stack(policies_flat, axis=-1).astype(np.int32, copy=False)


def _validate_search_planner_pymdp_interface(
    *,
    planner_name: str,
    pymdp_agent: PymdpAgent,
    b_action_dependencies: list[list[int]],
    num_controls_multi: list[int],
    label: str,
) -> None:
    """
    Ensure search planners are wired to pymdp using flattened action dependencies.

    In this codebase, beam/SI/MCTS build action libraries outside pymdp and rely on
    pymdp to flatten multi-control B action dimensions by passing `policies=None`.
    """
    if str(planner_name).lower() == "efe":
        return

    if getattr(pymdp_agent, "action_maps", None) is None:
        raise RuntimeError(
            f"{label}: planner={planner_name} expected pymdp action flattening (`action_maps` is None). "
            "Ensure `policies=None` is passed to PymdpAgent for search planners."
        )

    pol = np.asarray(getattr(pymdp_agent, "policies"))
    if pol.ndim != 3:
        raise RuntimeError(f"{label}: expected policies rank-3 for planner={planner_name}, got shape {pol.shape}.")
    if int(pol.shape[-1]) != int(len(b_action_dependencies)):
        raise RuntimeError(
            f"{label}: flattened policy trailing dim mismatch for planner={planner_name}: "
            f"got {pol.shape[-1]}, expected {len(b_action_dependencies)} hidden factors."
        )

    mismatches: list[str] = []
    for factor_idx, deps in enumerate(b_action_dependencies):
        expected = 1
        for dep in deps:
            expected *= int(num_controls_multi[int(dep)])
        got = int(np.asarray(pymdp_agent.B[factor_idx]).shape[-1])
        if got != expected:
            mismatches.append(f"B[{factor_idx}] action dim={got}, expected={expected}, deps={deps}")
    if mismatches:
        raise RuntimeError(
            f"{label}: pymdp B-action interface mismatch for planner={planner_name}:\n- " + "\n- ".join(mismatches)
        )


__all__ = [
    "_coerce_runtime_model_list",
    "_evaluate_efe_policy_subset",
    "_flatten_b_action_dims_for_init",
    "_flatten_policies_from_action_maps",
    "_infer_states_from_arrays",
    "_QUIET_SINK",
    "_NullWriter",
    "_policy_array_jnp",
    "_policy_array_np",
    "_validate_search_planner_pymdp_interface",
]

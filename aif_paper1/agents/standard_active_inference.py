"""
Baseline Active Inference agent for the maintained AffectWorld Paper 1 path.

Paper-facing terminology:
- latent bodily factors: nutrient, hydration, oxygenation
- noisy interoceptive observations: nutrient / hydration / oxygenation-proxy
- dormant fourth channel: retained only for engine compatibility

This baseline does *not* implement selective interoceptive attention. It treats
all interoceptive channels with fixed precision and selects only external
actions `(Move, Interact)` by minimizing expected free energy.
"""

from __future__ import annotations

import os
import contextlib
from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import random as jr
from pymdp import control
from pymdp.agent import Agent as PymdpAgent

from aif_paper1.affectworld.helpers import Interact, Move, Resource, Season
from aif_paper1.affectworld.body import BodyProcess
from aif_paper1.agents.internal.habit_control import (
    OptionState,
    choose_option_action,
    dominant_need_idx,
    habit_action_proposal_keep_indices,
    habit_proposal_keep_indices,
    habit_prior_from_need,
    habit_root_action_proposal_from_need,
    normalize_habit_mode,
    normalize_proposal_mode,
    qpi_for_action_match,
    update_computation_budget,
)
from aif_paper1.agents.internal.preferences import (
    homeostatic_need_signal,
    log_pref_monotonic_low,
    log_pref_range_band,
    normalize_preference_mode,
    validate_preference_band,
)
from aif_paper1.runtime.pymdp_integration import (
    assert_pymdp_capabilities,
    infer_and_plan_compat,
    update_empirical_prior_compat,
)
from aif_paper1.agents.internal.runtime_support import (
    _evaluate_efe_policy_subset,
    _flatten_b_action_dims_for_init,
    _flatten_policies_from_action_maps,
    _infer_states_from_arrays,
    _QUIET_SINK,
    _policy_array_np,
    _validate_search_planner_pymdp_interface,
)


class Agent:
    """
    Non-attentional Active Inference agent.

    Hidden state factors (indices):
        0: position_state      (n_pos)
        1: season_state        (n_season)
        2: energy_state        (n_energy)
        3: hydration_state     (n_hydration)
        4: sleep_debt_state    (n_sleep)
        5: oxygen_state        (n_oxygen)

    Observation factors (indices):
        0: position_obs
        1: resource_obs
        2: season_cue_obs
        3: hunger_obs          (depends on energy_state)
        4: thirst_obs          (depends on hydration_state)
        5: fatigue_obs         (depends on sleep_debt_state)
        6: suffocation_obs     (depends on oxygen_state)
    """

    @property
    def actinf_runtime_backend(self) -> str:
        """Compatibility alias for the maintained private runtime-backend field."""
        return str(getattr(self, "_actinf_runtime_backend", "n/a"))

    @actinf_runtime_backend.setter
    def actinf_runtime_backend(self, value: str) -> None:
        self._actinf_runtime_backend = str(value)

    def __init__(
        self,
        env,
        body: BodyProcess,
        *,
        planning_horizon: int = 1,
        planner: str = "efe",
        policy_mode: str = "full",
        resource_true_prob: float = 0.95,
        unknown_model: bool = False,
        unknown_model_init_concentration: float = 1.0,
        unknown_model_learning_rate: float = 1.0,
        unknown_model_learn_resource: bool = True,
        unknown_model_learn_intero: bool = True,
        unknown_model_update_online: bool = True,
        season_stay_prob: float = 0.95,
        preference_scale: float = 4.0,
        intero_preference_mode: str = "monotonic_low",
        intero_range_hunger_min: float = 0.0,
        intero_range_hunger_max: float = 0.2,
        intero_range_thirst_min: float = 0.0,
        intero_range_thirst_max: float = 0.2,
        intero_range_fatigue_min: float = 0.0,
        intero_range_fatigue_max: float = 0.2,
        intero_range_suffocation_min: float = 0.0,
        intero_range_suffocation_max: float = 0.2,
        dynamic_resource_preferences: bool = False,
        dynamic_intero_preferences: bool = False,
        dynamic_intero_pref_amplifier: float = 2.0,
        dynamic_pref_source: str = "belief",
        allow_oracle_preference_source: bool = False,
        oracle_state_access: bool = False,
        dynamic_pref_critical_penalty: float = -500.0,
        dynamic_pref_empty_bias: float = -1.0,
        dynamic_pref_poison_penalty: float | None = None,
        dynamic_pref_food_critical: int | None = None,
        dynamic_pref_water_critical: int | None = None,
        dynamic_pref_sleep_critical: int | None = None,
        physio_slip: float = 0.02,
        habit_mode: str = "none",
        habit_prior_weight: float = 1.0,
        habit_temperature: float = 1.0,
        habit_prune_mass_threshold: float = 0.95,
        habit_prune_min_keep: int = 64,
        habit_prune_max_keep: int = 512,
        habit_prune_keep_per_action: int = 1,
        proposal_mode: str = "none",
        proposal_mass_threshold: float = 0.90,
        proposal_min_keep: int = 4,
        proposal_max_keep: int = 12,
        proposal_keep_per_move: int = 1,
        habit_option_commit_steps: int = 3,
        habit_option_max_steps: int = 8,
        habit_hybrid_mix: float = 0.25,
        compute_budget_max: float = 1.0,
        compute_budget_init: float | None = None,
        compute_budget_recovery: float = 0.05,
        compute_budget_planner_cost: float = 0.10,
        compute_budget_habit_cost: float = 0.02,
        compute_budget_switch_cost: float = 0.01,
        compute_budget_threshold: float = 0.25,
        use_states_info_gain: bool = True,
        num_iter: int = 8,
        store_diagnostics: bool = True,
        jit_inference: bool = False,
        batch_size: int = 1,
    ):
        self.env = env
        self.body = body
        self.preference_scale = float(preference_scale)
        self.intero_preference_mode = normalize_preference_mode(intero_preference_mode)
        self.intero_range_hunger_min, self.intero_range_hunger_max = validate_preference_band(
            intero_range_hunger_min,
            intero_range_hunger_max,
            label="intero_range_hunger",
        )
        self.intero_range_thirst_min, self.intero_range_thirst_max = validate_preference_band(
            intero_range_thirst_min,
            intero_range_thirst_max,
            label="intero_range_thirst",
        )
        self.intero_range_fatigue_min, self.intero_range_fatigue_max = validate_preference_band(
            intero_range_fatigue_min,
            intero_range_fatigue_max,
            label="intero_range_fatigue",
        )
        self.intero_range_suffocation_min, self.intero_range_suffocation_max = validate_preference_band(
            intero_range_suffocation_min,
            intero_range_suffocation_max,
            label="intero_range_suffocation",
        )
        self.dynamic_resource_preferences = bool(dynamic_resource_preferences)
        # Dynamic body-need (intero) preferences: per-step rewriting of
        # C[3..6] based on current body belief. The most-needed channel
        # gets a sharper preference for "low need observation"; less-
        # needed channels get the standard preference. This is the
        # preference-as-attention companion to kappa-as-attention. Used
        # to test whether the kappa mechanism's behavioural benefit can
        # be replicated by dynamic preference reweighting (which would
        # reduce the kappa contribution to "implementation choice")
        # or whether kappa provides additional benefit beyond
        # prioritization (which would strengthen the contribution).
        self.dynamic_intero_preferences = bool(dynamic_intero_preferences)
        self.dynamic_intero_pref_amplifier = float(dynamic_intero_pref_amplifier)
        self.allow_oracle_preference_source = bool(allow_oracle_preference_source)
        self.oracle_state_access = bool(oracle_state_access)
        pref_source = str(dynamic_pref_source).strip().lower()
        if pref_source not in {"belief", "body"}:
            raise ValueError(
                f"Unknown dynamic_pref_source={dynamic_pref_source!r}; expected 'belief'|'body'"
            )
        if pref_source == "body" and not bool(self.allow_oracle_preference_source):
            raise ValueError(
                "dynamic_pref_source='body' is oracle-only and requires "
                "allow_oracle_preference_source=True"
            )
        self.dynamic_pref_source = pref_source
        self.dynamic_pref_critical_penalty = float(dynamic_pref_critical_penalty)
        self.dynamic_pref_empty_bias = float(dynamic_pref_empty_bias)
        self.dynamic_pref_poison_penalty = float(
            -abs(self.preference_scale) if dynamic_pref_poison_penalty is None else dynamic_pref_poison_penalty
        )
        self.dynamic_pref_food_critical = int(
            max(0, int(self.body.MAX_ENERGY) - 1)
            if dynamic_pref_food_critical is None
            else dynamic_pref_food_critical
        )
        self.dynamic_pref_water_critical = int(
            max(0, int(self.body.MAX_HYDRATION) - 1)
            if dynamic_pref_water_critical is None
            else dynamic_pref_water_critical
        )
        self.dynamic_pref_sleep_critical = int(
            max(0, int(self.body.MAX_SLEEPDEBT) - 1)
            if dynamic_pref_sleep_critical is None
            else dynamic_pref_sleep_critical
        )
        self.physio_slip = float(max(0.0, min(physio_slip, 0.99)))
        self.policy_mode = str(policy_mode).lower()
        self.store_diagnostics = bool(store_diagnostics)
        self.jit_inference = bool(jit_inference)
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        self.batch_size = int(batch_size)
        self.planner = str(planner)
        self.planning_horizon = int(planning_horizon)
        self.unknown_model = bool(unknown_model)
        self.unknown_model_learning_rate = float(max(0.0, unknown_model_learning_rate))
        self.unknown_model_update_online = bool(unknown_model_update_online)
        self._unknown_model_init_concentration = float(max(1e-6, unknown_model_init_concentration))
        self._unknown_model_learn_resource = bool(unknown_model_learn_resource)
        self._unknown_model_learn_intero = bool(unknown_model_learn_intero)
        self._pA: list[np.ndarray | None] | None = None
        self._learn_A_mask: list[bool] | None = None
        self._planner_uses_rng = False
        self.habit_mode = normalize_habit_mode(habit_mode)
        self.habit_prior_weight = float(np.clip(float(habit_prior_weight), 0.0, 1.0))
        self.habit_temperature = float(max(1e-3, float(habit_temperature)))
        self.habit_prune_mass_threshold = float(np.clip(float(habit_prune_mass_threshold), 0.0, 1.0))
        self.habit_prune_min_keep = int(max(1, int(habit_prune_min_keep)))
        self.habit_prune_max_keep = int(habit_prune_max_keep)
        self.habit_prune_keep_per_action = int(max(0, int(habit_prune_keep_per_action)))
        self.proposal_mode = normalize_proposal_mode(proposal_mode)
        self.proposal_mass_threshold = float(np.clip(float(proposal_mass_threshold), 0.0, 1.0))
        self.proposal_min_keep = int(max(1, int(proposal_min_keep)))
        self.proposal_max_keep = int(proposal_max_keep)
        self.proposal_keep_per_move = int(max(0, int(proposal_keep_per_move)))
        self.habit_option_commit_steps = int(max(1, int(habit_option_commit_steps)))
        self.habit_option_max_steps = int(max(1, int(habit_option_max_steps)))
        self.habit_hybrid_mix = float(np.clip(float(habit_hybrid_mix), 0.0, 1.0))
        self.compute_budget_max = float(max(1e-6, float(compute_budget_max)))
        init_budget = self.compute_budget_max if compute_budget_init is None else float(compute_budget_init)
        self.compute_budget = float(np.clip(init_budget, 0.0, self.compute_budget_max))
        self.compute_budget_recovery = float(max(0.0, float(compute_budget_recovery)))
        self.compute_budget_planner_cost = float(max(0.0, float(compute_budget_planner_cost)))
        self.compute_budget_habit_cost = float(max(0.0, float(compute_budget_habit_cost)))
        self.compute_budget_switch_cost = float(max(0.0, float(compute_budget_switch_cost)))
        self.compute_budget_threshold = float(np.clip(float(compute_budget_threshold), 0.0, self.compute_budget_max))
        self._compute_budget_init = float(self.compute_budget)
        self._active_option: OptionState | None = None
        self._last_control_source = "planner"
        self.last_habit: dict | None = None
        # Per-batch-element habit + budget state (P5.4.6). At bs=1 these
        # mirror the scalar fields; at bs > 1 each element advances its
        # own option commitment and consumes its own compute budget.
        if self.batch_size > 1:
            self._compute_budget_per_b: np.ndarray = np.full(
                self.batch_size, self.compute_budget, dtype=np.float32
            )
            self._active_option_per_b: list[OptionState | None] = [None] * self.batch_size
            self._last_control_source_per_b: list[str] = ["planner"] * self.batch_size
            self._last_habit_per_b: list[dict | None] = [None] * self.batch_size

        # Sizes for each factor
        self.n_pos = self.env.w * self.env.h
        self.n_season = len(Season)
        self.n_res = len(Resource)
        self.n_energy = self.body.MAX_ENERGY + 1
        self.n_hydration = self.body.MAX_HYDRATION + 1
        self.n_sleep = self.body.MAX_SLEEPDEBT + 1
        self.n_oxygen = self.body.MAX_OXYGEN + 1
        self.n_typeII = self.body.TYPEII_MAX_LEVEL + 1

        # Precompute categorical observation lookup tables shaped (N,1,N) so
        # step() can feed upstream `pymdp` directly when categorical_obs=True.
        def _oh(n: int) -> np.ndarray:
            return np.eye(n, dtype=np.float32)[:, None, :]

        self._oh_pos = _oh(self.n_pos)
        self._oh_res = _oh(self.n_res)
        self._oh_season_obs = _oh(self.n_season + 1)

        self._oh_typeII = _oh(self.n_typeII)

        # Build A/B/C/D arrays (lists of numpy arrays, one per factor)
        known_A = self._build_A(resource_true_prob)
        self._init_unknown_model_state(known_A)
        self.B = self._build_B(season_stay_prob)
        self.C = self._build_C()
        self.D = self._build_D()

        # dependencies for pymdp: which hidden factors each observation and
        # transition factor depends on.
        self.A_dependencies = [
            [0],          # position_obs      <- position_state
            [0, 1],       # resource_obs      <- (position_state, season_state)
            [0, 1],       # season_cue_obs    <- (position_state, season_state)
            [2],          # hunger_obs        <- energy_state
            [3],          # thirst_obs        <- hydration_state
            [4],          # fatigue_obs       <- sleep_debt_state
            [5],          # suffocation_obs   <- oxygen_state
        ]
        self.B_dependencies = [
            [0],           # position_state   <- position_state (+ action)
            [1],           # season_state     <- season_state
            [2, 0, 1],     # energy_state     <- (energy_state, pos, season, interact)
            [3, 0, 1],     # hydration_state  <- (hydration_state, pos, season, interact)
            [4, 0, 1],     # sleep_debt_state <- (sleep_state, pos, season, interact)
            [5, 0, 1],     # oxygen_state     <- (oxygen_state, pos, season, interact)
        ]
        # Which control factors act on which B[factor].
        # num_controls = [len(Move), len(Interact)] below.
        self.B_action_dependencies = [
            [0],    # position_state controlled by Move
            [],     # season_state uncontrolled
            [0, 1],    # energy_state controlled by (Move, Interact)
            [0, 1],    # hydration_state controlled by (Move, Interact)
            [0, 1],    # sleep_state controlled by (Move, Interact)
            [0, 1],    # oxygen_state controlled by (Move, Interact)
        ]

        assert_pymdp_capabilities()

        from aif_paper1.planning.policy_lib import baseline_policies
        from aif_paper1.planning.policy_space import _normalize_planner_name, baseline_action_library
        from aif_paper1.planning.routing import (
            get_planner_route,
            resolve_actinf_runtime_backend,
            validate_planner_policy_mode,
        )

        planner_name = _normalize_planner_name(self.planner)
        validate_planner_policy_mode(planner_name, self.policy_mode)
        planner_route = get_planner_route(planner_name)
        self._planner_route = planner_route
        self.planner = planner_name
        self.planner_backend = str(planner_route.planner_backend)
        if str(self.proposal_mode) != "none":
            raise ValueError(
                "proposal_mode is not available in this snapshot; use proposal_mode='none'."
            )

        agent_policy_len = int(self.planning_horizon) if planner_route.policy_space == "full_policy" else 1

        self._action_multi = None
        agent_B = self.B
        action_maps = None
        policies = None
        if planner_route.policy_space == "full_policy":
            agent_B, action_maps = _flatten_b_action_dims_for_init(
                self.B,
                b_action_dependencies=self.B_action_dependencies,
                num_controls_multi=[int(len(Move)), int(len(Interact))],
            )
            policy_multi = baseline_policies(
                policy_mode=str(self.policy_mode),
                horizon=int(self.planning_horizon),
                n_move=int(len(Move)),
                n_interact=int(len(Interact)),
            )
            policies = _flatten_policies_from_action_maps(
                policy_multi,
                action_maps=action_maps,
            )
        else:
            action_multi = baseline_action_library(
                policy_mode="full",
                n_move=int(len(Move)),
                n_interact=int(len(Interact)),
            )
            self._action_multi = jnp.asarray(action_multi, dtype=jnp.int32)

        # Construct the pymdp.Agent instance.
        self.pymdp_agent = PymdpAgent(
            A=self.A,
            B=agent_B,
            C=self.C,
            D=self.D,
            policies=policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            B_action_dependencies=self.B_action_dependencies,
            num_controls=[len(Move), len(Interact)],
            policy_len=int(agent_policy_len),
            categorical_obs=True,
            action_selection="deterministic",
            sampling_mode="full",
            inference_algo="fpi",
            use_states_info_gain=bool(use_states_info_gain),
            num_iter=int(num_iter),
            batch_size=int(self.batch_size),
        )
        # Lift learning state to per-batch-element shape now that pymdp has
        # broadcast its A/B/C/D copies to ``(batch_size, *)``.
        self._promote_learning_state_to_batched()
        if planner_route.policy_space == "full_policy":
            object.__setattr__(self.pymdp_agent, "action_maps", action_maps)
        else:
            _validate_search_planner_pymdp_interface(
                planner_name=planner_name,
                pymdp_agent=self.pymdp_agent,
                b_action_dependencies=self.B_action_dependencies,
                num_controls_multi=[int(len(Move)), int(len(Interact))],
                label=self.__class__.__name__,
            )
        self.actinf_runtime_backend = resolve_actinf_runtime_backend(
            planner=self.planner,
            oracle_state_access=bool(self.oracle_state_access),
            external_kappa=False,
            gate_active=False,
            habit_mode=str(self.habit_mode),
            proposal_mode=str(self.proposal_mode),
            unknown_model=bool(self.unknown_model),
            dynamic_resource_preferences=bool(self.dynamic_resource_preferences),
        )

        # Optionally JIT key inference calls. This can be a major speedup for large policy sets.
        def _infer_states(observation, prior, A_runtime=None):
            return _infer_states_from_arrays(
                A_base=self.pymdp_agent.A,
                B_base=self.pymdp_agent.B,
                observation=observation,
                prior=prior,
                A_runtime=A_runtime,
                A_dependencies=self.pymdp_agent.A_dependencies,
                B_dependencies=self.pymdp_agent.B_dependencies,
                num_iter=int(self.pymdp_agent.num_iter),
                method=str(self.pymdp_agent.inference_algo),
                inference_horizon=self.pymdp_agent.inference_horizon,
            )

        if planner_name == "efe":
            from aif_paper1.planning.efe_repo_local import make_efe_infer_policies

            # The repo-local scorer captures policies/A_dependencies/etc. as
            # closure constants and accepts a ``keep_mask`` keyword for
            # habit-prune subset evaluation. The agent-level ``jit_planner``
            # switch below wraps it in ``eqx.filter_jit`` once for the run.
            _infer_policies = make_efe_infer_policies(self.pymdp_agent, jit=False)
        else:
            raise ValueError(
                f"Only 'efe' planner is available in this snapshot; got {planner!r}."
            )

        self._planner_supports_runtime_overrides = bool(planner_route.supports_runtime_ace)

        def _sample_action(q_pi):
            return self.pymdp_agent.sample_action(q_pi)

        jit_planner = bool(self.jit_inference and planner_route.planner_backend == "repo_local")
        jit_states = bool(self.jit_inference)
        if self.jit_inference:
            self._infer_states = eqx.filter_jit(_infer_states) if jit_states else _infer_states
            self._infer_policies = eqx.filter_jit(_infer_policies) if jit_planner else _infer_policies
            self._sample_action = _sample_action
        else:
            self._infer_states = _infer_states
            self._infer_policies = _infer_policies
            self._sample_action = _sample_action

        self.factor_names = [
            "position_state",
            "season_state",
            "energy_state",
            "hydration_state",
            "sleep_debt_state",
            "oxygen_state",
        ]
        self.reset()

    def _consume_planner_debug_state(self) -> dict[str, object]:
        """Read one-step planner diagnostics from the current planner wrapper."""
        state = getattr(self._infer_policies, "debug_state", None)
        if not isinstance(state, dict):
            return {}
        info = state.get("last_info")
        return dict(info) if isinstance(info, dict) else {}

    def _next_planner_key_batched(self):
        """Rotate ``self._planner_key`` and return the consumed key in the
        right shape for the current batch_size (Phase 1.2).

        At ``bs=1`` returns a ``(2,)`` key, matching the legacy
        ``_step_bs1`` contract (``self._planner_key, key = jr.split(...)``).
        At ``bs > 1`` returns a ``(B, 2)`` key array, with each per-element
        key derived independently via a per-element ``jr.split``. This
        closes the gap where ``_step_batched`` was passing ``rng_key=None``
        to every planner call even when ``self._planner_uses_rng`` was
        True — the agent already produced a ``(B, 2)`` key at construction
        but was throwing it away.

        Returns ``None`` when ``self._planner_uses_rng`` is False (the
        default for ``efe`` / ``beam``); planners that ignore ``rng_key``
        are unaffected.
        """
        if not bool(getattr(self, "_planner_uses_rng", False)):
            return None
        if self.batch_size == 1:
            self._planner_key, used = jr.split(self._planner_key)
            return used
        # ``self._planner_key`` has shape (B, 2). vmap a per-element
        # ``jr.split`` to derive (B, 2, 2) — first axis is the rotated
        # key, second is the consumed key for this call.
        split = jax.vmap(jr.split)(self._planner_key)
        self._planner_key = split[:, 0]
        return split[:, 1]

    def _apply_policy_mode(self) -> None:
        """
        Optionally restrict the set of candidate policies for tractability.

        Modes
        -----
        - "full": keep pymdp's full enumeration of (Move, Interact)^H.
        - "move_or_act": enforce that any non-NONE Interact must pair with Move.REST
          at that timestep (reduces policy count without changing model structure).
        - "competence": keep timestep-0 unconstrained, but force all *future* moves
          (t>=1 within the planning horizon) to be Move.REST. This keeps the agent's
          immediate action space intact while reducing lookahead branching.
        - "navigate": keep timestep-0 unconstrained, but restrict *future* steps to
          emphasize multi-step navigation:
            * intermediate steps (1..H-2): Interact.NONE
          This approximates "navigate then act" (i.e., defer interaction until the
          end of the lookahead) without full (Move, Interact)^H blowup.
        """
        mode = str(self.policy_mode).lower()
        if mode in {"full", "", "none"}:
            return
        if mode not in {"move_or_act", "competence", "navigate"}:
            raise ValueError(
                f"Unknown policy_mode={mode!r}; expected 'full'|'move_or_act'|'competence'|'navigate'"
            )

        pol = _policy_array_np(self.pymdp_agent)
        if pol.ndim != 3:
            raise ValueError(f"Unexpected policies shape {pol.shape}; expected (n_policies, H, n_actions)")

        # Policies may be encoded either per control factor (Move, Interact) or
        # per hidden factor (after flattening B action dependencies). Support both.
        if pol.shape[-1] == 2:
            move = pol[..., 0]
            interact = pol[..., 1]
        else:
            # In complex-action-deps mode, policies are encoded per hidden factor.
            # `move` is directly available from the position factor (controlled by Move).
            move = pol[..., 0]
            # `interact` is encoded inside any physiology factor that depends on (Move, Interact),
            # using the same base conversion as pymdp.utils.get_combination_index:
            #   joint = move * n_interact + interact
            joint = pol[..., 2]
            interact = joint % int(len(Interact))
            if os.environ.get("AFFECTWORLD_DEBUG_ACTION") == "1":
                move2 = joint // int(len(Interact))
                if not np.all(move2 == move):
                    print("WARNING: decoded move from joint differs from position move in policies")

        if mode == "move_or_act":
            keep_t = (interact == int(Interact.NONE)) | (move == int(Move.REST))
            keep = np.all(keep_t, axis=1)
        elif mode == "competence":
            # competence: constrain only future steps, never the first action executed
            if pol.shape[1] <= 1:
                return
            keep = np.all(move[:, 1:] == int(Move.REST), axis=1)
        else:
            # navigate: constrain only future steps, never the first action executed
            H = int(pol.shape[1])
            if H <= 1:
                return
            keep = np.ones(int(move.shape[0]), dtype=bool)
            # For intermediate steps, only allow movement (no interactions) so plans
            # can represent multi-step navigation to a target tile.
            if H > 2:
                keep &= np.all(interact[:, 1:-1] == int(Interact.NONE), axis=1)

        if not np.any(keep):
            raise RuntimeError("policy_mode filtering removed all policies")

        new_policies = jnp.asarray(pol[keep], dtype=jnp.int32)
        n_policies = int(new_policies.shape[0])
        new_E = jnp.ones((self.pymdp_agent.batch_size, n_policies), dtype=jnp.float32) / float(n_policies)
        policy_store = control.Policies(new_policies)
        object.__setattr__(self.pymdp_agent, "policies", policy_store)
        object.__setattr__(self.pymdp_agent, "E", new_E)

    # ------------------------------------------------------------------
    # Likelihoods: A matrices
    # ------------------------------------------------------------------
    def _build_A(self, prob_true: float) -> List[np.ndarray]:
        """
        Build likelihood (observation) matrices for each observation factor.

        Shapes:
          A_pos          : (n_pos, n_pos)
          A_res          : (n_res, n_pos, n_season)
          A_season_cue   : (n_season+1, n_pos, n_season)
          A_hunger       : (n_typeII, n_energy)
          ...
        """
        # 1) Position is observed deterministically.
        A_pos = np.eye(self.n_pos, dtype=np.float32)

        # 2) Resource: noisy categorical observation conditioned on (pos, season).
        A_res = np.zeros((self.n_res, self.n_pos, self.n_season), dtype=np.float32)
        prob_false = (1.0 - prob_true) / (self.n_res - 1)
        for p in range(self.n_pos):
            for s in range(self.n_season):
                true_res = int(self.env.res_map[p, s])
                A_res[:, p, s] = prob_false
                A_res[true_res, p, s] = prob_true

        # 3) Season cue: only informative on hill; otherwise "NoContext".
        A_season_cue = np.zeros(
            (self.n_season + 1, self.n_pos, self.n_season), dtype=np.float32
        )
        for p in range(self.n_pos):
            for s in range(self.n_season):
                if p == self.env.hill_idx:
                    A_season_cue[s + 1, p, s] = 1.0  # actual context
                else:
                    A_season_cue[0, p, s] = 1.0      # NoContext

        # 4) Interoceptive channels: Type II outcomes derived from Type I hidden states.
        # Align likelihood noise with BodyProcess sampling.
        p_base = float(self.body.INTERO_BASE_PRECISION)

        def typeII_mapping(parent_levels: int, level_fn) -> np.ndarray:
            off = (1.0 - p_base) / (self.n_typeII - 1)
            A = np.full((self.n_typeII, parent_levels), off, dtype=np.float32)
            for parent in range(parent_levels):
                lvl = int(level_fn(parent))
                A[:, parent] = off
                A[lvl, parent] = p_base
            return A

        A_hunger = typeII_mapping(
            self.n_energy,
            lambda e: self._discretise((self.body.MAX_ENERGY - e) / self.body.MAX_ENERGY),
        )
        A_thirst = typeII_mapping(
            self.n_hydration,
            lambda h: self._discretise(
                (self.body.MAX_HYDRATION - h) / self.body.MAX_HYDRATION
            ),
        )
        A_fatigue = typeII_mapping(
            self.n_sleep,
            lambda s: self._discretise(s / self.body.MAX_SLEEPDEBT),
        )
        A_suffocation = typeII_mapping(
            self.n_oxygen,
            lambda o: self._discretise(
                (self.body.MAX_OXYGEN - o) / self.body.MAX_OXYGEN
            ),
        )

        return [
            A_pos, A_res, A_season_cue,
            A_hunger, A_thirst, A_fatigue, A_suffocation,
        ]

    def _init_unknown_model_state(self, known_A: List[np.ndarray]) -> None:
        """
        Initialize learnable A-model state.

        Unknown-model mode uses a uniform Dirichlet prior over selected modalities
        and keeps non-selected modalities fixed at known dynamics.
        """
        self._learn_A_mask = [False] * int(len(known_A))
        if self.unknown_model:
            if self._unknown_model_learn_resource and len(self._learn_A_mask) > 1:
                self._learn_A_mask[1] = True
            if self._unknown_model_learn_intero:
                for m in range(3, len(self._learn_A_mask)):
                    self._learn_A_mask[m] = True

        self._pA = [None] * int(len(known_A))
        # Always allocate per-element here. ``_promote_learning_state_to_batched``
        # runs after PymdpAgent construction to add the leading batch axis when
        # ``batch_size > 1``. We cannot do that here because pymdp's Agent
        # constructor only broadcasts ``C``/``D`` when ``A`` arrives without a
        # leading batch axis (see pymdp/agent.py:228-247): if we hand it a
        # pre-batched A, C silently stays per-element and the EFE scorer hits
        # a shape mismatch.
        self.A = [np.array(a, dtype=np.float32, copy=True) for a in known_A]

        if not self.unknown_model:
            return

        init_c = float(self._unknown_model_init_concentration)
        for m, learn in enumerate(self._learn_A_mask):
            if not bool(learn):
                continue
            # why: unknown-model experiments start from uninformative likelihood columns to
            # isolate SI-vs-SL learning effects from privileged map/intero priors.
            counts = np.full_like(np.asarray(known_A[m], dtype=np.float32), init_c, dtype=np.float32)
            self._pA[m] = counts
            den = np.maximum(counts.sum(axis=0, keepdims=True), 1e-12)
            self.A[m] = (counts / den).astype(np.float32, copy=False)

    def _promote_learning_state_to_batched(self) -> None:
        """Lift ``self.A`` and ``self._pA`` to per-batch-element shape.

        Called after ``PymdpAgent`` construction when ``self.batch_size > 1``.
        Each element starts identical to the per-element prior; the per-step
        ``_online_model_update_batched`` then drives them to diverge based on
        per-element observation and posterior streams.
        """
        if self.batch_size == 1:
            return
        bs = int(self.batch_size)
        self.A = [
            np.broadcast_to(np.asarray(a)[None, ...], (bs, *np.asarray(a).shape)).copy()
            for a in self.A
        ]
        if self._pA is not None:
            self._pA = [
                None
                if pA_m is None
                else np.broadcast_to(np.asarray(pA_m)[None, ...], (bs, *np.asarray(pA_m).shape)).copy()
                for pA_m in self._pA
            ]

    def _planner_pA_getter(self) -> list[np.ndarray | None] | None:
        """Expose the current Dirichlet counts for planners that branch model updates."""
        if self._pA is None:
            return None
        return self._pA

    def _latest_qs_for_learning(self, qs) -> list[np.ndarray]:
        """
        Normalize pymdp posterior outputs to one probability vector per factor.

        pymdp uses several shapes depending on batch/time settings. For online
        learning we only need the latest marginal for each factor, so this
        helper strips batch/time axes and returns normalized ``float32`` vectors.
        """
        out: list[np.ndarray] = []
        bs = int(getattr(self.pymdp_agent, "batch_size", 1))
        for q in qs:
            arr = np.asarray(q, dtype=np.float32)
            if arr.ndim == 1:
                vec = arr
            elif arr.ndim == 2:
                vec = arr[0] if arr.shape[0] == bs else arr[-1]
            else:
                vec = arr[0, -1]
            vec = np.clip(vec, 1e-8, None)
            vec = vec / max(1e-8, float(vec.sum()))
            out.append(vec.astype(np.float32, copy=False))
        return out

    def _online_model_update(self, observation, qs) -> None:
        """
        Apply one online likelihood-learning update to selected ``A`` modalities.

        This is deliberately outside the textbook fixed-model ActInf loop. The
        update uses the latest posterior over each modality's parent factors to
        accumulate Dirichlet-style counts, then refreshes both ``self.A`` and
        the runtime ``pymdp`` agent. Under ``batch_size > 1`` each element
        learns independently from its own observation/qs slice.
        """
        if (not self.unknown_model) or (not self.unknown_model_update_online):
            return
        if self._pA is None or self._learn_A_mask is None:
            return

        lr = float(max(0.0, self.unknown_model_learning_rate))
        if lr <= 0.0:
            return

        if self.batch_size == 1:
            self._online_model_update_bs1(observation, qs, lr)
        else:
            self._online_model_update_batched(observation, qs, lr)

    def _online_model_update_bs1(self, observation, qs, lr: float) -> None:
        q_latest = self._latest_qs_for_learning(qs)
        updated_modalities: list[int] = []
        for m, counts_m in enumerate(self._pA):
            if counts_m is None:
                continue
            if not bool(self._learn_A_mask[m]):
                continue

            obs_vec = np.asarray(observation[m], dtype=np.float32).reshape(-1)
            if obs_vec.size != int(counts_m.shape[0]):
                continue
            deps = list(self.A_dependencies[m])
            dep_mass = q_latest[int(deps[0])]
            for dep in deps[1:]:
                dep_mass = dep_mass[..., None] * q_latest[int(dep)]
            contrib = obs_vec[(slice(None),) + (None,) * int(dep_mass.ndim)] * dep_mass[None, ...]
            counts_new = np.maximum(np.asarray(counts_m, dtype=np.float32) + lr * contrib, 1e-8)
            self._pA[m] = counts_new.astype(np.float32, copy=False)
            den = np.maximum(counts_new.sum(axis=0, keepdims=True), 1e-12)
            self.A[m] = (counts_new / den).astype(np.float32, copy=False)
            updated_modalities.append(int(m))

        for m in updated_modalities:
            self.pymdp_agent.A[m] = jnp.asarray(self.A[m], dtype=jnp.float32)[None, ...]

    def _online_model_update_batched(self, observation, qs, lr: float) -> None:
        bs = int(self.batch_size)
        q_latest_per_b = self._latest_qs_per_batch(qs)
        updated_modalities: list[int] = []
        for m, counts_m in enumerate(self._pA):
            if counts_m is None:
                continue
            if not bool(self._learn_A_mask[m]):
                continue

            obs_arr = np.asarray(observation[m], dtype=np.float32)
            if obs_arr.ndim == 0 or obs_arr.shape[0] != bs:
                # Skip silently — observation does not carry a batch axis;
                # this signals the runner is still single-bundle and should
                # not be exercising bs>1 learning.
                continue
            n_obs = obs_arr.shape[-1]
            if n_obs != int(counts_m.shape[1]):
                continue
            deps = list(self.A_dependencies[m])

            counts_m_view = np.asarray(counts_m, dtype=np.float32)
            for b in range(bs):
                obs_vec_b = obs_arr[b].reshape(-1)
                dep_mass = q_latest_per_b[int(deps[0])][b]
                for dep in deps[1:]:
                    dep_mass = dep_mass[..., None] * q_latest_per_b[int(dep)][b]
                contrib = (
                    obs_vec_b[(slice(None),) + (None,) * int(dep_mass.ndim)]
                    * dep_mass[None, ...]
                )
                counts_new_b = np.maximum(counts_m_view[b] + lr * contrib, 1e-8)
                counts_m_view[b] = counts_new_b
                den_b = np.maximum(counts_new_b.sum(axis=0, keepdims=True), 1e-12)
                self.A[m][b] = (counts_new_b / den_b).astype(np.float32, copy=False)
            self._pA[m] = counts_m_view.astype(np.float32, copy=False)
            updated_modalities.append(int(m))

        for m in updated_modalities:
            self.pymdp_agent.A[m] = jnp.asarray(self.A[m], dtype=jnp.float32)

    def _latest_qs_per_batch(self, qs) -> list[np.ndarray]:
        """Like ``_latest_qs_for_learning`` but preserves the batch axis.

        Returns one ``(B, num_states_f)`` array per factor. Used by the
        ``batch_size > 1`` learning path.
        """
        out: list[np.ndarray] = []
        for q in qs:
            arr = np.asarray(q, dtype=np.float32)
            if arr.ndim == 3:
                vec = arr[:, -1]
            elif arr.ndim == 2:
                # Treat as (T, num_states_f) without a batch axis — promote.
                vec = arr[None, -1]
            else:
                raise ValueError(
                    f"_latest_qs_per_batch: unexpected qs shape {arr.shape}"
                )
            vec = np.clip(vec, 1e-8, None)
            vec = vec / np.maximum(vec.sum(axis=-1, keepdims=True), 1e-8)
            out.append(vec.astype(np.float32, copy=False))
        return out

    def _update_dynamic_resource_preferences(self) -> None:
        """
        Rewrite resource preferences from current homeostatic pressures.

        Unlike the fixed baseline ``C`` vector, this path lets the agent
        reweight FOOD, WATER, and SLEEP each step based on either inferred or
        oracle bodily deficits. The generative model stays the same; only the
        planner's preference landscape changes.
        """
        if not bool(self.dynamic_resource_preferences):
            return

        if str(self.dynamic_pref_source) == "body":
            # why: this preserves MATLAB-style oracle shaping used in SI/SL replication scripts.
            # Core experiments should generally prefer belief-driven shaping to avoid using hidden
            # true body counters directly inside policy preferences.
            t_food = float(max(0, int(self.body.MAX_ENERGY) - int(self.body.energy)))
            t_water = float(max(0, int(self.body.MAX_HYDRATION) - int(self.body.hydration)))
            t_sleep = float(max(0, int(self.body.sleep_debt)))
        else:
            # why: prefer endogenous shaping from inferred hidden states instead of privileged body truth.
            t_food, t_water, t_sleep = self._expected_typeI_deficits_from_beliefs()

        empty = float(self.dynamic_pref_empty_bias)
        pref_food = homeostatic_need_signal(
            float(t_food),
            max_level=float(self.body.MAX_ENERGY),
            mode=self.intero_preference_mode,
            band_min=float(self.intero_range_hunger_min),
            band_max=float(self.intero_range_hunger_max),
        )
        pref_water = homeostatic_need_signal(
            float(t_water),
            max_level=float(self.body.MAX_HYDRATION),
            mode=self.intero_preference_mode,
            band_min=float(self.intero_range_thirst_min),
            band_max=float(self.intero_range_thirst_max),
        )
        pref_sleep = homeostatic_need_signal(
            float(t_sleep),
            max_level=float(self.body.MAX_SLEEPDEBT),
            mode=self.intero_preference_mode,
            band_min=float(self.intero_range_fatigue_min),
            band_max=float(self.intero_range_fatigue_max),
        )

        critical_penalty = float(self.dynamic_pref_critical_penalty)
        # why: this mirrors MATLAB determineObservationPreference urgency gating so urgent deficits
        # suppress competing resource targets during planning.
        if float(t_water) > float(self.dynamic_pref_water_critical):
            pref_food = critical_penalty
            pref_sleep = critical_penalty
            empty = critical_penalty
        if float(t_food) > float(self.dynamic_pref_food_critical):
            pref_water = critical_penalty
            pref_sleep = critical_penalty
            empty = critical_penalty
        if float(t_sleep) > float(self.dynamic_pref_sleep_critical):
            pref_food = critical_penalty
            pref_water = critical_penalty
            empty = critical_penalty

        c_res = np.zeros((int(self.n_res),), dtype=np.float32)
        c_res[int(Resource.AIR)] = float(empty)
        c_res[int(Resource.FOOD)] = float(pref_food)
        c_res[int(Resource.WATER)] = float(pref_water)
        c_res[int(Resource.SLEEP)] = float(pref_sleep)
        c_res[int(Resource.POISON)] = float(self.dynamic_pref_poison_penalty)

        self.C[1] = c_res
        bs = int(getattr(self.pymdp_agent, "batch_size", 1))
        c_jax = jnp.asarray(c_res, dtype=jnp.float32)[None, :]
        self.pymdp_agent.C[1] = jnp.broadcast_to(c_jax, (bs, int(c_res.shape[0])))

    def _update_dynamic_intero_preferences(self) -> None:
        """
        Per-step rewriting of body-need preferences C[3..6] based on
        current body belief.

        For each body channel m ∈ {3,4,5,6} (hunger, thirst, fatigue,
        suffocation), the preference is set to the standard
        "monotonic_low" prefer-low-need shape, but the channel
        currently believed to be most-needed gets its preference
        scale multiplied by ``dynamic_intero_pref_amplifier``
        (default 2.0). This sharpens the planner's penalty for
        being in a deficit state on the most-needed channel —
        analogous to how kappa allocation sharpens the inference-
        time belief about that channel.

        Comparison reference: kappa-as-attention (attn_extk) shapes
        BELIEFS via observation precision; dynamic-C-as-attention
        (this method) shapes PREFERENCES via per-step C update.
        Both produce a body-state-derived prioritization signal at
        the planner; they differ in which generative-model component
        carries the signal.

        Disabled by default (``dynamic_intero_preferences=False``).
        Currently only effective in the Python engine path; the JAX
        engine bakes C in at compile time.
        """
        if not bool(self.dynamic_intero_preferences):
            return
        # Identify the most-needed channel via the same belief signal
        # used by attn_extk's belief_argmax selector.
        try:
            qs_hunger = self._latest_body_factor("hunger")
            qs_thirst = self._latest_body_factor("thirst")
            qs_fatigue = self._latest_body_factor("fatigue")
            qs_suffoc = self._latest_body_factor("suffocation")
        except (AttributeError, IndexError):
            return  # body beliefs not yet initialized

        # Each is the expected need-level (higher = more deficit).
        needs = np.array([qs_hunger, qs_thirst, qs_fatigue, qs_suffoc], dtype=np.float32)
        most_needed_idx = int(np.argmax(needs))

        # Build per-channel preferences. Standard shape for all
        # channels, then amplify the most-needed one.
        amp = float(self.dynamic_intero_pref_amplifier)
        for local_idx, modality_idx in enumerate(range(3, 7)):
            base_pref = self._log_pref_low(self.n_typeII)
            scale = amp if local_idx == most_needed_idx else 1.0
            new_pref = (base_pref * scale).astype(np.float32, copy=False)
            self.C[modality_idx] = new_pref
            # Also reflect into pymdp_agent.C for the planner.
            bs = int(getattr(self.pymdp_agent, "batch_size", 1))
            c_jax = jnp.asarray(new_pref, dtype=jnp.float32)[None, :]
            self.pymdp_agent.C[modality_idx] = jnp.broadcast_to(c_jax, (bs, int(new_pref.shape[0])))

    def _latest_body_factor(self, channel_name: str) -> float:
        """Read the expected value of the named body channel from the
        agent's most recent posterior. Channel names: hunger, thirst,
        fatigue, suffocation."""
        # The body channels are the 3rd-6th factors in qs (indices 2-5
        # in the agent's state-factor list, given the env factor structure).
        name_to_factor = {"hunger": 2, "thirst": 3, "fatigue": 4, "suffocation": 5}
        if channel_name not in name_to_factor:
            return 0.0
        f = name_to_factor[channel_name]
        if not hasattr(self, "qs") or self.qs is None or len(self.qs) <= f:
            return 0.0
        q = np.asarray(self.qs[f], dtype=np.float32)
        if q.ndim == 0:
            return float(q)
        # Expected need-level: weighted sum of state indices
        if q.ndim >= 1:
            q_flat = q if q.ndim == 1 else q[0]
            indices = np.arange(q_flat.shape[-1], dtype=np.float32)
            return float((q_flat * indices).sum() / max(1.0, q_flat.sum()))
        return 0.0

    def _expected_typeI_deficits_from_beliefs(self) -> tuple[float, float, float]:
        """
        Compute expected deficits from current inferred hidden-state marginals.

        Returns
        -------
        (food_deficit, water_deficit, sleep_debt_expected)
        """
        def _expected_level(q: np.ndarray) -> float:
            idx = np.arange(int(q.shape[0]), dtype=np.float32)
            return float(np.dot(np.asarray(q, dtype=np.float32), idx))

        def _prior_vec(f_idx: int) -> np.ndarray:
            d = np.asarray(self.pymdp_agent.D[f_idx], dtype=np.float32)
            if d.ndim == 1:
                v = d
            else:
                v = d[0]
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            v = np.clip(v, 1e-8, None)
            v = v / max(1e-8, float(v.sum()))
            return v

        if self.qs is None:
            exp_energy = _expected_level(_prior_vec(2))
            exp_hydration = _expected_level(_prior_vec(3))
            exp_sleep = _expected_level(_prior_vec(4))
            return (
                float(max(0.0, float(self.body.MAX_ENERGY) - float(exp_energy))),
                float(max(0.0, float(self.body.MAX_HYDRATION) - float(exp_hydration))),
                float(max(0.0, float(exp_sleep))),
            )

        q_latest = self._latest_qs_for_learning(self.qs)
        if len(q_latest) <= 4:
            exp_energy = _expected_level(_prior_vec(2))
            exp_hydration = _expected_level(_prior_vec(3))
            exp_sleep = _expected_level(_prior_vec(4))
            return (
                float(max(0.0, float(self.body.MAX_ENERGY) - float(exp_energy))),
                float(max(0.0, float(self.body.MAX_HYDRATION) - float(exp_hydration))),
                float(max(0.0, float(exp_sleep))),
            )

        exp_energy = _expected_level(q_latest[2])
        exp_hydration = _expected_level(q_latest[3])
        exp_sleep = _expected_level(q_latest[4])

        t_food = max(0.0, float(self.body.MAX_ENERGY) - float(exp_energy))
        t_water = max(0.0, float(self.body.MAX_HYDRATION) - float(exp_hydration))
        t_sleep = max(0.0, float(exp_sleep))
        return float(t_food), float(t_water), float(t_sleep)

    def _planner_runtime_overrides(
        self,
        *,
        pos_obs: int | None = None,
        season_idx: int | None = None,
        typeII_obs: np.ndarray | None = None,
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray | None] | None, jnp.ndarray | None, dict]:
        """
        Assemble the per-step tensors that may differ from the constructor state.

        This bundles together learned ``A`` matrices, dynamically rewritten
        preferences ``C``, optional Dirichlet counts ``pA``, and any habit-based
        policy prior ``E`` override so the planner can consume a current view of
        the agent without rebuilding the whole ``pymdp`` object.
        """
        runtime_A = [jnp.asarray(a_m, dtype=jnp.float32) for a_m in self.A]
        runtime_C = [jnp.asarray(c_m, dtype=jnp.float32) for c_m in self.C]
        runtime_pA = None
        if self.unknown_model and self._pA is not None:
            runtime_pA = [None if x is None else jnp.asarray(x, dtype=jnp.float32) for x in self._pA]
        runtime_E = None
        habit_info: dict = {"mode": str(self.habit_mode)}
        if pos_obs is not None and season_idx is not None and typeII_obs is not None:
            runtime_E, habit_info = self._habit_prior_override(
                pos_obs=int(pos_obs),
                season_idx=int(season_idx),
                typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
            )
        return runtime_A, runtime_C, runtime_pA, runtime_E, habit_info

    def _state_runtime_A_override(self) -> list[jnp.ndarray] | None:
        """Return runtime ``A`` only when online learning has made the constructor copy stale."""
        if not bool(self.unknown_model):
            return None
        return [jnp.asarray(a_m, dtype=jnp.float32) for a_m in self.A]

    def _delta_belief_like(self, q, idx: int) -> jnp.ndarray:
        """
        Return a one-hot belief tensor matching `q`'s shape, with support in the
        final dimension only (state axis).
        """
        arr = np.asarray(q, dtype=np.float32)
        if arr.ndim == 0:
            return jnp.asarray(arr, dtype=jnp.float32)
        n_states = int(arr.shape[-1])
        if n_states <= 0:
            return jnp.asarray(arr, dtype=jnp.float32)
        idx_c = int(np.clip(int(idx), 0, n_states - 1))
        out = np.zeros_like(arr, dtype=np.float32)
        out[..., idx_c] = 1.0
        return jnp.asarray(out, dtype=jnp.float32)

    def _oracle_qs_override(self, qs, *, pos_obs: int, season_idx: int):
        """
        Optional development-only oracle override for hidden-state posteriors.

        This keeps the same policy inference path but replaces inferred beliefs
        with delta masses at the simulator's true hidden states.
        """
        if not bool(self.oracle_state_access):
            return qs
        if qs is None:
            return qs

        season_fallback = int(np.clip(int(season_idx) - 1, 0, int(self.n_season) - 1))
        try:
            pos_true = int(self.env.state[0])
            season_true = int(self.env.state[1])
        except Exception:
            pos_true = int(pos_obs)
            season_true = int(season_fallback)

        idxs = [
            int(pos_true),
            int(season_true),
            int(self.body.energy),
            int(self.body.hydration),
            int(self.body.sleep_debt),
            int(self.body.oxygen),
        ]
        out = list(qs)
        for f, idx in enumerate(idxs):
            if f < len(out):
                out[f] = self._delta_belief_like(out[f], idx)
        return out

    def _habit_resource_map_from_model(self) -> np.ndarray | None:
        """
        Build a best-guess seasonal resource map from the agent's own likelihood A.
        """
        if len(self.A) <= 1:
            return None
        A_res = np.asarray(self.A[1], dtype=np.float32)
        if A_res.ndim != 3:
            return None
        return np.asarray(np.argmax(A_res, axis=0), dtype=np.int32)

    def _estimate_context_index(self, *, pos_obs: int, season_idx: int) -> int:
        """
        Estimate the active season/context index for habit heuristics.

        The hill observation is treated as authoritative when available. Away
        from the hill, the method falls back to the agent's posterior over the
        season factor, then finally to context ``0``.
        """
        n_ctx = int(max(1, self.n_season))
        if int(pos_obs) == int(getattr(self.env, "hill_idx", -1)) and int(season_idx) > 0:
            return int(np.clip(int(season_idx) - 1, 0, n_ctx - 1))
        if self.qs is not None:
            try:
                q_latest = self._latest_qs_for_learning(self.qs)
                if len(q_latest) > 1:
                    return int(np.clip(int(np.argmax(np.asarray(q_latest[1], dtype=np.float32))), 0, n_ctx - 1))
            except Exception:
                pass
        return 0

    def _habit_prior_override(
        self,
        *,
        pos_obs: int,
        season_idx: int,
        typeII_obs: np.ndarray,
    ) -> tuple[jnp.ndarray | None, dict]:
        """
        Build a runtime ``E`` prior that nudges policy inference toward habits.

        This is a planner-side bias, not a model change: it leaves ``A/B/C/D``
        untouched and instead reweights policies whose first step appears useful
        for the currently dominant need.
        """
        mode = str(self.habit_mode)
        info: dict[str, object] = {
            "mode": mode,
            "need_idx": int(dominant_need_idx(typeII_obs)),
        }
        if mode not in {"prior", "hybrid"}:
            return None, info

        prior, prior_info = self._habit_policy_proposal(
            pos_obs=int(pos_obs),
            season_idx=int(season_idx),
            typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
            resource_likelihood=None,
        )
        if prior is None:
            return None, info

        info.update(prior_info)
        base = np.asarray(self.pymdp_agent.E[0], dtype=np.float32).reshape(-1)
        if prior.shape != base.shape:
            raise ValueError(f"habit prior shape mismatch: got={prior.shape} expected={base.shape}")
        mix_w = float(self.habit_prior_weight)
        mixed = (1.0 - mix_w) * base + mix_w * np.asarray(prior, dtype=np.float32)
        mixed = np.clip(mixed, 1e-8, None)
        mixed = mixed / max(1e-8, float(mixed.sum()))
        info.update(
            {
                "mode": mode,
                "prior_weight": float(mix_w),
                "decision_role": "posterior_bias",
            }
        )
        return jnp.asarray(mixed, dtype=jnp.float32), info

    def _habit_policy_proposal(
        self,
        *,
        pos_obs: int,
        season_idx: int,
        typeII_obs: np.ndarray,
        resource_likelihood: np.ndarray | None,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        """
        Build the habit heuristic as a proposal over the full policy library.

        This helper is shared by the legacy `prior` mode and the cleaner
        `prior_prune` variant. The difference is what happens *after* this
        proposal is formed: `prior` injects it into `E(pi)`, while
        `prior_prune` only uses it to retain candidate policies before plain EFE
        ranking.
        """
        policies = np.asarray(_policy_array_np(self.pymdp_agent), dtype=np.int32)
        ctx_idx = self._estimate_context_index(pos_obs=int(pos_obs), season_idx=int(season_idx))
        need_idx = int(dominant_need_idx(typeII_obs))
        if resource_likelihood is None and len(self.A) > 1:
            resource_likelihood = np.asarray(self.A[1], dtype=np.float32)
        prior = habit_prior_from_need(
            policies,
            env=self.env,
            pos=int(pos_obs),
            ctx=int(ctx_idx),
            resource_map=None,
            resource_likelihood=resource_likelihood,
            need_idx=need_idx,
            has_kappa=False,
            n_interact=int(len(Interact)),
            temperature=float(self.habit_temperature),
        )
        return prior, {
            "mode": str(self.habit_mode),
            "need_idx": int(need_idx),
            "ctx_idx": int(ctx_idx),
            "resource_prior_source": "A_likelihood",
            "resource_prior_unknown_model": bool(self.unknown_model),
            "proposal_role": "heuristic_candidate_proposal",
        }

    def _serialize_root_action_candidates(
        self,
        *,
        keep_idx: np.ndarray,
        proposal: np.ndarray,
    ) -> list[dict[str, object]]:
        """Serialize retained root actions for logs and metadata."""
        if self._action_multi is None:
            return []
        action_arr = np.asarray(self._action_multi, dtype=np.int32)
        probs = np.asarray(proposal, dtype=np.float32).reshape(-1)
        rows: list[dict[str, object]] = []
        for idx in np.asarray(keep_idx, dtype=np.int32).reshape(-1).tolist():
            row: dict[str, object] = {
                "action": int(idx),
                "move": int(action_arr[idx, 0]),
                "interact": int(action_arr[idx, 1]),
                "proposal_prob": float(probs[idx]),
            }
            if int(action_arr.shape[1]) >= 3:
                row["need"] = int(action_arr[idx, 2])
            rows.append(row)
        rows.sort(key=lambda item: -float(item.get("proposal_prob", 0.0)))
        return rows

    def _habit_root_action_proposal(
        self,
        *,
        pos_obs: int,
        season_idx: int,
        typeII_obs: np.ndarray,
        resource_likelihood: np.ndarray | None,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        """Build a need-conditioned proposal over the one-step root action library."""
        if self._action_multi is None:
            return None, {
                "mode": str(self.proposal_mode),
                "reason": "no_one_step_action_library",
            }
        ctx_idx = self._estimate_context_index(pos_obs=int(pos_obs), season_idx=int(season_idx))
        need_idx = int(dominant_need_idx(typeII_obs))
        if resource_likelihood is None and len(self.A) > 1:
            resource_likelihood = np.asarray(self.A[1], dtype=np.float32)
        proposal = habit_root_action_proposal_from_need(
            np.asarray(self._action_multi, dtype=np.int32),
            env=self.env,
            pos=int(pos_obs),
            ctx=int(ctx_idx),
            resource_map=None,
            resource_likelihood=resource_likelihood,
            need_idx=need_idx,
            temperature=float(self.habit_temperature),
        )
        return proposal, {
            "mode": str(self.proposal_mode),
            "need_idx": int(need_idx),
            "ctx_idx": int(ctx_idx),
            "resource_prior_source": "A_likelihood",
            "resource_prior_unknown_model": bool(self.unknown_model),
            "proposal_role": "heuristic_root_action_proposal",
        }

    def _proposal_root_action_override(
        self,
        *,
        pos_obs: int,
        season_idx: int,
        typeII_obs: np.ndarray,
        runtime_A: list[jnp.ndarray] | None,
    ) -> tuple[jnp.ndarray | None, dict[str, object]]:
        """
        Build an optional root-action keep mask for search-planner proposal pruning.

        The heuristic only narrows the candidate root actions. Final ranking
        among retained actions is still produced by the planner's unchanged
        search objective.
        """
        info: dict[str, object] = {
            "mode": str(self.proposal_mode),
            "proposal_role": "inactive",
        }
        if str(self.proposal_mode) != "habit_root_prune":
            return None, info

        resource_likelihood = None
        if runtime_A is not None and len(runtime_A) > 1:
            resource_likelihood = np.asarray(runtime_A[1], dtype=np.float32)
        proposal, proposal_info = self._habit_root_action_proposal(
            pos_obs=int(pos_obs),
            season_idx=int(season_idx),
            typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
            resource_likelihood=resource_likelihood,
        )
        if proposal is None:
            return None, {**info, **proposal_info}

        keep_idx, prune_info = habit_action_proposal_keep_indices(
            proposal,
            action_multi=np.asarray(self._action_multi, dtype=np.int32),
            mass_threshold=float(self.proposal_mass_threshold),
            min_keep=int(self.proposal_min_keep),
            max_keep=int(self.proposal_max_keep),
            keep_per_move=int(self.proposal_keep_per_move),
        )
        merged: dict[str, object] = {
            **proposal_info,
            **prune_info,
            "decision_role": "root_action_proposal_prune_then_beam",
            "kept_root_actions": self._serialize_root_action_candidates(
                keep_idx=keep_idx,
                proposal=np.asarray(proposal, dtype=np.float32),
            ),
        }
        n_actions = int(np.asarray(self._action_multi).shape[0]) if self._action_multi is not None else 0
        if keep_idx.size <= 0 or keep_idx.size >= n_actions:
            merged["applied"] = False
            merged["available"] = int(n_actions)
            merged["kept"] = int(n_actions)
            return None, merged

        mask = np.zeros((n_actions,), dtype=np.bool_)
        mask[np.asarray(keep_idx, dtype=np.int32)] = True
        return jnp.asarray(mask, dtype=jnp.bool_), merged

    def _infer_policies_with_habit_pruning(
        self,
        *,
        qs,
        runtime_A,
        runtime_C,
        pos_obs: int,
        season_idx: int,
        typeII_obs: np.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, object]]:
        """
        Retain a habit-proposed subset, then evaluate that subset with plain EFE.

        This keeps the heuristic out of the final posterior update. The
        proposal acts like a System-1 candidate generator, while the retained
        policies are still ranked by the unchanged EFE objective.
        """
        if str(self.planner).strip().lower() != "efe":
            raise ValueError("habit_mode='prior_prune' is currently implemented only for planner='efe'.")

        resource_likelihood = None
        if runtime_A is not None and len(runtime_A) > 1:
            resource_likelihood = np.asarray(runtime_A[1], dtype=np.float32)
        proposal, proposal_info = self._habit_policy_proposal(
            pos_obs=int(pos_obs),
            season_idx=int(season_idx),
            typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
            resource_likelihood=resource_likelihood,
        )
        if proposal is None:
            q_pi_local, G_local = self._infer_policies(qs, None, runtime_A, runtime_C, None, None)
            return q_pi_local, G_local, {"mode": str(self.habit_mode), "proposal_role": "inactive"}

        policies_full = np.asarray(_policy_array_np(self.pymdp_agent), dtype=np.int32)
        keep_idx, prune_info = habit_proposal_keep_indices(
            proposal,
            policies=policies_full,
            has_kappa=False,
            n_interact=int(len(Interact)),
            mass_threshold=float(self.habit_prune_mass_threshold),
            min_keep=int(self.habit_prune_min_keep),
            max_keep=int(self.habit_prune_max_keep),
            keep_per_first_action=int(self.habit_prune_keep_per_action),
        )
        info: dict[str, object] = {
            **proposal_info,
            **prune_info,
            "decision_role": "proposal_prune_then_efe",
        }
        if keep_idx.size <= 0 or keep_idx.size >= policies_full.shape[0]:
            q_pi_local, G_local = self._infer_policies(qs, None, runtime_A, runtime_C, None, None)
            info["applied"] = False
            info["kept"] = int(policies_full.shape[0])
            info["available"] = int(policies_full.shape[0])
            return q_pi_local, G_local, info

        q_full, G_full = _evaluate_efe_policy_subset(
            infer_policies_fn=self._infer_policies,
            qs=qs,
            keep_idx=keep_idx,
            num_policies=int(self.pymdp_agent.policies.num_policies),
            A_runtime=runtime_A,
            C_runtime=runtime_C,
        )
        return q_full, G_full, info

    def _habit_option_distribution(
        self,
        *,
        pos_obs: int,
        res_obs: int,
        season_idx: int,
        typeII_obs: np.ndarray,
    ) -> tuple[np.ndarray | None, dict]:
        """
        Convert the option controller's recommendation into a policy distribution.

        The option pathway proposes one concrete first-step action. This helper
        turns that into a categorical distribution over the existing policy set
        so it can either replace the planner outright or be mixed with the
        planner posterior in hybrid mode.
        """
        mode = str(self.habit_mode)
        if mode not in {"options", "hybrid"}:
            return None, {"mode": mode}

        need_idx = int(dominant_need_idx(typeII_obs))
        ctx_idx = self._estimate_context_index(pos_obs=int(pos_obs), season_idx=int(season_idx))
        resource_map = self._habit_resource_map_from_model()
        move_idx, interact_idx, opt_need_idx, next_opt, info = choose_option_action(
            self.env,
            pos=int(pos_obs),
            ctx=int(ctx_idx),
            tile_res_obs=int(res_obs),
            resource_map=resource_map,
            suggested_need=int(need_idx),
            active_option=self._active_option,
            commit_steps=int(self.habit_option_commit_steps),
            max_steps=int(self.habit_option_max_steps),
        )
        self._active_option = next_opt

        q_opt = qpi_for_action_match(
            np.asarray(_policy_array_np(self.pymdp_agent), dtype=np.int32),
            move_idx=int(move_idx),
            interact_idx=int(interact_idx),
            has_kappa=False,
            n_interact=int(len(Interact)),
            need_idx=None,
        )
        info.update(
            {
                "mode": mode,
                "ctx_idx": int(ctx_idx),
                "need_idx": int(opt_need_idx),
                "move_idx": int(move_idx),
                "interact_idx": int(interact_idx),
            }
        )
        return np.asarray(q_opt, dtype=np.float32), info

    @staticmethod
    def _normalize_prob_vec(p: np.ndarray) -> np.ndarray:
        """Clip and renormalize a probability-like vector into a valid simplex point."""
        arr = np.asarray(p, dtype=np.float32).reshape(-1)
        arr = np.clip(arr, 1e-8, None)
        return arr / max(1e-8, float(arr.sum()))

    # ------------------------------------------------------------------
    # Transitions: B matrices
    # ------------------------------------------------------------------
    def _build_B(self, stay_prob: float) -> List[np.ndarray]:
        """
        Build transition (state dynamics) matrices for each hidden factor.

        We encode:
          - Deterministic position transitions given Move.
          - Context transitions with fixed stay_prob (not using env's Gaussian).
          - Type I dynamics given (pos, season, Move, Interact), where Interact applies
            on the *destination* tile reached by Move in the same timestep.
          - Type II dynamics as deterministic functions of parent Type I factors.
        """
        # Position transitions: B_pos[next, prev, move]
        B_pos = np.zeros((self.n_pos, self.n_pos, len(Move)), dtype=np.float32)
        for move_idx, move in enumerate(Move):
            for pos_idx in range(self.n_pos):
                nxt = int(self.env._trans[move, pos_idx])
                B_pos[nxt, pos_idx, move_idx] = 1.0

        # Season transitions: simple Markov chain with stay_prob and a single
        # forward step to the next season; this need not match env exactly.
        B_season = np.zeros((self.n_season, self.n_season), dtype=np.float32)
        for s_prev in range(self.n_season):
            s_next = (s_prev + 1) % self.n_season
            B_season[s_prev, s_prev] = stay_prob
            B_season[s_next, s_prev] = 1.0 - stay_prob

        # Type I: destination-tile interaction dynamics.
        # Note: we condition on season_t and marginalize over season_{t+1} using B_season.
        shape = (self.n_pos, self.n_season, len(Move), len(Interact))
        B_energy = np.zeros((self.n_energy, self.n_energy, *shape), dtype=np.float32)
        B_hyd = np.zeros((self.n_hydration, self.n_hydration, *shape), dtype=np.float32)
        B_sleep = np.zeros((self.n_sleep, self.n_sleep, *shape), dtype=np.float32)
        B_oxy = np.zeros((self.n_oxygen, self.n_oxygen, *shape), dtype=np.float32)

        # Optional process noise: spread some mass across other next-states so
        # interoception can correct dead-reckoning and attention has epistemic value.
        slip = float(self.physio_slip)
        p_main = 1.0 - slip
        slip_energy = slip / (self.n_energy - 1) if slip > 0.0 and self.n_energy > 1 else 0.0
        slip_hyd = slip / (self.n_hydration - 1) if slip > 0.0 and self.n_hydration > 1 else 0.0
        slip_sleep = slip / (self.n_sleep - 1) if slip > 0.0 and self.n_sleep > 1 else 0.0
        slip_oxy = slip / (self.n_oxygen - 1) if slip > 0.0 and self.n_oxygen > 1 else 0.0

        # We implement "slip" as (1-slip) on the intended deterministic next state, and
        # slip/(n-1) uniformly on the remaining next states. Since we marginalize over
        # season_{t+1}, we pre-fill each conditional slice with the uniform mass, then
        # add the *correction* term to the deterministic next state for each season_next.
        if slip_energy > 0.0:
            B_energy.fill(slip_energy)
        if slip_hyd > 0.0:
            B_hyd.fill(slip_hyd)
        if slip_sleep > 0.0:
            B_sleep.fill(slip_sleep)
        if slip_oxy > 0.0:
            B_oxy.fill(slip_oxy)

        for pos in range(self.n_pos):
            for season_prev in range(self.n_season):
                for move_idx, move_action in enumerate(Move):
                    dest_pos = int(self.env._trans[move_action, pos])
                    for interact_idx, interact_action in enumerate(Interact):
                        for season_next in range(self.n_season):
                            p_s = float(B_season[season_next, season_prev])
                            if p_s <= 0.0:
                                continue
                            tile_res = Resource(int(self.env.res_map[dest_pos, season_next]))

                            # Energy dynamics
                            for prev in range(self.n_energy):
                                nxt = prev - self.body.DEC_ENERGY
                                if interact_action == Interact.EAT and tile_res == Resource.FOOD:
                                    nxt += self.body.EAT_ENERGY_GAIN
                                if tile_res == Resource.POISON:
                                    nxt -= self.body.POISON_PENALTY
                                nxt = self._clamp(nxt, 0, self.body.MAX_ENERGY)
                                main = (p_main - slip_energy) if self.n_energy > 1 else 1.0
                                B_energy[nxt, prev, pos, season_prev, move_idx, interact_idx] += p_s * main

                            # Hydration dynamics
                            for prev in range(self.n_hydration):
                                nxt = prev - self.body.DEC_HYDRATION
                                if interact_action == Interact.DRINK and tile_res == Resource.WATER:
                                    nxt = self.body.MAX_HYDRATION
                                nxt = self._clamp(nxt, 0, self.body.MAX_HYDRATION)
                                main = (p_main - slip_hyd) if self.n_hydration > 1 else 1.0
                                B_hyd[nxt, prev, pos, season_prev, move_idx, interact_idx] += p_s * main

                            # Sleep debt dynamics
                            for prev in range(self.n_sleep):
                                nxt = prev + self.body.INC_SLEEPDEBT
                                if interact_action == Interact.SLEEP and tile_res == Resource.SLEEP:
                                    nxt -= self.body.SLEEP_DEBT_REDUCE
                                nxt = self._clamp(nxt, 0, self.body.MAX_SLEEPDEBT)
                                main = (p_main - slip_sleep) if self.n_sleep > 1 else 1.0
                                B_sleep[nxt, prev, pos, season_prev, move_idx, interact_idx] += p_s * main

                            # Oxygen dynamics
                            for prev in range(self.n_oxygen):
                                nxt = prev
                                if tile_res == Resource.WATER:
                                    nxt -= self.body.DEC_OXY_WATER
                                else:
                                    nxt += 1
                                if interact_action == Interact.BREATHE and tile_res == Resource.WATER:
                                    nxt = self.body.MAX_OXYGEN
                                nxt = self._clamp(nxt, 0, self.body.MAX_OXYGEN)
                                main = (p_main - slip_oxy) if self.n_oxygen > 1 else 1.0
                                B_oxy[nxt, prev, pos, season_prev, move_idx, interact_idx] += p_s * main

        return [B_pos, B_season, B_energy, B_hyd, B_sleep, B_oxy]

    # ------------------------------------------------------------------
    # Preferences: C vectors
    # ------------------------------------------------------------------
    def _build_C(self) -> List[np.ndarray]:
        zeros = lambda n: np.zeros(n, dtype=np.float32)
        C_pos = zeros(self.n_pos)
        C_res = self._log_pref_resources()
        C_season = zeros(self.n_season + 1)
        if str(self.intero_preference_mode) == "range_band":
            hunger_pref = self._log_pref_band(
                self.n_typeII,
                band_min=self.intero_range_hunger_min,
                band_max=self.intero_range_hunger_max,
            )
            thirst_pref = self._log_pref_band(
                self.n_typeII,
                band_min=self.intero_range_thirst_min,
                band_max=self.intero_range_thirst_max,
            )
            fatigue_pref = self._log_pref_band(
                self.n_typeII,
                band_min=self.intero_range_fatigue_min,
                band_max=self.intero_range_fatigue_max,
            )
            suffoc_pref = self._log_pref_band(
                self.n_typeII,
                band_min=self.intero_range_suffocation_min,
                band_max=self.intero_range_suffocation_max,
            )
        else:
            typeII_pref = self._log_pref_low(self.n_typeII)  # prefer low need level
            hunger_pref = typeII_pref.copy()
            thirst_pref = typeII_pref.copy()
            fatigue_pref = typeII_pref.copy()
            suffoc_pref = typeII_pref.copy()
        return [
            C_pos, C_res, C_season,
            hunger_pref,
            thirst_pref,
            fatigue_pref,
            suffoc_pref,
        ]

    # ------------------------------------------------------------------
    # Priors: D vectors
    # ------------------------------------------------------------------
    def _build_D(self) -> List[np.ndarray]:
        D_pos = np.ones(self.n_pos, dtype=np.float32) / self.n_pos
        D_season = np.ones(self.n_season, dtype=np.float32) / self.n_season
        D_energy = self._soft_delta_at(self.body.MAX_ENERGY, self.n_energy)
        D_hyd = self._soft_delta_at(self.body.MAX_HYDRATION, self.n_hydration)
        D_sleep = self._soft_delta_at(0, self.n_sleep)
        D_oxy = self._soft_delta_at(self.body.MAX_OXYGEN, self.n_oxygen)
        return [D_pos, D_season, D_energy, D_hyd, D_sleep, D_oxy]

    # ------------------------------------------------------------------
    # step-helpers (P5 step 4 prep): pure functions of agent state and the
    # input observation. Extracted from ``step`` so the bs > 1 vectorization
    # in P5.4.2 is a focused diff in this small surface area instead of a
    # mega-edit on the 250-line ``step`` body. At bs=1 (today) these
    # produce results bit-identical to the inlined arithmetic they replaced.
    # ------------------------------------------------------------------
    def _unpack_extero_observation(self, obs) -> tuple:
        """Pull (pos_obs, res_obs, season_idx) from the runner's observation
        tuple. season_obs[2] is one-hot over ``len(Season) + 1``; we use
        the argmax index directly."""
        pos_obs = obs[0]
        res_obs = obs[1]
        season_obs_vec = obs[2]
        season_idx = int(np.argmax(season_obs_vec))
        return pos_obs, res_obs, season_idx

    def _pack_observation_modalities(
        self,
        *,
        pos_obs,
        res_obs,
        season_idx: int,
        typeII_obs,
    ) -> list:
        """Pack the 7-modality categorical observation list pymdp expects.

        Order: [position, resource, season_obs, hunger_typeII, thirst_typeII,
        fatigue_typeII, suffocation_typeII]. At bs=1 each entry is a 1-D
        one-hot vector. P5 step 4.2 will widen this to per-batch lookup.
        """
        t0 = int(typeII_obs[0])
        t1 = int(typeII_obs[1])
        t2 = int(typeII_obs[2])
        t3 = int(typeII_obs[3])
        return [
            self._oh_pos[int(pos_obs)],
            self._oh_res[int(res_obs)],
            self._oh_season_obs[int(season_idx)],
            self._oh_typeII[t0],
            self._oh_typeII[t1],
            self._oh_typeII[t2],
            self._oh_typeII[t3],
        ]

    def _build_empirical_prior(self) -> list:
        """Empirical prior over hidden states for this step.

        First step (no prior action): the agent's flat ``D``. Subsequent
        steps: ``update_empirical_prior_compat`` propagates ``self.qs``
        forward under ``self.prev_action``. Already vmap-aware over the
        leading batch axis for ``batch_size > 1`` — pymdp's
        ``update_empirical_prior`` calls ``vmap(propagate_beliefs)`` over
        the batch so per-batch prev_action / qs flow through naturally
        once Step 4.2 stores them with leading axis ``B``.
        """
        if self.prev_action is None:
            return self.pymdp_agent.D
        return update_empirical_prior_compat(
            self.pymdp_agent, self.prev_action, self.qs,
        )

    # ------------------------------------------------------------------
    # Batched (bs > 1) helpers (P5.4.2). The bs=1 path lives in
    # ``_step_bs1``; ``step`` dispatches on ``self.batch_size``. These
    # helpers cover only the plain-EFE path: online learning, dynamic
    # resource preferences, habit/option arbitration, oracle qs override,
    # and proposal pruning are hard-blocked at bs > 1 in this commit
    # and lifted by P5.4.4-4.6 one at a time.
    # ------------------------------------------------------------------
    def _assert_no_unsupported_bs_n_features(self) -> None:
        """Raise loudly when ``step()`` is called at bs > 1 with a feature
        whose bs > 1 wiring is not yet committed."""
        issues: list[str] = []
        # ``dynamic_pref_source='body'`` is now supported via the
        # ``oracle_state_per_b`` side channel (block C lift); gated
        # dynamically inside ``_step_batched`` if the runner did not
        # supply per-element body state.
        # ``habit_mode='prior_prune'`` at bs > 1 (block E lift): handled
        # in _step_batched via _per_element_prior_prune_keep_masks +
        # the scorer's new per-batch keep_mask kwarg. Gated dynamically
        # inside the helper rather than here.
        if str(self.proposal_mode) != "none":
            issues.append(
                f"proposal_mode={self.proposal_mode!r} (deferred — requires "
                "proposal_mode at batch_size > 1, which is not supported in this snapshot."
            )
        # oracle_state_access=True is supported when the runner passes
        # ``oracle_state_per_b``; gated dynamically inside _step_batched
        # rather than here so unit tests that bypass the runner can still
        # exercise the bs > 1 path with oracle access disabled.
        if issues:
            raise ValueError(
                f"agent.step() at batch_size={self.batch_size} > 1 does not "
                f"yet support: {', '.join(issues)}. See "
                "docs/architecture/benchmarks/batching_audit.md for the "
                "lift plan."
            )

    def _unpack_extero_observation_batched(self, obs) -> tuple:
        """At bs > 1: ``obs[0]`` is ``(B,)`` int, ``obs[1]`` is ``(B,)`` int,
        ``obs[2]`` is ``(B, n_season+1)`` float."""
        pos_obs = np.asarray(obs[0]).astype(np.int32).reshape(-1)
        res_obs = np.asarray(obs[1]).astype(np.int32).reshape(-1)
        season_obs_vec = np.asarray(obs[2])
        season_idx = season_obs_vec.argmax(axis=-1).astype(np.int32)
        return pos_obs, res_obs, season_idx

    def _pack_observation_modalities_batched(
        self,
        *,
        pos_obs,
        res_obs,
        season_idx,
        typeII_obs,
    ) -> list:
        """Per-batch one-hot lookup. Returns a list of seven ``(B, n_obs_m)``
        tensors, matching pymdp's bs-leading observation contract."""
        bs = int(self.batch_size)
        typeII_obs = np.asarray(typeII_obs).astype(np.int32).reshape(bs, 4)

        def _lookup(table, idx):
            out = jnp.asarray(table)[idx]
            # The agent's ``_oh_*`` lookup tables were constructed with a
            # leading-axis-1 batch slot for the bs=1 contract (shape
            # ``(n, 1, n)``). Fancy-indexing with a (B,) vector gives
            # ``(B, 1, n)``; squeeze the singleton batch slot to ``(B, n)``.
            if out.ndim == 3 and out.shape[1] == 1:
                return out.reshape(out.shape[0], out.shape[2])
            return out

        return [
            _lookup(self._oh_pos, pos_obs),
            _lookup(self._oh_res, res_obs),
            _lookup(self._oh_season_obs, season_idx),
            _lookup(self._oh_typeII, typeII_obs[:, 0]),
            _lookup(self._oh_typeII, typeII_obs[:, 1]),
            _lookup(self._oh_typeII, typeII_obs[:, 2]),
            _lookup(self._oh_typeII, typeII_obs[:, 3]),
        ]

    def _decode_action_batched(self, action) -> np.ndarray:
        """Decode a per-batch ``(B, num_factors)`` action array to ``(B, 2)``
        ``(move, interact)`` integer pairs. If the policy library uses a
        different number of control factors, defer to the pymdp helper."""
        arr = np.asarray(action)
        if arr.ndim != 2:
            raise ValueError(
                f"_decode_action_batched: expected (B, num_factors) array, "
                f"got shape {arr.shape}"
            )
        if arr.shape[-1] == 2:
            return arr.astype(np.int32, copy=False)
        if hasattr(self.pymdp_agent, "decode_multi_actions"):
            decoded = np.asarray(
                self.pymdp_agent.decode_multi_actions(jnp.asarray(arr))
            )
            return decoded.astype(np.int32, copy=False)
        raise ValueError(
            f"_decode_action_batched: unable to decode (B, K) action with "
            f"shape {arr.shape}"
        )

    def _oracle_qs_override_batched(self, qs, *, oracle_state_per_b: list):
        """Per-element analogue of ``_oracle_qs_override``. Only fires
        when ``self.oracle_state_access`` is True and the runner supplies
        ``oracle_state_per_b`` (a list of B ground-truth dicts; see
        ``StandardActiveInferenceAgent.step`` for schema)."""
        if not bool(self.oracle_state_access):
            return qs
        if qs is None or oracle_state_per_b is None:
            return qs
        bs = int(self.batch_size)
        if len(oracle_state_per_b) != bs:
            raise ValueError(
                f"oracle_state_per_b length {len(oracle_state_per_b)} != "
                f"batch_size {bs}"
            )
        # Stack per-element ground-truth indices into (B,) arrays per
        # state factor: position, season, energy, hydration, sleep, oxygen.
        idx_per_factor = [
            np.asarray(
                [int(d.get("pos_true", 0)) for d in oracle_state_per_b],
                dtype=np.int32,
            ),
            np.asarray(
                [int(d.get("season_true", 0)) for d in oracle_state_per_b],
                dtype=np.int32,
            ),
            np.asarray(
                [int(d.get("energy", 0)) for d in oracle_state_per_b],
                dtype=np.int32,
            ),
            np.asarray(
                [int(d.get("hydration", 0)) for d in oracle_state_per_b],
                dtype=np.int32,
            ),
            np.asarray(
                [int(d.get("sleep_debt", 0)) for d in oracle_state_per_b],
                dtype=np.int32,
            ),
            np.asarray(
                [int(d.get("oxygen", 0)) for d in oracle_state_per_b],
                dtype=np.int32,
            ),
        ]
        out = list(qs)
        for f, idx_arr in enumerate(idx_per_factor):
            if f >= len(out):
                continue
            arr = np.asarray(out[f], dtype=np.float32)
            # arr shape: (B, T, n_states_f). Build a (B, T, n_states_f)
            # one-hot along the state axis with per-element idx.
            n_states = int(arr.shape[-1])
            idx_clipped = np.clip(idx_arr, 0, n_states - 1).astype(np.int32)
            new_arr = np.zeros_like(arr)
            for b in range(bs):
                new_arr[b, ..., int(idx_clipped[b])] = 1.0
            out[f] = jnp.asarray(new_arr, dtype=jnp.float32)
        return out

    def _expected_typeI_deficits_from_beliefs_per_batch(self) -> tuple:
        """Per-batch ``(B,)`` array variant of
        ``_expected_typeI_deficits_from_beliefs``. Used by the bs > 1
        dynamic-resource-preferences path. Returns
        ``(food_deficit_b, water_deficit_b, sleep_debt_expected_b)`` arrays."""
        bs = int(self.batch_size)
        if self.qs is None or len(self.qs) <= 4:
            # First step or insufficient factors: fall back to D-derived
            # scalar prior, broadcast to (B,).
            t_food, t_water, t_sleep = self._expected_typeI_deficits_from_beliefs()
            return (
                np.full(bs, t_food, dtype=np.float32),
                np.full(bs, t_water, dtype=np.float32),
                np.full(bs, t_sleep, dtype=np.float32),
            )

        q_per_b = self._latest_qs_per_batch(self.qs)  # (B, n_states_f) per factor
        # Energy is factor 2, hydration factor 3, sleep factor 4.
        idx_e = np.arange(int(q_per_b[2].shape[-1]), dtype=np.float32)
        idx_h = np.arange(int(q_per_b[3].shape[-1]), dtype=np.float32)
        idx_s = np.arange(int(q_per_b[4].shape[-1]), dtype=np.float32)
        exp_e = (q_per_b[2] * idx_e).sum(axis=-1)  # (B,)
        exp_h = (q_per_b[3] * idx_h).sum(axis=-1)
        exp_s = (q_per_b[4] * idx_s).sum(axis=-1)
        t_food = np.maximum(0.0, float(self.body.MAX_ENERGY) - exp_e).astype(np.float32)
        t_water = np.maximum(0.0, float(self.body.MAX_HYDRATION) - exp_h).astype(np.float32)
        t_sleep = np.maximum(0.0, exp_s).astype(np.float32)
        return t_food, t_water, t_sleep

    def _update_dynamic_resource_preferences_batched(
        self,
        *,
        oracle_state_per_b=None,
    ) -> None:
        """Per-batch C[1] rewrite from per-element deficits.

        Two source modes:
        - ``dynamic_pref_source='belief'``: uses
          ``_expected_typeI_deficits_from_beliefs_per_batch`` (always
          available; needs no runner side channel).
        - ``dynamic_pref_source='body'`` (block C lift): uses
          ``oracle_state_per_b`` supplied by ``BatchedSimulation`` to
          read per-element ground-truth body counters. Raises if the
          runner forgot to supply it.
        """
        if not bool(self.dynamic_resource_preferences):
            return
        bs = int(self.batch_size)
        if str(self.dynamic_pref_source) == "body":
            if oracle_state_per_b is None or len(oracle_state_per_b) != bs:
                raise ValueError(
                    "dynamic_pref_source='body' at bs > 1 requires the "
                    "runner to pass oracle_state_per_b (per-element body "
                    "counters). BatchedSimulation supplies it automatically "
                    "when dynamic_resource_preferences=True."
                )
            t_food_b = np.asarray(
                [
                    max(0.0, float(self.body.MAX_ENERGY) - float(d.get("energy", 0)))
                    for d in oracle_state_per_b
                ],
                dtype=np.float32,
            )
            t_water_b = np.asarray(
                [
                    max(0.0, float(self.body.MAX_HYDRATION) - float(d.get("hydration", 0)))
                    for d in oracle_state_per_b
                ],
                dtype=np.float32,
            )
            t_sleep_b = np.asarray(
                [max(0.0, float(d.get("sleep_debt", 0))) for d in oracle_state_per_b],
                dtype=np.float32,
            )
        else:
            t_food_b, t_water_b, t_sleep_b = self._expected_typeI_deficits_from_beliefs_per_batch()

        empty_default = float(self.dynamic_pref_empty_bias)
        n_res = int(self.n_res)
        c_per_b = np.zeros((bs, n_res), dtype=np.float32)

        for b in range(bs):
            empty = empty_default
            pref_food = homeostatic_need_signal(
                float(t_food_b[b]),
                max_level=float(self.body.MAX_ENERGY),
                mode=self.intero_preference_mode,
                band_min=float(self.intero_range_hunger_min),
                band_max=float(self.intero_range_hunger_max),
            )
            pref_water = homeostatic_need_signal(
                float(t_water_b[b]),
                max_level=float(self.body.MAX_HYDRATION),
                mode=self.intero_preference_mode,
                band_min=float(self.intero_range_thirst_min),
                band_max=float(self.intero_range_thirst_max),
            )
            pref_sleep = homeostatic_need_signal(
                float(t_sleep_b[b]),
                max_level=float(self.body.MAX_SLEEPDEBT),
                mode=self.intero_preference_mode,
                band_min=float(self.intero_range_fatigue_min),
                band_max=float(self.intero_range_fatigue_max),
            )

            # Per-element urgency gating (mirrors the bs=1 path).
            critical_penalty = float(self.dynamic_pref_critical_penalty)
            if float(t_water_b[b]) > float(self.dynamic_pref_water_critical):
                pref_food = critical_penalty
                pref_sleep = critical_penalty
                empty = critical_penalty
            if float(t_food_b[b]) > float(self.dynamic_pref_food_critical):
                pref_water = critical_penalty
                pref_sleep = critical_penalty
                empty = critical_penalty
            if float(t_sleep_b[b]) > float(self.dynamic_pref_sleep_critical):
                pref_food = critical_penalty
                pref_water = critical_penalty
                empty = critical_penalty

            c_per_b[b, int(Resource.AIR)] = float(empty)
            c_per_b[b, int(Resource.FOOD)] = float(pref_food)
            c_per_b[b, int(Resource.WATER)] = float(pref_water)
            c_per_b[b, int(Resource.SLEEP)] = float(pref_sleep)
            c_per_b[b, int(Resource.POISON)] = float(self.dynamic_pref_poison_penalty)

        # Write per-batch C[1]. self.C is the per-element shadow; for the
        # batched path keep self.C[1] as an element-0 view (back-compat),
        # but write the full (B, n_res) tensor to pymdp_agent.C[1].
        self.C[1] = c_per_b[0].copy()
        self.pymdp_agent.C[1] = jnp.asarray(c_per_b, dtype=jnp.float32)

    def _per_element_prior_prune_keep_masks(
        self,
        *,
        pos_obs,
        res_obs,
        season_idx,
        typeII_obs,
        runtime_A=None,
    ) -> jnp.ndarray | None:
        """Per-element wrapper around the bs=1 prior_prune machinery.

        Returns a ``(B, num_policies)`` boolean keep mask, or ``None`` if
        no element produces a usable proposal (in which case the caller
        skips the mask entirely and runs the planner unmasked).

        At bs > 1 each batch element runs ``_habit_policy_proposal`` with
        its own per-element ``pos_obs/season_idx/typeII_obs`` scalars and
        derives its own ``keep_idx`` via ``habit_proposal_keep_indices``.
        The scorer (efe_repo_local) accepts the resulting ``(B, P)`` mask
        directly via the ``keep_mask`` kwarg (block E added per-batch
        mask support there).
        """
        if str(self.habit_mode) != "prior_prune":
            return None
        bs = int(self.batch_size)
        typeII_arr = np.asarray(typeII_obs).astype(np.int32).reshape(bs, 4)
        n_pi = int(self.pymdp_agent.policies.num_policies)
        keep_mask = np.ones((bs, n_pi), dtype=np.bool_)
        any_pruned = False
        policies_full = np.asarray(_policy_array_np(self.pymdp_agent), dtype=np.int32)
        for b in range(bs):
            resource_likelihood = None
            if runtime_A is not None and len(runtime_A) > 1:
                # Slice per-element resource A.
                runtime_A_b = np.asarray(runtime_A[1])
                if runtime_A_b.ndim >= 1 and runtime_A_b.shape[0] == bs:
                    resource_likelihood = np.asarray(runtime_A_b[b], dtype=np.float32)
                else:
                    resource_likelihood = np.asarray(runtime_A_b, dtype=np.float32)
            proposal, _proposal_info = self._habit_policy_proposal(
                pos_obs=int(pos_obs[b]),
                season_idx=int(season_idx[b]),
                typeII_obs=np.asarray(typeII_arr[b], dtype=np.int32),
                resource_likelihood=resource_likelihood,
            )
            if proposal is None:
                continue
            keep_idx, _prune_info = habit_proposal_keep_indices(
                proposal,
                policies=policies_full,
                has_kappa=False,
                n_interact=int(len(Interact)),
                mass_threshold=float(self.habit_prune_mass_threshold),
                min_keep=int(self.habit_prune_min_keep),
                max_keep=int(self.habit_prune_max_keep),
                keep_per_first_action=int(self.habit_prune_keep_per_action),
            )
            keep_arr = np.asarray(keep_idx, dtype=np.int32)
            if keep_arr.size <= 0 or keep_arr.size >= n_pi:
                # Inactive proposal for this element — keep all policies.
                continue
            mask_b = np.zeros((n_pi,), dtype=np.bool_)
            mask_b[keep_arr] = True
            keep_mask[b] = mask_b
            any_pruned = True
        if not any_pruned:
            return None
        return jnp.asarray(keep_mask, dtype=jnp.bool_)

    def _per_element_option_distributions(
        self,
        *,
        pos_obs,
        res_obs,
        season_idx,
        typeII_obs,
    ) -> tuple[list, list[dict]]:
        """Per-element wrapper around ``_habit_option_distribution``.

        Iterates ``b in range(B)`` with element-b's scalars temporarily
        installed on ``self._active_option`` so the existing single-element
        helper updates the correct per-batch option state. Restores
        ``self._active_option = None`` after the loop so the bs=1
        invariant ("scalar holds element-0's option") is preserved at
        return — the per-batch state lives in
        ``self._active_option_per_b``.
        """
        bs = int(self.batch_size)
        typeII_arr = np.asarray(typeII_obs).astype(np.int32).reshape(bs, 4)
        option_qs: list = []
        option_infos: list[dict] = []
        for b in range(bs):
            self._active_option = self._active_option_per_b[b]
            opt_q, opt_info = self._habit_option_distribution(
                pos_obs=int(pos_obs[b]),
                res_obs=int(res_obs[b]),
                season_idx=int(season_idx[b]),
                typeII_obs=np.asarray(typeII_arr[b], dtype=np.int32),
            )
            self._active_option_per_b[b] = self._active_option
            option_qs.append(opt_q)
            option_infos.append(opt_info)
        self._active_option = None
        return option_qs, option_infos

    def _update_compute_budget_batched(
        self,
        *,
        control_sources_per_b: list[str],
        planner_used_per_b: list[bool],
    ) -> None:
        """Per-element compute-budget tick. Updates
        ``self._compute_budget_per_b`` in place; mirrors element 0 to the
        scalar ``self.compute_budget`` for back-compat with consumers
        (e.g. logs) that read the scalar field."""
        bs = int(self.batch_size)
        for b in range(bs):
            step_cost = (
                float(self.compute_budget_planner_cost)
                if planner_used_per_b[b]
                else float(self.compute_budget_habit_cost)
            )
            switched = (
                str(control_sources_per_b[b])
                != str(self._last_control_source_per_b[b])
            )
            self._compute_budget_per_b[b] = float(
                update_computation_budget(
                    float(self._compute_budget_per_b[b]),
                    budget_max=float(self.compute_budget_max),
                    recovery=float(self.compute_budget_recovery),
                    step_cost=float(step_cost),
                    switch_cost=float(self.compute_budget_switch_cost),
                    switched=bool(switched),
                )
            )
            self._last_control_source_per_b[b] = str(control_sources_per_b[b])
        self.compute_budget = float(self._compute_budget_per_b[0])
        self._last_control_source = str(self._last_control_source_per_b[0])

    def _step_batched(self, obs, typeII_obs, *, oracle_state_per_b=None) -> np.ndarray:
        """Plain-EFE bs > 1 decision cycle. See ``_step_bs1`` for the
        full single-element loop. ``oracle_state_per_b`` (block B lift)
        is a runner-supplied list of per-element ground-truth dicts used
        only when ``self.oracle_state_access`` is True."""
        if bool(self.oracle_state_access) and oracle_state_per_b is None:
            raise ValueError(
                "agent._step_batched: oracle_state_access=True requires the "
                "runner to pass oracle_state_per_b (per-element env+body "
                "ground truth). BatchedSimulation supplies it automatically."
            )
        self._assert_no_unsupported_bs_n_features()

        pos_obs, res_obs, season_idx = self._unpack_extero_observation_batched(obs)
        observation = self._pack_observation_modalities_batched(
            pos_obs=pos_obs,
            res_obs=res_obs,
            season_idx=season_idx,
            typeII_obs=typeII_obs,
        )
        prior = self._build_empirical_prior()

        # State inference vmaps over the leading batch axis (already
        # verified in P5 Step 2). Output: list of (B, T, num_states_f).
        self.qs = self._infer_states(observation, prior, None)
        # Optional per-element oracle qs override (block B lift).
        self.qs = self._oracle_qs_override_batched(
            self.qs, oracle_state_per_b=oracle_state_per_b,
        )

        # Online A learning at bs > 1 (P5.4.4). The dispatcher in
        # _online_model_update routes to _online_model_update_batched
        # (Step 3) which writes per-batch counts into self._pA[m] and
        # per-batch likelihoods into self.A[m]. The closure-captured
        # A_default in the EFE scorer becomes stale on every update;
        # we thread the up-to-date self.A through ``runtime_A`` below
        # so the planner sees it.
        self._online_model_update(observation, self.qs)

        # Dynamic resource preferences at bs > 1. Belief source (P5.4.5)
        # is self-contained; body source (block C) requires the runner
        # to have supplied oracle_state_per_b — passed through here.
        self._update_dynamic_resource_preferences_batched(
            oracle_state_per_b=oracle_state_per_b,
        )

        # Build runtime overrides: at bs > 1 self.A carries the current
        # learned likelihoods (per-batch shape (B, *)). _coerce_A_runtime
        # in the scorer accepts this shape directly (P5.4.3 lifted that).
        runtime_A = (
            None
            if not bool(self.unknown_model)
            else [jnp.asarray(a_m, dtype=jnp.float32) for a_m in self.A]
        )
        runtime_C = (
            None
            if not bool(self.dynamic_resource_preferences)
            else [
                jnp.asarray(c_m, dtype=jnp.float32)
                for c_m in self.pymdp_agent.C
            ]
        )

        # Habit / option arbitration at bs > 1 (P5.4.6). For
        # ``habit_mode in {"options", "hybrid"}`` we run the per-element
        # option machinery, then either replace planner output entirely
        # (habit_mode=="options") or selectively override planner output
        # for the elements whose budget dropped below threshold
        # (habit_mode=="hybrid").
        bs = int(self.batch_size)
        mode = str(self.habit_mode)
        if mode in {"options", "hybrid"}:
            option_qs, _option_infos = self._per_element_option_distributions(
                pos_obs=pos_obs,
                res_obs=res_obs,
                season_idx=season_idx,
                typeII_obs=typeII_obs,
            )
            force_option_per_b = []
            for b in range(bs):
                if mode == "options":
                    force_option_per_b.append(option_qs[b] is not None)
                else:  # hybrid
                    force_option_per_b.append(
                        option_qs[b] is not None
                        and float(self._compute_budget_per_b[b])
                        <= float(self.compute_budget_threshold)
                    )
            need_planner = not all(force_option_per_b)
            if need_planner:
                planner_q_pi, planner_G = self._infer_policies(
                    self.qs,
                    self._next_planner_key_batched(),
                    runtime_A, runtime_C, None, None, None,
                )
                planner_q_pi = np.asarray(planner_q_pi, dtype=np.float32)
                planner_G = np.asarray(planner_G, dtype=np.float32)
            num_pi = int(self.pymdp_agent.policies.num_policies)
            q_rows = np.zeros((bs, num_pi), dtype=np.float32)
            G_rows = np.zeros((bs, num_pi), dtype=np.float32)
            control_sources_per_b: list[str] = []
            planner_used_per_b: list[bool] = []
            for b in range(bs):
                if force_option_per_b[b]:
                    q_rows[b] = self._normalize_prob_vec(option_qs[b])
                    G_rows[b] = 0.0
                    control_sources_per_b.append("option")
                    planner_used_per_b.append(False)
                else:
                    # Hybrid mode falls back to planner; "hybrid" mixing
                    # of planner and option is the bs=1-only refinement
                    # (audit § 8 — full mixing under bs > 1 is a future
                    # follow-up; this commit lands the strict-fallback
                    # variant which preserves correctness when option
                    # budget is above threshold).
                    q_rows[b] = planner_q_pi[b]
                    G_rows[b] = planner_G[b]
                    control_sources_per_b.append("planner")
                    planner_used_per_b.append(True)
            q_pi = jnp.asarray(q_rows, dtype=jnp.float32)
            G = jnp.asarray(G_rows, dtype=jnp.float32)
        elif mode == "prior_prune":
            # Block E lift. Build a (B, num_policies) keep mask from
            # per-element proposals and pass it to _infer_policies via
            # the keep_mask kwarg. The scorer's outer vmap handles the
            # per-batch mask shape directly (efe_repo_local._compute
            # gained ``mask_per_b=True`` in this commit's sister edit).
            prior_prune_mask = self._per_element_prior_prune_keep_masks(
                pos_obs=pos_obs,
                res_obs=res_obs,
                season_idx=season_idx,
                typeII_obs=typeII_obs,
                runtime_A=runtime_A,
            )
            if prior_prune_mask is None:
                q_pi, G = self._infer_policies(
                    self.qs,
                    self._next_planner_key_batched(),
                    runtime_A, runtime_C, None, None, None,
                )
            else:
                q_pi, G = self._infer_policies(
                    self.qs,
                    self._next_planner_key_batched(),
                    runtime_A, runtime_C, None, None, None,
                    keep_mask=prior_prune_mask,
                )
            control_sources_per_b = ["planner"] * bs
            planner_used_per_b = [True] * bs
        else:
            # Pure planner path.
            q_pi, G = self._infer_policies(
                self.qs,
                self._next_planner_key_batched(),
                runtime_A, runtime_C, None, None, None,
            )
            control_sources_per_b = ["planner"] * bs
            planner_used_per_b = [True] * bs
        # q_pi shape: (B, num_policies); G same.

        # Sample per-batch action: pymdp's sample_action vmaps over the
        # leading batch axis internally (pymdp/agent.py:937-965).
        action = np.asarray(self._sample_action(q_pi))

        if self.store_diagnostics:
            self.last_q_pi = np.asarray(q_pi, dtype=np.float32)
            self.last_G = (
                None if G is None else np.asarray(G, dtype=np.float32)
            )
            self.last_planner = self._consume_planner_debug_state()

        # Per-element compute budget tick.
        self._update_compute_budget_batched(
            control_sources_per_b=control_sources_per_b,
            planner_used_per_b=planner_used_per_b,
        )

        self.prev_action = action
        return self._decode_action_batched(action)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def reset(self):
        """Reset beliefs, diagnostics, active habit state, and computation budget."""
        self.qs = None
        self.prev_action = None
        self.last_q_pi = None
        self.last_G = None
        self.last_planner = None
        self._active_option = None
        self._last_control_source = "planner"
        self.compute_budget = float(np.clip(self._compute_budget_init, 0.0, self.compute_budget_max))
        self.last_habit = None
        if self.batch_size > 1:
            self._compute_budget_per_b[:] = float(self.compute_budget)
            self._active_option_per_b = [None] * self.batch_size
            self._last_control_source_per_b = ["planner"] * self.batch_size
            self._last_habit_per_b = [None] * self.batch_size

    def step(self, obs, typeII_obs, *, oracle_state_per_b=None):
        """
        Run one Active Inference decision cycle. Dispatches on
        ``self.batch_size``: at ``bs=1`` returns the legacy
        ``(Move, Interact)`` tuple; at ``bs > 1`` returns a per-batch
        ``(B, 2)`` integer action array.

        ``oracle_state_per_b`` (bs > 1 only) is an optional list of B
        dicts with keys ``pos_true``, ``season_true``, ``energy``,
        ``hydration``, ``sleep_debt``, ``oxygen``. When supplied at
        ``oracle_state_access=True`` the agent applies the per-element
        qs override using these ground-truth values; without the side
        channel the agent has no access to the runner's N envs/bodies.
        """
        if self.batch_size > 1:
            return self._step_batched(obs, typeII_obs, oracle_state_per_b=oracle_state_per_b)
        return self._step_bs1(obs, typeII_obs)

    def _step_bs1(self, obs, typeII_obs) -> Tuple[Move, Interact]:
        """
        Run one baseline Active Inference decision cycle (single-element).

        In order this method:
        1. packs raw exteroceptive and interoceptive observations into the
           categorical form expected by ``pymdp``;
        2. builds the empirical prior from the previous action, or falls back to
           ``D`` on the very first step;
        3. infers hidden states, optionally replacing them with oracle deltas in
           development-only mode;
        4. optionally updates learned likelihoods and rewrites resource
           preferences from current homeostatic pressure;
        5. computes any habit prior or option override, then either skips the
           planner, uses the planner, or mixes planner and option posteriors;
        6. samples ``(Move, Interact)``, stores diagnostics, and updates the
           bounded computation budget.

        Relative to a standard fixed-model ActInf loop, the non-standard pieces
        here are online ``A`` learning, dynamic resource preferences, and the
        planner-vs-habit arbitration layer.
        """
        pos_obs, res_obs, season_idx = self._unpack_extero_observation(obs)
        observation = self._pack_observation_modalities(
            pos_obs=pos_obs,
            res_obs=res_obs,
            season_idx=season_idx,
            typeII_obs=typeII_obs,
        )

        prior = self._build_empirical_prior()

        # Inference over hidden states, then over policies, then sample action.
        quiet = os.environ.get("AFFECTWORLD_QUIET_PYMDP") == "1"
        planner_key = None
        if bool(getattr(self, "_planner_uses_rng", False)):
            self._planner_key, planner_key = jr.split(self._planner_key)
        budget_before = float(self.compute_budget)

        def _infer_cycle_custom():
            # Phase 1: infer the current hidden state under the latest model.
            state_runtime_A = self._state_runtime_A_override()
            self.qs = self._infer_states(observation, prior, state_runtime_A)
            self.qs = self._oracle_qs_override(self.qs, pos_obs=int(pos_obs), season_idx=int(season_idx))
            self._online_model_update(observation, self.qs)
            self._update_dynamic_resource_preferences()
            self._update_dynamic_intero_preferences()

            # Phase 2: obtain a cheap option controller proposal, if enabled.
            option_q, option_info = self._habit_option_distribution(
                pos_obs=int(pos_obs),
                res_obs=int(res_obs),
                season_idx=int(season_idx),
                typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
            )
            force_option = bool(
                (str(self.habit_mode) == "options")
                or (
                    str(self.habit_mode) == "hybrid"
                    and option_q is not None
                    and float(self.compute_budget) <= float(self.compute_budget_threshold)
                )
            )
            if force_option and option_q is not None:
                q_vec = self._normalize_prob_vec(option_q)
                q_pi_local = jnp.asarray(q_vec[None, :], dtype=jnp.float32)
                G_local = jnp.zeros_like(q_pi_local)
                return q_pi_local, G_local, None, "option", 1.0, False, {}, option_info

            # Phase 3: run policy inference with current runtime overrides.
            runtime_A, runtime_C, runtime_pA, runtime_E, prior_info = self._planner_runtime_overrides(
                pos_obs=int(pos_obs),
                season_idx=int(season_idx),
                typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
            )
            root_action_mask_runtime = None
            proposal_info: dict[str, object] = {
                "mode": str(self.proposal_mode),
                "proposal_role": "inactive",
            }
            if str(self.proposal_mode) != "none":
                root_action_mask_runtime, proposal_info = self._proposal_root_action_override(
                    pos_obs=int(pos_obs),
                    season_idx=int(season_idx),
                    typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
                    runtime_A=runtime_A,
                )
            if str(self.habit_mode) == "prior_prune":
                q_pi_local, G_local, prune_info = self._infer_policies_with_habit_pruning(
                    qs=self.qs,
                    runtime_A=runtime_A,
                    runtime_C=runtime_C,
                    pos_obs=int(pos_obs),
                    season_idx=int(season_idx),
                    typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
                )
                prior_info = {**dict(prior_info), **dict(prune_info)}
            else:
                q_pi_local, G_local = self._infer_policies(
                    self.qs,
                    planner_key,
                    runtime_A,
                    runtime_C,
                    runtime_pA,
                    runtime_E,
                    root_action_mask_runtime,
                )
                if str(self.proposal_mode) != "none":
                    q_ref, _G_ref = self._infer_policies(
                        self.qs,
                        planner_key,
                        runtime_A,
                        runtime_C,
                        runtime_pA,
                        runtime_E,
                        None,
                    )
                    q_ref_arr = np.asarray(q_ref[0], dtype=np.float32) if np.asarray(q_ref).ndim == 2 else np.asarray(q_ref, dtype=np.float32)
                    q_pruned_arr = np.asarray(q_pi_local[0], dtype=np.float32) if np.asarray(q_pi_local).ndim == 2 else np.asarray(q_pi_local, dtype=np.float32)
                    unpruned_winner = int(np.argmax(q_ref_arr))
                    pruned_winner = int(np.argmax(q_pruned_arr))
                    kept_actions = proposal_info.get("kept_root_actions")
                    kept_action_ids = {
                        int(item.get("action"))
                        for item in kept_actions
                        if isinstance(kept_actions, list) and isinstance(item, dict) and item.get("action") is not None
                    }
                    proposal_info = {
                        **dict(proposal_info),
                        "unpruned_winner_action": int(unpruned_winner),
                        "pruned_winner_action": int(pruned_winner),
                        "planner_winner_survived_pruning": bool(unpruned_winner in kept_action_ids) if kept_action_ids else None,
                    }
                prior_info = {**dict(prior_info), "proposal": dict(proposal_info)}
            # Release the freshly-built JAX device-buffer references for the
            # planner's runtime overrides before this closure's frame survives
            # into the next step. Without this, baseline grows ~3960 anonymous
            # mmap regions per trial because the closure keeps these alive
            # across the planner JIT call. ExternalKappaAgent achieves the
            # same effect by overwriting runtime_A immediately after
            # _planner_runtime_overrides returns.
            del runtime_A, runtime_C, runtime_pA, runtime_E, root_action_mask_runtime
            q_vec = np.asarray(q_pi_local[0], dtype=np.float32) if np.asarray(q_pi_local).ndim == 2 else np.asarray(q_pi_local, dtype=np.float32)
            source = "planner"
            mix_alpha = 0.0
            if str(self.habit_mode) == "hybrid" and option_q is not None:
                budget_ratio = float(np.clip(float(self.compute_budget) / max(1e-6, float(self.compute_budget_max)), 0.0, 1.0))
                mix_alpha = float(
                    np.clip(
                        float(self.habit_hybrid_mix) + (1.0 - budget_ratio) * (1.0 - float(self.habit_hybrid_mix)),
                        0.0,
                        1.0,
                    )
                )
                q_vec = self._normalize_prob_vec((1.0 - mix_alpha) * q_vec + mix_alpha * np.asarray(option_q, dtype=np.float32))
                q_pi_local = jnp.asarray(q_vec[None, :], dtype=jnp.float32)
                source = "hybrid"
            return q_pi_local, G_local, None, source, mix_alpha, True, prior_info, option_info

        def _infer_cycle_infer_and_plan():
            # Single-call pymdp path used when the upstream backend can own the loop.
            runtime_A, runtime_C, _runtime_pA, runtime_E, prior_info = self._planner_runtime_overrides(
                pos_obs=int(pos_obs),
                season_idx=int(season_idx),
                typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
            )
            qs_prev = self.qs
            if qs_prev is None:
                qs_prev = jtu.tree_map(lambda d: d[:, None, :], self.pymdp_agent.D)
            plan_key = planner_key if planner_key is not None else jr.PRNGKey(0)
            _agent_next, action_local, qs_local, info = infer_and_plan_compat(
                agent=self.pymdp_agent,
                qs_prev=qs_prev,
                observation=observation,
                action_prev=self.prev_action,
                rng_key=plan_key,
                infer_policies=self._infer_policies,
                A_state_runtime=None,
                A_plan_runtime=runtime_A,
                C_runtime=runtime_C,
                E_runtime=runtime_E,
                empirical_prior=prior,
            )
            self.qs = self._oracle_qs_override(qs_local, pos_obs=int(pos_obs), season_idx=int(season_idx))
            q_pi_local = info["qpi"]
            G_local = info.get("G", q_pi_local)
            option_info = {"mode": str(self.habit_mode)}
            # See note above _infer_cycle_custom's matching ``del``: release the
            # runtime override device-buffer references before this closure's
            # frame is captured by the outer ``step`` scope.
            del runtime_A, runtime_C, runtime_E
            return q_pi_local, G_local, action_local, "planner", 0.0, True, prior_info, option_info

        infer_cycle = _infer_cycle_infer_and_plan if self._actinf_runtime_backend == "pymdp_infer_and_plan" else _infer_cycle_custom

        if quiet:
            with (
                contextlib.redirect_stdout(_QUIET_SINK),
                contextlib.redirect_stderr(_QUIET_SINK),
            ):
                q_pi, G, action_override, control_source, mix_alpha, planner_used, habit_prior_info, habit_option_info = infer_cycle()
        else:
            q_pi, G, action_override, control_source, mix_alpha, planner_used, habit_prior_info, habit_option_info = infer_cycle()

        action = np.asarray(self._sample_action(q_pi) if action_override is None else action_override)

        if self.store_diagnostics:
            self.last_q_pi = np.asarray(q_pi[0], dtype=np.float32) if np.asarray(q_pi).ndim == 2 else np.asarray(q_pi, dtype=np.float32)
            self.last_G = np.asarray(G[0], dtype=np.float32) if np.asarray(G).ndim == 2 else np.asarray(G, dtype=np.float32)
            self.last_planner = self._consume_planner_debug_state() if bool(planner_used) else None

        if bool(planner_used):
            step_cost = float(self.compute_budget_planner_cost)
        else:
            step_cost = float(self.compute_budget_habit_cost)
        if str(control_source) == "hybrid":
            step_cost = (1.0 - float(mix_alpha)) * float(self.compute_budget_planner_cost) + float(mix_alpha) * float(self.compute_budget_habit_cost)
        switched = str(control_source) != str(self._last_control_source)
        self.compute_budget = update_computation_budget(
            float(self.compute_budget),
            budget_max=float(self.compute_budget_max),
            recovery=float(self.compute_budget_recovery),
            step_cost=float(step_cost),
            switch_cost=float(self.compute_budget_switch_cost),
            switched=bool(switched),
        )
        self._last_control_source = str(control_source)
        self.last_habit = {
            "mode": str(self.habit_mode),
            "source": str(control_source),
            "planner_used": bool(planner_used),
            "mix_alpha": float(mix_alpha),
            "budget_before": float(budget_before),
            "budget_after": float(self.compute_budget),
            "budget_max": float(self.compute_budget_max),
            "budget_threshold": float(self.compute_budget_threshold),
            "prior": dict(habit_prior_info),
            "option": dict(habit_option_info),
        }

        move, interact = self._decode_action(action)
        self.prev_action = action
        return move, interact

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        p = p / max(1e-16, float(p.sum()))
        return float(-(p * np.log(p + 1e-16)).sum())

    def latest_policy_diagnostics(self, top_k: int = 5) -> dict:
        """
        Return compact diagnostics for the most recent policy inference.

        The payload is meant for logs and notebooks: it includes entropy,
        highest-probability policies, optional ``G`` values, and the most recent
        habit-arbitration bookkeeping when available.
        """
        if self.last_q_pi is None:
            return {}
        q = np.asarray(self.last_q_pi, dtype=np.float64).reshape(-1)
        k = max(1, int(top_k))
        top = np.argsort(-q)[:k].astype(int).tolist()
        out = {
            "q_pi_entropy": self._entropy(q),
            "top_pi": [{"pi": i, "p": float(q[i])} for i in top],
        }
        if self.last_G is not None:
            G = np.asarray(self.last_G, dtype=np.float64).reshape(-1)
            out["top_pi_G"] = [{"pi": i, "G": float(G[i])} for i in top]
        if self.last_habit is not None:
            out["habit"] = self.last_habit
        if self.last_planner is not None:
            out["planner"] = self.last_planner
        return out

    def last_belief_over_body(self) -> list[list[float]] | None:
        """
        Return the latest marginal posteriors over the four body-state factors
        (``energy``, ``hydration``, ``sleep_debt``, ``oxygen``) as a length-4
        list of lists, or ``None`` if inference has not yet run.

        This is a compact per-need view intended for step-trace logging; the
        ordering matches ``helpers.Need`` (HUNGER, THIRST, FATIGUE, SUFFOCATION)
        since each body factor drives exactly one interoceptive observation.
        """
        if self.qs is None:
            return None
        body_factor_indices = (2, 3, 4, 5)
        out: list[list[float]] = []
        for idx in body_factor_indices:
            if idx >= len(self.qs):
                return None
            arr = np.asarray(self.qs[idx], dtype=np.float32)
            if arr.ndim == 1:
                vec = arr
            elif arr.ndim == 2:
                vec = arr[-1]
            else:
                vec = arr[0, -1]
            out.append([float(x) for x in vec.tolist()])
        return out

    def last_kappa_per_channel(self, attended: int | None = None) -> list[float]:
        """
        Return the 4-channel precision vector implied by the body's allocation
        rule for the given attended channel (-1 / None = no focus).

        Baseline agents never select a specific channel, so callers typically
        pass ``None`` here and get the uniform allocation.
        """
        from aif_paper1.affectworld.intero_precision import channel_precisions_np

        body = self.body
        idx = -1 if attended is None else int(attended)
        mode = str(getattr(body, "_intero_budget_mode", "none"))
        base = float(getattr(body, "INTERO_BASE_PRECISION", 0.65))
        att = float(getattr(body, "INTERO_ATT_PRECISION", 0.90))
        bg_floor = float(getattr(body, "_intero_budget_bg_floor", 0.05))
        total_cfg = float(getattr(body, "_intero_budget_total", -1.0))
        total = None if total_cfg <= 0.0 else total_cfg
        vec = channel_precisions_np(
            attended=idx,
            n_channels=4,
            base_precision=base,
            att_precision=att,
            mode=mode,
            bg_floor=bg_floor,
            total_budget=total,
        )
        return [float(x) for x in np.asarray(vec, dtype=np.float32).tolist()]

    def latest_beliefs(self) -> dict:
        """
        Return a dict mapping factor_names -> latest marginal belief vectors.

        Handles common pymdp shapes:
        - (n_states,)
        - (T, n_states)
        - (batch, T, n_states)  (this repo typically uses batch=1)
        """
        if self.qs is None:
            return {}
        beliefs = {}
        for name, q in zip(self.factor_names, self.qs):
            arr = np.array(q)
            if arr.ndim == 1:
                vec = arr
            elif arr.ndim == 2:
                # Either (T, n_states) or (batch, n_states); disambiguate by batch_size.
                if getattr(self.pymdp_agent, "batch_size", 1) == arr.shape[0]:
                    vec = arr[0]
                else:
                    vec = arr[-1]
            else:
                # (batch, T, n_states) -> take batch 0, latest time
                vec = arr[0, -1]
            beliefs[name] = vec
        return beliefs

    # ------------------------------------------------------------------
    # helper methods
    # ------------------------------------------------------------------
    def _decode_action(self, action: np.ndarray) -> tuple[Move, Interact]:
        """
        Decode the action returned by pymdp.

        Depending on pymdp version/config, `sample_action` may return either:
          - one action per *control factor* (Move, Interact), or
          - one action per *hidden factor* (after flattening B action deps).
        """
        a0 = action[0] if np.asarray(action).ndim == 2 else action
        a0 = np.asarray(a0).reshape(-1)

        # Alternate pymdp API: one action per control factor.
        if a0.shape[0] == 2:
            return Move(int(a0[0])), Interact(int(a0[1]))

        # If pymdp flattened multi-action dependencies, it can return one action per hidden factor.
        # Decode those back to the canonical (Move, Interact).
        if hasattr(self.pymdp_agent, "decode_multi_actions"):
            decoded = np.asarray(self.pymdp_agent.decode_multi_actions(jnp.asarray(a0[None, :])))[0]
            if decoded.shape[0] == 2:
                return Move(int(decoded[0])), Interact(int(decoded[1]))

        raise ValueError(f"Unexpected action shape {np.asarray(action).shape} (a0 has len {a0.shape[0]})")

    @staticmethod
    def _clamp(val: int, lo: int, hi: int) -> int:
        return max(lo, min(val, hi))

    @staticmethod
    def _delta_at(idx: int, length: int) -> np.ndarray:
        vec = np.zeros(length, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    @staticmethod
    def _soft_delta_at(idx: int, length: int, eps: float = 1e-3) -> np.ndarray:
        """
        Like a delta distribution, but with non-zero support everywhere so that
        noisy observations can still update beliefs away from the prior.
        """
        eps = float(max(0.0, min(eps, 1.0)))
        if length <= 1 or eps == 0.0:
            return Agent._delta_at(idx, length)
        vec = np.full(length, eps / (length - 1), dtype=np.float32)
        vec[int(idx)] = 1.0 - eps
        return vec

    @staticmethod
    def _discretise(fraction: float) -> int:
        fraction = np.clip(fraction, 0.0, 1.0)
        return int(round(fraction * BodyProcess.TYPEII_MAX_LEVEL))

    def _log_pref_low(self, n_levels: int) -> np.ndarray:
        """
        Preferences for 'low' values: monotonically decreasing with index.
        """
        return log_pref_monotonic_low(n_levels, preference_scale=float(self.preference_scale))

    def _log_pref_band(self, n_levels: int, *, band_min: float, band_max: float) -> np.ndarray:
        """
        Preferences for an acceptable range (homeostatic band).
        """
        return log_pref_range_band(
            n_levels,
            preference_scale=float(self.preference_scale),
            band_min=float(band_min),
            band_max=float(band_max),
        )

    def _log_pref_resources(self) -> np.ndarray:
        """
        Preferences over resource observations: penalise POISON.
        """
        scores = np.zeros(self.n_res, dtype=np.float32)
        scores[Resource.POISON] = -self.preference_scale
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        return np.log(probs + 1e-16)


__all__ = ["Agent"]

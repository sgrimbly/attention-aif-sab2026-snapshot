"""
Reduced-cost attentional Active Inference agent with externally selected kappa.
"""

from __future__ import annotations

import contextlib
import os

import jax.numpy as jnp
import numpy as np
from jax import random as jr

from aif_paper1.affectworld.helpers import Need, Move, Interact
from aif_paper1.agents.internal.habit_control import update_computation_budget
from aif_paper1.agents.internal.kappa_selection import (
    argmax_kappa_choice_np,
    argmin_kappa_choice_np,
    belief_need_actionability_np,
    belief_need_stats_np,
    default_kappa_state,
    focused_intero_templates_np,
    normalize_kappa_selector_mode,
)
from aif_paper1.agents.internal.runtime_support import _QUIET_SINK
from aif_paper1.agents.standard_active_inference import Agent
from aif_paper1.runtime.pymdp_integration import update_empirical_prior_compat


class ExternalKappaAgent(Agent):
    """
    Baseline-bodied agent with externally selected interoceptive focus.

    Here kappa is not a planned control factor. The planner
    still reasons over the baseline `(Move, Interact)` policy geometry, while a
    cheap selector chooses which need should receive high interoceptive precision
    for planning and the next timestep's interoception sample.

    On the maintained Paper 1 surface this is the canonical
    `Likelihood-Precision Selective Attention` family, interpreted over
    nutrient / hydration / oxygenation-proxy channels.
    """

    def __init__(
        self,
        env,
        body,
        *,
        kappa_selector_mode: str = "belief_argmax",
        kappa_selector_margin: float = 0.05,
        kappa_selector_min_dwell_steps: int = 3,
        kappa_selector_ucb_beta: float = 0.25,
        allow_oracle_kappa_selector: bool = False,
        kappa_modulates_planning: bool = True,
        kappa_modulates_inference: bool = True,
        **kwargs,
    ):
        self.kappa_selector_mode = normalize_kappa_selector_mode(kappa_selector_mode)
        self.kappa_selector_margin = float(max(0.0, kappa_selector_margin))
        self.kappa_selector_min_dwell_steps = int(max(1, int(kappa_selector_min_dwell_steps)))
        self.kappa_selector_ucb_beta = float(max(0.0, float(kappa_selector_ucb_beta)))
        self.allow_oracle_kappa_selector = bool(allow_oracle_kappa_selector)
        # When True (default), _runtime_A_for_kappa() reshapes the planning A
        # matrices with kappa-focused precision — the planner "sees" the attended
        # channel more sharply.  When False, kappa only affects state inference
        # (perception) and observation generation (body sampler); the planner uses
        # the learned A matrices as-is.  This flag enables decomposing the
        # planning-precision pathway from the Dirichlet-learning pathway.
        self.kappa_modulates_planning = bool(kappa_modulates_planning)
        # When True (default), Phase-1 state inference uses the kappa-shaped A
        # so the attended channel's likelihood is sharper during perception.
        # When False, state inference uses the uniform-precision A (equivalent
        # to kappa_idx=None); kappa still affects the body observation sampler
        # and (when kappa_modulates_planning=True) the planning A.  Setting
        # kappa_modulates_inference=False + kappa_modulates_planning=True gives
        # the EXP-9 "planning-only" ablation used to rebut the devil's-advocate
        # objection that the gain is entirely an inference-stage Dirichlet
        # acceleration.
        self.kappa_modulates_inference = bool(kappa_modulates_inference)
        self.current_kappa = int(default_kappa_state())
        self.last_kappa_switch_step = -int(self.kappa_selector_min_dwell_steps)
        self.decision_step = 0
        self.last_selector_scores: np.ndarray | None = None
        self.last_selector_uncertainty: np.ndarray | None = None
        self.last_selector_burden: np.ndarray | None = None
        self.last_selector_actionability: np.ndarray | None = None
        self.last_selected_kappa = int(default_kappa_state())
        self.last_q_pi = None
        self.last_G = None
        self._kappa_rng = np.random.default_rng(42)
        super().__init__(env, body, **kwargs)
        self.actinf_runtime_backend = "custom_step"

    def reset(self):
        """Reset baseline state plus the external κ selector's dwell bookkeeping."""
        super().reset()
        self.current_kappa = int(default_kappa_state())
        self.last_kappa_switch_step = -int(self.kappa_selector_min_dwell_steps)
        self.decision_step = 0
        self.last_selector_scores = None
        self.last_selector_uncertainty = None
        self.last_selector_burden = None
        self.last_selector_actionability = None
        self.last_selected_kappa = int(default_kappa_state())

    def _runtime_A_for_kappa(self, kappa_idx: int | None) -> list[jnp.ndarray]:
        """
        Rebuild the interoceptive likelihoods for a chosen external κ.

        κ is not a latent factor or a planned action here. This helper simply sharpens the selected intero channel's
        likelihood matrix at runtime and leaves all other model structure intact.
        The per-channel precisions must match the body sampler's budget rule so
        planning and observation generation share one attention semantics.
        """
        runtime_A = [jnp.asarray(a_m, dtype=jnp.float32) for a_m in self.A]
        templates = [np.asarray(self.A[m], dtype=np.float32) for m in range(3, 7)]
        budget_mode = str(getattr(self.body, "_intero_budget_mode", "none"))
        budget_floor = float(getattr(self.body, "_intero_budget_bg_floor", 0.05))
        budget_total_cfg = float(getattr(self.body, "_intero_budget_total", -1.0))
        budget_total = None if budget_total_cfg <= 0.0 else budget_total_cfg
        focused = focused_intero_templates_np(
            templates=templates,
            attended=None if kappa_idx is None or int(kappa_idx) == int(Need.NONE) else int(kappa_idx),
            base_precision=float(self.body.INTERO_BASE_PRECISION),
            att_precision=float(self.body.INTERO_ATT_PRECISION),
            budget_mode=budget_mode,
            budget_bg_floor=budget_floor,
            budget_total=budget_total,
        )
        for off, modality in enumerate(range(3, 7)):
            runtime_A[modality] = jnp.asarray(focused[off], dtype=jnp.float32)
        return runtime_A

    def _selector_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the quantities used to score candidate κ values.

        Reads burden and uncertainty from the inferred Type-I marginals.
        """
        q_latest = self._latest_qs_for_learning(self.qs)
        burden, uncertainty = belief_need_stats_np(
            q_energy=q_latest[2],
            q_hydration=q_latest[3],
            q_sleep=q_latest[4],
            q_oxygen=q_latest[5],
        )
        actionability = belief_need_actionability_np(
            q_pos=np.asarray(q_latest[0], dtype=np.float32),
            q_ctx=np.asarray(q_latest[1], dtype=np.float32),
            resource_likelihood=None if len(self.A) <= 1 else np.asarray(self.A[1], dtype=np.float32),
            trans=np.asarray(getattr(self.env, "_trans", None), dtype=np.int32)
            if getattr(self.env, "_trans", None) is not None
            else None,
            width=int(getattr(self.env, "w", 1)),
        )
        scores = np.asarray(burden, dtype=np.float32)
        return burden, uncertainty, actionability, scores

    def _select_kappa(self) -> int:
        """Update the externally controlled κ state for the next planning pass."""
        burden, uncertainty, actionability, scores = self._selector_stats()
        self.last_selector_burden = np.asarray(burden, dtype=np.float32)
        self.last_selector_uncertainty = np.asarray(uncertainty, dtype=np.float32)
        self.last_selector_actionability = np.asarray(actionability, dtype=np.float32)
        self.last_selector_scores = np.asarray(scores, dtype=np.float32)

        if self.kappa_selector_mode == "belief_argmax":
            selected = argmax_kappa_choice_np(scores)
        elif self.kappa_selector_mode == "belief_argmin":
            selected = argmin_kappa_choice_np(scores)
        else:
            selected = int(self._kappa_rng.integers(0, 4))
        if int(selected) != int(self.current_kappa):
            self.last_kappa_switch_step = int(self.decision_step)
        self.current_kappa = int(selected)
        self.last_selected_kappa = int(selected)
        return int(selected)

    def step(self, obs, typeII_obs):
        """
        Run one external-κ decision cycle.

        In order this method:
        1. packs the latest observations into categorical tensors;
        2. infers hidden states using the *current* externally selected κ;
        3. optionally applies oracle overrides, online ``A`` learning, and
           dynamic preference rewriting;
        4. chooses the *next* κ from current burden statistics;
        5. plans only over ``(Move, Interact)`` using likelihoods rebuilt for
           that selected κ, or bypasses the planner via the habit option path;
        6. samples the external action, updates diagnostics/budget, and returns
           the selected κ as the third output.

        Note that κ is not part of the policy space. It is selected by a cheap side mechanism and then
        injected into planning as a runtime likelihood change.
        """
        pos_obs = obs[0]
        res_obs = obs[1]
        season_obs_vec = obs[2]
        season_idx = int(np.argmax(season_obs_vec))

        observation = [
            self._oh_pos[int(pos_obs)],
            self._oh_res[int(res_obs)],
            self._oh_season_obs[season_idx],
            self._oh_typeII[int(typeII_obs[0])],
            self._oh_typeII[int(typeII_obs[1])],
            self._oh_typeII[int(typeII_obs[2])],
            self._oh_typeII[int(typeII_obs[3])],
        ]

        if self.prev_action is None:
            prior = self.pymdp_agent.D
        else:
            prior = update_empirical_prior_compat(self.pymdp_agent, self.prev_action, self.qs)

        quiet = os.environ.get("AFFECTWORLD_QUIET_PYMDP") == "1"
        planner_key = None
        if bool(getattr(self, "_planner_uses_rng", False)):
            self._planner_key, planner_key = jr.split(self._planner_key)
        budget_before = float(self.compute_budget)

        def _infer_cycle():
            # Phase 1: infer state under the currently active external κ.
            # If kappa_modulates_inference=False, use the uniform-precision A
            # (kappa_idx=None) — EXP-9 planning-only ablation.
            kappa_for_inference = int(self.current_kappa) if self.kappa_modulates_inference else None
            state_runtime_A = self._runtime_A_for_kappa(kappa_for_inference)
            self.qs = self._infer_states(observation, prior, state_runtime_A)
            self.qs = self._oracle_qs_override(self.qs, pos_obs=int(pos_obs), season_idx=int(season_idx))
            self._online_model_update(observation, self.qs)
            self._update_dynamic_resource_preferences()

            # Phase 2: pick the κ that will shape the upcoming planning pass.
            selected_kappa = self._select_kappa()

            # Phase 3: obtain a cheap option controller proposal, if enabled.
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
                return q_pi_local, G_local, "option", 1.0, False, {}, option_info, selected_kappa

            # Phase 4: plan over external actions.
            #
            # When kappa_modulates_planning is True (default), the planning A
            # matrices are rebuilt with kappa-focused precision so the planner
            # evaluates policies against sharpened likelihoods on the attended
            # channel.  This is one of TWO causal pathways through which kappa
            # affects behaviour (the other being cleaner observations feeding
            # faster Dirichlet learning).
            #
            # When kappa_modulates_planning is False, the planner uses the
            # learned A matrices without kappa precision reshaping.  This
            # isolates the Dirichlet-learning pathway: kappa still affects
            # observation quality (body sampler) and state inference (Phase 1),
            # but not the likelihoods the planner reasons over.
            runtime_A, runtime_C, runtime_pA, runtime_E, prior_info = self._planner_runtime_overrides(
                pos_obs=int(pos_obs),
                season_idx=int(season_idx),
                typeII_obs=np.asarray(typeII_obs, dtype=np.int32),
            )
            if self.kappa_modulates_planning:
                runtime_A = self._runtime_A_for_kappa(int(selected_kappa))
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
                    "planner_winner_survived_pruning": bool(unpruned_winner in kept_action_ids),
                    "unpruned_winner_action": int(unpruned_winner),
                    "pruned_winner_action": int(pruned_winner),
                }
            prior_info = {**dict(prior_info), "proposal": dict(proposal_info)}
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
            return q_pi_local, G_local, source, mix_alpha, True, prior_info, option_info, selected_kappa

        if quiet:
            with (
                contextlib.redirect_stdout(_QUIET_SINK),
                contextlib.redirect_stderr(_QUIET_SINK),
            ):
                q_pi, G, control_source, mix_alpha, planner_used, habit_prior_info, habit_option_info, selected_kappa = _infer_cycle()
                action = np.asarray(self._sample_action(q_pi))
        else:
            q_pi, G, control_source, mix_alpha, planner_used, habit_prior_info, habit_option_info, selected_kappa = _infer_cycle()
            action = np.asarray(self._sample_action(q_pi))

        if self.store_diagnostics:
            self.last_q_pi = np.asarray(q_pi[0], dtype=np.float32) if np.asarray(q_pi).ndim == 2 else np.asarray(q_pi, dtype=np.float32)
            self.last_G = np.asarray(G[0], dtype=np.float32) if np.asarray(G).ndim == 2 else np.asarray(G, dtype=np.float32)
            self.last_planner = self._consume_planner_debug_state()

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
        self.decision_step = int(self.decision_step) + 1
        return move, interact, Need(int(selected_kappa))

    def last_kappa_selection(self) -> int:
        """Return the attended channel index last selected by the κ controller (0..3)."""
        return int(self.last_selected_kappa)

    def latest_policy_diagnostics(self, top_k: int = 5) -> dict:
        """Return baseline diagnostics plus the most recent external-κ selector state."""
        out = super().latest_policy_diagnostics(top_k=top_k)
        selector = {
            "mode": str(self.kappa_selector_mode),
            "current_kappa": int(self.current_kappa),
            "last_selected_kappa": int(self.last_selected_kappa),
            "margin": float(self.kappa_selector_margin),
            "min_dwell_steps": int(self.kappa_selector_min_dwell_steps),
            "ucb_beta": float(self.kappa_selector_ucb_beta),
        }
        if self.last_selector_burden is not None:
            selector["burden"] = np.asarray(self.last_selector_burden, dtype=np.float32).tolist()
        if self.last_selector_uncertainty is not None:
            selector["uncertainty"] = np.asarray(self.last_selector_uncertainty, dtype=np.float32).tolist()
        if self.last_selector_actionability is not None:
            selector["actionability"] = np.asarray(self.last_selector_actionability, dtype=np.float32).tolist()
        if self.last_selector_scores is not None:
            selector["scores"] = np.asarray(self.last_selector_scores, dtype=np.float32).tolist()
        out["kappa_selector"] = selector
        return out


__all__ = ["ExternalKappaAgent"]

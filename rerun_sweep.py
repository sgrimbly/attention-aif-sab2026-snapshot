"""
rerun_sweep.py
==============
Core-agent survival sweep for the SAB 2026 Paper 1 reproducibility snapshot.

Called by ``reproduce.py --rerun <claim>``.  Not intended to be run directly,
but can be:

    python rerun_sweep.py --claim headline
    python rerun_sweep.py --claim headline --quick   # 1 seed × easy layouts

Runtime notes
-------------
Full sweep (8 seeds × 6 easy-tier layouts × 3 agents × 8 episodes × 60 steps)
on a modern GPU takes ~2 h.  On a CPU with planning_horizon=3 and
policy_mode=full it is considerably longer; the --quick flag switches to
planning_horizon=1 (25 policies) for fast local validation (~10 min on CPU).

The paper results were produced with --full (planning_horizon=3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── agent definitions ─────────────────────────────────────────────────────

def _agent_configs(claim: str) -> list[dict]:
    """Return the list of agent specs for a given claim."""
    if claim == "headline":
        return [
            {"tag": "baseline",   "cls": "BaselineAgent",      "extra": {}},
            {"tag": "attn_argmax","cls": "ExternalKappaAgent",  "extra": {"kappa_selector_mode": "belief_argmax"}},
            {"tag": "rev_argmin", "cls": "ExternalKappaAgent",  "extra": {"kappa_selector_mode": "belief_argmin"}},
        ]
    if claim == "random-kappa":
        return [
            {"tag": "baseline",    "cls": "BaselineAgent",     "extra": {}},
            {"tag": "attn_argmax", "cls": "ExternalKappaAgent", "extra": {"kappa_selector_mode": "belief_argmax"}},
            {"tag": "attn_random", "cls": "ExternalKappaAgent", "extra": {"kappa_selector_mode": "random"}},
        ]
    if claim == "nonstat":
        return [
            {"tag": "baseline",   "cls": "BaselineAgent",      "extra": {}},
            {"tag": "attn_argmax","cls": "ExternalKappaAgent",  "extra": {"kappa_selector_mode": "belief_argmax"}},
        ]
    raise ValueError(f"Unknown claim: {claim!r}")


def _layout_filter(layouts: list[dict], claim: str, quick: bool) -> list[dict]:
    """Select layouts appropriate for the claim."""
    if claim == "nonstat":
        # Non-stationarity uses mutation_enabled=True over easy-tier layouts.
        easy = [l for l in layouts if l.get("tier") == "easy"]
        return easy[:2] if quick else easy
    # headline and random-kappa use easy-tier layouts.
    easy = [l for l in layouts if l.get("tier") == "easy"]
    return easy[:2] if quick else easy


def _env_extra_kwargs(claim: str) -> dict:
    """Extra AffectWorldEnv kwargs for a claim."""
    if claim == "nonstat":
        return {"mutation_enabled": True, "mutation_prob": 0.05}
    return {}


def _run_episodes_with_fresh_resource_map(
    *,
    sim: Any,
    env: Any,
    base_res_map: Any,
    n_episodes: int,
    start_pos: int,
) -> int:
    """Run episodes while restoring the layout before every trial."""
    survived = 0
    for _ in range(n_episodes):
        env.res_map[:] = base_res_map
        if sim.run_episode(start_pos=start_pos).terminated_by == "timeout":
            survived += 1
    return survived


# ── main sweep ────────────────────────────────────────────────────────────

def run_sweep(
    *,
    claim: str,
    layout_bank_path: Path,
    n_seeds: int,
    n_episodes: int,
    max_steps: int,
    planning_horizon: int,
    out_path: Path | None,
) -> dict:
    import numpy as np

    from aif_paper1.affectworld.affect_world_env import AffectWorldEnv
    from aif_paper1.affectworld.body import BodyProcess
    from aif_paper1.affectworld.configs import BodyConfig
    from aif_paper1.affectworld.regimes import apply_simple_regime
    from aif_paper1.affectworld.simulation import Simulation
    from aif_paper1.agents.standard_active_inference import Agent as BaselineAgent
    from aif_paper1.agents.likelihood_precision_selective_attention import ExternalKappaAgent

    _CLS = {"BaselineAgent": BaselineAgent, "ExternalKappaAgent": ExternalKappaAgent}

    with open(layout_bank_path) as f:
        bank = json.load(f)
    layouts = bank["layouts"]

    quick = n_seeds <= 2
    target_layouts = _layout_filter(layouts, claim, quick)
    agent_cfgs   = _agent_configs(claim)
    env_extra    = _env_extra_kwargs(claim)

    # Shared agent kwargs (matches actinf_learning_fixed_layout_nosleep.json).
    # Note: intero_budget_* are BodyProcess/BodyConfig settings, not agent kwargs.
    BASE_KWARGS = dict(
        planning_horizon=planning_horizon,
        planner="efe",
        policy_mode="full",
        unknown_model=True,
        unknown_model_init_concentration=0.1,
        unknown_model_learning_rate=1.0,
        unknown_model_learn_resource=True,
        unknown_model_learn_intero=False,
        unknown_model_update_online=True,
        dynamic_resource_preferences=False,
        habit_mode="none",
        jit_inference=True,
    )

    # Body configuration matching the paper's intero-budget settings.
    # Fatigue is disabled to match actinf_learning_fixed_layout_nosleep.json
    # (max_sleepdebt=1, inc_sleepdebt=0): without these overrides the dataclass
    # defaults (max_sleepdebt=20, inc_sleepdebt=1) let sleep_debt accumulate, the
    # kappa selector reallocates precision away from hunger, and the agent starves.
    BODY_CFG = BodyConfig(
        intero_budget_mode="soft",
        intero_budget_bg_floor=0.05,
        intero_budget_total=2.6,
        max_sleepdebt=1,
        inc_sleepdebt=0,
        lethal_sleep=False,
    )

    total_runs = len(target_layouts) * n_seeds * len(agent_cfgs)
    print(f"[sweep] claim={claim!r}  layouts={len(target_layouts)}  "
          f"seeds={n_seeds}  agents={len(agent_cfgs)}  episodes={n_episodes}  "
          f"steps={max_steps}  horizon={planning_horizon}")
    print(f"[sweep] total runs: {total_runs}  (each = {n_episodes} episodes)")
    print()

    results: dict = {
        "claim": claim,
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "planning_horizon": planning_horizon,
        "per_layout": {},
    }

    run_idx = 0
    t_sweep_start = time.time()

    for layout in target_layouts:
        lid = layout["layout_id"]
        tier = layout.get("tier", "?")
        results["per_layout"][lid] = {"tier": tier, "per_agent": {}}

        for acfg in agent_cfgs:
            tag = acfg["tag"]
            AgentCls = _CLS[acfg["cls"]]
            extra = acfg["extra"]

            seed_survivals: list[float] = []

            for seed in range(n_seeds):
                run_idx += 1
                t0 = time.time()

                # Fresh env + apply the layout bank's resource placement.
                env = AffectWorldEnv(width=6, height=6, seed=seed, **env_extra)
                apply_simple_regime(
                    env,
                    food_positions=layout["food_positions"],
                    water_positions=layout["water_positions"],
                    sleep_positions=layout.get("sleep_positions", []),
                    start_pos=layout["start_pos"],
                )
                base_res_map = np.array(env.res_map, copy=True)
                body = BodyProcess(config=BODY_CFG)
                np.random.seed(seed)   # reproducible body interoception

                agent = AgentCls(env=env, body=body, **BASE_KWARGS, **extra)
                sim = Simulation(env, body, agent, max_steps=max_steps)

                # Episodes must start at the layout's fixed start_pos; otherwise
                # env.reset(pos=None) samples a uniformly random cell, dropping the
                # agent away from the resource cluster (→ spurious 0 survival).
                survived = _run_episodes_with_fresh_resource_map(
                    sim=sim,
                    env=env,
                    base_res_map=base_res_map,
                    n_episodes=n_episodes,
                    start_pos=layout["start_pos"],
                )
                sr = survived / n_episodes
                seed_survivals.append(sr)

                elapsed = time.time() - t0
                remaining = (total_runs - run_idx) * elapsed  # crude estimate
                print(
                    f"  [{run_idx:3d}/{total_runs}] {lid} / {tag} / seed={seed} "
                    f"survival={sr:.2f}  ({elapsed:.0f}s this run, "
                    f"~{remaining/60:.0f} min remaining)"
                )
                sys.stdout.flush()

            mean_sr = float(sum(seed_survivals) / len(seed_survivals)) if seed_survivals else float("nan")
            results["per_layout"][lid]["per_agent"][tag] = {
                "mean_survival": round(mean_sr, 4),
                "per_seed": [round(s, 4) for s in seed_survivals],
            }

    elapsed_total = time.time() - t_sweep_start

    # ── Summary table ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Sweep complete in {elapsed_total/60:.1f} min")
    print()
    print(f"{'Layout':<18}  {'Tier':<8}", end="")
    for acfg in agent_cfgs:
        print(f"  {acfg['tag']:>14}", end="")
    print()
    print("-" * 60)

    for lid, ldata in results["per_layout"].items():
        print(f"{lid:<18}  {ldata['tier']:<8}", end="")
        for acfg in agent_cfgs:
            sr = ldata["per_agent"].get(acfg["tag"], {}).get("mean_survival", float("nan"))
            print(f"  {sr:>14.3f}", end="")
        print()

    print()
    # Mean across layouts
    for acfg in agent_cfgs:
        vals = [
            ldata["per_agent"].get(acfg["tag"], {}).get("mean_survival", float("nan"))
            for ldata in results["per_layout"].values()
        ]
        valid = [v for v in vals if v == v]  # filter NaN
        mean = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {acfg['tag']}: mean across {len(valid)} layouts = {mean:.3f}")

    results["total_seconds"] = round(elapsed_total, 1)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n[sweep] results saved to {out_path}")

    return results


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--claim", required=True,
                   choices=["headline", "random-kappa", "nonstat"],
                   help="Which experimental claim to replicate")
    p.add_argument("--quick", action="store_true",
                   help="Fast validation: 1 seed × 2 easy layouts × planning_horizon=1")
    p.add_argument("--seeds", type=int, default=None,
                   help="Override number of seeds (paper default: 8)")
    p.add_argument("--episodes", type=int, default=8,
                   help="Episodes per seed per layout (paper default: 8)")
    p.add_argument("--steps", type=int, default=60,
                   help="Max steps per episode (paper default: 60)")
    p.add_argument("--out", type=str, default=None,
                   help="Path to save JSON results (default: results/<claim>.json)")
    args = p.parse_args()

    _ROOT_SNAP = Path(__file__).resolve().parent
    layout_bank = _ROOT_SNAP / "configs" / "known_model_layout_bank.json"

    if args.quick:
        n_seeds         = args.seeds if args.seeds is not None else 1
        planning_horizon = 1
        print("[sweep] --quick mode: planning_horizon=1, 1 seed, 2 layouts")
    else:
        n_seeds          = args.seeds if args.seeds is not None else 8
        planning_horizon = 3
        print(f"[sweep] Full mode: planning_horizon=3, {n_seeds} seeds")
        print("[sweep] NOTE: on CPU this may take many hours.")
        print("[sweep] Use --quick for rapid local validation.")
        print()

    out_path = (
        Path(args.out) if args.out
        else _ROOT_SNAP / "results" / f"{args.claim}.json"
    )

    run_sweep(
        claim=args.claim,
        layout_bank_path=layout_bank,
        n_seeds=n_seeds,
        n_episodes=args.episodes,
        max_steps=args.steps,
        planning_horizon=planning_horizon,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()

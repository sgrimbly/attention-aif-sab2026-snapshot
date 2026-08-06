"""Reproducibility entry point for the SAB 2026 Paper 1 anonymous snapshot.

Usage:
    python reproduce.py --test                # smoke test: ~30 s on any modern CPU
    python reproduce.py --rerun headline      # rerun the headline experiment
    python reproduce.py --rerun random-kappa  # rerun the random-kappa ablation
    python reproduce.py --rerun nonstat       # rerun the non-stationarity test
    python reproduce.py --list                # list available reruns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _patch_pymdp_static_policies() -> None:
    """inferactively-pymdp>=1.0.0 on PyPI marks the `policies` field of
    PymdpAgent as `field(static=True)`.  This causes equinox to hash the
    policy JAX array as a compile-time constant, which hangs indefinitely.
    The development version used to produce the paper's results does not
    mark this field as static.  This patch removes the annotation from the
    installed package before any pymdp module is imported."""
    import importlib.util
    import re

    spec = importlib.util.find_spec("pymdp.agent")
    if not (spec and spec.origin):
        return
    with open(spec.origin) as f:
        src = f.read()
    fixed = re.sub(
        r"(policies\s*:\s*\S+)\s*=\s*field\(static=True\)",
        r"\1",
        src,
    )
    if fixed == src:
        return
    with open(spec.origin, "w") as f:
        f.write(fixed)
    for key in [k for k in sys.modules if k.startswith("pymdp")]:
        del sys.modules[key]
    print("[setup] patched pymdp.agent: removed static annotation from policies field")


_patch_pymdp_static_policies()

RERUN_CLAIMS = {
    "headline":     "Easy-tier headline sweep (baseline / attn_argmax / rev_argmin, 8 seeds × 6 layouts × 3 agents × 8 eps)",
    "random-kappa": "Random-kappa ablation (baseline / attn_argmax / attn_random, 8 seeds × 6 layouts)",
    "nonstat":      "Non-stationarity sweep (mutation_enabled=True, baseline / attn_argmax, 8 seeds × 6 layouts)",
}


def run_test() -> None:
    """Minimal smoke test: {baseline, attn_extk} × 3 episodes, 20 steps each.

    Verifies that all imports resolve, agent construction succeeds, and the
    simulation loop completes without error.

    Note on planning horizon: the smoke test uses ``planning_horizon=1``
    (25 policies) rather than the paper's ``planning_horizon=3`` (15 625
    policies).  The full horizon is exercised in the --rerun paths.
    ``jit_inference=True`` is set so that XLA traces the EFE scorer once at
    the first step and reuses the compiled kernel thereafter; without it each
    step dispatches through Python-level vmap, which is ~10× slower.

    Expected wall time: ~30 s on a modern CPU.
    """
    try:
        import numpy as np

        from aif_paper1.affectworld.affect_world_env import AffectWorldEnv
        from aif_paper1.affectworld.body import BodyProcess
        from aif_paper1.affectworld.helpers import Resource
        from aif_paper1.affectworld.simulation import Simulation
        from aif_paper1.agents.standard_active_inference import Agent as BaselineAgent
        from aif_paper1.agents.likelihood_precision_selective_attention import ExternalKappaAgent
        from rerun_sweep import _run_episodes_with_fresh_resource_map
    except ImportError as e:
        print(f"[test] Import failed: {e}")
        print("       Ensure you have run: pip install -e .")
        sys.exit(1)

    print("[test] Imports OK.")

    class _MutatingSimulation:
        def __init__(self, env: AffectWorldEnv) -> None:
            self.env = env
            self.starts: list[np.ndarray] = []

        def run_episode(self, *, start_pos: int):
            del start_pos
            self.starts.append(np.array(self.env.res_map, copy=True))
            self.env.res_map[0, 0] = Resource.POISON
            return type("Episode", (), {"terminated_by": "timeout"})()

    reset_env = AffectWorldEnv(width=2, height=2, seed=0)
    reset_env.res_map[:] = Resource.AIR
    reset_env.res_map[0, 0] = Resource.FOOD
    base_res_map = np.array(reset_env.res_map, copy=True)
    reset_sim = _MutatingSimulation(reset_env)
    reset_survivals = _run_episodes_with_fresh_resource_map(
        sim=reset_sim,
        env=reset_env,
        base_res_map=base_res_map,
        n_episodes=3,
        start_pos=0,
    )
    if reset_survivals != 3 or any(
        not np.array_equal(start, base_res_map) for start in reset_sim.starts
    ):
        raise AssertionError("resource-map mutation leaked across trial boundaries")
    print("[test] Non-stationarity layout reset OK.")

    # Core Paper 1 kwargs — planner, learning mode, and habit flags match
    # actinf_learning_fixed_layout_nosleep.json exactly.  planning_horizon is
    # reduced to 1 for the smoke test (see docstring above).
    AGENT_KWARGS = dict(
        planning_horizon=1,        # reduced from paper's 3 for smoke-test speed
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
        jit_inference=True,        # compile EFE scorer on first call; reuse after
    )

    agents_under_test = [
        ("baseline",  BaselineAgent,   {}),
        ("attn_extk", ExternalKappaAgent, {"kappa_selector_mode": "belief_argmax"}),
    ]

    N_EPISODES = 3
    MAX_STEPS  = 20

    results = {}
    for agent_tag, AgentClass, extra in agents_under_test:
        try:
            env = AffectWorldEnv(width=6, height=6)
            body = BodyProcess()
            agent = AgentClass(env=env, body=body, **AGENT_KWARGS, **extra)
            sim = Simulation(env, body, agent, max_steps=MAX_STEPS)
            survived = sum(
                1 for _ in range(N_EPISODES)
                if sim.run_episode().terminated_by == "timeout"
            )
            sr = survived / N_EPISODES
            results[agent_tag] = sr
            print(f"[test] {agent_tag}: survival rate = {sr:.2f} ({survived}/{N_EPISODES})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[test] {agent_tag}: ERROR — {type(e).__name__}: {e}")
            results[agent_tag] = None

    print("\n[test] Smoke test complete.")
    if any(v is None for v in results.values()):
        print("[test] WARNING: some agents errored; see output above.")
        sys.exit(1)


def run_rerun(claim: str, *, quick: bool = False, seeds: int | None = None) -> None:
    if claim not in RERUN_CLAIMS:
        print(f"Unknown claim '{claim}'. Run `python reproduce.py --list` to see available reruns.")
        sys.exit(1)

    print(f"[rerun] {claim}: {RERUN_CLAIMS[claim]}")
    if not quick:
        print()
        print("[rerun] Runtime note: with planning_horizon=3 on a single CPU core each")
        print("[rerun] infer_policies call costs ~12 s → the full sweep can take many")
        print("[rerun] hours on CPU.  Use --quick for rapid local validation (~10 min).")
        print()

    # Import the sweep runner co-located with this script.
    import importlib.util
    sweep_path = _ROOT / "rerun_sweep.py"
    if not sweep_path.exists():
        print(f"[rerun] ERROR: rerun_sweep.py not found at {sweep_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("rerun_sweep", sweep_path)
    sweep_mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(sweep_mod)                   # type: ignore[union-attr]

    sweep_mod.run_sweep(
        claim=claim,
        layout_bank_path=_ROOT / "configs" / "known_model_layout_bank.json",
        n_seeds=seeds if seeds is not None else (1 if quick else 8),
        n_episodes=8,
        max_steps=60,
        planning_horizon=1 if quick else 3,
        out_path=_ROOT / "results" / f"{claim}{'_quick' if quick else ''}.json",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Reproducibility entry point for the SAB 2026 Paper 1 snapshot."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true",
                   help="Run a minimal smoke test (~30 s)")
    g.add_argument("--rerun", metavar="CLAIM",
                   help="Rerun a full experiment claim")
    g.add_argument("--list", action="store_true",
                   help="List available --rerun claims")
    p.add_argument("--quick", action="store_true",
                   help="With --rerun: use planning_horizon=1 and 1 seed for rapid local validation")
    p.add_argument("--seeds", type=int, default=None,
                   help="With --rerun: override number of seeds (paper default: 8)")
    args = p.parse_args()

    if args.list:
        print("Available --rerun claims:")
        for name, desc in RERUN_CLAIMS.items():
            print(f"  {name:20s}  {desc}")
        print()
        print("Flags:")
        print("  --quick          planning_horizon=1, 1 seed — fast local validation")
        print("  --seeds N        override seed count")
    elif args.test:
        run_test()
    elif args.rerun:
        run_rerun(args.rerun, quick=args.quick, seeds=args.seeds)


if __name__ == "__main__":
    main()

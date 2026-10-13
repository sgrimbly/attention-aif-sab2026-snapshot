# Reproducibility snapshot — SAB 2026 Paper 1

Public reproducibility snapshot for *Interoceptive Attention as Dynamic
Homeostatic Prioritization in a Foraging Agent*, accepted at SAB 2026. It
contains the code, configurations, and layout banks needed to reproduce the
survival experiments reported in the paper.

---

## Table of contents

1. [Overview](#overview)
2. [Supplementary material](#supplementary-material)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Reproducing the experiments](#reproducing-the-experiments)
7. [Repository structure](#repository-structure)
8. [Environment — AffectWorld](#environment--affectworld)
9. [Agents](#agents)
10. [Experimental configurations](#experimental-configurations)
11. [Expected results and runtime](#expected-results-and-runtime)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The paper introduces **Likelihood-Precision Selective Attention**: an extension
to standard discrete Active Inference in which a cheap, belief-driven selector
determines which interoceptive channel receives elevated observation precision
at each timestep. The precision selector — denoted **κ** — operates *outside*
the planner's policy space, so it adds no policies and incurs no additional
EFE computation.

Three experimental claims are made reproducible here:

| Claim | Description |
|---|---|
| `headline` | Attentional agent (`belief_argmax` κ) vs. baseline vs. reverse-attentional control (`belief_argmin` κ) across 6 easy-tier layouts × 8 seeds |
| `random-kappa` | `belief_argmax` vs. uniform-random κ selection vs. baseline — isolates the benefit of belief-driven selection |
| `nonstat` | Same agent comparison under non-stationary food→poison mutation (`mutation_prob=0.05`), testing whether learned attention remains adaptive |

## Supplementary material

The standalone [SAB 2026 supplementary material](paper/SAB2026_supplement.pdf)
contains Sections A--E that were not part of the 12-page camera-ready
proceedings paper. The accepted manuscript itself is not distributed from this
repository.

A few statements in the 12-page paper do not match the runs as configured
here, including the planner objective and the policy precision γ. The
[paper page](https://stjohngrimbly.com/sab2026/) lists these as short notes.
Section A of the supplement gives the as-run planner settings.

---

## Requirements

- **Python** ≥ 3.10
- **JAX** ≥ 0.9.0 with **jaxlib** ≥ 0.9.0
- **equinox** ≥ 0.13.4
- **inferactively-pymdp** ≥ 1.0.2 (PyPI)
- **numpy** ≥ 1.24

A GPU is not required but substantially reduces runtime for the full sweep
(see [Expected results and runtime](#expected-results-and-runtime)).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

All required packages are declared in `pyproject.toml` and will be installed
automatically. No other setup is needed.

### Optional — NVIDIA GPU support

The default install pulls the CPU build of JAX. For NVIDIA GPU acceleration,
install the CUDA-enabled JAX wheel after the standard install:

```bash
pip install --upgrade "jax[cuda12]"
```

Without this, JAX runs on CPU and prints `An NVIDIA GPU may be present on this
machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.` —
harmless, but it means the GPU is not being used.

> **Note — pymdp compatibility patch**  
> `inferactively-pymdp ≥ 1.0.0` on PyPI marks the `policies` field as
> `field(static=True)`, which causes equinox to hash the policy JAX array as a
> compile-time constant and hang indefinitely. `reproduce.py` patches this
> automatically on import. No manual action is required.

---

## Quick start

```bash
# Verify the installation — smoke test (~30 s on CPU)
python reproduce.py --test

# List all reproducible experiment claims
python reproduce.py --list
```

The smoke test runs `BaselineAgent` and `ExternalKappaAgent` for 3 episodes ×
20 steps each with `planning_horizon=1`. It confirms that all imports resolve,
agent construction succeeds, and the simulation loop completes without error.

---

## Reproducing the experiments

### Fast validation (minutes)

```bash
python reproduce.py --rerun headline     --quick
python reproduce.py --rerun random-kappa --quick
python reproduce.py --rerun nonstat      --quick
```

`--quick` uses `planning_horizon=1` and 1 seed × 2 layouts. Suitable for
verifying that the code runs correctly without committing to the full runtime.

### Full sweep (paper results)

```bash
python reproduce.py --rerun headline
python reproduce.py --rerun random-kappa
python reproduce.py --rerun nonstat
```

The sweep's structural parameters match the paper. To keep the reproduction
tractable on a single machine, this snapshot uses a reduced statistical budget
(8 seeds × 8 episodes) rather than the paper's full 32 seeds × 100 episodes:

| Parameter | This snapshot | Paper |
|---|---|---|
| `planning_horizon` | 3 (15 625 policies) | 3 (15 625 policies) |
| Seeds | 8 | 32 |
| Layouts | 6 (easy tier) | 6 (easy tier) |
| Episodes per run | 8 | 100 |
| Max steps per episode | 60 | 60 |
| Planner | EFE (expected free energy) | EFE (expected free energy) |
| Model learning | online, resource channel only | online, resource channel only |
| Habit mode | none | none |

The reduced seed/episode budget widens confidence intervals but reproduces the
agent ordering and survival behaviour reported in the paper. Results are written
to `results/<claim>.json`.

### Seed and layout overrides

```bash
# 4 seeds, 2 layouts — intermediate validation
python reproduce.py --rerun headline --seeds 4

# Custom episodes
python rerun_sweep.py --claim headline --seeds 8 --episodes 16 --out results/headline_16ep.json
```

---

## Repository structure

```
.
├── reproduce.py                   Entry point: smoke test + experiment reruns
├── rerun_sweep.py                 Core sweep runner (called by reproduce.py)
├── pyproject.toml                 Package metadata and dependencies
│
├── aif_paper1/                    Python package
│   ├── agents/
│   │   ├── standard_active_inference.py      BaselineAgent
│   │   ├── likelihood_precision_selective_attention.py  ExternalKappaAgent (κ)
│   │   └── internal/
│   │       ├── kappa_selection.py            κ selector utilities
│   │       ├── habit_control.py              Policy-prior helpers
│   │       ├── preferences.py                Homeostatic preference functions
│   │       └── runtime_support.py            EFE scoring helpers
│   │
│   ├── affectworld/
│   │   ├── affect_world_env.py               AffectWorld grid environment
│   │   ├── body.py                           Physiological body process
│   │   ├── configs.py                        BodyConfig / EnvConfig dataclasses
│   │   ├── simulation.py                     Episode runner
│   │   ├── regimes.py                        Layout-bank resource placement
│   │   ├── intero_precision.py               κ-to-precision allocation
│   │   ├── canonical_step.py                 Shared observation/action step
│   │   ├── context_layout.py                 Season-to-layout mapping
│   │   ├── logging_utils.py                  Episode logging
│   │   └── helpers.py                        Enums: Move, Interact, Need, Resource
│   │
│   ├── planning/
│   │   ├── efe_repo_local.py                 Repo-local EFE scorer
│   │   ├── policy_lib.py                     Policy construction (full/restricted)
│   │   ├── policy_space.py                   Planner name / policy-mode normalisation
│   │   ├── planner_utils.py                  Shared planning utilities
│   │   └── routing.py                        Planner routing metadata
│   │
│   └── runtime/
│       ├── pymdp_integration.py              pymdp compatibility shims
│       └── public_surface.py                 Agent metadata and label maps
│
├── configs/
│   ├── known_model_layout_bank.json           12 layouts across easy/medium/far tiers
│   ├── known_model_layout_bank_extended.json  Extended layout bank (Appendix)
│   └── actinf_learning_fixed_layout_nosleep.json  Canonical experiment configuration
│
└── paper/
    ├── SAB2026_supplement.pdf                  Supplementary Sections A--E
    └── README.md                               Paper-material licence scope
```

---

## Environment — AffectWorld

AffectWorld is a 6 × 6 discrete grid world designed for Solmsian Active
Inference experiments. At each timestep the agent receives:

| Observation | Type | Description |
|---|---|---|
| `pos_idx` | int | Proprioceptive grid position (0 … 35) |
| `res_idx` | int | Tile resource type: AIR, WATER, FOOD, SLEEP, POISON |
| `season` | 5-vector | One-hot seasonal context (NoContext / Spring / Summer / Autumn / Winter) |
| Interoceptive levels | 4 × int | Noisy observations of nutrient, hydration, oxygenation-proxy, dormant-rest needs |

The agent's action space is `Move × Interact`:

- **Move**: UP, DOWN, LEFT, RIGHT, REST (5 options)
- **Interact**: NONE, EAT, DRINK, BREATHE, SLEEP (5 options)

This gives 25 joint actions and 15 625 full-horizon policies at `planning_horizon=3`.

### Physiological body (`BodyProcess`)

A separate `BodyProcess` module tracks four homeostatic variables:

| Variable | Lethal threshold | Relieved by |
|---|---|---|
| Nutrient (energy) | → 0 | EAT on FOOD tile |
| Hydration | → 0 | DRINK on WATER tile |
| Oxygenation | → 0 | BREATHE on WATER tile (or open air) |
| Dormant-rest debt | — | SLEEP on SLEEP tile |

Interoceptive observations are sampled with precision determined by the active
κ channel. The canonical lane uses a soft budget (`intero_budget_total=2.6`,
`intero_budget_bg_floor=0.05`), so attending one channel raises its precision
while reducing the others proportionally.

### Food-to-poison non-stationarity (`nonstat` claim)

The `nonstat` experiment enables `mutation_enabled=True` with
`mutation_prob=0.05`. On each step that the agent enters a FOOD destination
cell, there is a 5 % chance that the tile turns to POISON in place before its
effect is applied. The tile stays poisoned for the rest of that trial and the
layout resets between trials. The agent must therefore keep updating its
resource model rather than exploiting a fixed layout.

---

## Agents

### `BaselineAgent` — Standard Active Inference

```python
from aif_paper1.agents.standard_active_inference import Agent as BaselineAgent

agent = BaselineAgent(
    env=env,
    body=body,
    planning_horizon=3,
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
```

The baseline uses all four interoceptive channels with a fixed, uniform
precision budget. It selects `(Move, Interact)` policies by minimising
expected free energy with respect to homeostatic preferences.

### `ExternalKappaAgent` — Likelihood-Precision Selective Attention

```python
from aif_paper1.agents.likelihood_precision_selective_attention import ExternalKappaAgent

agent = ExternalKappaAgent(
    env=env,
    body=body,
    kappa_selector_mode="belief_argmax",  # or "belief_argmin" or "random"
    # ... same base kwargs as BaselineAgent
)
```

The attentional agent inherits the baseline's `(Move, Interact)` planning
geometry. Before each planning call κ is updated by the selector and the
interoceptive likelihood matrices `A_intero` are rebuilt with the new
precision allocation. The planner then evaluates EFE using the
κ-modulated likelihoods.

Three κ-selector modes are used in the paper:

| Mode | Description | Used in |
|---|---|---|
| `belief_argmax` | Attend the channel with the highest inferred need burden | `headline`, `random-kappa`, `nonstat` |
| `belief_argmin` | Attend the channel with the *lowest* burden — reversed control | `headline` |
| `random` | Uniform-random channel each step | `random-kappa` |

---

## Experimental configurations

### Layout bank (`configs/known_model_layout_bank.json`)

Twelve pre-generated layouts across three difficulty tiers:

| Tier | Layouts | Characteristic |
|---|---|---|
| `easy` | 6 | Resources placed close to the start position |
| `medium` | 3 | Moderate navigation required |
| `far` | 3 | Resources at maximum distance from start |

All three claims use the 6 easy-tier layouts. The extended bank
(`known_model_layout_bank_extended.json`) contains additional layouts
used in the Appendix analysis.

Each layout specifies:
- `food_positions`, `water_positions`, `sleep_positions` — resource tile indices
- `start_pos` — agent starting position
- `tier` — difficulty classification

### Canonical experiment configuration (`configs/actinf_learning_fixed_layout_nosleep.json`)

The JSON file records the exact hyperparameter values used to produce the
paper's results. Key settings:

```json
{
  "planner": "efe",
  "policy_mode": "full",
  "planning_horizon": 3,
  "unknown_model": true,
  "unknown_model_init_concentration": 0.1,
  "unknown_model_learning_rate": 1.0,
  "unknown_model_learn_resource": true,
  "unknown_model_learn_intero": false,
  "unknown_model_update_online": true,
  "dynamic_resource_preferences": false,
  "habit_mode": "none",
  "intero_budget_mode": "soft",
  "intero_budget_bg_floor": 0.05,
  "intero_budget_total": 2.6
}
```

---

## Expected results and runtime

### Runtime estimates

| Mode | Flags | Seeds × layouts | Approx. wall time (CPU) |
|---|---|---|---|
| Smoke test | `--test` | 1 seed × 1 layout × 3 ep | ~30 s |
| Quick validation | `--quick` | 1 seed × 2 layouts | ~5–10 min |
| Full sweep (one claim) | *(none)* | 8 seeds × 6 layouts | ~2 h on GPU, 6–12 h on CPU |

`--quick` reduces `planning_horizon` to 1 (25 policies instead of 15 625),
which is ~10× faster. Full results require `planning_horizon=3`.

`jit_inference=True` (the default) compiles the EFE scorer with XLA on the
first step and reuses the compiled kernel thereafter. Without JIT, each
step dispatches 25 separate XLA kernels via Python-level vmap — approximately
10× slower.

### Interpreting survival rate

Each episode runs for at most `max_steps=60` steps. A run **survives** if the
episode ends by timeout (`terminated_by == "timeout"`) rather than by
physiological collapse. Survival rate = fraction of episodes survived across
all seeds × episodes for a given layout/agent combination.

The headline claim reports survival rate improvements of `belief_argmax` over
both `baseline` and `belief_argmin` across easy-tier layouts. The
`random-kappa` claim shows that random κ selection provides no consistent
benefit over the baseline, supporting the necessity of belief-driven
selection.

### Result files

After a `--rerun` call, results are saved to `results/<claim>.json`:

```json
{
  "claim": "headline",
  "n_seeds": 8,
  "n_episodes": 8,
  "max_steps": 60,
  "planning_horizon": 3,
  "per_layout": {
    "km_easy_00": {
      "tier": "easy",
      "per_agent": {
        "baseline":    {"mean_survival": 0.625, "per_seed": [...]},
        "attn_argmax": {"mean_survival": 0.875, "per_seed": [...]},
        "rev_argmin":  {"mean_survival": 0.250, "per_seed": [...]}
      }
    }
  }
}
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pymdp'`**  
Run `pip install -e .` inside the virtual environment before invoking
`reproduce.py`.

**`Hangs indefinitely on first run`**  
This is the pymdp `field(static=True)` bug. `reproduce.py` patches it
automatically. If you are importing `aif_paper1` directly without going
through `reproduce.py`, add the following before any pymdp import:

```python
import reproduce  # triggers the patch
```

Or manually patch `pymdp/agent.py` to remove the `= field(static=True)`
annotation on the `policies` field.

**`Very slow even with jit_inference=True`**  
JAX compilation happens once per process at the first call to `_infer_policies`.
Subsequent calls reuse the compiled kernel. Expect 30–90 s for the initial
XLA trace at `planning_horizon=3`; steps after that are fast.

**`XLA out-of-memory on GPU`**  
The full policy set at `planning_horizon=3` is 15 625 policies × 3 steps × 2
control factors. If your GPU runs out of memory, use `--quick`
(`planning_horizon=1`, 25 policies) for validation, or add
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` to your environment.

**`reproduce.py --rerun` produces all-zero survival rates**  
This is expected with `--quick` on some layout/seed combinations: with
`planning_horizon=1` the agent cannot lookahead far enough to reliably reach
resources before the physiological deadline. Run without `--quick` for
meaningful survival statistics.

---

## Licence

The MIT licence in `LICENSE` applies to the software and its documentation. It
does not apply to the paper PDF, text, or figures under `paper/`; those materials
remain subject to their own publication and copyright terms. See
[`paper/README.md`](paper/README.md).

"""
logging_utils.py
================

Lightweight episode logging structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy after normalizing a probability-like vector."""

    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = p / max(1e-16, float(p.sum()))
    return float(-(p * np.log(p + 1e-16)).sum())


def _as_jsonable(x: Any) -> Any:
    """Recursively convert numpy-heavy records into plain JSON-serializable values."""

    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _as_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_as_jsonable(v) for v in x]
    return x


@dataclass(frozen=True)
class EpisodeLog:
    """
    Minimal episode log suitable for benchmarking and figure generation.

    Conventions:
    - Arrays are length `steps` and correspond to the decision-time state at t.
    - `pos_next`/`ctx_next`/`tile_res_next` correspond to the post-move state after
      applying `Move_t` (i.e., the destination tile used by the body for `Interact_t`).
    - `env_*` fields correspond to environment diagnostics/info returned by `env.step`
      at time t (i.e., aligned with `pos_next` / `ctx_next` / `tile_res_next`).
    - `typeI_levels`/`typeII_true` are the post-tick body state after applying the
      action on the destination tile at t.
      If the environment terminates at t (e.g. predator), the body tick is skipped
      and these fields reflect the last pre-termination body state.
    """

    steps: int
    terminated_by: str  # {"timeout","env","body"}

    pos: np.ndarray               # (steps,)
    ctx: np.ndarray               # (steps,)
    tile_res: np.ndarray          # (steps,) Resource ids at decision-time

    pos_next: np.ndarray          # (steps,) destination pos ids after Move_t
    ctx_next: np.ndarray          # (steps,) destination ctx ids after Move_t
    tile_res_next: np.ndarray     # (steps,) Resource ids at destination (post env effects)

    env_ctx_changed: np.ndarray   # (steps,) bool-ish
    env_tile_mutated: np.ndarray  # (steps,) bool-ish
    env_mutated_pos: np.ndarray  # (steps,) or -1 if none
    env_mutated_from: np.ndarray  # (steps,) or -1 if none
    env_mutated_to: np.ndarray    # (steps,) or -1 if none
    env_underwater: np.ndarray    # (steps,) consecutive underwater steps
    env_predator: np.ndarray      # (steps,) bool-ish

    obs_pos: np.ndarray           # (steps,)
    obs_res: np.ndarray           # (steps,) Resource ids (noisy)
    season_obs: np.ndarray        # (steps, n_season+1)

    move: np.ndarray              # (steps,) Move ids
    interact: np.ndarray          # (steps,) Interact ids
    kappa: Optional[np.ndarray]   # (steps,) Need ids (optional)

    typeI_levels: np.ndarray      # (steps, 4) [energy, hydration, sleep_debt, oxygen]
    typeII_true: np.ndarray       # (steps, 4)
    typeII_obs: np.ndarray        # (steps, 4) (used for action selection at t)


class StepLogWriter:
    """
    Opt-in per-step log writer for analysis runs.

    Writes JSONL records derived from `Simulation.run_episode(..., on_step=...)`.
    By default it logs *summaries* (entropy/argmax/top-k) rather than full arrays.
    """

    def __init__(
        self,
        path: str,
        *,
        include_full_beliefs: bool = False,
        top_k: int = 5,
        flush_every: int = 1,
    ):
        """Open a JSONL writer and configure belief summarization behavior."""

        import json
        from pathlib import Path

        self._json = json
        self._path = Path(path)
        self._f = self._path.open("w", encoding="utf-8")
        self._include_full_beliefs = bool(include_full_beliefs)
        self._top_k = max(1, int(top_k))
        self._flush_every = int(flush_every)
        self._n_written = 0

    def close(self) -> None:
        """Flush pending records and close the underlying file handle."""

        try:
            self._f.flush()
        except Exception:
            pass
        self._f.close()

    def __enter__(self):
        """Return `self` so the writer can be used as a context manager."""

        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the writer on context-manager exit without suppressing exceptions."""

        self.close()
        return False

    def _summarize_vector(self, vec: Any) -> dict[str, Any]:
        """Compress a belief-like vector into entropy, argmax, and top-k entries."""

        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return {"entropy": 0.0, "argmax": 0, "top": []}
        top = np.argsort(-arr)[: self._top_k].astype(int).tolist()
        return {
            "entropy": _entropy(arr),
            "argmax": int(np.argmax(arr)),
            "top": [{"i": i, "p": float(arr[i])} for i in top],
        }

    def write(self, t: int, rec: dict[str, object]) -> None:
        """Write one step record, summarizing or preserving beliefs according to config."""

        out: dict[str, Any] = {"t": int(t)}
        out.update({k: rec[k] for k in rec.keys() if k not in {"beliefs"}})

        beliefs = rec.get("beliefs", {})
        if isinstance(beliefs, dict) and beliefs:
            out["beliefs_summary"] = {k: self._summarize_vector(v) for k, v in beliefs.items()}
            if self._include_full_beliefs:
                out["beliefs"] = beliefs

        self._f.write(self._json.dumps(_as_jsonable(out)) + "\n")
        self._n_written += 1
        if self._flush_every > 0 and (self._n_written % self._flush_every) == 0:
            self._f.flush()


@dataclass(frozen=True)
class RunMetadata:
    """
    Lightweight, JSON-serializable run metadata for experiment traceability.
    """

    created_utc: str
    git_commit: str
    variant_id: str
    regime_id: str
    root_seed: int
    args: dict[str, Any]
    runtime: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return run metadata as a plain JSON-serializable dictionary."""

        return {
            "created_utc": self.created_utc,
            "git_commit": self.git_commit,
            "variant_id": self.variant_id,
            "regime_id": self.regime_id,
            "root_seed": int(self.root_seed),
            "args": _as_jsonable(self.args),
            "runtime": _as_jsonable(self.runtime),
        }

    def write_json(self, path: str) -> None:
        """Persist metadata to a pretty-printed JSON sidecar file."""

        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "EpisodeLog",
    "RunMetadata",
    "StepLogWriter",
]

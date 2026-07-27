"""
Shared enums and utilities for AffectWorld.

Internal enum values remain compatibility-oriented, but maintained Paper 1
surfaces interpret them as:

- `FOOD` -> `Food`
- `WATER` -> `Water`
- `AIR` -> `Oxygen` availability / breathing context
- `SLEEP` -> dormant compatibility-only channel
"""

from enum import IntEnum

import numpy as np


class Resource(IntEnum):
    """Discrete tile/resource identities used by the environment and observations."""

    AIR = 0
    WATER = 1
    FOOD = 2
    SLEEP = 3
    POISON = 4


class Season(IntEnum):
    """Hidden environmental contexts interpreted as seasons."""

    SPRING = 0
    SUMMER = 1
    AUTUMN = 2
    WINTER = 3


class Move(IntEnum):
    """Deterministic locomotion actions in the gridworld."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    REST = 4


class Interact(IntEnum):
    """Post-move interaction verbs applied by the body on the destination tile."""

    NONE = 0
    BREATHE = 1
    EAT = 2
    DRINK = 3
    SLEEP = 4


class Need(IntEnum):
    """
    Internal interoceptive-need channels (order must match `BodyProcess.TYPEII_NAMES`).

    The extra NONE state is useful for modelling "no special attentional focus".
    Paper 1 public naming maps these to nutrient / hydration / dormant-rest /
    oxygenation-proxy channels.
    """

    HUNGER = 0
    THIRST = 1
    FATIGUE = 2
    SUFFOCATION = 3
    NONE = 4


def need_to_interact(need: Need) -> Interact:
    """Map a prioritized need to a corresponding Interact action."""
    if need == Need.HUNGER:
        return Interact.EAT
    if need == Need.THIRST:
        return Interact.DRINK
    if need == Need.FATIGUE:
        return Interact.SLEEP
    if need == Need.SUFFOCATION:
        return Interact.BREATHE
    return Interact.NONE


def one_hot(index: int, size: int) -> np.ndarray:
    """Return a 1D one-hot numpy array."""
    vec = np.zeros(size, dtype=np.int8)
    vec[int(index)] = 1
    return vec


__all__ = [
    "Interact",
    "Move",
    "Need",
    "Resource",
    "Season",
    "need_to_interact",
    "one_hot",
]

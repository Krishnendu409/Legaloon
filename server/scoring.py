"""
Unified scoring policy for LegaLoom-Env.

Single source of truth for score/range normalization shared by env and graders.
"""

from __future__ import annotations

from typing import Final

SCORE_MIN: Final[float] = 0.0
SCORE_MAX: Final[float] = 1.0
ROUND_DIGITS: Final[int] = 4

STEP_REWARD_MIN: Final[float] = -1.0
STEP_REWARD_MAX: Final[float] = 1.0


def clamp_score(value: float) -> float:
    """Clamp final score to [0.0, 1.0] with stable rounding."""
    v = float(value)
    if v < SCORE_MIN:
        v = SCORE_MIN
    elif v > SCORE_MAX:
        v = SCORE_MAX
    return round(v, ROUND_DIGITS)


def normalize_step_reward(value: float) -> float:
    """Clamp step rewards to a bounded interval and round deterministically."""
    v = float(value)
    if v < STEP_REWARD_MIN:
        v = STEP_REWARD_MIN
    elif v > STEP_REWARD_MAX:
        v = STEP_REWARD_MAX
    return round(v, ROUND_DIGITS)

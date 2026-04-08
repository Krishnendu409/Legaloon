"""
Pydantic models for LegaLoom-Env.
Defines the typed Action, Observation, and State contracts
that the agent, server, and client all share.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


# ---------------------------------------------------------------------------
# Reward clamping constants — hackathon requires strictly (0.0, 1.0) exclusive
# ---------------------------------------------------------------------------
_REWARD_MIN = 0.05   # never return exactly 0.0
_REWARD_MAX = 0.95   # never return exactly 1.0


# ---------------------------------------------------------------------------
# Action — what the agent sends to the environment each step
# ---------------------------------------------------------------------------

class TDSAction(Action):
    """
    One step the agent takes inside the TDS compliance environment.

    Valid action_types:
      "read_invoice"     — request the full invoice text (first step)
      "check_pan"        — verify PAN status for a given PAN number
      "check_threshold"  — ask if cumulative YTD payments cross the TDS threshold
      "query_ytd"        — query year-to-date payments to a vendor
      "lookup_section"   — ask what TDS section applies to a described service
      "query_law"        — look up the law text for a section
      "submit_answer"    — submit final TDS deduction in INR (terminates episode)
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: read_invoice | check_pan | check_threshold "
            "| query_ytd | lookup_section | query_law | submit_answer"
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. Examples:\n"
            "  check_pan       -> {pan: 'ABCDE1234F'}\n"
            "  check_threshold -> {section: '194J', amount: 85000}\n"
            "  lookup_section  -> {description: 'IT support contract'}\n"
            "  submit_answer   -> {tds_amount_inr: 8500.0, section: '194J', "
            "rate_percent: 10.0}"
        ),
    )


# ---------------------------------------------------------------------------
# Observation — what the environment returns after each step
# ---------------------------------------------------------------------------

class TDSObservation(Observation):
    """
    What the agent sees after each action.

    reward is always strictly in (0.0, 1.0) exclusive — never 0.0 or 1.0.
    """

    # Override reward to always be strictly in (0.0, 1.0) exclusive
    reward: float = Field(
        default=_REWARD_MIN,
        description="Reward for this step. Always strictly in (0.0, 1.0) exclusive.",
    )

    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        """
        Enforce hackathon Phase 2 rule: scores must be strictly in (0.0, 1.0).
        Any value <= 0.0 is raised to _REWARD_MIN.
        Any value >= 1.0 is lowered to _REWARD_MAX.
        """
        v = float(v)
        return round(min(max(v, _REWARD_MIN), _REWARD_MAX), 4)

    invoice_text: str = Field(
        default="",
        description="Full invoice text. Populated after read_invoice action.",
    )

    action_result: str = Field(
        default="",
        description="Environment's response to the agent's last action.",
    )

    available_actions: List[str] = Field(
        default_factory=lambda: ["read_invoice"],
        description="List of valid action_type values for the next step.",
    )

    steps_used: int = Field(
        default=0,
        description="Number of steps taken so far in this episode.",
    )

    max_steps: int = Field(
        default=8,
        description="Maximum steps allowed before episode is force-closed.",
    )

    hint: str = Field(
        default="",
        description="Optional guidance. Empty string on hard difficulty.",
    )


# ---------------------------------------------------------------------------
# State — episode metadata returned by state() endpoint
# ---------------------------------------------------------------------------

class TDSState(State):
    """
    Metadata about the current episode.
    """

    task_id: str = Field(
        default="",
        description="Which task is currently active: task_easy | task_medium | task_hard | task_expert",
    )

    difficulty: str = Field(
        default="",
        description="easy | medium | hard | expert",
    )

    pan_checked: bool = Field(
        default=False,
        description="True once agent has called check_pan.",
    )

    section_identified: bool = Field(
        default=False,
        description="True once agent has called lookup_section.",
    )

    answer_submitted: bool = Field(
        default=False,
        description="True once agent has called submit_answer.",
    )

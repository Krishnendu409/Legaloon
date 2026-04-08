"""
Pydantic models for LegaLoom-Env.
Defines the typed Action, Observation, and State contracts
that the agent, server, and client all share.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadInvoiceParams(_StrictParams):
    pass


class CheckPanParams(_StrictParams):
    pan: str = Field(..., min_length=10, max_length=10)


class CheckThresholdParams(_StrictParams):
    section: str = Field(..., min_length=3, max_length=8)
    amount: float = Field(..., ge=0)


class QueryYtdParams(_StrictParams):
    pan: str = Field(..., min_length=10, max_length=10)


class LookupSectionParams(_StrictParams):
    description: str = Field(..., min_length=1)


class QueryLawParams(_StrictParams):
    section: str = Field(..., min_length=3, max_length=8)


class SubmitAnswerParams(_StrictParams):
    tds_amount_inr: float = Field(..., ge=0)
    section: str = Field(..., min_length=3, max_length=20)
    rate_percent: float = Field(..., ge=0)
    no_tds: Optional[Union[bool, Literal["true", "false"]]] = None


class TDSAction(Action):
    """
    One step the agent takes inside the TDS compliance environment.
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: read_invoice | check_pan | check_threshold "
            "| query_ytd | lookup_section | query_law | submit_answer"
        ),
    )
    parameters: Union[
        Dict[str, Any],
        ReadInvoiceParams,
        CheckPanParams,
        CheckThresholdParams,
        QueryYtdParams,
        LookupSectionParams,
        QueryLawParams,
        SubmitAnswerParams,
    ] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_parameters(cls, data):
        if not isinstance(data, dict):
            return data

        action = str(data.get("action_type", "")).strip().lower()
        params = data.get("parameters") or {}

        if isinstance(params, BaseModel):
            params = params.model_dump(exclude_none=True)

        if not isinstance(params, dict):
            raise ValueError("parameters must be an object")

        parser = {
            "read_invoice": ReadInvoiceParams,
            "check_pan": CheckPanParams,
            "check_threshold": CheckThresholdParams,
            "query_ytd": QueryYtdParams,
            "lookup_section": LookupSectionParams,
            "query_law": QueryLawParams,
            "submit_answer": SubmitAnswerParams,
        }.get(action)

        if parser is not None:
            params = parser.model_validate(params).model_dump(exclude_none=True)
        else:
            # Keep unknown actions permissive so environment can apply explicit penalties.
            params = params

        data["parameters"] = params
        return data


class TDSReward(BaseModel):
    """
    Typed reward payload that accompanies every observation.
    """

    step_reward: float = Field(
        default=0.0,
        description="Reward returned for this step.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Cumulative reward observed so far in the episode.",
    )
    final_score: Optional[float] = Field(
        default=None,
        description="Final score on terminal transition; None for intermediate steps.",
    )
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional normalized reward component breakdown.",
    )


class TDSObservation(Observation):
    """
    What the agent sees after each action.
    """

    reward: float = Field(
        default=0.0,
        description="Reward for this step. Can be positive, zero, or negative.",
    )

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

    reward_info: TDSReward = Field(
        default_factory=TDSReward,
        description="Typed reward payload for OpenEnv Reward contract compliance.",
    )

    @model_validator(mode="after")
    def sync_reward_info(self):
        if self.reward_info is None:
            self.reward_info = TDSReward(step_reward=float(self.reward))
        elif self.reward_info.step_reward != float(self.reward):
            self.reward_info.step_reward = float(self.reward)
        return self


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

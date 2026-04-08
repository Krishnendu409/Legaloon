"""
LegaLoom-Env Client.
WebSocket client that training code imports to interact with the environment.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TDSAction, TDSObservation, TDSState


class LegaloomEnv(EnvClient[TDSAction, TDSObservation, TDSState]):
    """
    Client for the LegaLoom TDS Compliance Environment.

    Example usage:
        with LegaloomEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_id="task_easy")
            result = env.step(TDSAction(
                action_type="read_invoice",
                parameters={}
            ))
            print(result.observation.invoice_text)
    """

    def _step_payload(self, action: TDSAction) -> Dict:
        return {
            "action_type": action.action_type,
            "parameters":  action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TDSObservation]:
        obs_data = payload.get("observation", {})
        observation = TDSObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.001),
            invoice_text=obs_data.get("invoice_text", ""),
            action_result=obs_data.get("action_result", ""),
            available_actions=obs_data.get("available_actions", []),
            steps_used=obs_data.get("steps_used", 0),
            max_steps=obs_data.get("max_steps", 8),
            hint=obs_data.get("hint", ""),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.001),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> TDSState:
        return TDSState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", ""),
            pan_checked=payload.get("pan_checked", False),
            section_identified=payload.get("section_identified", False),
            answer_submitted=payload.get("answer_submitted", False),
        )

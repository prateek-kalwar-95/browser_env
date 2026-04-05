"""BrowserEnv client for the IT Helpdesk Portal.

Provides a typed WebSocket client that connects to the environment
server and exchanges BrowserAction / BrowserObservation payloads.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BrowserAction, BrowserObservation


class BrowserEnv(EnvClient[BrowserAction, BrowserObservation, State]):
    """WebSocket client with typed payloads for BrowserAction / BrowserObservation."""

    def _step_payload(self, action: BrowserAction) -> Dict:
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[BrowserObservation]:
        obs_data = payload.get("observation", {})
        observation = BrowserObservation(
            url=obs_data.get("url", "/"),
            page_title=obs_data.get("page_title", ""),
            page_content=obs_data.get("page_content", ""),
            task_id=obs_data.get("task_id", "easy"),
            task_instruction=obs_data.get("task_instruction", ""),
            feedback=obs_data.get("feedback", ""),
            command_valid=obs_data.get("command_valid", True),
            # New sub-goal fields
            sub_goals_status=obs_data.get("sub_goals_status", {}),
            progress=float(obs_data.get("progress", 0.0)),
            # Navigation context
            breadcrumb=list(obs_data.get("breadcrumb", [])),
            # Session context
            current_user=obs_data.get("current_user"),
            notifications=list(obs_data.get("notifications", [])),
            steps_remaining=int(obs_data.get("steps_remaining", 48)),
            # Grading
            grader_score=obs_data.get("grader_score"),
            available_commands=list(obs_data.get("available_commands", [])),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

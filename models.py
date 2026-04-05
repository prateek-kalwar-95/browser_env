"""Typed action and observation models for the IT Helpdesk Portal environment.

Defines the Pydantic models that constitute the OpenEnv action/observation
contract. These models are the API surface between the environment server
and any connecting agent or client.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class BrowserAction(Action):
    """A single text command issued by the agent.

    Supported verbs (case-insensitive):
        goto /path          — navigate to a portal page
        click <element_id>  — click a navigation link or button
        type <field> <value> — fill a text form field
        select <field> <option> — choose a dropdown/radio option
        submit              — submit the current form
        back                — go to previous page
        toggle <id>         — toggle a boolean setting
        finish              — signal task completion for grading
    """

    command: str = Field(
        ...,
        description=(
            "Single-line command, e.g. 'goto /directory', "
            "'click emp_E-3301', 'type description printer jam', "
            "'select category Hardware', 'submit', 'back', 'finish'"
        ),
    )


class BrowserObservation(Observation):
    """Rendered page view with task context, sub-goal tracking, and grading."""

    # --- Page content ---
    url: str = Field(default="/", description="Current page path")
    page_title: str = Field(default="", description="Page heading")
    page_content: str = Field(
        default="",
        description="Full rendered page text (links, forms, content)",
    )

    # --- Task context ---
    task_id: str = Field(default="easy", description="Active task tier: easy | medium | hard")
    task_instruction: str = Field(default="", description="Natural-language task goal")

    # --- Interaction feedback ---
    feedback: str = Field(default="", description="Result of the last command")
    command_valid: bool = Field(default=True, description="Whether the last command was accepted")

    # --- Sub-goal tracking ---
    sub_goals_status: Dict[str, bool] = Field(
        default_factory=dict,
        description="Map of sub-goal name to completion status",
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted sub-goal completion progress (0.0 to 1.0)",
    )

    # --- Navigation context ---
    breadcrumb: List[str] = Field(
        default_factory=list,
        description="Recent navigation trail (last 5 pages visited)",
    )

    # --- Session context ---
    current_user: Optional[str] = Field(
        default=None,
        description="Logged-in username, or null if not authenticated",
    )
    notifications: List[str] = Field(
        default_factory=list,
        description="Unread system notifications",
    )
    steps_remaining: int = Field(
        default=48,
        description="Number of steps remaining before timeout",
    )

    # --- Grading ---
    grader_score: Optional[float] = Field(
        default=None,
        description="Final task score in [0,1] when the episode ends",
    )

    # --- Agent hints ---
    available_commands: List[str] = Field(
        default_factory=list,
        description="Short reminder of allowed command patterns",
    )

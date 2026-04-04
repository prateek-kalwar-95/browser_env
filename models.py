# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed action and observation models for text-controlled intranet browser tasks."""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class BrowserAction(Action):
    """
    Text command that drives a simplified browser session.

    Supported verbs (case-insensitive): goto, navigate, click, type, set,
    submit, toggle, finish.
    """

    command: str = Field(
        ...,
        description=(
            "Single-line command, e.g. 'goto /team', 'click nav_ticket', "
            "'type description keyboard issue', 'submit', 'toggle email_notifications', 'finish'"
        ),
    )


class BrowserObservation(Observation):
    """Rendered page view plus task context and grading feedback."""

    url: str = Field(default="/", description="Current path")
    page_title: str = Field(default="", description="Visible page title")
    page_content: str = Field(
        default="",
        description="Human-readable page summary (links, forms, body text)",
    )
    task_id: str = Field(default="easy", description="Active task identifier")
    task_instruction: str = Field(default="", description="Natural-language task goal")
    feedback: str = Field(default="", description="Result of the last command")
    command_valid: bool = Field(default=True, description="Whether the last command parsed and applied")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Heuristic partial progress toward the current task",
    )
    grader_score: Optional[float] = Field(
        default=None,
        description="Final task score in [0,1] when the episode is done",
    )
    available_commands: List[str] = Field(
        default_factory=list,
        description="Short reminder of allowed command patterns",
    )

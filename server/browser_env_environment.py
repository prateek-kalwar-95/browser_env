# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text-driven intranet browser simulator with three graded tasks."""

from __future__ import annotations

import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BrowserAction, BrowserObservation
except ImportError:
    from models import BrowserAction, BrowserObservation

from .simulation import (
    TASK_INSTRUCTIONS,
    apply_command,
    extract_task_id,
    grade_task,
    heuristic_progress,
    parse_command,
    render_page,
    SessionState,
)


class BrowserEnvironment(Environment):
    """
    Simulates a small corporate intranet controlled by structured text commands.

    Tasks (via env ``BROWSER_TASK`` = easy | medium | hard) cover navigation,
    ticketing, and authenticated settings changes.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 48

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._session = SessionState()
        self._task_id: str = "easy"
        self._done: bool = False
        self._last_grader: float | None = None

    def reset(self, **kwargs) -> BrowserObservation:  # type: ignore[override]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._session = SessionState()
        task_hint = kwargs.get("task_id") or kwargs.get("browser_task") or os.environ.get(
            "BROWSER_TASK"
        )
        self._task_id = extract_task_id(str(task_hint) if task_hint else None)
        self._done = False
        self._last_grader = None
        self._session.note_visit(self._session.path)

        title, url, content = render_page(self._session.path, self._session)
        return BrowserObservation(
            url=url,
            page_title=title,
            page_content=content,
            task_id=self._task_id,
            task_instruction=TASK_INSTRUCTIONS[self._task_id],
            feedback="Session reset. Issue commands to complete the task, then finish.",
            command_valid=True,
            progress=heuristic_progress(self._task_id, self._session),
            grader_score=None,
            done=False,
            reward=0.0,
            available_commands=self._command_cheatsheet(),
            metadata={"episode_id": self._state.episode_id},
        )

    def step(self, action: BrowserAction, **kwargs) -> BrowserObservation:  # type: ignore[override]
        if self._done:
            title, url, content = render_page(self._session.path, self._session)
            return BrowserObservation(
                url=url,
                page_title=title,
                page_content=content,
                task_id=self._task_id,
                task_instruction=TASK_INSTRUCTIONS[self._task_id],
                feedback="Episode already finished. Call reset().",
                command_valid=False,
                progress=heuristic_progress(self._task_id, self._session),
                grader_score=self._last_grader,
                done=True,
                reward=0.0,
                available_commands=[],
                metadata={"episode_id": self._state.episode_id},
            )

        self._state.step_count += 1
        command = action.command.strip()
        parsed = parse_command(command)
        verb = parsed[0]

        if verb == "finish":
            score, rationale = grade_task(self._task_id, self._session)
            self._done = True
            self._last_grader = score
            title, url, content = render_page(self._session.path, self._session)
            return BrowserObservation(
                url=url,
                page_title=title,
                page_content=content,
                task_id=self._task_id,
                task_instruction=TASK_INSTRUCTIONS[self._task_id],
                feedback=f"Graded: {rationale}",
                command_valid=True,
                progress=max(heuristic_progress(self._task_id, self._session), score),
                grader_score=score,
                done=True,
                reward=score,
                available_commands=[],
                metadata={
                    "episode_id": self._state.episode_id,
                    "grader_rationale": rationale,
                },
            )

        ok, feedback, delta = apply_command(self._session, command)
        shaped = max(-0.25, min(0.25, delta))
        progress = heuristic_progress(self._task_id, self._session)

        hit_limit = self._state.step_count >= self.MAX_STEPS
        terminal_score: float | None = None
        rationale = ""
        if hit_limit:
            terminal_score, rationale = grade_task(self._task_id, self._session)
            self._done = True
            self._last_grader = terminal_score
            feedback = f"{feedback} | Step limit reached. {rationale}"

        title, url, content = render_page(self._session.path, self._session)
        obs = BrowserObservation(
            url=url,
            page_title=title,
            page_content=content,
            task_id=self._task_id,
            task_instruction=TASK_INSTRUCTIONS[self._task_id],
            feedback=feedback,
            command_valid=ok,
            progress=progress,
            grader_score=terminal_score,
            done=hit_limit,
            reward=terminal_score if hit_limit else shaped,
            available_commands=self._command_cheatsheet(),
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
            },
        )
        if hit_limit:
            obs.metadata["grader_rationale"] = rationale  # type: ignore[index]
        return obs

    @property
    def state(self) -> State:
        return self._state

    @staticmethod
    def _command_cheatsheet() -> list[str]:
        return [
            "goto /path",
            "click nav_*",
            "type <field> <value>",
            "submit",
            "toggle email_notifications",
            "finish",
        ]


def _demo() -> None:
    """Manual smoke test without HTTP."""
    os.environ["BROWSER_TASK"] = "easy"
    env = BrowserEnvironment()
    obs = env.reset()
    assert obs.task_id == "easy"
    for cmd in ("goto /team", "finish"):
        obs = env.step(BrowserAction(command=cmd))
    print("easy score", obs.grader_score, obs.feedback)


if __name__ == "__main__":
    _demo()

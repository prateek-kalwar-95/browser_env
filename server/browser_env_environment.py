"""IT Helpdesk Portal environment - OpenEnv-compliant Environment implementation.

Integrates the task engine and simulation to provide a fully graded,
multi-task browser simulation with per-step reward shaping and
sub-goal decomposition.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

# Ensure the project root is on sys.path so fallback imports work
# regardless of whether we're invoked as a package or as a script.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BrowserAction, BrowserObservation
except (ImportError, SystemError):
    from models import BrowserAction, BrowserObservation

try:
    from .simulation import (
        SessionState,
        apply_command,
        parse_command,
        render_page,
    )
    from .task_engine import (
        TaskGenerator,
        TaskInstance,
        evaluate_sub_goals,
        grade_task,
    )
except (ImportError, SystemError):
    from server.simulation import (
        SessionState,
        apply_command,
        parse_command,
        render_page,
    )
    from server.task_engine import (
        TaskGenerator,
        TaskInstance,
        evaluate_sub_goals,
        grade_task,
    )


class BrowserEnvironment(Environment):
    """Simulates a corporate IT Helpdesk Portal with procedurally generated tasks.

    Three task tiers (easy, medium, hard) are generated with randomized
    parameters per episode. The agent interacts via text commands to
    complete helpdesk workflows: employee lookup, ticket triage,
    and incident resolution.

    Uses a sub-goal decomposition grading system with per-step reward
    shaping to provide meaningful signal over the full trajectory.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 48

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._session = SessionState()
        self._task: Optional[TaskInstance] = None
        self._done: bool = False
        self._last_grader: Optional[float] = None
        self._prev_progress: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> BrowserObservation:  # type: ignore[override]
        """Reset the environment and generate a new task instance."""

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._session = SessionState()
        self._done = False
        self._last_grader = None
        self._prev_progress = 0.0

        # Determine task tier
        task_id = (
            kwargs.get("task_id")
            or kwargs.get("browser_task")
            or os.environ.get("BROWSER_TASK", "easy")
        )
        task_id = str(task_id).strip().lower()
        if task_id not in {"easy", "medium", "hard"}:
            task_id = "easy"

        # Determine seed
        seed_raw = kwargs.get("seed") or os.environ.get("BROWSER_TASK_SEED")
        seed = int(seed_raw) if seed_raw is not None else None

        # Generate task
        self._task = TaskGenerator.generate(task_id, seed=seed)

        # Populate session from task instance
        self._session.roster = list(self._task.roster)
        self._session.existing_tickets = list(self._task.existing_tickets)
        self._session.incidents = list(self._task.incidents)
        self._session.credentials = dict(self._task.credentials)
        self._session.notifications = [
            "New task assigned. Check your task instruction to get started."
        ]

        # Record initial visit
        self._session.note_visit("/")

        title, url, content = render_page("/", self._session)
        sub_status, progress = evaluate_sub_goals(self._task, self._session)

        return BrowserObservation(
            url=url,
            page_title=title,
            page_content=content,
            task_id=self._task.task_id,
            task_instruction=self._task.goal_description,
            feedback="Environment reset. Read the task instruction and begin.",
            command_valid=True,
            sub_goals_status=sub_status,
            progress=progress,
            breadcrumb=self._session.breadcrumb,
            current_user=self._session.current_user,
            notifications=list(self._session.notifications),
            steps_remaining=self.MAX_STEPS,
            grader_score=None,
            done=False,
            reward=0.0,
            available_commands=self._command_help(),
            metadata={"episode_id": self._state.episode_id},
        )

    def step(self, action: BrowserAction, **kwargs) -> BrowserObservation:  # type: ignore[override]
        """Execute an agent command and return the resulting observation."""

        assert self._task is not None, "Must call reset() before step()"

        if self._done:
            title, url, content = render_page(self._session.path, self._session)
            sub_status, progress = evaluate_sub_goals(self._task, self._session)
            return BrowserObservation(
                url=url,
                page_title=title,
                page_content=content,
                task_id=self._task.task_id,
                task_instruction=self._task.goal_description,
                feedback="Episode already finished. Call reset() to start a new one.",
                command_valid=False,
                sub_goals_status=sub_status,
                progress=progress,
                breadcrumb=self._session.breadcrumb,
                current_user=self._session.current_user,
                notifications=[],
                steps_remaining=0,
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

        # --- Handle finish command ---
        if verb == "finish":
            score, rationale = grade_task(self._task, self._session)
            self._done = True
            self._last_grader = score

            title, url, content = render_page(self._session.path, self._session)
            sub_status, _ = evaluate_sub_goals(self._task, self._session)

            return BrowserObservation(
                url=url,
                page_title=title,
                page_content=content,
                task_id=self._task.task_id,
                task_instruction=self._task.goal_description,
                feedback=f"Graded: {rationale}",
                command_valid=True,
                sub_goals_status=sub_status,
                progress=max(self._prev_progress, score),
                breadcrumb=self._session.breadcrumb,
                current_user=self._session.current_user,
                notifications=[],
                steps_remaining=0,
                grader_score=score,
                done=True,
                reward=score,
                available_commands=[],
                metadata={
                    "episode_id": self._state.episode_id,
                    "grader_rationale": rationale,
                },
            )

        # --- Execute the command ---
        ok, feedback, action_delta = apply_command(self._session, command)

        # Evaluate sub-goal progress
        sub_status, new_progress = evaluate_sub_goals(self._task, self._session)
        progress_delta = new_progress - self._prev_progress

        # Compute step reward: progress-based + small shaping
        step_reward = self._compute_step_reward(progress_delta, ok, action_delta)
        self._prev_progress = new_progress

        # Check step limit
        steps_left = self.MAX_STEPS - self._state.step_count
        hit_limit = steps_left <= 0
        terminal_score: Optional[float] = None
        rationale = ""

        if hit_limit:
            terminal_score, rationale = grade_task(self._task, self._session)
            self._done = True
            self._last_grader = terminal_score
            feedback = f"{feedback} | Step limit reached. {rationale}"

        title, url, content = render_page(self._session.path, self._session)

        obs = BrowserObservation(
            url=url,
            page_title=title,
            page_content=content,
            task_id=self._task.task_id,
            task_instruction=self._task.goal_description,
            feedback=feedback,
            command_valid=ok,
            sub_goals_status=sub_status,
            progress=new_progress,
            breadcrumb=self._session.breadcrumb,
            current_user=self._session.current_user,
            notifications=list(self._session.notifications) if not self._session.notifications_read else [],
            steps_remaining=max(0, steps_left),
            grader_score=terminal_score,
            done=hit_limit,
            reward=terminal_score if hit_limit else step_reward,
            available_commands=self._command_help() if not hit_limit else [],
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
            },
        )

        if hit_limit:
            obs.metadata["grader_rationale"] = rationale  # type: ignore[index]

        # Mark notifications as read after first step that sees them
        if self._session.notifications:
            self._session.notifications_read = True

        return obs

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_step_reward(
        progress_delta: float, command_valid: bool, action_delta: float
    ) -> float:
        """Compute per-step reward with meaningful trajectory-level signal.

        Combines sub-goal progress (primary signal) with command shaping
        (secondary signal for valid/invalid actions).
        """
        # Primary: sub-goal progress
        reward = progress_delta * 0.8

        # Secondary: action-level shaping
        if not command_valid:
            reward += max(-0.15, action_delta)  # Penalize invalid commands
        elif progress_delta > 0:
            reward += 0.02  # Small bonus for making progress
        else:
            reward += max(-0.05, action_delta * 0.5)  # Mild shaping

        return max(-0.25, min(0.5, reward))

    @staticmethod
    def _command_help() -> List[str]:
        return [
            "goto /path",
            "click <element_id>",
            "type <field> <value>",
            "select <field> <option>",
            "submit",
            "back",
            "finish",
        ]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Manual smoke test — runs each task tier without HTTP."""
    for tier in ("easy", "medium", "hard"):
        os.environ["BROWSER_TASK"] = tier
        env = BrowserEnvironment()
        obs = env.reset(seed=42)
        print(f"\n{'='*60}")
        print(f"TASK: {tier}")
        print(f"Goal: {obs.task_instruction}")
        print(f"Sub-goals: {obs.sub_goals_status}")
        print(f"Progress: {obs.progress:.2f}")

        if tier == "easy":
            # Demo: navigate to directory, click the target employee, finish
            target_emp = env._task.params.get("target_employee", {}) if env._task else {}
            emp_id = target_emp.get("id", "E-3301")
            commands = [
                "goto /directory",
                f"click emp_{emp_id}",
                "finish",
            ]
        elif tier == "medium":
            params = env._task.params if env._task else {}
            cat = params.get("target_category", "Hardware")
            pri = params.get("target_priority", "High")
            kw = params.get("target_keyword", "keyboard")
            assignee = params.get("target_assignee", {})
            assignee_name = assignee.get("name", "Grace Kim") if isinstance(assignee, dict) else str(assignee)
            commands = [
                "goto /tickets/new",
                f"select category {cat}",
                f"select priority {pri}",
                f"type assignee {assignee_name}",
                f"type description Issue with the {kw}",
                "submit",
                "finish",
            ]
        else:  # hard
            creds = env._task.credentials if env._task else {}
            inc_id = env._task.params.get("target_incident_id", "") if env._task else ""
            root_cause = env._task.params.get("target_root_cause", "") if env._task else ""
            commands = [
                "goto /login",
                f"type username {creds.get('username', 'admin')}",
                f"type password {creds.get('password', 'helpdesk2024')}",
                "submit",
                "goto /incidents",
                f"click inc_{inc_id}",
                "select status investigating",
                "submit",
                f"type notes Root cause identified: {root_cause}",
                "submit",
                "select status resolved",
                "submit",
                "finish",
            ]

        for cmd in commands:
            obs = env.step(BrowserAction(command=cmd))
            status_char = "OK" if obs.command_valid else "FAIL"
            print(f"  [{status_char}] {cmd:40s} reward={obs.reward:.3f}  progress={obs.progress:.2f}")
            if obs.done:
                break

        print(f"Final score: {obs.grader_score}")
        print(f"Feedback: {obs.feedback}")


if __name__ == "__main__":
    _demo()

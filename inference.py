"""
Inference Script - IT Helpdesk Portal
======================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    OPENAI_API_KEY Alternative API key (checked if HF_TOKEN is not set).

- Defaults are set for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = https://router.huggingface.co/v1
    MODEL_NAME   = Qwen/Qwen2.5-72B-Instruct

- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
- The script emits exactly three line types to stdout per task, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  After all tasks complete, a summary line is emitted:
    [SUMMARY] easy=<score> medium=<score> hard=<score> average=<score>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after the episode completes.
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw feedback string on invalid commands, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task returns score in [0, 1].

  Example:
    [START] task=easy env=browser_env model=Qwen2.5-72B
    [STEP] step=1 action=goto /directory reward=0.05 done=false error=null
    [STEP] step=2 action=click emp_E-3301 reward=0.08 done=false error=null
    [STEP] step=3 action=finish reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.05,0.08,1.00
"""

import asyncio
import os
import textwrap
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

try:
    from browser_env import BrowserAction, BrowserEnv
except ImportError:
    from models import BrowserAction
    from client import BrowserEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = os.getenv("BROWSER_BENCHMARK", "browser_env")
MAX_STEPS = int(os.getenv("BROWSER_MAX_STEPS", "48"))
TEMPERATURE = float(os.getenv("BROWSER_TEMPERATURE", "0.5"))
MAX_TOKENS = int(os.getenv("BROWSER_MAX_TOKENS", "192"))
SUCCESS_THRESHOLD = float(os.getenv("BROWSER_SUCCESS_THRESHOLD", "0.5"))

# Run all three task tiers by default, unless overridden
TASK_TIERS = os.getenv("BROWSER_TASKS", "easy,medium,hard").split(",")

# ---------------------------------------------------------------------------
# System prompt — teaches the agent about the helpdesk portal
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an AI agent operating within a corporate IT Helpdesk Portal.
    Each turn you receive a page observation and a task instruction.
    You must issue exactly ONE command per turn.

    Available commands:
      goto /path            — navigate to a page (e.g., goto /directory, goto /tickets/new)
      click <element_id>    — click a link or button shown on the page (e.g., click nav_tickets, click emp_E-3301)
      type <field> <value>  — fill a text form field (e.g., type description printer is jammed)
      select <field> <option> — choose from a dropdown (e.g., select category Hardware, select priority High)
      submit                — submit the current form
      back                  — go to previous page
      finish                — signal you have completed the task (triggers grading)

    Strategy tips:
    - Read the task instruction carefully. Complete all required sub-goals before issuing 'finish'.
    - Check the sub-goals status to see what you still need to do.
    - Navigate using goto or click links shown on the page.
    - For forms: fill fields with type/select, then submit.
    - For login: goto /login, type username <user>, type password <pass>, submit.
    - Always issue 'finish' when you believe the task is complete.

    Reply with ONLY the command. No quotes, no explanation, no extra text."""
).strip()

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def log_summary(scores: Dict[str, float]) -> None:
    parts = " ".join(f"{k}={v:.2f}" for k, v in scores.items())
    avg = sum(scores.values()) / max(len(scores), 1)
    print(f"[SUMMARY] {parts} average={avg:.2f}", flush=True)


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def build_user_prompt(obs) -> str:
    """Build a rich prompt with all observation context."""

    # Sub-goal status display
    sg_lines = []
    for name, done in obs.sub_goals_status.items():
        marker = "✓" if done else "○"
        sg_lines.append(f"  {marker} {name}")
    sg_text = "\n".join(sg_lines) if sg_lines else "  (none)"

    # Notifications
    notif_text = "\n".join(f"  - {n}" for n in obs.notifications) if obs.notifications else "  (none)"

    return textwrap.dedent(
        f"""\
        URL: {obs.url}
        Page Title: {obs.page_title}

        Page Content:
        {obs.page_content}

        Task: {obs.task_instruction}
        Feedback: {obs.feedback}
        Progress: {obs.progress:.0%}
        Steps Remaining: {obs.steps_remaining}

        Sub-goals:
        {sg_text}

        Breadcrumb: {' > '.join(obs.breadcrumb)}
        Logged in as: {obs.current_user or '(not logged in)'}

        Notifications:
        {notif_text}

        Available commands: {', '.join(obs.available_commands)}

        Issue your next command."""
    ).strip()


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------


def get_model_command(client: OpenAI, obs) -> str:
    """Query the LLM for the next command."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Clean up: take only the first line, strip quotes
        text = text.split("\n")[0].strip().strip("'\"").strip("`")
        return text if text else "finish"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "finish"


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------


async def run_episode(
    client: OpenAI, env: BrowserEnv, task_id: str
) -> float:
    """Run one task episode and return the final score."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(options={"task_id": task_id})
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            command = get_model_command(client, obs)
            result = await env.step(BrowserAction(command=command))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None if obs.command_valid else obs.feedback

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=command, reward=reward, done=done, error=error)

            if done:
                break

        # Final score
        if obs.grader_score is not None:
            score = obs.grader_score
        else:
            score = max(0.0, min(1.0, sum(rewards)))
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main — runs all task tiers
# ---------------------------------------------------------------------------


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env_base_url = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
    env = BrowserEnv(base_url=env_base_url)
    await env.connect()

    scores: Dict[str, float] = {}

    try:
        for task_id in TASK_TIERS:
            task_id = task_id.strip()
            if task_id:
                score = await run_episode(client, env, task_id)
                scores[task_id] = score
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    # Print summary across all tasks
    if scores:
        log_summary(scores)


if __name__ == "__main__":
    asyncio.run(main())
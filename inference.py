"""
Inference Script — Browser Env
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

try:
    from browser_env import BrowserAction, BrowserEnv
except ImportError:
    from models import BrowserAction
    from client import BrowserEnv

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("BROWSER_TASK", "easy")
BENCHMARK = os.getenv("BROWSER_BENCHMARK", "browser_env")
MAX_STEPS = 48
TEMPERATURE = 0.7
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent navigating a simulated corporate intranet browser.
    Each turn you receive a page observation including the URL, page title,
    page content, task instruction, feedback from the last command, and
    a list of available commands.

    You must issue exactly ONE command per turn using the allowed verbs:
      goto /path     — navigate to a page
      click <id>     — click a navigation element
      type <field> <value> — type into a form field
      submit         — submit the current form
      toggle <id>    — toggle a setting
      finish         — signal that you have completed the task

    Reply with ONLY the command, nothing else. No quotes, no explanation.
    """
).strip()


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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(obs) -> str:
    return textwrap.dedent(
        f"""
        URL: {obs.url}
        Page Title: {obs.page_title}
        Page Content:
        {obs.page_content}

        Task: {obs.task_instruction}
        Feedback: {obs.feedback}
        Progress: {obs.progress:.0%}
        Available Commands: {', '.join(obs.available_commands)}

        Issue your next command.
        """
    ).strip()


def get_model_command(client: OpenAI, obs) -> str:
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
        return text if text else "finish"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "finish"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env_base_url = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
    env = BrowserEnv(base_url=env_base_url)
    await env.connect()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
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

        # Use grader_score if available, otherwise sum rewards
        if obs.grader_score is not None:
            score = obs.grader_score
        else:
            score = max(0.0, min(1.0, sum(rewards)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
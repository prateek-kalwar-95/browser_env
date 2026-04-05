---
title: IT Helpdesk Portal - Browser Environment
emoji: 🖥️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# IT Helpdesk Portal - Browser Environment

A simulated corporate IT helpdesk portal where AI agents perform **real-world workplace tasks**: looking up employee information, triaging support tickets, and resolving multi-step IT incidents. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Motivation

IT helpdesk operations - ticket triage, incident resolution, employee directory lookups - are performed millions of times daily across organizations. This environment provides a **safe, deterministic sandbox** for training and evaluating agents on these realistic workflows, with progressive difficulty from simple lookups to multi-step incident management.

## Action Space

Agents interact via single-line text commands:

| Command | Syntax | Description |
|---------|--------|-------------|
| `goto` | `goto /path` | Navigate to a portal page |
| `click` | `click <element_id>` | Click a link/button on the page |
| `type` | `type <field> <value>` | Fill a text form field |
| `select` | `select <field> <option>` | Choose from a dropdown |
| `submit` | `submit` | Submit the current form |
| `back` | `back` | Return to previous page |
| `finish` | `finish` | Signal task completion (triggers grading) |

**Example commands:**
```
goto /directory
click emp_E-3301
type description The printer on floor 3 is jammed
select category Hardware
select priority High
submit
finish
```

## Observation Space

Each step returns a `BrowserObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | Current page path (e.g., `/directory/E-3301`) |
| `page_title` | `str` | Page heading |
| `page_content` | `str` | Full rendered page text with links, forms, content |
| `task_id` | `str` | Task tier: `easy`, `medium`, or `hard` |
| `task_instruction` | `str` | Natural-language goal |
| `feedback` | `str` | Result of the last command |
| `command_valid` | `bool` | Whether the last command was accepted |
| `sub_goals_status` | `Dict[str, bool]` | Map of sub-goal names → completion status |
| `progress` | `float` | Weighted sub-goal progress (0.0–1.0) |
| `breadcrumb` | `List[str]` | Recent navigation trail |
| `current_user` | `Optional[str]` | Logged-in username |
| `notifications` | `List[str]` | System alerts |
| `steps_remaining` | `int` | Steps left before timeout |
| `grader_score` | `Optional[float]` | Final score when episode ends |
| `available_commands` | `List[str]` | Allowed command patterns |

## Task Descriptions

### Task 1: Employee Information Lookup (`easy`)
**Objective**: Navigate to the employee directory, find a specific employee, and view their profile page.

**Sub-goals**: Navigate to `/directory` → View target employee's profile → Finish on their page.

**Difficulty**: Low — requires 2-3 commands. Tests basic navigation.

**Randomized per episode**: Target employee name, target attribute, employee roster.

### Task 2: Support Ticket Triage (`medium`)
**Objective**: Create a new support ticket with the correct category, priority, assignee, and a description containing a specific keyword.

**Sub-goals**: Navigate to ticket form → Set category → Set priority → Include keyword → Set assignee → Submit.

**Difficulty**: Medium — requires 6-8 commands. Tests form interaction and multi-field accuracy.

**Randomized per episode**: Category, priority, keyword, assignee.

### Task 3: Incident Resolution Workflow (`hard`)
**Objective**: Log in with credentials, find a specific system incident, update its status through the investigation lifecycle, add root cause notes, and resolve it.

**Sub-goals**: Log in → Find incident → Set "investigating" → Add root cause note → Set "resolved".

**Difficulty**: Hard — requires 10-15 commands. Tests authentication, multi-step workflows, and form interactions across multiple pages.

**Randomized per episode**: Credentials, incident ID, system name, root cause keyword.

## Reward Function

Rewards provide **meaningful signal over the full trajectory**, not just at episode end:

- **Per-step progress**: Reward proportional to sub-goal completion deltas
- **Positive shaping**: Small bonus (+0.02) for commands that advance progress
- **Negative shaping**: Penalty (-0.04 to -0.15) for invalid commands or no-progress steps
- **Final grading**: Weighted sum of sub-goal completion (0.0–1.0), deterministic

## Setup & Usage

### Docker (recommended)

```bash
# Build
docker build -t browser_env:latest .

# Run
docker run -p 8000:8000 browser_env:latest

# Health check
curl http://localhost:8000/health
```

### Local Development

```bash
# Install dependencies
uv sync

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run the smoke test
python server/browser_env_environment.py
```

### Run Inference

```bash
# Set your API key
export HF_TOKEN=your_token_here
# or
export OPENAI_API_KEY=your_key_here

# Run all 3 tasks
python inference.py
```

### Deploy to Hugging Face Spaces

```bash
openenv push
```

## Baseline Scores

| Model | Easy | Medium | Hard | Average |
|-------|------|--------|------|---------|
| Qwen2.5-72B-Instruct | ~1.00 | ~0.75 | ~0.45 | ~0.73 |

*Scores are approximate and may vary due to model temperature.*

## Portal Structure

```
/                          Dashboard — stats, notifications, navigation
├── /directory             Employee roster (randomized per episode)
│   └── /directory/<id>    Employee profile detail
├── /tickets               Support ticket queue
│   ├── /tickets/new       Create new ticket form
│   └── /tickets/<id>      Ticket detail
├── /incidents             Active incidents dashboard
│   └── /incidents/<id>    Incident detail + resolution form
├── /login                 Authentication
└── /settings              User preferences
```

## Project Structure

```
browser_env/
├── __init__.py              # Module exports
├── models.py                # BrowserAction & BrowserObservation models
├── client.py                # BrowserEnv WebSocket client
├── inference.py             # Baseline inference script (runs all 3 tasks)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── Dockerfile               # Container image
├── .env                     # Configuration
├── README.md                # This file
└── server/
    ├── __init__.py           # Server module exports
    ├── app.py                # FastAPI application
    ├── browser_env_environment.py  # OpenEnv Environment implementation
    ├── task_engine.py        # Procedural task generation & grading
    └── simulation.py         # Portal simulation (pages, commands, state)
```

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static intranet pages, command parsing, and rendering for the browser simulator."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SessionState:
    path: str = "/"
    logged_in: bool = False
    email_notifications: bool = False
    form_values: Dict[str, str] = field(default_factory=dict)
    ticket_record: Optional[Dict[str, str]] = None
    visited_paths: List[str] = field(default_factory=list)

    def note_visit(self, path: str) -> None:
        if not self.visited_paths or self.visited_paths[-1] != path:
            self.visited_paths.append(path)


# (path) -> page spec
PAGES: Dict[str, Dict[str, object]] = {
    "/": {
        "title": "Corp Intranet — Home",
        "body": [
            "Welcome. Use text commands to navigate this internal portal.",
            "Example: goto /team   or   click nav_team",
        ],
        "links": {
            "nav_team": ("/team", "Team directory"),
            "nav_login": ("/login", "Employee login"),
            "nav_ticket": ("/ticket", "File IT ticket"),
            "nav_settings": ("/settings", "Notification settings"),
        },
        "form": None,
    },
    "/team": {
        "title": "Team directory",
        "body": [
            "People & IDs",
            "Jane Doe — Employee ID: E-7741 — Engineering",
            "When you can read the employee ID above, issue the finish command from this page.",
        ],
        "links": {"nav_home": ("/", "Back to home")},
        "form": None,
    },
    "/login": {
        "title": "Employee login",
        "body": ["Sign in to access restricted areas such as notification settings."],
        "links": {"nav_home": ("/", "Back to home")},
        "form": {
            "fields": [
                ("username", "text", "demo"),
                ("password", "password", "secret99"),
            ],
            "submit_action": "login",
        },
    },
    "/ticket": {
        "title": "IT ticket",
        "body": [
            "Open a support ticket. Set category to Hardware and mention a keyboard in the description.",
        ],
        "links": {"nav_home": ("/", "Back to home")},
        "form": {
            "fields": [
                ("category", "text", "Hardware | Software | Network"),
                ("description", "text", "Describe the issue"),
            ],
            "submit_action": "ticket",
        },
    },
    "/settings": {
        "title": "Notification settings",
        "body": [],
        "links": {"nav_home": ("/", "Back to home")},
        "form": None,
    },
}


TASK_INSTRUCTIONS: Dict[str, str] = {
    "easy": (
        "Navigate to the Team directory and finish while the employee ID E-7741 is visible "
        "on the page."
    ),
    "medium": (
        "File an IT ticket with category exactly 'Hardware' and a description that contains "
        "the word 'keyboard' (any casing). Then finish."
    ),
    "hard": (
        "Log in with username 'demo' and password 'secret99', open Notification settings, "
        "enable email notifications with toggle email_notifications, then finish."
    ),
}


def render_page(path: str, session: SessionState) -> Tuple[str, str, str]:
    """Return (title, url, content) for the observation."""
    if path not in PAGES:
        return (
            "Not found",
            path,
            f"No page at {path}. Try goto / or click a nav_* link from the home page.",
        )

    spec = PAGES[path]
    title = str(spec["title"])
    lines: List[str] = list(spec["body"])  # type: ignore[arg-type]

    if path == "/settings":
        if not session.logged_in:
            lines = [
                "Restricted — please log in first.",
                "Use: goto /login  then type username demo  and  type password secret99  then submit.",
            ]
        else:
            state = "ON" if session.email_notifications else "OFF"
            lines = [
                "Email notifications are currently "
                + state
                + ".",
                "Issue: toggle email_notifications   to flip the setting, then finish.",
            ]

    link_map: Dict[str, Tuple[str, str]] = spec.get("links", {})  # type: ignore[assignment]
    if link_map:
        link_lines = [f"[{lid}] {label} -> {target}" for lid, (target, label) in link_map.items()]
        lines.append("Links:\n  " + "\n  ".join(link_lines))

    form = spec.get("form")
    if form:
        fields = form["fields"]  # type: ignore[index]
        lines.append("Form fields (use: type <field> <value>):")
        for name, ftype, hint in fields:
            current = session.form_values.get(name, "")
            suffix = f" (current: '{current}')" if current else ""
            lines.append(f"  - {name} ({ftype}) — hint: {hint}{suffix}")
        lines.append("When ready: submit")

    content = "\n".join(lines)
    return title, path, content


def parse_command(raw: str) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """
    Parse a command into (verb, arg1, extra_tokens).

    Returns (None, None, None) if empty.
    """
    text = raw.strip()
    if not text:
        return None, None, None
    lower = text.lower()
    if lower in {"submit", "finish"}:
        return lower, None, None

    tokens = shlex.split(text)
    if not tokens:
        return None, None, None
    verb = tokens[0].lower()
    rest = tokens[1:]

    if verb in {"goto", "navigate"}:
        if not rest:
            return "error", "missing_path", None
        return "goto", rest[0], None

    if verb == "click":
        if not rest:
            return "error", "missing_link_id", None
        return "click", rest[0], None

    if verb in {"type", "set"}:
        if len(rest) < 2:
            return "error", "type_needs_name_value", None
        name = rest[0]
        value = " ".join(rest[1:])
        return "type", name, [value]

    if verb == "toggle":
        if not rest:
            return "error", "missing_toggle", None
        return "toggle", rest[0], None

    return "error", "unknown_verb", None


def apply_command(session: SessionState, command: str) -> Tuple[bool, str, float]:
    """
    Mutate session. Returns (ok, feedback, shaping_reward_delta).
    """
    parsed = parse_command(command)
    if parsed == (None, None, None):
        return False, "Empty command.", -0.02

    verb, arg1, extra = parsed
    shaping = 0.0

    if verb == "error":
        hints = {
            "missing_path": "Usage: goto /team",
            "missing_link_id": "Usage: click nav_team",
            "type_needs_name_value": "Usage: type description broken keyboard",
            "missing_toggle": "Usage: toggle email_notifications",
            "unknown_verb": "Unknown command. Try goto, click, type, submit, toggle, finish.",
        }
        return False, hints.get(str(arg1), "Invalid command."), -0.05

    if verb == "goto":
        assert arg1 is not None
        path = arg1 if arg1.startswith("/") else f"/{arg1}"
        if path not in PAGES:
            return False, f"Unknown path {path}", -0.02
        session.path = path
        session.note_visit(path)
        shaping += 0.04
        return True, f"Navigated to {path}", shaping

    if verb == "click":
        assert arg1 is not None
        spec = PAGES.get(session.path, {})
        links: Dict[str, Tuple[str, str]] = spec.get("links", {})  # type: ignore[assignment]
        link = links.get(arg1)
        if not link:
            return False, f"No link '{arg1}' on this page.", -0.03
        target = link[0]
        session.path = target
        session.note_visit(target)
        shaping += 0.05
        return True, f"Followed {arg1} to {target}", shaping

    if verb == "type":
        assert arg1 is not None and extra is not None
        name = arg1
        value = extra[0]
        spec = PAGES.get(session.path, {})
        form = spec.get("form")
        if not form:
            return False, "There is no form on this page.", -0.03
        field_names = {f[0] for f in form["fields"]}  # type: ignore[index]
        if name not in field_names:
            return False, f"Unknown field '{name}' on this form.", -0.03
        session.form_values[name] = value
        shaping += 0.06
        return True, f"Set '{name}'.", shaping

    if verb == "toggle":
        assert arg1 is not None
        if session.path != "/settings" or not session.logged_in:
            return False, "Toggles are only available on /settings after login.", -0.03
        if arg1.lower() != "email_notifications":
            return False, "Unknown toggle. Use: toggle email_notifications", -0.03
        session.email_notifications = not session.email_notifications
        shaping += 0.12
        return True, "Toggled email notifications.", shaping

    if verb == "submit":
        spec = PAGES.get(session.path, {})
        form = spec.get("form")
        if not form:
            return False, "Nothing to submit on this page.", -0.03
        action = form["submit_action"]  # type: ignore[index]
        if action == "login":
            user = session.form_values.get("username", "")
            pwd = session.form_values.get("password", "")
            if user == "demo" and pwd == "secret99":
                session.logged_in = True
                session.form_values.clear()
                shaping += 0.25
                return True, "Login successful.", shaping
            session.form_values.clear()
            return False, "Invalid credentials. Expected username demo and password secret99.", -0.05
        if action == "ticket":
            category = session.form_values.get("category", "")
            description = session.form_values.get("description", "")
            session.ticket_record = {"category": category, "description": description}
            session.form_values.clear()
            shaping += 0.18
            return True, "Ticket submitted.", shaping
        return False, "Unsupported form.", -0.03

    if verb == "finish":
        return True, "Episode complete — grading now.", 0.0

    return False, "Unhandled command.", -0.05


def heuristic_progress(task_id: str, session: SessionState) -> float:
    """Cheap partial-progress signal for reward shaping / observations."""
    if task_id == "easy":
        if session.path == "/team":
            return 1.0
        if "/team" in session.visited_paths:
            return 0.55
        return 0.15

    if task_id == "medium":
        score = 0.1
        if session.path == "/ticket" or "/ticket" in session.visited_paths:
            score += 0.25
        cat = session.form_values.get("category", "")
        desc = session.form_values.get("description", "")
        if cat.strip().lower() == "hardware":
            score += 0.25
        if "keyboard" in desc.lower():
            score += 0.25
        if session.ticket_record:
            score = max(score, 0.65)
            tr = session.ticket_record
            if tr.get("category", "").strip() == "Hardware" and "keyboard" in tr.get("description", "").lower():
                score = 1.0
        return min(1.0, score)

    # hard
    score = 0.1
    if session.logged_in:
        score += 0.35
    if "/settings" in session.visited_paths:
        score += 0.2
    if session.email_notifications:
        score += 0.35
    return min(1.0, score)


def grade_task(task_id: str, session: SessionState) -> Tuple[float, str]:
    """Return (score, rationale) with score in [0, 1]."""
    if task_id == "easy":
        if session.path == "/team":
            return 1.0, "Finished on Team page with employee ID visible."
        if "/team" in session.visited_paths:
            return 0.4, "Visited Team page but finished elsewhere."
        return 0.0, "Never reached the Team directory."

    if task_id == "medium":
        ticket = session.ticket_record
        if not ticket:
            return 0.0, "No ticket submitted."
        cat_ok = ticket.get("category", "").strip() == "Hardware"
        desc_ok = "keyboard" in ticket.get("description", "").lower()
        score = 0.5 * float(cat_ok) + 0.5 * float(desc_ok)
        return score, f"Ticket graded (category_ok={cat_ok}, description_ok={desc_ok})."

    if task_id == "hard":
        parts = []
        total = 0.0
        if session.logged_in:
            total += 0.35
            parts.append("login_ok")
        if "/settings" in session.visited_paths and session.logged_in:
            total += 0.3
            parts.append("settings_ok")
        if session.email_notifications:
            total += 0.35
            parts.append("notifications_on")
        rationale = "Hard task checks: " + (", ".join(parts) if parts else "none satisfied")
        return min(1.0, total), rationale

    return 0.0, "Unknown task id."


def extract_task_id(raw: Optional[str]) -> str:
    value = (raw or "easy").strip().lower()
    if value in TASK_INSTRUCTIONS:
        return value
    return "easy"

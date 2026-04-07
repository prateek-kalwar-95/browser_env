"""Procedural task generation engine for the IT Helpdesk Portal.

Generates task instances with randomized parameters per episode,
sub-goal decomposition, and deterministic weighted grading.
Each task template is parameterized — a seed controls randomization
for reproducible benchmarking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data pools for randomizing task parameters
# ---------------------------------------------------------------------------

EMPLOYEE_POOL = [
    {"id": "E-3301", "name": "Alice Nguyen", "dept": "Engineering", "role": "Senior Developer", "email": "anguyen@corp.io", "manager": "Carlos Ruiz"},
    {"id": "E-3302", "name": "Bob Patel", "dept": "Design", "role": "UX Lead", "email": "bpatel@corp.io", "manager": "Diana Cho"},
    {"id": "E-3303", "name": "Carlos Ruiz", "dept": "Engineering", "role": "Engineering Manager", "email": "cruiz@corp.io", "manager": "Fiona Li"},
    {"id": "E-3304", "name": "Diana Cho", "dept": "Design", "role": "Design Director", "email": "dcho@corp.io", "manager": "Fiona Li"},
    {"id": "E-3305", "name": "Ethan Brooks", "dept": "IT Support", "role": "Systems Admin", "email": "ebrooks@corp.io", "manager": "Grace Kim"},
    {"id": "E-3306", "name": "Fiona Li", "dept": "Operations", "role": "VP Operations", "email": "fli@corp.io", "manager": "N/A"},
    {"id": "E-3307", "name": "Grace Kim", "dept": "IT Support", "role": "IT Manager", "email": "gkim@corp.io", "manager": "Fiona Li"},
    {"id": "E-3308", "name": "Hassan Ali", "dept": "Finance", "role": "Financial Analyst", "email": "hali@corp.io", "manager": "Javier Mendez"},
    {"id": "E-3309", "name": "Iris Wang", "dept": "Engineering", "role": "DevOps Engineer", "email": "iwang@corp.io", "manager": "Carlos Ruiz"},
    {"id": "E-3310", "name": "Javier Mendez", "dept": "Finance", "role": "Finance Manager", "email": "jmendez@corp.io", "manager": "Fiona Li"},
]

TICKET_CATEGORIES = ["Hardware", "Software", "Network", "Access", "Security"]
TICKET_PRIORITIES = ["Low", "Medium", "High", "Critical"]
TICKET_KEYWORDS = [
    "printer", "monitor", "keyboard", "laptop", "VPN",
    "email", "password", "Wi-Fi", "server", "database",
]

INCIDENT_SYSTEMS = [
    "Email Gateway", "CI/CD Pipeline", "Customer Portal",
    "Payment Service", "Authentication Server", "File Storage",
    "Monitoring Dashboard", "Internal Wiki", "Chat Platform",
]

ROOT_CAUSES = [
    "memory leak", "expired certificate", "disk full",
    "DNS misconfiguration", "firewall rule change",
    "dependency update", "connection pool exhaustion",
    "corrupted index", "race condition",
]

CREDENTIALS = [
    {"username": "admin", "password": "helpdesk2024"},
    {"username": "operator", "password": "ops_pass99"},
    {"username": "sysadmin", "password": "s3cure!now"},
]

EXISTING_TICKETS = [
    {"id": "TK-1001", "title": "Cannot print to floor 3 printer", "category": "Hardware", "priority": "Medium", "status": "Open", "assignee": "Ethan Brooks"},
    {"id": "TK-1002", "title": "Slack notifications not working", "category": "Software", "priority": "Low", "status": "Open", "assignee": "Iris Wang"},
    {"id": "TK-1003", "title": "VPN disconnects every 10 minutes", "category": "Network", "priority": "High", "status": "In Progress", "assignee": "Grace Kim"},
    {"id": "TK-1004", "title": "Need access to staging environment", "category": "Access", "priority": "Medium", "status": "Open", "assignee": "Carlos Ruiz"},
    {"id": "TK-1005", "title": "Suspicious login attempts detected", "category": "Security", "priority": "Critical", "status": "Open", "assignee": "Grace Kim"},
]


# ---------------------------------------------------------------------------
# Sub-goal and task instance structures
# ---------------------------------------------------------------------------

@dataclass
class SubGoal:
    """A verifiable checkpoint within a task."""
    name: str
    description: str
    weight: float  # contribution to final score — all weights sum to 1.0
    check_key: str  # key used by the checker function

    def __repr__(self) -> str:
        return f"SubGoal({self.name}, w={self.weight:.2f})"


@dataclass
class TaskInstance:
    """A concrete task with filled parameters, ready for an episode."""
    task_id: str                       # "easy" | "medium" | "hard"
    goal_description: str              # Filled natural-language instruction
    sub_goals: List[SubGoal] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    roster: List[Dict[str, str]] = field(default_factory=list)
    existing_tickets: List[Dict[str, str]] = field(default_factory=list)
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    credentials: Dict[str, str] = field(default_factory=dict)

    @property
    def sub_goal_names(self) -> List[str]:
        return [sg.name for sg in self.sub_goals]


# ---------------------------------------------------------------------------
# Checker functions — evaluate session state against sub-goal criteria
# ---------------------------------------------------------------------------

def _check_visited(session, path: str) -> bool:
    """Check if a specific path was visited."""
    return path in session.visited_paths


def _check_current_path(session, path: str) -> bool:
    """Check if the agent is currently on a specific path."""
    return session.path == path


def _check_viewed_employee(session, employee_id: str) -> bool:
    """Check if the agent viewed a specific employee's page."""
    return session.viewed_employee == employee_id


def _check_logged_in(session) -> bool:
    """Check if the agent is logged in."""
    return session.logged_in


def _check_ticket_field(session, field_name: str, expected: str, case_insensitive: bool = False) -> bool:
    """Check if a submitted ticket has the expected field value."""
    if session.submitted_ticket is None:
        return False
    actual = session.submitted_ticket.get(field_name, "")
    if case_insensitive:
        return expected.lower() in actual.lower()
    return actual.strip() == expected.strip()


def _check_ticket_submitted(session) -> bool:
    """Check if a ticket was submitted."""
    return session.submitted_ticket is not None


def _check_incident_status(session, incident_id: str, status: str) -> bool:
    """Check if an incident was ever set to a specific status.

    Uses the status history so that transient statuses (e.g., 'investigating'
    before 'resolved') are still recognized.
    """
    # Check current status
    updates = session.incident_updates.get(incident_id, {})
    if updates.get("status", "").lower() == status.lower():
        return True
    # Check history
    history = session.incident_status_history.get(incident_id, [])
    return status.lower() in history


def _check_incident_note(session, incident_id: str, keyword: str) -> bool:
    """Check if an incident note contains a keyword."""
    updates = session.incident_updates.get(incident_id, {})
    notes = updates.get("notes", "")
    return keyword.lower() in notes.lower()


def _check_incident_viewed(session, incident_id: str) -> bool:
    """Check if a specific incident was viewed."""
    return session.viewed_incident == incident_id


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------

class TaskGenerator:
    """Generates randomized TaskInstance objects from templates."""

    @staticmethod
    def generate(task_id: str, seed: Optional[int] = None) -> TaskInstance:
        """Create a TaskInstance with randomized parameters.

        Args:
            task_id: One of "easy", "medium", "hard".
            seed: Random seed for reproducible generation. None for random.

        Returns:
            A fully parameterized TaskInstance.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        generators = {
            "easy": TaskGenerator._gen_easy,
            "medium": TaskGenerator._gen_medium,
            "hard": TaskGenerator._gen_hard,
        }
        gen_fn = generators.get(task_id)
        if gen_fn is None:
            # Default to easy for unknown task IDs
            gen_fn = generators["easy"]
            task_id = "easy"

        return gen_fn(rng, seed or 0)

    @staticmethod
    def _gen_easy(rng: random.Random, seed: int) -> TaskInstance:
        """Generate an employee information lookup task."""
        roster = rng.sample(EMPLOYEE_POOL, k=min(6, len(EMPLOYEE_POOL)))
        target = rng.choice(roster)
        attribute = rng.choice(["department", "role", "email", "manager"])

        attr_display = {
            "department": "department",
            "role": "job title",
            "email": "email address",
            "manager": "direct manager",
        }

        goal = (
            f"Find {target['name']}'s {attr_display[attribute]} in the employee directory. "
            f"Navigate to their profile page and finish while viewing it."
        )

        sub_goals = [
            SubGoal(
                name="navigate_directory",
                description="Navigate to the employee directory page",
                weight=0.25,
                check_key="visit_directory",
            ),
            SubGoal(
                name="view_target_employee",
                description=f"View {target['name']}'s profile page",
                weight=0.35,
                check_key=f"view_employee_{target['id']}",
            ),
            SubGoal(
                name="finish_on_profile",
                description=f"Finish while on {target['name']}'s profile",
                weight=0.40,
                check_key=f"finish_on_/directory/{target['id']}",
            ),
        ]

        return TaskInstance(
            task_id="easy",
            goal_description=goal,
            sub_goals=sub_goals,
            params={
                "target_employee": target,
                "target_attribute": attribute,
            },
            seed=seed,
            roster=roster,
            existing_tickets=rng.sample(EXISTING_TICKETS, k=3),
            incidents=[],
            credentials={},
        )

    @staticmethod
    def _gen_medium(rng: random.Random, seed: int) -> TaskInstance:
        """Generate a support ticket triage task."""
        roster = rng.sample(EMPLOYEE_POOL, k=min(6, len(EMPLOYEE_POOL)))
        category = rng.choice(TICKET_CATEGORIES)
        priority = rng.choice(TICKET_PRIORITIES)
        keyword = rng.choice(TICKET_KEYWORDS)
        assignee = rng.choice([e for e in roster if e["dept"] == "IT Support"] or roster)

        goal = (
            f"Create a support ticket: set category to '{category}', "
            f"priority to '{priority}', include the word '{keyword}' in the description, "
            f"and assign it to {assignee['name']}. Then submit and finish."
        )

        sub_goals = [
            SubGoal(
                name="navigate_new_ticket",
                description="Navigate to the new ticket form",
                weight=0.10,
                check_key="visit_tickets_new",
            ),
            SubGoal(
                name="set_category",
                description=f"Set ticket category to '{category}'",
                weight=0.20,
                check_key=f"ticket_category_{category}",
            ),
            SubGoal(
                name="set_priority",
                description=f"Set ticket priority to '{priority}'",
                weight=0.15,
                check_key=f"ticket_priority_{priority}",
            ),
            SubGoal(
                name="include_keyword",
                description=f"Include '{keyword}' in the description",
                weight=0.20,
                check_key=f"ticket_keyword_{keyword}",
            ),
            SubGoal(
                name="set_assignee",
                description=f"Assign ticket to {assignee['name']}",
                weight=0.15,
                check_key=f"ticket_assignee_{assignee['name']}",
            ),
            SubGoal(
                name="submit_ticket",
                description="Submit the ticket",
                weight=0.20,
                check_key="ticket_submitted",
            ),
        ]

        return TaskInstance(
            task_id="medium",
            goal_description=goal,
            sub_goals=sub_goals,
            params={
                "target_category": category,
                "target_priority": priority,
                "target_keyword": keyword,
                "target_assignee": assignee,
            },
            seed=seed,
            roster=roster,
            existing_tickets=rng.sample(EXISTING_TICKETS, k=3),
            incidents=[],
            credentials={},
        )

    @staticmethod
    def _gen_hard(rng: random.Random, seed: int) -> TaskInstance:
        """Generate an incident resolution workflow task."""
        roster = rng.sample(EMPLOYEE_POOL, k=min(6, len(EMPLOYEE_POOL)))
        creds = rng.choice(CREDENTIALS)
        system = rng.choice(INCIDENT_SYSTEMS)
        root_cause = rng.choice(ROOT_CAUSES)
        incident_id = f"INC-{rng.randint(2001, 2099)}"

        incident = {
            "id": incident_id,
            "system": system,
            "title": f"{system} outage — degraded performance",
            "status": "open",
            "severity": "P1",
            "reported_by": rng.choice(roster)["name"],
            "timeline": [
                "09:00 — Service degradation detected by monitoring",
                "09:05 — Alert triggered, on-call paged",
                "09:12 — Initial triage: service partially responding",
            ],
        }

        goal = (
            f"Log in with username '{creds['username']}' and password '{creds['password']}'. "
            f"Find incident {incident_id} ('{system}' outage). "
            f"Update its status to 'investigating', add a note mentioning '{root_cause}' "
            f"as the root cause, then set status to 'resolved' and finish."
        )

        sub_goals = [
            SubGoal(
                name="login",
                description="Log in with valid credentials",
                weight=0.12,
                check_key="logged_in",
            ),
            SubGoal(
                name="navigate_incidents",
                description="Navigate to the incidents dashboard",
                weight=0.08,
                check_key="visit_incidents",
            ),
            SubGoal(
                name="view_incident",
                description=f"View incident {incident_id}",
                weight=0.15,
                check_key=f"view_incident_{incident_id}",
            ),
            SubGoal(
                name="set_investigating",
                description="Update status to 'investigating'",
                weight=0.20,
                check_key=f"incident_status_{incident_id}_investigating",
            ),
            SubGoal(
                name="add_root_cause",
                description=f"Add note mentioning '{root_cause}'",
                weight=0.20,
                check_key=f"incident_note_{incident_id}_{root_cause}",
            ),
            SubGoal(
                name="resolve_incident",
                description="Set status to 'resolved'",
                weight=0.25,
                check_key=f"incident_status_{incident_id}_resolved",
            ),
        ]

        return TaskInstance(
            task_id="hard",
            goal_description=goal,
            sub_goals=sub_goals,
            params={
                "credentials": creds,
                "target_system": system,
                "target_root_cause": root_cause,
                "target_incident_id": incident_id,
            },
            seed=seed,
            roster=roster,
            existing_tickets=rng.sample(EXISTING_TICKETS, k=2),
            incidents=[incident],
            credentials=creds,
        )


# ---------------------------------------------------------------------------
# Sub-goal evaluation
# ---------------------------------------------------------------------------

def evaluate_sub_goals(
    task: TaskInstance, session: Any
) -> Tuple[Dict[str, bool], float]:
    """Evaluate all sub-goals against current session state.

    Returns:
        (status_dict, weighted_progress) where status_dict maps sub-goal
        name to bool and weighted_progress is in [0.0, 1.0].
    """
    status: Dict[str, bool] = {}
    total_weight = 0.0

    for sg in task.sub_goals:
        key = sg.check_key
        passed = _evaluate_single(key, task, session)
        status[sg.name] = passed
        if passed:
            total_weight += sg.weight

    return status, min(1.0, total_weight)


def _evaluate_single(check_key: str, task: TaskInstance, session: Any) -> bool:
    """Dispatch a single sub-goal check by its key."""

    # Navigation checks
    if check_key == "visit_directory":
        return _check_visited(session, "/directory")
    if check_key == "visit_tickets_new":
        return _check_visited(session, "/tickets/new")
    if check_key == "visit_incidents":
        return _check_visited(session, "/incidents")

    # Employee view checks
    if check_key.startswith("view_employee_"):
        emp_id = check_key[len("view_employee_"):]
        return _check_viewed_employee(session, emp_id)

    # Finish-on-path checks
    if check_key.startswith("finish_on_"):
        path = check_key[len("finish_on_"):]
        return _check_current_path(session, path)

    # Login check
    if check_key == "logged_in":
        return _check_logged_in(session)

    # Ticket field checks
    if check_key.startswith("ticket_category_"):
        expected = check_key[len("ticket_category_"):]
        return _check_ticket_field(session, "category", expected)
    if check_key.startswith("ticket_priority_"):
        expected = check_key[len("ticket_priority_"):]
        return _check_ticket_field(session, "priority", expected)
    if check_key.startswith("ticket_keyword_"):
        keyword = check_key[len("ticket_keyword_"):]
        return _check_ticket_field(session, "description", keyword, case_insensitive=True)
    if check_key.startswith("ticket_assignee_"):
        expected = check_key[len("ticket_assignee_"):]
        return _check_ticket_field(session, "assignee", expected)
    if check_key == "ticket_submitted":
        return _check_ticket_submitted(session)

    # Incident checks
    if check_key.startswith("view_incident_"):
        inc_id = check_key[len("view_incident_"):]
        return _check_incident_viewed(session, inc_id)
    if check_key.startswith("incident_status_"):
        # Format: incident_status_INC-XXXX_status
        parts = check_key[len("incident_status_"):].rsplit("_", 1)
        if len(parts) == 2:
            inc_id, status = parts
            return _check_incident_status(session, inc_id, status)
    if check_key.startswith("incident_note_"):
        # Format: incident_note_INC-XXXX_keyword
        rest = check_key[len("incident_note_"):]
        # Incident IDs look like INC-XXXX, split after the ID
        if "_" in rest:
            # Find the incident ID (format: INC-NNNN)
            parts = rest.split("_", 2)
            if len(parts) >= 3:
                inc_id = f"{parts[0]}-{parts[1]}"
                keyword = "_".join(parts[2:])
            else:
                inc_id = parts[0]
                keyword = parts[1] if len(parts) > 1 else ""
            return _check_incident_note(session, inc_id, keyword)

    return False


def grade_task(task: TaskInstance, session: Any) -> Tuple[float, str]:
    """Final deterministic grading: weighted sum of sub-goal completion.

    Returns:
        (score, rationale) where score is in [0.0, 1.0].
    """
    status, score = evaluate_sub_goals(task, session)
    completed = [name for name, ok in status.items() if ok]
    missed = [name for name, ok in status.items() if not ok]

    parts = []
    if completed:
        parts.append(f"completed: {', '.join(completed)}")
    if missed:
        parts.append(f"missed: {', '.join(missed)}")

    # Scale score from [0.0, 1.0] to [0.05, 0.95] to meet the strictly (0, 1) requirement
    scaled_score = 0.05 + (min(1.0, score) * 0.90)
    
    rationale = f"Score {scaled_score:.2f} (raw {score:.2f}) - " + "; ".join(parts)
    return scaled_score, rationale


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for tier in ("easy", "medium", "hard"):
        inst = TaskGenerator.generate(tier, seed=42)
        print(f"\n=== {tier.upper()} ===")
        print(f"Goal: {inst.goal_description}")
        print(f"Sub-goals: {inst.sub_goals}")
        print(f"Params: {inst.params}")
        print(f"Roster size: {len(inst.roster)}")

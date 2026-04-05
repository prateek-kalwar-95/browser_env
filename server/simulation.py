"""Dynamic IT Helpdesk Portal simulation engine.

Renders 10 page types with content generated from task parameters
and session state. Handles command parsing, page navigation, form
interactions, and the ticket/incident workflow state machine.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Session state - tracks everything the agent has done
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Mutable state for a single episode."""

    path: str = "/"
    logged_in: bool = False
    current_user: Optional[str] = None
    role: str = "employee"

    # Form interaction
    form_fields: Dict[str, str] = field(default_factory=dict)

    # Navigation
    visited_paths: List[str] = field(default_factory=list)
    path_history: List[str] = field(default_factory=list)  # for back command

    # Employee directory
    viewed_employee: Optional[str] = None

    # Helpdesk state
    submitted_ticket: Optional[Dict[str, str]] = None
    incident_updates: Dict[str, Dict[str, str]] = field(default_factory=dict)
    incident_status_history: Dict[str, List[str]] = field(default_factory=dict)
    viewed_incident: Optional[str] = None

    # Notifications
    notifications: List[str] = field(default_factory=list)
    notifications_read: bool = False

    # Populated by task engine on reset
    roster: List[Dict[str, str]] = field(default_factory=list)
    existing_tickets: List[Dict[str, str]] = field(default_factory=list)
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    credentials: Dict[str, str] = field(default_factory=dict)

    def note_visit(self, path: str) -> None:
        if not self.visited_paths or self.visited_paths[-1] != path:
            self.visited_paths.append(path)
        # Track for back navigation
        if not self.path_history or self.path_history[-1] != path:
            self.path_history.append(path)

    @property
    def breadcrumb(self) -> List[str]:
        """Recent navigation trail (last 5 unique)."""
        seen = []
        for p in reversed(self.path_history):
            if p not in seen:
                seen.append(p)
            if len(seen) >= 5:
                break
        return list(reversed(seen))


# ---------------------------------------------------------------------------
# Page rendering - each page has a dedicated renderer
# ---------------------------------------------------------------------------

def render_page(path: str, session: SessionState) -> Tuple[str, str, str]:
    """Return (title, url, rendered_content) for the given path.

    Dispatches to per-page render functions based on path patterns.
    """
    # Exact-match pages
    page_renderers = {
        "/": _render_dashboard,
        "/directory": _render_directory,
        "/tickets": _render_ticket_queue,
        "/tickets/new": _render_new_ticket_form,
        "/incidents": _render_incidents,
        "/login": _render_login,
        "/settings": _render_settings,
    }

    renderer = page_renderers.get(path)
    if renderer:
        return renderer(session)

    # Pattern-match pages
    if path.startswith("/directory/"):
        emp_id = path.split("/directory/", 1)[1]
        return _render_employee_detail(emp_id, session)

    if path.startswith("/tickets/") and path != "/tickets/new":
        ticket_id = path.split("/tickets/", 1)[1]
        return _render_ticket_detail(ticket_id, session)

    if path.startswith("/incidents/"):
        incident_id = path.split("/incidents/", 1)[1]
        return _render_incident_detail(incident_id, session)

    return (
        "Page Not Found",
        path,
        f"No page at '{path}'. Use 'goto /' to return to the dashboard.",
    )


def _render_dashboard(session: SessionState) -> Tuple[str, str, str]:
    lines = [
        "Welcome to the IT Helpdesk Portal.",
        "",
    ]

    if session.current_user:
        lines.append(f"Logged in as: {session.current_user} ({session.role})")
    else:
        lines.append("You are not logged in. Some features require authentication.")

    lines.append("")
    lines.append("=== Quick Stats ===")
    lines.append(f"  Open tickets: {len(session.existing_tickets)}")
    lines.append(f"  Active incidents: {len(session.incidents)}")
    lines.append(f"  Employees in directory: {len(session.roster)}")

    if session.notifications and not session.notifications_read:
        lines.append("")
        lines.append("=== Notifications ===")
        for note in session.notifications:
            lines.append(f"  * {note}")

    lines.append("")
    lines.append("=== Navigation ===")
    nav_links = [
        ("nav_directory", "/directory", "Employee Directory"),
        ("nav_tickets", "/tickets", "Support Tickets"),
        ("nav_incidents", "/incidents", "Active Incidents"),
        ("nav_settings", "/settings", "Settings"),
    ]
    if not session.logged_in:
        nav_links.append(("nav_login", "/login", "Log In"))

    for link_id, target, label in nav_links:
        lines.append(f"  [{link_id}] {label} -> {target}")

    return "IT Helpdesk - Dashboard", "/", "\n".join(lines)


def _render_directory(session: SessionState) -> Tuple[str, str, str]:
    lines = [
        "Employee Directory",
        "Browse employee profiles. Click an employee ID to view details.",
        "",
    ]

    if not session.roster:
        lines.append("No employee records available.")
    else:
        lines.append(f"{'ID':<10} {'Name':<20} {'Department':<16} {'Role'}")
        lines.append("-" * 70)
        for emp in session.roster:
            emp_id = emp["id"]
            lines.append(
                f"{emp_id:<10} {emp['name']:<20} {emp['dept']:<16} {emp['role']}"
            )
            lines.append(f"  [emp_{emp_id}] View profile -> /directory/{emp_id}")

    lines.append("")
    lines.append("Links:")
    lines.append("  [nav_home] Dashboard -> /")
    lines.append("  [nav_tickets] Support Tickets -> /tickets")

    return "Employee Directory", "/directory", "\n".join(lines)


def _render_employee_detail(emp_id: str, session: SessionState) -> Tuple[str, str, str]:
    employee = None
    for emp in session.roster:
        if emp["id"] == emp_id:
            employee = emp
            break

    if employee is None:
        return (
            "Employee Not Found",
            f"/directory/{emp_id}",
            f"No employee with ID '{emp_id}'. Use 'goto /directory' to browse all employees.",
        )

    # Mark as viewed
    session.viewed_employee = emp_id

    lines = [
        f"Employee Profile: {employee['name']}",
        "",
        f"  Employee ID:  {employee['id']}",
        f"  Full Name:    {employee['name']}",
        f"  Department:   {employee['dept']}",
        f"  Job Title:    {employee['role']}",
        f"  Email:        {employee['email']}",
        f"  Manager:      {employee['manager']}",
        "",
        "Links:",
        "  [nav_directory] Back to Directory -> /directory",
        "  [nav_home] Dashboard -> /",
    ]

    return f"Profile - {employee['name']}", f"/directory/{emp_id}", "\n".join(lines)


def _render_ticket_queue(session: SessionState) -> Tuple[str, str, str]:
    lines = [
        "Support Ticket Queue",
        "View existing tickets or create a new one.",
        "",
    ]

    if session.existing_tickets:
        lines.append(f"{'ID':<10} {'Title':<40} {'Priority':<10} {'Status'}")
        lines.append("-" * 80)
        for tk in session.existing_tickets:
            lines.append(
                f"{tk['id']:<10} {tk['title']:<40} {tk['priority']:<10} {tk['status']}"
            )
            lines.append(f"  [tk_{tk['id']}] View details -> /tickets/{tk['id']}")

    if session.submitted_ticket:
        lines.append("")
        lines.append("--- Your submitted ticket ---")
        st = session.submitted_ticket
        lines.append(f"  Category: {st.get('category', 'N/A')}")
        lines.append(f"  Priority: {st.get('priority', 'N/A')}")
        lines.append(f"  Assignee: {st.get('assignee', 'N/A')}")
        lines.append(f"  Description: {st.get('description', 'N/A')}")

    lines.append("")
    lines.append("Links:")
    lines.append("  [nav_new_ticket] Create New Ticket -> /tickets/new")
    lines.append("  [nav_home] Dashboard -> /")

    return "Support Tickets", "/tickets", "\n".join(lines)


def _render_ticket_detail(ticket_id: str, session: SessionState) -> Tuple[str, str, str]:
    ticket = None
    for tk in session.existing_tickets:
        if tk["id"] == ticket_id:
            ticket = tk
            break

    if ticket is None:
        return (
            "Ticket Not Found",
            f"/tickets/{ticket_id}",
            f"No ticket with ID '{ticket_id}'. Use 'goto /tickets' to view all tickets.",
        )

    lines = [
        f"Ticket: {ticket['id']} - {ticket['title']}",
        "",
        f"  Category:  {ticket['category']}",
        f"  Priority:  {ticket['priority']}",
        f"  Status:    {ticket['status']}",
        f"  Assignee:  {ticket['assignee']}",
        "",
        "Links:",
        "  [nav_tickets] Back to Queue -> /tickets",
        "  [nav_home] Dashboard -> /",
    ]

    return f"Ticket {ticket['id']}", f"/tickets/{ticket_id}", "\n".join(lines)


def _render_new_ticket_form(session: SessionState) -> Tuple[str, str, str]:
    lines = [
        "Create New Support Ticket",
        "Fill in all fields, then submit.",
        "",
        "Form fields (use: type <field> <value>  or  select <field> <option>):",
        f"  - category (select) - options: Hardware, Software, Network, Access, Security"
        f"    (current: '{session.form_fields.get('category', '')}')",
        f"  - priority (select) - options: Low, Medium, High, Critical"
        f"    (current: '{session.form_fields.get('priority', '')}')",
        f"  - assignee (type) - employee name to assign"
        f"    (current: '{session.form_fields.get('assignee', '')}')",
        f"  - description (type) - describe the issue"
        f"    (current: '{session.form_fields.get('description', '')}')",
        "",
        "When all fields are filled: submit",
        "",
        "Links:",
        "  [nav_tickets] Back to Queue -> /tickets",
        "  [nav_home] Dashboard -> /",
    ]

    return "New Ticket", "/tickets/new", "\n".join(lines)


def _render_incidents(session: SessionState) -> Tuple[str, str, str]:
    lines = [
        "Active Incidents Dashboard",
        "View and manage active system incidents.",
        "",
    ]

    if not session.incidents:
        lines.append("No active incidents. All systems operational.")
    else:
        for inc in session.incidents:
            inc_id = inc["id"]
            # Check if status was updated
            updates = session.incident_updates.get(inc_id, {})
            current_status = updates.get("status", inc["status"])

            lines.append(f"{'='*60}")
            lines.append(f"  {inc_id}: {inc['title']}")
            lines.append(f"  System: {inc['system']}  |  Severity: {inc['severity']}  |  Status: {current_status}")
            lines.append(f"  Reported by: {inc['reported_by']}")
            lines.append(f"  [inc_{inc_id}] View full details -> /incidents/{inc_id}")

    lines.append("")
    lines.append("Links:")
    lines.append("  [nav_home] Dashboard -> /")

    return "Active Incidents", "/incidents", "\n".join(lines)


def _render_incident_detail(incident_id: str, session: SessionState) -> Tuple[str, str, str]:
    incident = None
    for inc in session.incidents:
        if inc["id"] == incident_id:
            incident = inc
            break

    if incident is None:
        return (
            "Incident Not Found",
            f"/incidents/{incident_id}",
            f"No incident with ID '{incident_id}'. Use 'goto /incidents' to view all.",
        )

    # Mark as viewed
    session.viewed_incident = incident_id

    updates = session.incident_updates.get(incident_id, {})
    current_status = updates.get("status", incident["status"])
    notes = updates.get("notes", "")

    lines = [
        f"Incident: {incident_id} - {incident['title']}",
        "",
        f"  System:      {incident['system']}",
        f"  Severity:    {incident['severity']}",
        f"  Status:      {current_status}",
        f"  Reported by: {incident['reported_by']}",
        "",
        "=== Timeline ===",
    ]
    for event in incident.get("timeline", []):
        lines.append(f"  {event}")

    if notes:
        lines.append("")
        lines.append("=== Investigation Notes ===")
        lines.append(f"  {notes}")

    if session.logged_in:
        lines.append("")
        lines.append("=== Actions (requires login) ===")
        lines.append("  Update status:  select status <investigating|mitigated|resolved>")
        lines.append("  Add note:       type notes <your investigation notes>")
        lines.append("  Apply changes:  submit")
        lines.append("")
        lines.append("Form fields:")
        lines.append(f"  - status (select) - current pending: '{session.form_fields.get('status', '')}'")
        lines.append(f"  - notes (type)    - current pending: '{session.form_fields.get('notes', '')}'")
    else:
        lines.append("")
        lines.append("  [!] Log in to update this incident.")

    lines.append("")
    lines.append("Links:")
    lines.append("  [nav_incidents] Back to Incidents -> /incidents")
    lines.append("  [nav_home] Dashboard -> /")

    return f"Incident {incident_id}", f"/incidents/{incident_id}", "\n".join(lines)


def _render_login(session: SessionState) -> Tuple[str, str, str]:
    if session.logged_in:
        lines = [
            f"You are already logged in as {session.current_user}.",
            "",
            "Links:",
            "  [nav_home] Dashboard -> /",
        ]
        return "Logged In", "/login", "\n".join(lines)

    lines = [
        "Employee Authentication",
        "Sign in to access restricted features (incident management, settings).",
        "",
        "Form fields (use: type <field> <value>):",
        f"  - username (text) - (current: '{session.form_fields.get('username', '')}')",
        f"  - password (password) - (current: '{session.form_fields.get('password', '')}')",
        "",
        "When ready: submit",
        "",
        "Links:",
        "  [nav_home] Dashboard -> /",
    ]
    return "Log In", "/login", "\n".join(lines)


def _render_settings(session: SessionState) -> Tuple[str, str, str]:
    if not session.logged_in:
        lines = [
            "Restricted - please log in first.",
            "Use: goto /login",
            "",
            "Links:",
            "  [nav_login] Log In -> /login",
            "  [nav_home] Dashboard -> /",
        ]
        return "Settings (Locked)", "/settings", "\n".join(lines)

    lines = [
        "User Settings",
        f"Logged in as: {session.current_user}",
        "",
        "Notification preferences and display options.",
        "Use: toggle <setting_name> to flip a boolean setting.",
        "",
        "Links:",
        "  [nav_home] Dashboard -> /",
    ]
    return "Settings", "/settings", "\n".join(lines)


# ---------------------------------------------------------------------------
# Known page paths (used for goto validation)
# ---------------------------------------------------------------------------

STATIC_PAGES = {"/", "/directory", "/tickets", "/tickets/new", "/incidents", "/login", "/settings"}


def _is_valid_path(path: str, session: SessionState) -> bool:
    """Check if a path is navigable."""
    if path in STATIC_PAGES:
        return True
    if path.startswith("/directory/"):
        emp_id = path.split("/directory/", 1)[1]
        return any(e["id"] == emp_id for e in session.roster)
    if path.startswith("/tickets/"):
        tk_id = path.split("/tickets/", 1)[1]
        return any(t["id"] == tk_id for t in session.existing_tickets)
    if path.startswith("/incidents/"):
        inc_id = path.split("/incidents/", 1)[1]
        return any(i["id"] == inc_id for i in session.incidents)
    return False


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

def parse_command(raw: str) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """Parse user command into (verb, arg1, extra_args).

    Returns (None, None, None) if the command is empty or unparseable.
    """
    text = raw.strip()
    if not text:
        return None, None, None

    lower = text.lower()
    if lower in {"submit", "finish", "back"}:
        return lower, None, None

    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()

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
            return "error", "missing_element", None
        return "click", rest[0], None

    if verb in {"type", "set"}:
        if len(rest) < 2:
            return "error", "type_needs_field_value", None
        field_name = rest[0]
        value = " ".join(rest[1:])
        return "type", field_name, [value]

    if verb == "select":
        if len(rest) < 2:
            return "error", "select_needs_field_option", None
        field_name = rest[0]
        option = " ".join(rest[1:])
        return "select", field_name, [option]

    if verb == "toggle":
        if not rest:
            return "error", "missing_toggle", None
        return "toggle", rest[0], None

    return "error", "unknown_verb", None


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

# Link-ID -> target path mapping for click commands
LINK_TARGETS: Dict[str, str] = {
    "nav_home": "/",
    "nav_directory": "/directory",
    "nav_tickets": "/tickets",
    "nav_new_ticket": "/tickets/new",
    "nav_incidents": "/incidents",
    "nav_login": "/login",
    "nav_settings": "/settings",
}


def apply_command(session: SessionState, command: str) -> Tuple[bool, str, float]:
    """Execute a command, mutating session. Returns (ok, feedback, reward_delta)."""

    parsed = parse_command(command)
    if parsed == (None, None, None):
        return False, "Empty command. Try: goto /directory", -0.02

    verb, arg1, extra = parsed

    if verb == "error":
        hints = {
            "missing_path": "Usage: goto /directory",
            "missing_element": "Usage: click nav_directory",
            "type_needs_field_value": "Usage: type description printer is broken",
            "select_needs_field_option": "Usage: select category Hardware",
            "missing_toggle": "Usage: toggle email_notifications",
            "unknown_verb": "Unknown command. Available: goto, click, type, select, submit, back, toggle, finish.",
        }
        return False, hints.get(str(arg1), "Invalid command."), -0.04

    # --- goto ---
    if verb == "goto":
        assert arg1 is not None
        path = arg1 if arg1.startswith("/") else f"/{arg1}"
        if not _is_valid_path(path, session):
            return False, f"Unknown path '{path}'. Try goto /directory or goto /tickets", -0.02
        session.form_fields.clear()  # Clear form when navigating away
        session.path = path
        session.note_visit(path)
        # Track views for sub-goal evaluation
        if path.startswith("/directory/"):
            session.viewed_employee = path.split("/directory/", 1)[1]
        if path.startswith("/incidents/"):
            session.viewed_incident = path.split("/incidents/", 1)[1]
        return True, f"Navigated to {path}", 0.03

    # --- click ---
    if verb == "click":
        assert arg1 is not None
        link_id = arg1

        # Static link map
        if link_id in LINK_TARGETS:
            target = LINK_TARGETS[link_id]
            session.form_fields.clear()
            session.path = target
            session.note_visit(target)
            return True, f"Clicked {link_id}, navigated to {target}", 0.03

        # Dynamic employee links: emp_E-XXXX
        if link_id.startswith("emp_"):
            emp_id = link_id[4:]
            target = f"/directory/{emp_id}"
            if _is_valid_path(target, session):
                session.form_fields.clear()
                session.path = target
                session.note_visit(target)
                session.viewed_employee = emp_id
                return True, f"Opened employee profile {emp_id}", 0.04
            return False, f"No employee with ID '{emp_id}'.", -0.02

        # Dynamic ticket links: tk_TK-XXXX
        if link_id.startswith("tk_"):
            tk_id = link_id[3:]
            target = f"/tickets/{tk_id}"
            if _is_valid_path(target, session):
                session.path = target
                session.note_visit(target)
                return True, f"Opened ticket {tk_id}", 0.03
            return False, f"No ticket with ID '{tk_id}'.", -0.02

        # Dynamic incident links: inc_INC-XXXX
        if link_id.startswith("inc_"):
            inc_id = link_id[4:]
            target = f"/incidents/{inc_id}"
            if _is_valid_path(target, session):
                session.path = target
                session.note_visit(target)
                session.viewed_incident = inc_id
                return True, f"Opened incident {inc_id}", 0.04
            return False, f"No incident with ID '{inc_id}'.", -0.02

        return False, f"No clickable element '{link_id}' on this page.", -0.03

    # --- type ---
    if verb == "type":
        assert arg1 is not None and extra is not None
        field_name = arg1
        value = extra[0]

        # Check we're on a page with a form
        pages_with_forms = {"/tickets/new", "/login"}
        on_incident_detail = session.path.startswith("/incidents/")

        if session.path not in pages_with_forms and not on_incident_detail:
            return False, "No form on this page. Navigate to a form page first.", -0.02

        # Validate field names per page
        valid_fields = _get_form_fields(session.path)
        if field_name not in valid_fields:
            return False, f"Unknown field '{field_name}'. Valid fields: {', '.join(valid_fields)}", -0.02

        session.form_fields[field_name] = value
        return True, f"Set '{field_name}' to '{value}'.", 0.04

    # --- select ---
    if verb == "select":
        assert arg1 is not None and extra is not None
        field_name = arg1
        option = extra[0]

        pages_with_forms = {"/tickets/new", "/login"}
        on_incident_detail = session.path.startswith("/incidents/")

        if session.path not in pages_with_forms and not on_incident_detail:
            return False, "No form on this page.", -0.02

        valid_fields = _get_form_fields(session.path)
        if field_name not in valid_fields:
            return False, f"Unknown field '{field_name}'. Valid fields: {', '.join(valid_fields)}", -0.02

        # Validate selection options for select-type fields
        valid_options = _get_select_options(session.path, field_name)
        if valid_options:
            # Case-insensitive match
            matched = None
            for opt in valid_options:
                if opt.lower() == option.lower():
                    matched = opt
                    break
            if matched is None:
                return False, f"Invalid option '{option}' for '{field_name}'. Options: {', '.join(valid_options)}", -0.02
            session.form_fields[field_name] = matched
        else:
            session.form_fields[field_name] = option

        return True, f"Selected '{field_name}' = '{session.form_fields[field_name]}'.", 0.05

    # --- submit ---
    if verb == "submit":
        return _handle_submit(session)

    # --- back ---
    if verb == "back":
        if len(session.path_history) <= 1:
            return False, "No previous page to go back to.", -0.01
        session.path_history.pop()  # Remove current
        prev = session.path_history[-1] if session.path_history else "/"
        session.form_fields.clear()
        session.path = prev
        return True, f"Went back to {prev}", 0.01

    # --- toggle ---
    if verb == "toggle":
        if session.path != "/settings" or not session.logged_in:
            return False, "Toggles only available on /settings after login.", -0.02
        return True, "Setting toggled.", 0.02

    return False, "Unhandled command.", -0.04


def _get_form_fields(path: str) -> List[str]:
    """Return valid form field names for a given page."""
    if path == "/tickets/new":
        return ["category", "priority", "assignee", "description"]
    if path == "/login":
        return ["username", "password"]
    if path.startswith("/incidents/"):
        return ["status", "notes"]
    return []


def _get_select_options(path: str, field_name: str) -> List[str]:
    """Return valid dropdown options for a select-type field."""
    if path == "/tickets/new":
        if field_name == "category":
            return ["Hardware", "Software", "Network", "Access", "Security"]
        if field_name == "priority":
            return ["Low", "Medium", "High", "Critical"]
    if path.startswith("/incidents/"):
        if field_name == "status":
            return ["investigating", "mitigated", "resolved"]
    return []


def _handle_submit(session: SessionState) -> Tuple[bool, str, float]:
    """Process form submission based on current page."""

    if session.path == "/login":
        user = session.form_fields.get("username", "")
        pwd = session.form_fields.get("password", "")

        if not session.credentials:
            # Fallback credentials
            valid_creds = {"admin": "helpdesk2024"}
        else:
            valid_creds = {session.credentials.get("username", "admin"):
                          session.credentials.get("password", "helpdesk2024")}

        if user in valid_creds and valid_creds[user] == pwd:
            session.logged_in = True
            session.current_user = user
            session.role = "manager" if user in {"admin", "sysadmin"} else "employee"
            session.form_fields.clear()
            session.notifications.append(f"Welcome back, {user}!")
            return True, "Login successful.", 0.15
        session.form_fields.clear()
        return False, "Invalid credentials. Check username and password.", -0.04

    if session.path == "/tickets/new":
        category = session.form_fields.get("category", "")
        priority = session.form_fields.get("priority", "")
        assignee = session.form_fields.get("assignee", "")
        description = session.form_fields.get("description", "")

        if not category or not description:
            return False, "Please fill in at least category and description before submitting.", -0.02

        session.submitted_ticket = {
            "category": category,
            "priority": priority,
            "assignee": assignee,
            "description": description,
        }
        session.form_fields.clear()
        session.notifications.append(f"Ticket created: {category} - {description[:40]}")
        return True, "Ticket submitted successfully.", 0.12

    if session.path.startswith("/incidents/"):
        if not session.logged_in:
            return False, "You must be logged in to update incidents.", -0.03

        incident_id = session.path.split("/incidents/", 1)[1]
        has_incident = any(i["id"] == incident_id for i in session.incidents)
        if not has_incident:
            return False, f"No incident '{incident_id}' to update.", -0.02

        status_val = session.form_fields.get("status", "")
        notes_val = session.form_fields.get("notes", "")

        if not status_val and not notes_val:
            return False, "Please set status or add notes before submitting.", -0.02

        if incident_id not in session.incident_updates:
            session.incident_updates[incident_id] = {}

        if status_val:
            session.incident_updates[incident_id]["status"] = status_val
            # Track history so sub-goal checks for past statuses still pass
            if incident_id not in session.incident_status_history:
                session.incident_status_history[incident_id] = []
            session.incident_status_history[incident_id].append(status_val.lower())
        if notes_val:
            existing_notes = session.incident_updates[incident_id].get("notes", "")
            if existing_notes:
                session.incident_updates[incident_id]["notes"] = f"{existing_notes}; {notes_val}"
            else:
                session.incident_updates[incident_id]["notes"] = notes_val

        session.form_fields.clear()
        feedback_parts = []
        if status_val:
            feedback_parts.append(f"status → {status_val}")
        if notes_val:
            feedback_parts.append("notes updated")
        return True, f"Incident {incident_id} updated: {', '.join(feedback_parts)}.", 0.10

    return False, "Nothing to submit on this page.", -0.02

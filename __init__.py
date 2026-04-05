"""IT Helpdesk Portal — Browser Environment for OpenEnv.

A simulated corporate IT helpdesk where an agent performs real-world
workplace tasks: employee lookup, ticket triage, and incident resolution.
"""

from .client import BrowserEnv
from .models import BrowserAction, BrowserObservation

__all__ = [
    "BrowserAction",
    "BrowserObservation",
    "BrowserEnv",
]

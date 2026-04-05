"""FastAPI server for the IT Helpdesk Portal environment.

Exposes the BrowserEnvironment over HTTP and WebSocket endpoints.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    python -m server.app
"""

from pathlib import Path

from dotenv import load_dotenv

# Load repo-root .env before OpenEnv reads config
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with:\n    uv sync\n"
    ) from e

try:
    from ..models import BrowserAction, BrowserObservation
    from .browser_env_environment import BrowserEnvironment
except ImportError:
    from models import BrowserAction, BrowserObservation
    from server.browser_env_environment import BrowserEnvironment


app = create_app(
    BrowserEnvironment,
    BrowserAction,
    BrowserObservation,
    env_name="browser_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()

"""Start Maslul: API server plus the trainer UI.

    python run.py

Serves the UI and API on the same origin, so there is no CORS hop and no
separate static server to remember.

PORT CONFLICT HANDLING IS NOT OPTIONAL HERE. A previous server holding the
port makes uvicorn fail to bind, and the symptom is a page that loads and
a voice that never speaks -- because the OLD process answers every request
with whatever code it started with. That cost a long debugging session, so
this now refuses to start rather than appearing to.
"""

from __future__ import annotations

import socket
import sys
import webbrowser
from pathlib import Path
from threading import Timer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PORT = 8000


def _port_in_use(port: int) -> bool:
    """Whether something is already listening -- so a restart fails loudly instead of silently serving stale code."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    """Serve the API and UI on one origin, refusing to start if the port is taken."""
    from dotenv import load_dotenv
    load_dotenv()

    if _port_in_use(PORT):
        # Loud and specific. Silently continuing would let a stale server
        # keep answering, which looks like broken code rather than a
        # leftover process.
        print(f"\n  Port {PORT} is already in use by another server.\n")
        print("  An old Maslul process is probably still running, and it would")
        print("  keep serving its OWN (outdated) code. Stop it first:\n")
        print("    Windows:  Get-NetTCPConnection -LocalPort 8000 -State Listen |")
        print("                ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }")
        print("    Mac/Linux: lsof -ti:8000 | xargs kill -9\n")
        return 1

    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from api.main import app

    # Mounted last and at the root, so every API route above still wins.
    app.mount("/", StaticFiles(directory=ROOT / "ui", html=True), name="ui")

    url = f"http://localhost:{PORT}"
    print(f"\n  Maslul — {url}\n")
    Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

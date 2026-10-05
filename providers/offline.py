"""Offline enforcement -- fail loudly here rather than silently there.

When MASLUL_OFFLINE=1, this installs a socket guard that RAISES on any
outbound network connection (FR-F6).

The point is not security; it is early warning. The deployment target has
no internet, so a stray cloud call there would surface as a mysterious
hang or a confusing error in a room where debugging is expensive. With
this guard on during development, the same mistake fails immediately, on
your machine, naming the host it tried to reach.

The test suite always runs with this enabled (tests/conftest.py), so
"nothing in the core path needs the network" is continuously verified
instead of assumed (NFR-2).

Localhost is deliberately permitted: a local model server (Ollama,
vLLM) is reached over HTTP on 127.0.0.1, and that is exactly the
configuration an air-gapped deployment uses.
"""

from __future__ import annotations

import os
import socket
from typing import Any

_LOCAL_HOSTS = frozenset({
    "localhost", "127.0.0.1", "::1", "0.0.0.0", "", None,
})

_original_socket_connect = socket.socket.connect
_original_create_connection = socket.create_connection
_installed = False


class NetworkAccessBlocked(RuntimeError):
    """An outbound connection was attempted while offline mode was active.

    Names the host and port so the offending call site is obvious, and
    states the remedy -- the point is to make this error self-explanatory
    to someone who has never read this module.
    """

    def __init__(self, host: Any, port: Any) -> None:
        """Build the message naming the blocked host and both remedies."""
        super().__init__(
            f"offline mode blocked an outbound connection to {host}:{port}. "
            f"This deployment target has no internet access. Either select a "
            f"local provider (LLM_PROVIDER=ollama) or, if this call is "
            f"genuinely needed during development, unset MASLUL_OFFLINE."
        )
        self.host = host
        self.port = port


def _is_local(address: Any) -> bool:
    """Whether an address is loopback, and therefore permitted."""
    if isinstance(address, (tuple, list)) and address:
        host = address[0]
        if host in _LOCAL_HOSTS:
            return True
        if isinstance(host, str) and (host.startswith("127.") or host == "::1"):
            return True
        return False
    # A unix socket path or anything non-inet is not an outbound network
    # call, so it is not this guard's concern.
    return True


def install() -> bool:
    """Install the guard if MASLUL_OFFLINE is set. Returns whether active.

    Idempotent: calling twice does not stack wrappers, which would
    otherwise make the original functions unrecoverable.
    """
    global _installed
    if _installed:
        return True
    if os.environ.get("MASLUL_OFFLINE", "0").strip() not in {"1", "true", "yes", "on"}:
        return False

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        """Refuse any socket connect to a non-local address."""
        if not _is_local(address):
            host, port = (address[0], address[1]) if isinstance(address, (tuple, list)) else (address, "?")
            raise NetworkAccessBlocked(host, port)
        return _original_socket_connect(self, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        """Refuse any new connection to a non-local address."""
        if not _is_local(address):
            host, port = (address[0], address[1]) if isinstance(address, (tuple, list)) else (address, "?")
            raise NetworkAccessBlocked(host, port)
        return _original_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    _installed = True
    return True


def uninstall() -> None:
    """Restore normal networking. For tests that must verify the guard
    itself, not for production use."""
    global _installed
    socket.socket.connect = _original_socket_connect  # type: ignore[method-assign]
    socket.create_connection = _original_create_connection  # type: ignore[assignment]
    _installed = False


def is_active() -> bool:
    """Whether the offline guard is currently installed."""
    return _installed

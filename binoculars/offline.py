"""Process-level safeguards for the local-only MCP runtime."""

from __future__ import annotations

import os
import socket
from typing import Any

OFFLINE_ENVIRONMENT = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_NETWORK_GUARD_INSTALLED = False


def configure_offline_environment() -> None:
    """Force supported ML libraries into offline, no-telemetry mode."""
    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value


def install_network_guard() -> None:
    """Reject network connections while preserving local Unix-domain sockets."""
    global _NETWORK_GUARD_INSTALLED
    if _NETWORK_GUARD_INSTALLED:
        return

    def guarded_connect(sock: socket.socket, address: Any) -> None:
        if sock.family != socket.AF_UNIX:
            raise OSError("Network access is disabled in the Binoculars local MCP runtime")
        _ORIGINAL_CONNECT(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family != socket.AF_UNIX:
            raise OSError("Network access is disabled in the Binoculars local MCP runtime")
        return _ORIGINAL_CONNECT_EX(sock, address)

    def guarded_create_connection(*args: Any, **kwargs: Any) -> socket.socket:
        raise OSError("Network access is disabled in the Binoculars local MCP runtime")

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection
    _NETWORK_GUARD_INSTALLED = True


def network_guard_probe() -> bool:
    """Return True when an attempted TCP connection is blocked by the guard."""
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    except OSError as exc:
        return "Network access is disabled" in str(exc)
    return False


configure_offline_environment()

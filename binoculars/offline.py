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
_ORIGINAL_SENDTO = socket.socket.sendto
_ORIGINAL_SENDMSG = getattr(socket.socket, "sendmsg", None)
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

    def guarded_sendto(sock: socket.socket, *args: Any, **kwargs: Any) -> int:
        if sock.family != socket.AF_UNIX:
            raise OSError("Network access is disabled in the Binoculars local MCP runtime")
        return _ORIGINAL_SENDTO(sock, *args, **kwargs)

    def guarded_sendmsg(sock: socket.socket, *args: Any, **kwargs: Any) -> int:
        if sock.family != socket.AF_UNIX:
            raise OSError("Network access is disabled in the Binoculars local MCP runtime")
        if _ORIGINAL_SENDMSG is None:
            raise OSError("socket.sendmsg is unavailable")
        return _ORIGINAL_SENDMSG(sock, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = guarded_sendto  # type: ignore[method-assign]
    if _ORIGINAL_SENDMSG is not None:
        socket.socket.sendmsg = guarded_sendmsg  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection
    _NETWORK_GUARD_INSTALLED = True


def network_guard_probe() -> bool:
    """Return True when attempted TCP and UDP egress are blocked by the guard."""
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    except OSError as exc:
        tcp_blocked = "Network access is disabled" in str(exc)
    else:
        tcp_blocked = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.sendto(b"probe", ("127.0.0.1", 9))
    except OSError as exc:
        udp_blocked = "Network access is disabled" in str(exc)
    else:
        udp_blocked = False
    return tcp_blocked and udp_blocked


configure_offline_environment()

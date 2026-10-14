"""Public package surface with lazy ML dependency loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .detector import Binoculars

__all__ = ["Binoculars"]


def __getattr__(name: str) -> Any:
    """Load the detector only when the public class is requested."""
    if name != "Binoculars":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .detector import Binoculars

    return Binoculars

"""ALMA3 foundation and diagnostic inference runtime."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime import ALMA3

__all__ = ["ALMA3", "__version__"]
__version__ = "3.0.0"


def __getattr__(name: str) -> Any:
    if name == "ALMA3":
        from .runtime import ALMA3

        return ALMA3
    raise AttributeError(name)

import sys
from typing import Any, Union, cast

if sys.version_info >= (3, 11):
    from typing import LiteralString
else:
    try:
        from typing_extensions import LiteralString
    except ImportError:
        # Fallback to str if typing_extensions is not installed.
        # This allows the code to run, but loses some static analysis depth.
        LiteralString = str  # type: ignore

__all__ = ["LiteralString", "Union", "cast", "Any"]

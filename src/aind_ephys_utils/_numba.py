"""Lazy numba import shim used by modules that need optional JIT.

Modules that JIT-compile inner kernels (``ccg``, ``spike_utils``)
import ``njit`` and ``prange`` from here instead of from ``numba``
directly so that ``aind-ephys-utils`` remains importable when the
``[numba]`` extra is not installed.  Decorated functions raise a clear
``ImportError`` at call time when numba is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover

    def njit(**kwargs: Any) -> Callable[[F], F]:
        """Stub that raises at call time when numba is missing.

        Mimics ``numba.njit``'s decorator-with-kwargs form
        (``@njit(cache=True, parallel=True)``).  Bare ``@njit`` (no
        parentheses) is not supported by this stub — none of the
        in-tree consumers use that form.
        """

        def _decorator(fn: F) -> F:
            """Return a wrapper that raises when the kernel is called."""

            def _wrapper(*a: Any, **kw: Any) -> NoReturn:
                """Raise ImportError because numba is not installed."""
                raise ImportError(
                    "numba is required for this kernel. Install "
                    "with: pip install 'aind-ephys-utils[numba]'"
                )

            return _wrapper  # type: ignore[return-value]

        return _decorator

    prange = range


__all__ = ["njit", "prange"]

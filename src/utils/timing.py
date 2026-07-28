from __future__ import annotations

from time import perf_counter
from types import TracebackType


class Timer:
    """Context manager measuring wall-clock duration in milliseconds."""

    def __init__(self) -> None:
        self._start = 0.0
        self._end: float | None = None

    def __enter__(self) -> Timer:
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._end = perf_counter()

    @property
    def elapsed_ms(self) -> int:
        end = self._end if self._end is not None else perf_counter()
        return int((end - self._start) * 1000)

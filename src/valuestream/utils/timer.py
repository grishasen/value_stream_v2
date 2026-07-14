import inspect
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from valuestream.utils.logger import get_logger

logger = get_logger(__name__)
_P = ParamSpec("_P")
_R = TypeVar("_R")


def timed(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorator that logs the execution time of a sync or async function in milliseconds."""

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
            """Time an async function call and log the elapsed duration."""
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            elapsed_ms = (end - start) * 1000
            logger.debug(f"{func.__qualname__} executed in {elapsed_ms:.2f} ms (async)")
            return result

        return cast(Callable[_P, _R], async_wrapper)

    @wraps(func)
    def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Time a synchronous function call and log the elapsed duration."""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed_ms = (end - start) * 1000
        logger.debug(f"{func.__qualname__} executed in {elapsed_ms:.2f} ms")
        return result

    return sync_wrapper

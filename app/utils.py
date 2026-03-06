"""Shared utilities for the LegacyLens application."""

import logging
import time
from typing import Callable, TypeVar

from app.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T")


def retry_on_rate_limit(
    fn: Callable[..., T],
    *args,
    exc_type: type[Exception] = Exception,
    exc_match: str | None = None,
    **kwargs,
) -> T:
    """Call *fn* with retry on rate-limit errors.

    Args:
        fn: The callable to invoke.
        exc_type: The exception class to catch (e.g. ``RateLimitError``).
        exc_match: If set, only retry when ``str(e)`` contains this substring.
                   Useful for broad exception types like ``ClientError``.
        *args, **kwargs: Forwarded to *fn*.

    Returns:
        The return value of *fn* on success.

    Raises:
        The caught exception if all retries are exhausted.
    """
    max_retries = settings.max_retries
    delay = settings.retry_delay

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except exc_type as e:
            if exc_match and exc_match not in str(e):
                raise
            if attempt < max_retries - 1:
                wait = delay * (attempt + 1)
                logger.warning(
                    "%s rate limit, retrying in %.1fs (attempt %d/%d)",
                    fn.__qualname__, wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
                continue
            raise

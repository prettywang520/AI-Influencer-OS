"""Polite pacing and backoff -- never evasion. `RateLimiter` sleeps a
randomized delay before each navigation and an escalating backoff
after a retryable failure, then hard-stops once
`max_retries_per_navigation` is exhausted. Every test injects a no-op
`sleep_fn` so the suite never actually sleeps.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from .exceptions import RetryBudgetExhaustedError


@dataclass(slots=True)
class RateLimiter:
    minimum_delay_seconds: float
    maximum_delay_seconds: float
    backoff_seconds: tuple[float, ...] = ()
    max_retries_per_navigation: int = 2
    sleep_fn: Callable[[float], None] = field(default=time.sleep)
    random_fn: Callable[[float, float], float] = field(default=random.uniform)

    def polite_delay(self) -> float:
        """Sleeps a randomized delay in [minimum_delay_seconds,
        maximum_delay_seconds] before a navigation. Returns the delay
        actually used."""
        delay = self.random_fn(self.minimum_delay_seconds, self.maximum_delay_seconds)
        self.sleep_fn(delay)
        return delay

    def backoff_delay(self, attempt: int) -> float:
        """Sleeps the configured backoff for `attempt` (0-indexed),
        clamped to the last configured value if attempt exceeds the
        list. Returns the delay actually used. Raises
        RetryBudgetExhaustedError if `attempt` is beyond
        max_retries_per_navigation -- callers should check
        retries_exhausted() before calling this."""
        if self.retries_exhausted(attempt):
            raise RetryBudgetExhaustedError(
                f"attempt {attempt} exceeds max_retries_per_navigation={self.max_retries_per_navigation}"
            )
        if not self.backoff_seconds:
            return 0.0
        index = min(attempt, len(self.backoff_seconds) - 1)
        delay = self.backoff_seconds[index]
        self.sleep_fn(delay)
        return delay

    def retries_exhausted(self, attempt: int) -> bool:
        return attempt >= self.max_retries_per_navigation

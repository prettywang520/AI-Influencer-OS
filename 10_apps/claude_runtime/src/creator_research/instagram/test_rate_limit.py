import unittest

from src.creator_research.instagram.exceptions import RetryBudgetExhaustedError
from src.creator_research.instagram.rate_limit import RateLimiter


def _no_sleep(_seconds):
    pass


def _fixed_random(low, high):
    return (low + high) / 2


class PoliteDelayTests(unittest.TestCase):
    def test_delay_within_configured_bounds(self):
        limiter = RateLimiter(
            minimum_delay_seconds=1.5, maximum_delay_seconds=4.0, sleep_fn=_no_sleep, random_fn=_fixed_random
        )
        delay = limiter.polite_delay()
        self.assertGreaterEqual(delay, 1.5)
        self.assertLessEqual(delay, 4.0)

    def test_never_actually_sleeps_when_sleep_fn_is_noop(self):
        calls = []
        limiter = RateLimiter(
            minimum_delay_seconds=1.5, maximum_delay_seconds=4.0, sleep_fn=calls.append, random_fn=_fixed_random
        )
        limiter.polite_delay()
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0], 2.75)

    def test_random_fn_receives_configured_bounds(self):
        seen = []

        def recording_random(low, high):
            seen.append((low, high))
            return low

        limiter = RateLimiter(
            minimum_delay_seconds=2.0, maximum_delay_seconds=3.0, sleep_fn=_no_sleep, random_fn=recording_random
        )
        limiter.polite_delay()
        self.assertEqual(seen, [(2.0, 3.0)])


class BackoffDelayTests(unittest.TestCase):
    def test_escalating_backoff(self):
        limiter = RateLimiter(
            minimum_delay_seconds=1.0, maximum_delay_seconds=2.0,
            backoff_seconds=(10.0, 30.0, 60.0), max_retries_per_navigation=3, sleep_fn=_no_sleep,
        )
        self.assertEqual(limiter.backoff_delay(0), 10.0)
        self.assertEqual(limiter.backoff_delay(1), 30.0)
        self.assertEqual(limiter.backoff_delay(2), 60.0)

    def test_attempt_beyond_list_clamps_to_last_value(self):
        limiter = RateLimiter(
            minimum_delay_seconds=1.0, maximum_delay_seconds=2.0,
            backoff_seconds=(10.0,), max_retries_per_navigation=5, sleep_fn=_no_sleep,
        )
        self.assertEqual(limiter.backoff_delay(3), 10.0)

    def test_empty_backoff_list_returns_zero(self):
        limiter = RateLimiter(
            minimum_delay_seconds=1.0, maximum_delay_seconds=2.0,
            backoff_seconds=(), max_retries_per_navigation=5, sleep_fn=_no_sleep,
        )
        self.assertEqual(limiter.backoff_delay(0), 0.0)

    def test_exhausted_retries_raises(self):
        limiter = RateLimiter(
            minimum_delay_seconds=1.0, maximum_delay_seconds=2.0,
            backoff_seconds=(10.0,), max_retries_per_navigation=1, sleep_fn=_no_sleep,
        )
        with self.assertRaises(RetryBudgetExhaustedError):
            limiter.backoff_delay(1)

    def test_backoff_sleeps_via_injected_sleep_fn(self):
        calls = []
        limiter = RateLimiter(
            minimum_delay_seconds=1.0, maximum_delay_seconds=2.0,
            backoff_seconds=(10.0, 30.0), max_retries_per_navigation=2, sleep_fn=calls.append,
        )
        limiter.backoff_delay(0)
        self.assertEqual(calls, [10.0])


class RetriesExhaustedTests(unittest.TestCase):
    def test_below_max_not_exhausted(self):
        limiter = RateLimiter(minimum_delay_seconds=1.0, maximum_delay_seconds=2.0, max_retries_per_navigation=2)
        self.assertFalse(limiter.retries_exhausted(0))
        self.assertFalse(limiter.retries_exhausted(1))

    def test_at_max_is_exhausted(self):
        limiter = RateLimiter(minimum_delay_seconds=1.0, maximum_delay_seconds=2.0, max_retries_per_navigation=2)
        self.assertTrue(limiter.retries_exhausted(2))

    def test_never_evades_by_retrying_past_configured_budget(self):
        # Structural guarantee: retries_exhausted() is a pure boolean
        # check with no side channel to raise the budget at runtime.
        limiter = RateLimiter(minimum_delay_seconds=1.0, maximum_delay_seconds=2.0, max_retries_per_navigation=1)
        self.assertTrue(limiter.retries_exhausted(1))
        self.assertTrue(limiter.retries_exhausted(100))


if __name__ == "__main__":
    unittest.main()

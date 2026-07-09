from llm_language_limits.ratelimit import RateLimiter


def test_no_interval_is_noop():
    rl = RateLimiter({})
    rl.acquire("x")
    rl.acquire("x")  # no error, no wait


def test_spaces_calls_by_interval():
    t = [0.0]
    slept = []

    clock = lambda: t[0]

    def sleep(s):
        slept.append(s)
        t[0] += s  # advance fake clock by the sleep

    rl = RateLimiter({"azure": 4.0}, clock=clock, sleep=sleep)
    rl.acquire("azure")  # first call: immediate (now=0 >= 0)
    rl.acquire("azure")  # must wait ~4s
    assert slept and abs(sum(slept) - 4.0) < 1e-6


def test_separate_keys_independent():
    t = [0.0]
    clock = lambda: t[0]
    sleep = lambda s: None
    rl = RateLimiter({"a": 5.0, "b": 5.0}, clock=clock, sleep=sleep)
    rl.acquire("a")
    rl.acquire("b")  # different keys don't block each other at t=0

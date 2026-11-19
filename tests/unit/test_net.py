"""The adaptive request rate (no network needed)."""
import time

from collect.net import AdaptiveThrottle, Throttle, feedback, platform_throttle


def test_rises_slowly_and_stops_at_the_cap():
    seen = []
    t = AdaptiveThrottle(2.0, cap=5.0, floor=2.0, step=0.5, every=10, on_change=lambda r, why: seen.append(r))
    for _ in range(10):
        t.ok(0.3)
    assert t.rate == 2.5                       # one step after `every` quick replies
    for _ in range(200):
        t.ok(0.3)
    assert t.rate == 5.0 and seen[-1] == 5.0   # never above the cap
    assert abs(t.gap - 0.2) < 1e-9


def test_strain_halves_below_the_floor_and_slow_replies_step_down_to_it():
    t = AdaptiveThrottle(4.0, cap=5.0, floor=2.0, min_rate=0.5, step=0.5, every=10)
    for _ in range(5):
        t.ok(0.3)
    t.ok(2.0)                                  # more than twice the usual time: one step down
    assert t.rate == 3.5
    t.strain("HTTP 429")
    assert t.rate == 1.75                      # halved, below the floor: the server is struggling
    t.strain("HTTP 503")
    t.strain("timeout")
    t.strain("timeout")
    assert t.rate == 0.5                       # never below min_rate
    for _ in range(5):
        t.ok(5.0)                              # slow replies never push the rate down past the floor rule
    assert t.rate == 0.5


def test_retry_after_pauses_the_next_request():
    t = AdaptiveThrottle(2.0, cap=2.0, floor=2.0)
    t.strain("HTTP 429", retry_after=30)
    assert t.last >= time.time() + 29          # the next wait() sleeps until then


def test_plain_throttle_ignores_feedback_and_cap_needs_approval():
    feedback(Throttle(2.0), False, why="HTTP 429")   # no error, nothing to adapt

    class C:
        def __init__(self, approved):
            self.d = {"platforms.kartaview": {"per_s": 2.0, "bulk_access_approved": approved,
                                              "adaptive": {"enabled": True, "max_per_s": 5.0}}}

        def get(self, k, default=None):
            return self.d.get(k, default)
    assert platform_throttle(C(False), "kartaview").cap == 2.0   # without approval it can only slow down
    assert platform_throttle(C(True), "kartaview").cap == 5.0
    assert type(platform_throttle(C(True), "mapillary")) is Throttle

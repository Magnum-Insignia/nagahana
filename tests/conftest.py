"""Shared test configuration.

Hypothesis property tests run without a per-example deadline. The default 200 ms deadline
measures wall-clock time, so a busy machine (or a cold import of torch) can fail a correct test
once and pass it on the next run. That happened on 2026-09-30: one run of
`test_field_value_status_invariants` failed and the same seed passed on replay. Correctness is
still checked on every generated example; only the timing check is off.
"""

from hypothesis import settings

settings.register_profile("nagahana", deadline=None)
settings.load_profile("nagahana")

"""pytest glue for the check()-style test modules.

tests/test_ingest.py counts failures in a module-level `failed` counter
instead of raising, so it can run without pytest:

    python3 -m tests.test_ingest

Under pytest that style reports every test as passed even when a check
failed. This hook compares the module's `failed` counter before and after
each test body and fails the test if it went up. Modules written with plain
`assert` have no counter and are unaffected.

Run everything:  python3 -m pytest
"""

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    module = getattr(item, "module", None)
    before = getattr(module, "failed", None)
    result = yield
    after = getattr(module, "failed", None)
    if isinstance(before, int) and isinstance(after, int) and after > before:
        pytest.fail(f"{after - before} check() call(s) failed - see the FAIL lines above",
                    pytrace=False)
    return result

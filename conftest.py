"""pytest bootstrap: make the project importable and install dependency shims
before any test imports the vendored upstream module."""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compat import install_dep_stubs  # noqa: E402

install_dep_stubs()


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_judge_cache():
    """The judge cache lives for ONE scoring run (see score_v31.RUBRIC_VERSION).

    In production that is a single ``main()`` invocation, which resets it on entry.
    Under pytest the process is shared, so without this a judgment cached by one test
    would be served to the next -- which is exactly the stale-result failure the cache
    exists to prevent, turned inward.
    """
    from score.score_v31 import reset_judge_cache
    reset_judge_cache()
    yield
    reset_judge_cache()

"""Vendored upstream AgentClinic package.

``agentclinic.py`` here is a byte-for-byte copy of the original project and must
never be edited (see the build brief invariants). This package initializer only
guarantees that importing ``upstream.agentclinic`` succeeds even when the heavy
ML dependencies are absent, by installing *fallback* import shims first. When the
real dependencies are installed they take precedence and this is a no-op.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compat import install_dep_stubs  # noqa: E402

install_dep_stubs()

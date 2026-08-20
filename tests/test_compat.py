"""The compat shims are OFFLINE-ONLY and must be inert wherever the real package exists.

``compat.install_dep_stubs()`` lets the harness import and test without the heavy ML
stack (openai / anthropic / transformers / replicate). That convenience becomes a
correctness hazard the moment a stub shadows a real, importable provider: a "live"
run would then be served by a fake, and either explode confusingly or — worse —
silently produce results that never touched a model.

The load-bearing test here is ``test_no_stub_shadows_an_importable_real_module``: it
FAILS if a stub is active while the real module is importable from disk. On a fully
provisioned machine every stub must be gone.
"""

import sys
import types
from importlib.machinery import PathFinder

import pytest

import compat
from core import backbones


def _real_module_exists(name: str) -> bool:
    """Is the real package importable from disk?

    ``PathFinder`` searches ``sys.path`` directly, deliberately bypassing
    ``sys.modules`` — otherwise an installed stub would answer for the real thing.
    """
    return PathFinder().find_spec(name) is not None


def _is_stub(name: str) -> bool:
    return getattr(sys.modules.get(name), "__stub__", False)


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------
def test_no_stub_shadows_an_importable_real_module():
    """A shim must never stand in front of a real provider package."""
    compat.install_dep_stubs()                       # idempotent; conftest already ran it
    shadowed = [name for name in compat._BUILDERS
                if _is_stub(name) and _real_module_exists(name)]
    assert not shadowed, (
        "compat stub(s) {} are shadowing packages that ARE installed. The shims are "
        "offline-only; a live run would be served by a fake.".format(shadowed))


def test_install_never_overwrites_an_already_imported_module():
    """Real installations always win — even ones imported after the first call."""
    sentinel = types.ModuleType("openai")
    sentinel.api_key = "real-module-marker"
    saved = sys.modules.get("openai")
    sys.modules["openai"] = sentinel
    try:
        compat.install_dep_stubs()
        assert sys.modules["openai"] is sentinel
        assert not getattr(sys.modules["openai"], "__stub__", False)
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            del sys.modules["openai"]


def test_active_stubs_reports_only_genuinely_stubbed_providers():
    compat.install_dep_stubs()
    for provider, module in compat._PROVIDER_MODULE.items():
        assert (provider in compat.active_stubs()) == _is_stub(module)


# ---------------------------------------------------------------------------
# A stub must be loud, never silently "successful"
# ---------------------------------------------------------------------------
def test_a_stubbed_provider_cannot_serve_a_live_run():
    """configure_providers refuses to run a model whose provider is only a shim."""
    if "openai" not in compat.active_stubs():
        pytest.skip("real openai is installed here, so there is no stub to reject")
    with pytest.raises(backbones.StubbedProvider, match="offline-only"):
        backbones.configure_providers(["gpt4o"], env={"OPENAI_API_KEY": "sk-test"})


def test_stub_network_entry_points_raise_instead_of_faking_a_result():
    if not _is_stub("openai"):
        pytest.skip("real openai is installed here")
    with pytest.raises(RuntimeError, match="stub active"):
        sys.modules["openai"].ChatCompletion.create(model="gpt-4o", messages=[])

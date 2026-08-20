"""Dependency fallback shims.

The vendored ``upstream/agentclinic.py`` imports ``openai``, ``anthropic``,
``transformers`` and ``replicate`` at module load. Those are only exercised for
*live* API runs; every code path in this harness routes LLM calls through
``query_model``, which the tests monkeypatch and the mock replaces entirely.

``install_dep_stubs`` injects a minimal stand-in into ``sys.modules`` *only* for
a dependency that is not already importable. Real installations always win (we
never overwrite a module that imports successfully), so this is a no-op in a
fully provisioned environment and merely lets the harness import + test without
the heavy ML stack. Any stub whose network entry point is actually called raises
a clear, actionable error instead of failing obscurely.
"""

import builtins
import sys
import types

_UTF8_OPEN_INSTALLED = False


def _ensure_utf8_open():
    """Default text-mode ``open()`` to UTF-8 when no encoding is given.

    The vendored upstream loaders call ``open(path, "r")`` with no encoding, so on
    Windows they would decode the UTF-8 scenario ``.jsonl`` files with the locale
    codepage (cp1252/cp949) and crash. We cannot edit upstream, so we make the
    process default to UTF-8 for text opens that don't specify an encoding — this
    is exactly the behavior of ``PYTHONUTF8=1`` / PEP 686, and it never overrides
    a caller that passed an explicit encoding or opened in binary mode.
    """
    global _UTF8_OPEN_INSTALLED
    if _UTF8_OPEN_INSTALLED:
        return
    _real_open = builtins.open

    def _utf8_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if encoding is None and "b" not in mode:
            encoding = "utf-8"
        return _real_open(file, mode, buffering, encoding, *args, **kwargs)

    builtins.open = _utf8_open
    _UTF8_OPEN_INSTALLED = True


def _stub_missing(name, build):
    """Install ``build()`` as ``sys.modules[name]`` iff the real module is absent."""
    if name in sys.modules:
        return False
    try:
        __import__(name)
        return False
    except Exception:
        module = build()
        module.__stub__ = True
        sys.modules[name] = module
        return True


def _build_openai():
    mod = types.ModuleType("openai")

    class ChatCompletion:
        @staticmethod
        def create(*args, **kwargs):
            raise RuntimeError(
                "openai stub active: the real 'openai' package is not installed, "
                "so live API calls are unavailable. Install 'openai==0.28.0' for "
                "real runs. (Tests/mocks never reach this path.)"
            )

    mod.ChatCompletion = ChatCompletion
    mod.api_key = None
    return mod


def _build_anthropic():
    mod = types.ModuleType("anthropic")

    class Anthropic:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "anthropic stub active: install the real 'anthropic' package for "
                "Claude backbones. (Tests/mocks never reach this path.)"
            )

    mod.Anthropic = Anthropic
    return mod


def _build_transformers():
    mod = types.ModuleType("transformers")

    def pipeline(*args, **kwargs):
        raise RuntimeError(
            "transformers stub active: install the real 'transformers' package "
            "for HuggingFace backbones. (Tests/mocks never reach this path.)"
        )

    mod.pipeline = pipeline
    return mod


def _build_replicate():
    mod = types.ModuleType("replicate")

    def run(*args, **kwargs):
        raise RuntimeError(
            "replicate stub active: install the real 'replicate' package for "
            "Replicate-hosted backbones. (Tests/mocks never reach this path.)"
        )

    mod.run = run
    return mod


_BUILDERS = {
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "transformers": _build_transformers,
    "replicate": _build_replicate,
}


def install_dep_stubs():
    """Fill in fallback shims for any missing upstream dependency.

    Returns the sorted list of module names that were stubbed (empty when the
    real dependencies are all present). Also ensures UTF-8 default text opens.
    """
    _ensure_utf8_open()
    stubbed = [name for name, build in _BUILDERS.items() if _stub_missing(name, build)]
    return sorted(stubbed)


# provider name (as used by core.backbones) -> the module its live path imports
_PROVIDER_MODULE = {
    "openai": "openai",
    "anthropic": "anthropic",
    "replicate": "replicate",
    "huggingface": "transformers",
}


def active_stubs():
    """Providers currently served by a shim rather than the real package.

    A live run must never touch one of these — ``core.backbones.configure_providers``
    refuses to proceed when a provider it needs appears here. In a fully provisioned
    environment this is always empty.
    """
    return sorted(
        provider for provider, module in _PROVIDER_MODULE.items()
        if getattr(sys.modules.get(module), "__stub__", False)
    )

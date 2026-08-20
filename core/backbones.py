"""Backbone (model-string) routing and provider-key resolution.

Every LLM call goes through the vendored upstream ``query_model``, which
dispatches on a model string (``"gpt4o"``, ``"claude3.5sonnet"``, ...). This
module is the glue around that:

* ``SUPPORTED_MODELS`` mirrors upstream's whitelist so a bad model string is
  caught at config time rather than after N scenarios. Upstream stays the source
  of truth -- an unknown string only warns, dispatch is never re-implemented.
* ``provider_of`` maps a model string to its provider, and hence to the
  environment variable holding its key.
* ``configure_providers`` reads those keys **from the environment only**; no key
  is read from or written into source. It makes no network calls.

Offline runs monkeypatch ``query_model`` and need no keys at all.

Mistral is the one extension beyond upstream's dispatch. Upstream does not know
those model strings and must stay byte-for-byte unmodified, so
``configure_providers`` installs a runtime wrapper that peels off Mistral strings
and forwards everything else to the original function. It is installed only for
live runs that actually use a Mistral model; the golden path never sees it.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from typing import Dict, Iterable, List, Optional

# The two Mistral API models the harness routes itself — exactly the ids available
# on the account (verified against api.mistral.ai's model list, not guessed):
# mistral-medium-2505 (backbone/doctor-capable) and mistral-small-2506 (judge).
MISTRAL_MODELS = (
    "mistral-medium-2505",
    "mistral-small-2506",
)

# Mirrors the whitelist in upstream.agentclinic.query_model (plus its "_HF" suffix
# rule), extended with the harness-routed MISTRAL_MODELS, which never reach the
# upstream dispatch. Kept as a flat tuple for config-time validation only.
SUPPORTED_MODELS = (
    "gpt4",
    "gpt3.5",
    "gpt4o",
    "gpt-4o-mini",
    "gpt4v",
    "o1-preview",
    "claude3.5sonnet",
    "llama-2-70b-chat",
    "llama-3-70b-instruct",
    "mixtral-8x7b",
) + MISTRAL_MODELS

# provider -> environment variable that must hold its key
PROVIDER_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "mistral": "MISTRAL_API_KEY",
    "huggingface": None,          # local weights; no key
}

_PROVIDER_OF = {
    "gpt4": "openai",
    "gpt3.5": "openai",
    "gpt4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt4v": "openai",
    "o1-preview": "openai",
    "claude3.5sonnet": "anthropic",
    "llama-2-70b-chat": "replicate",
    "llama-3-70b-instruct": "replicate",
    "mixtral-8x7b": "replicate",          # NOT the Mistral API: served via Replicate
    "mistral-medium-2505": "mistral",
    "mistral-small-2506": "mistral",
}


class MissingProviderKey(RuntimeError):
    """A live run needs a provider key that is not present in the environment."""


class StubbedProvider(RuntimeError):
    """A live run needs a provider whose real package is not installed (see compat.py)."""


def is_supported(model: str) -> bool:
    """True if upstream ``query_model`` will accept this model string."""
    return bool(model) and (model in SUPPORTED_MODELS or "_HF" in model)


def model_family(model_str: str) -> str:
    """Provider family of a model string, by prefix.

    Panickssery et al. (NeurIPS 2024): self-preference bias applies across a model
    FAMILY, not only when two model strings are identical. This is what the
    same-family judge/moderator guards key on; anything unrecognized is "unknown"
    (never warned on, since we cannot tell whether it is a sibling). Prefix-based on
    purpose, so an unseen sibling (e.g. a new ``gpt``/``claude`` string) still maps
    to its family without touching the whitelist above.
    """
    s = (model_str or "").strip().lower()
    if s.startswith("gpt"):
        return "openai"
    if s.startswith("claude"):
        return "anthropic"
    if s.startswith("mistral"):
        return "mistral"
    return "unknown"


def provider_of(model: str) -> Optional[str]:
    """Provider serving ``model`` (``None`` if the model string is unknown)."""
    if model and "_HF" in model:
        return "huggingface"
    return _PROVIDER_OF.get(model)


def providers_for(models: Iterable[str]) -> List[str]:
    """Distinct providers needed to serve ``models``, in a stable order."""
    out: List[str] = []
    for m in models:
        p = provider_of(m)
        if p is not None and p not in out:
            out.append(p)
    return out


def missing_keys(models: Iterable[str], env: Optional[Dict[str, str]] = None) -> List[str]:
    """Env vars that ``models`` require but that are absent/empty in ``env``."""
    env = os.environ if env is None else env
    out = []
    for provider in providers_for(models):
        var = PROVIDER_ENV_KEY.get(provider)
        if var and not env.get(var):
            out.append(var)
    return out


# CLI flag -> environment variable. Passing a key on the command line keeps it out of
# source and out of config JSON: the flag is copied into the environment and every
# consumer reads the environment, so no key is ever written down in the repo.
KEY_ARGS = {
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "replicate_api_key": "REPLICATE_API_TOKEN",
    "mistral_api_key": "MISTRAL_API_KEY",
}


def add_provider_key_args(ap) -> None:
    """Add ``--openai_api_key`` / ``--anthropic_api_key`` / ``--replicate_api_key``
    / ``--mistral_api_key``.

    Upstream ``main()`` takes its keys the same way. The alternative is to export the
    matching environment variable directly; either way the key stays out of the code.
    """
    ap.add_argument("--openai_api_key", type=str, default=None,
                    help="OpenAI key (gpt4o, gpt4, gpt3.5, ...); "
                         "or export OPENAI_API_KEY instead")
    ap.add_argument("--anthropic_api_key", type=str, default=None,
                    help="Anthropic key (claude3.5sonnet); or export ANTHROPIC_API_KEY")
    ap.add_argument("--replicate_api_key", type=str, default=None,
                    help="Replicate token (mixtral-8x7b, llama-3-70b-instruct); "
                         "or export REPLICATE_API_TOKEN")
    ap.add_argument("--mistral_api_key", type=str, default=None,
                    help="Mistral key (mistral-medium-2505, mistral-small-2506); "
                         "or export MISTRAL_API_KEY")


def apply_provider_key_args(args, env: Optional[Dict[str, str]] = None) -> List[str]:
    """Copy any key given on the command line into the environment.

    This is the ONLY place a CLI-supplied key is handled: from here on every consumer
    (including ``configure_providers`` below and upstream ``query_model``) reads the
    environment, so there is exactly one path and no key literal in source. A flag
    overrides an already-exported variable. Returns the env vars that were set.
    """
    env = os.environ if env is None else env
    applied = []
    for attr, var in KEY_ARGS.items():
        value = getattr(args, attr, None)
        if value:
            env[var] = value
            applied.append(var)
    return applied


# =============================================================================
# Mistral (api.mistral.ai) — the one provider upstream query_model does not know.
#
# Upstream must stay byte-for-byte unmodified, so Mistral model strings are peeled
# off BEFORE the upstream dispatch by a runtime wrapper over
# ``upstream.agentclinic.query_model`` (installed by configure_providers only when
# a live run actually uses a Mistral model). The endpoint is OpenAI-compatible
# chat completions; the key is read from MISTRAL_API_KEY at call time and is never
# logged, printed, or embedded in an error message.
# =============================================================================

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def _mistral_transport():
    """Pick the HTTP transport for the Mistral endpoint: ``(kind, module)``.

    Prefers the already-installed ``openai`` SDK pointed at a different base_url +
    api_key (legacy 0.28 takes both as per-call kwargs; >=1.0 as client kwargs), so
    no new HTTP client is written. Falls back to a minimal ``requests`` POST, and
    returns ``(None, None)`` when neither is available. A compat.py shim of
    ``openai`` (``__stub__``) is offline-only and never counts as a transport.
    """
    try:
        import openai as mod
    except Exception:
        mod = None
    if mod is not None and not getattr(mod, "__stub__", False):
        if hasattr(mod, "ChatCompletion"):            # legacy SDK (0.x)
            return "openai_legacy", mod
        if hasattr(mod, "OpenAI"):                    # modern SDK (>=1.0)
            return "openai_v1", mod
    try:
        import requests as req
    except Exception:
        return None, None
    return "requests", req


def mistral_chat(model_id: str, prompt: str, system_prompt: str,
                 max_tokens: int, temperature: float) -> str:
    """One OpenAI-compatible chat completion against ``MISTRAL_BASE_URL``.

    The key comes from the environment only, is passed per-call (the global
    ``openai.api_key`` — the OpenAI key — is never touched), and never appears in
    any raised message. Raises ``MissingProviderKey`` / ``StubbedProvider`` for the
    two non-retryable setup problems; anything else is a transport error the
    caller may retry.
    """
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise MissingProviderKey(
            "MISTRAL_API_KEY is not set. Pass --mistral_api_key on the command "
            "line, or export the variable. The key is never read from code or "
            "config, and never logged.")
    kind, mod = _mistral_transport()
    if kind is None:
        raise StubbedProvider(
            "No transport for the Mistral endpoint: neither the real 'openai' "
            "package nor 'requests' is installed. Install either one.")
    messages = [{"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt}]
    if kind == "openai_legacy":
        resp = mod.ChatCompletion.create(
            model=model_id, messages=messages, temperature=temperature,
            max_tokens=max_tokens, api_base=MISTRAL_BASE_URL, api_key=key)
        return resp["choices"][0]["message"]["content"]
    if kind == "openai_v1":
        client = mod.OpenAI(base_url=MISTRAL_BASE_URL, api_key=key)
        resp = client.chat.completions.create(
            model=model_id, messages=messages, temperature=temperature,
            max_tokens=max_tokens)
        return resp.choices[0].message.content
    resp = mod.post(
        MISTRAL_BASE_URL + "/chat/completions",
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
        json={"model": model_id, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def mistral_query_model(model_str, prompt, system_prompt, tries=30, timeout=20.0,
                        image_requested=False, scene=None, max_prompt_len=2**14,
                        clip_prompt=False):
    """Drop-in for upstream ``query_model``, serving only Mistral model strings.

    Deliberately mirrors the upstream agent path so a Mistral backbone behaves
    like every other backbone: same signature, ``temperature=0.05``,
    ``max_tokens=200`` (the agent cap — the scorer's judge does NOT go through
    here; it has its own uncapped caller, see ``score_v31.default_judge``), the
    same whitespace collapse, and the same retry loop. Setup problems (missing
    key, no transport) raise immediately instead of burning the retry budget.
    """
    for _ in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            answer = mistral_chat(model_str, prompt, system_prompt,
                                  max_tokens=200, temperature=0.05)
            return re.sub(r"\s+", " ", answer)
        except (MissingProviderKey, StubbedProvider):
            raise
        except Exception:
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


def install_mistral_route():
    """Wrap ``upstream.agentclinic.query_model`` so Mistral strings route here.

    Every other model string is forwarded, argument-for-argument, to the original
    (still byte-for-byte unmodified) upstream function, so the golden path is
    unchanged. Idempotent: installing twice does not double-wrap. Returns the
    installed wrapper.
    """
    import upstream.agentclinic as ac
    if getattr(ac.query_model, "__mistral_route__", False):
        return ac.query_model
    orig = ac.query_model

    def query_model_with_mistral_route(model_str, prompt, system_prompt,
                                       *args, **kwargs):
        if provider_of(model_str) == "mistral":
            return mistral_query_model(model_str, prompt, system_prompt,
                                       *args, **kwargs)
        return orig(model_str, prompt, system_prompt, *args, **kwargs)

    query_model_with_mistral_route.__mistral_route__ = True
    query_model_with_mistral_route.__wrapped__ = orig
    ac.query_model = query_model_with_mistral_route
    return query_model_with_mistral_route


def uninstall_mistral_route() -> bool:
    """Restore the original upstream ``query_model`` (used by tests). True if removed."""
    import upstream.agentclinic as ac
    if getattr(ac.query_model, "__mistral_route__", False):
        ac.query_model = ac.query_model.__wrapped__
        return True
    return False


# ------------------------------------------------------- budgeted callers
# Upstream ``query_model`` hardcodes ``max_tokens=200`` on every provider branch (and
# so does ``mistral_query_model`` above, deliberately, to mirror it). That is the right
# cap for an agent turn and the wrong one for any component that must emit a structured
# object: at 200 tokens the response is cut off mid-object and there is nothing to
# parse. ``score_v31.default_judge`` already solves this for the judge with its own
# per-provider callers; this is the same thing, factored so a non-judge component (the
# kernel) can have its own budget without importing the scorer -- the scorer
# grades the system, the kernel is part of it, and that separation is worth keeping.
#
# ``model string -> provider model id``, mirroring upstream's dispatch so a budgeted
# caller names the same model upstream would have named for that string.
_OPENAI_IDS = {
    "gpt4o": "gpt-4o",
    "gpt4": "gpt-4-turbo-preview",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt3.5": "gpt-3.5-turbo",
}
_ANTHROPIC_IDS = {
    "claude3.5sonnet": "claude-3-5-sonnet-20240620",
}
# Mistral is harness-routed, so the model string IS the API model id.
_MISTRAL_IDS = {m: m for m in MISTRAL_MODELS}


def budgeted_query(backbone: str, max_tokens: int, temperature: float = 0.05):
    """A ``query_model``-shaped callable for ``backbone`` at an explicit token budget.

    The signature is ``(model_str, prompt, system_prompt) -> str``, identical to
    upstream ``query_model``, so it drops into any call site that already takes a
    ``query``. It differs in exactly one way: ``max_tokens`` is the caller's, not the
    hardcoded 200.

    ``temperature`` defaults to 0.05, which is what the OpenAI branches of
    ``query_model`` hardcode and what ``mistral_query_model`` mirrors, so callers
    wanting to match an agent turn get it without asking. **There is no single
    "deployed temperature"** -- the Anthropic, o1-preview and Replicate branches set
    none at all, so an arm that swaps in a second backbone is also changing the
    sampling condition, and that has to be stated rather than assumed away.

    Providers with no direct caller here (Replicate, HuggingFace) fall back to
    ``query_model`` and are warned about: they stay capped at 200 tokens and at
    whatever temperature that branch hardcodes, so an explicit ``temperature`` is
    silently ignored on those paths.
    """
    if max_tokens is None or int(max_tokens) <= 0:
        raise ValueError("max_tokens must be a positive int, got {!r}".format(max_tokens))
    max_tokens = int(max_tokens)

    if backbone in _OPENAI_IDS:
        model_id = _OPENAI_IDS[backbone]

        def call_openai(model_str, prompt, system_prompt=None, *args, **kwargs):
            import openai
            messages = [{"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": prompt}]
            if hasattr(openai, "OpenAI"):                       # modern SDK (>=1.0)
                resp = openai.OpenAI().chat.completions.create(
                    model=model_id, messages=messages,
                    temperature=temperature, max_tokens=max_tokens)
                return resp.choices[0].message.content
            resp = openai.ChatCompletion.create(                # legacy SDK
                model=model_id, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            return resp["choices"][0]["message"]["content"]
        return call_openai

    if backbone in _ANTHROPIC_IDS:
        model_id = _ANTHROPIC_IDS[backbone]

        def call_anthropic(model_str, prompt, system_prompt=None, *args, **kwargs):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            message = client.messages.create(
                model=model_id, system=system_prompt or "", max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
            return message.content[0].text
        return call_anthropic

    if backbone in _MISTRAL_IDS:
        model_id = _MISTRAL_IDS[backbone]

        def call_mistral(model_str, prompt, system_prompt=None, *args, **kwargs):
            return mistral_chat(model_id, prompt, system_prompt or "",
                                max_tokens=max_tokens, temperature=temperature)
        return call_mistral

    warnings.warn(
        "backbone '{}' has no direct budgeted caller, so it falls back to upstream "
        "query_model, which caps completions at 200 tokens. A structured response "
        "will be truncated mid-object and fail to parse. Use an OpenAI ({}), "
        "Anthropic ({}) or Mistral ({}) backbone instead.".format(
            backbone, "/".join(sorted(_OPENAI_IDS)), "/".join(sorted(_ANTHROPIC_IDS)),
            "/".join(sorted(_MISTRAL_IDS))), UserWarning)

    def call_upstream(model_str, prompt, system_prompt=None, *args, **kwargs):
        import upstream.agentclinic as ac
        return ac.query_model(model_str, prompt, system_prompt)
    return call_upstream


def configure_providers(models: Iterable[str], env: Optional[Dict[str, str]] = None) -> dict:
    """Prepare provider SDKs for a **live** run of ``models``. No network calls.

    Keys are read from the environment (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``
    / ``REPLICATE_API_TOKEN`` / ``MISTRAL_API_KEY``) and never from source or
    config files. Raises
    ``MissingProviderKey`` if a needed key is absent, and ``StubbedProvider`` if the
    provider's real package is not installed (i.e. ``compat.py`` shimmed it — those
    shims are offline-only and must never serve a live run).

    ``anthropic`` and ``replicate`` read their key out of ``os.environ`` themselves
    inside ``query_model``; only the legacy ``openai`` SDK needs the key assigned
    onto the module, which is what this does.
    """
    env = os.environ if env is None else env
    models = [m for m in models if m]

    unknown = [m for m in models if not is_supported(m)]
    if unknown:
        raise ValueError(
            "Unsupported model string(s) {}: upstream query_model accepts {} "
            "(or any '*_HF' name).".format(sorted(set(unknown)), list(SUPPORTED_MODELS))
        )

    needed = providers_for(models)

    from compat import active_stubs  # local import: keeps this module import-light
    stubbed = [p for p in needed if p in active_stubs()]
    if stubbed:
        raise StubbedProvider(
            "Provider package(s) {} are not installed, so compat.py installed an "
            "offline-only shim for them. A live run cannot use a shim. Install the "
            "real package(s) before running models {}.".format(sorted(stubbed), models)
        )

    absent = missing_keys(models, env)
    if absent:
        flags = sorted({"--{}".format(a) for a, var in KEY_ARGS.items() if var in absent})
        raise MissingProviderKey(
            "Missing key(s) {} required by model(s) {}. Pass {} on the command line, or "
            "export the variable(s). Keys are never read from code or config.".format(
                sorted(absent), models, " / ".join(flags))
        )

    if "openai" in needed:
        import openai
        openai.api_key = env["OPENAI_API_KEY"]

    if "mistral" in needed:
        # Fail fast (like the shim check above) if there is nothing to carry the
        # HTTP call, then route Mistral strings around the unmodified upstream
        # dispatch. Installed only here, so offline/golden runs never see it.
        if _mistral_transport()[0] is None:
            raise StubbedProvider(
                "The Mistral provider needs the real 'openai' package or "
                "'requests' installed to reach {}; neither is available.".format(
                    MISTRAL_BASE_URL))
        install_mistral_route()

    return {"providers": needed, "models": models}

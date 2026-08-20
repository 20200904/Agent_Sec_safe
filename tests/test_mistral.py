"""Mistral provider (api.mistral.ai): resolution, routing, key handling, judge budget.

Entirely offline. The route wrapper, the chat transport and the judge caller are all
monkeypatched or fed fakes, so no key is needed and nothing hits the network. The
invariants under test:

* ``mistral-medium-2505`` / ``mistral-small-2506`` resolve (``--list_models``,
  ``is_supported``, provider mapping) and select the Mistral route, while upstream
  ``query_model`` stays byte-for-byte unmodified (the route is a runtime wrapper
  installed only for live runs that use a Mistral model).
* ``--mistral_api_key`` lands in ``MISTRAL_API_KEY`` and the key value is never
  echoed anywhere.
* A Mistral ``judge_llm`` goes through the scorer's own non-capped caller
  (``JUDGE_MAX_TOKENS`` = 1100, temperature 0), never the 200-token agent path.
"""

import argparse
import os
import types
import warnings

import pytest

import runner
import upstream.agentclinic as ac
from core import backbones
from core.backbones import (
    MISTRAL_BASE_URL,
    MISTRAL_MODELS,
    MissingProviderKey,
    add_provider_key_args,
    apply_provider_key_args,
    install_mistral_route,
    uninstall_mistral_route,
)
from core.config import RunConfig
from core.loader import load_scenarios
from core.orchestrator import Orchestrator
from runner import build_registry
from tests.mockllm import MockLLM

# A recognizable placeholder that must never appear in any output. (Deliberately not
# shaped like a real provider secret: test_no_api_key_literal_is_committed_anywhere
# scans this tree for key-shaped literals.)
FAKE_KEY = "mistral-test-key-never-echoed-XYZ"
FAKE_OPENAI_KEY = "openai-test-key-never-echoed-XYZ"

_HARM_VERDICT = ('{"evidence_quotes": [], "reasoning_steps": ["mock"], '
                 '"harm_gate": "harmed", "ncc_merp": "E", "direction": "under"}')


@pytest.fixture(autouse=True)
def _no_route_leak():
    """Whatever a test does, the unwrapped upstream ``query_model`` comes back."""
    yield
    uninstall_mistral_route()


# ===========================================================================
# resolution: the model strings are accepted and mapped to the mistral provider
# ===========================================================================
def test_mistral_models_resolve():
    for model in ("mistral-medium-2505", "mistral-small-2506"):
        assert model in MISTRAL_MODELS
        assert model in backbones.SUPPORTED_MODELS
        assert backbones.is_supported(model)
        assert backbones.provider_of(model) == "mistral"
    assert backbones.PROVIDER_ENV_KEY["mistral"] == "MISTRAL_API_KEY"
    # EXACTLY the two account model ids — no guessed "-latest" style aliases
    assert MISTRAL_MODELS == ("mistral-medium-2505", "mistral-small-2506")
    for guess in ("mistral-large-latest", "mistral-small-latest",
                  "mistral-large", "mistral-medium", "mistral-small"):
        assert not backbones.is_supported(guess)
        assert backbones.provider_of(guess) is None
    # mixtral-8x7b is a DIFFERENT model served via Replicate; it must not have moved
    assert backbones.provider_of("mixtral-8x7b") == "replicate"

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any config warning fails the test
        cfg = RunConfig(doctor_llm="mistral-medium-2505", judge_llm="gpt4o")
    assert cfg.config_warnings() == []


def test_list_models_shows_exactly_the_two_mistral_strings(capsys):
    runner.main(["--list_models"])
    printed = capsys.readouterr().out
    assert "mistral-medium-2505" in printed
    assert "mistral-small-2506" in printed
    assert "--mistral_api_key" in printed
    assert "mistral-large" not in printed              # removed guesses stay removed
    assert "latest" not in printed
    # the mistral provider section lists exactly the two account models
    mistral_block = printed.split("mistral    ")[1].split("replicate")[0]
    listed = [ln.strip() for ln in mistral_block.splitlines()
              if ln.strip().startswith("mistral-")]
    assert listed == ["mistral-medium-2505", "mistral-small-2506"]


# ===========================================================================
# routing: mistral strings peel off to the harness route; upstream is untouched
# ===========================================================================
def test_route_is_not_installed_on_the_clean_path():
    """Importing the harness must not wrap upstream: only a live mistral run does."""
    assert not getattr(ac.query_model, "__mistral_route__", False)


def test_mistral_strings_select_the_mistral_route(monkeypatch):
    upstream_calls, mistral_calls = [], []

    def upstream_stub(model_str, prompt, system_prompt, *a, **k):
        upstream_calls.append(model_str)
        return "upstream"

    def fake_mistral(model_str, prompt, system_prompt, *a, **k):
        mistral_calls.append(model_str)
        return "mistral"

    monkeypatch.setattr(ac, "query_model", upstream_stub)
    monkeypatch.setattr(backbones, "mistral_query_model", fake_mistral)

    wrapper = install_mistral_route()
    assert ac.query_model is wrapper
    assert ac.query_model("mistral-medium-2505", "p", "s") == "mistral"
    assert ac.query_model("mistral-small-2506", "p", "s") == "mistral"
    # every non-mistral string is forwarded, untouched, to the wrapped original
    assert ac.query_model("gpt4o", "p", "s") == "upstream"
    assert upstream_calls == ["gpt4o"]
    assert mistral_calls == ["mistral-medium-2505", "mistral-small-2506"]

    # idempotent: a second install must not double-wrap
    assert install_mistral_route() is wrapper
    assert wrapper.__wrapped__ is upstream_stub
    assert uninstall_mistral_route()
    assert ac.query_model is upstream_stub


def test_configure_providers_installs_the_route_for_a_live_mistral_run(monkeypatch):
    monkeypatch.setattr(backbones, "_mistral_transport", lambda: ("requests", object()))
    out = backbones.configure_providers(["mistral-medium-2505"],
                                        env={"MISTRAL_API_KEY": FAKE_KEY})
    assert out["providers"] == ["mistral"]
    assert getattr(ac.query_model, "__mistral_route__", False)


def test_configure_providers_without_the_key_names_the_flag(monkeypatch):
    monkeypatch.setattr(backbones, "_mistral_transport", lambda: ("requests", object()))
    with pytest.raises(MissingProviderKey, match=r"--mistral_api_key"):
        backbones.configure_providers(["mistral-medium-2505"], env={})
    # a failed configure must not have wrapped upstream
    assert not getattr(ac.query_model, "__mistral_route__", False)


def test_mistral_doctor_reaches_query_model_with_its_string(monkeypatch):
    """End-to-end through the orchestrator: the configured string is what arrives."""
    cfg = RunConfig(dataset="MedQA", n_scenarios=1, total_inferences=4,
                    backbone="gpt3.5", doctor_llm="mistral-medium-2505",
                    judge_llm="gpt4o", elicit_management=True)
    mock = MockLLM(scripts={"doctor": ["Doctor: Tell me more.",
                                       "Doctor: DIAGNOSIS READY: Something"]})
    with monkeypatch.context() as m:
        m.setattr(ac, "query_model", mock)
        scenarios, indices = load_scenarios(cfg)
        Orchestrator(scenarios[0], cfg, build_registry(cfg), lambda ev: None,
                     scenario_id=indices[0]).run()
    assert set(mock.models_for("doctor")) == {"mistral-medium-2505"}
    assert set(mock.models_for("patient")) == {"gpt3.5"}


# ===========================================================================
# transport: OpenAI-compatible chat completions against the Mistral base_url
# ===========================================================================
def test_mistral_chat_reuses_the_openai_client_with_the_mistral_base_url(monkeypatch):
    captured = {}
    fake = types.ModuleType("openai")

    class ChatCompletion:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    fake.ChatCompletion = ChatCompletion
    monkeypatch.setattr(backbones, "_mistral_transport", lambda: ("openai_legacy", fake))
    monkeypatch.setenv("MISTRAL_API_KEY", FAKE_KEY)

    out = backbones.mistral_chat("mistral-small-2506", "p", "s",
                                 max_tokens=200, temperature=0.05)
    assert out == "ok"
    assert captured["model"] == "mistral-small-2506"
    assert captured["api_base"] == MISTRAL_BASE_URL          # not the OpenAI endpoint
    assert captured["api_key"] == FAKE_KEY                   # per-call, from the env
    # the per-call key must not have leaked onto the global openai module (which
    # holds the OPENAI key for every other backbone)
    import sys
    assert getattr(sys.modules.get("openai"), "api_key", None) != FAKE_KEY


def test_agent_path_keeps_the_upstream_200_token_cap(monkeypatch):
    """A mistral AGENT behaves like every other backbone: 200 tokens, temp 0.05."""
    captured = {}

    def fake_chat(model_id, prompt, system_prompt, max_tokens, temperature):
        captured.update(max_tokens=max_tokens, temperature=temperature)
        return "answer   with\nwhitespace"

    monkeypatch.setattr(backbones, "mistral_chat", fake_chat)
    out = backbones.mistral_query_model("mistral-medium-2505", "p", "s")
    assert captured["max_tokens"] == 200
    assert captured["temperature"] == 0.05
    assert out == "answer with whitespace"       # upstream's whitespace collapse


# ===========================================================================
# key handling: --mistral_api_key -> MISTRAL_API_KEY, and the value is never echoed
# ===========================================================================
def test_key_flag_lands_in_the_environment_and_nowhere_else():
    ap = argparse.ArgumentParser()
    add_provider_key_args(ap)
    args = ap.parse_args(["--mistral_api_key", FAKE_KEY])

    env = {}
    applied = apply_provider_key_args(args, env)
    assert applied == ["MISTRAL_API_KEY"]                    # var NAMES only, no value
    assert env == {"MISTRAL_API_KEY": FAKE_KEY}
    assert backbones.missing_keys(MISTRAL_MODELS, env) == []
    assert backbones.missing_keys(MISTRAL_MODELS, {}) == ["MISTRAL_API_KEY"]


def test_mistral_chat_without_key_fails_without_echoing_anything(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(MissingProviderKey, match=r"--mistral_api_key"):
        backbones.mistral_chat("mistral-medium-2505", "p", "s", 200, 0.05)


def test_dry_run_resolves_mistral_and_never_echoes_the_key(monkeypatch, capsys):
    # setenv first so pytest restores the variables after the CLI overwrites them
    monkeypatch.setenv("MISTRAL_API_KEY", "sentinel-before-flag")
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-before-flag")
    # A mistral doctor now gets a cross-family moderator (gpt4o), so this run also
    # needs an OpenAI key -- that IS the defect-#1 fix (the accuracy axis is not graded
    # by the doctor's own family). Supply both keys via flags; neither may be echoed.
    out = runner.main([
        "--dataset", "MedQA", "--backbone", "mistral-medium-2505",
        "--judge_llm", "gpt4o", "--n_scenarios", "1",
        "--mistral_api_key", FAKE_KEY, "--openai_api_key", FAKE_OPENAI_KEY, "--dry_run",
    ])
    printed = capsys.readouterr()
    text = printed.out + printed.err

    assert os.environ["MISTRAL_API_KEY"] == FAKE_KEY         # flag -> env, the only path
    assert FAKE_KEY not in text                              # value never printed
    assert FAKE_OPENAI_KEY not in text                       # nor the moderator's key
    assert "MISTRAL_API_KEY SET" in text                     # only the var name + status
    assert out["unknown_models"] == []                       # the strings resolve
    assert out["missing_keys"] == []
    assert "OK: every model resolves" in text
    # a --dry_run makes no call and must not have wrapped upstream either
    assert not getattr(ac.query_model, "__mistral_route__", False)


# ===========================================================================
# judge: a mistral judge_llm takes the 1100-token judge caller, not the agent path
# ===========================================================================
def test_mistral_judge_uses_the_uncapped_judge_caller(monkeypatch):
    import score.score_v31 as sv

    captured = {}

    def fake_chat(model_id, prompt, system_prompt, max_tokens, temperature):
        captured.update(model=model_id, max_tokens=max_tokens, temperature=temperature)
        return _HARM_VERDICT

    monkeypatch.setattr(backbones, "mistral_chat", fake_chat)
    monkeypatch.setattr(backbones, "_mistral_transport", lambda: ("requests", object()))
    monkeypatch.setenv("MISTRAL_API_KEY", FAKE_KEY)

    with warnings.catch_warnings():
        warnings.simplefilter("error")     # the capped-fallback warning must NOT fire
        judge = sv.default_judge("mistral-medium-2505")
    assert judge is not ac.query_model                       # not the agent path

    verdict = sv.judge_harm("dx", "true dx", "ctx", None, judge, "mistral-medium-2505")
    assert verdict["ncc_merp"] == "E"                        # a full verdict parses
    assert captured["model"] == "mistral-medium-2505"
    assert captured["max_tokens"] == sv.JUDGE_MAX_TOKENS == 1100
    assert captured["max_tokens"] > 200                      # more than the agent cap
    assert captured["temperature"] == 0                      # adjudicator: reproducible


def test_mistral_judge_equal_to_mistral_doctor_still_warns():
    with pytest.warns(UserWarning, match="self-preference"):
        cfg = RunConfig(doctor_llm="mistral-medium-2505",
                        judge_llm="mistral-medium-2505")
    assert any("judge_llm == doctor" in w for w in cfg.config_warnings())

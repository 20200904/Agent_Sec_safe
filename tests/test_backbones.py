"""Multi-backbone routing (Change 1).

Every role's configured model string must be the one that actually reaches
``query_model``. These tests are offline: ``query_model`` is monkeypatched to a
MockLLM that records the ``model_str`` of each call, so no provider key is needed
and no network call is made.
"""

import argparse
import os
import re
import warnings

import pytest

import compat
import upstream.agentclinic as ac
from core import backbones
from core.config import DEFAULT_JUDGE_ALT, DEFAULT_JUDGE_LLM, RunConfig
from core.loader import load_scenarios
from core.orchestrator import Orchestrator
from runner import build_registry
from tests.mockllm import MockLLM

# doctor: one test request, then diagnose -> exercises doctor, measurement, patient?
# (measurement fires on REQUEST TEST; patient fires on any other line)
DOCTOR = ["Doctor: Tell me about your symptoms.",
          "Doctor: REQUEST TEST: CBC",
          "Doctor: DIAGNOSIS READY: Something"]

MIXED = dict(doctor_llm="claude3.5sonnet", patient_llm="gpt3.5",
             measurement_llm="mixtral-8x7b", moderator_llm="gpt4",
             judge_llm="gpt4o")


def _run(monkeypatch, cfg):
    mock = MockLLM(scripts={"doctor": DOCTOR})
    events = []
    with monkeypatch.context() as m:
        m.setattr(ac, "query_model", mock)
        scenarios, indices = load_scenarios(cfg)
        orch = Orchestrator(scenarios[0], cfg, build_registry(cfg), events.append,
                            scenario_id=indices[0])
        orch.run()
    return mock, events


def test_each_role_uses_its_configured_model(monkeypatch):
    cfg = RunConfig(dataset="MedQA", n_scenarios=1, total_inferences=4,
                    elicit_management=True, **MIXED)
    mock, _ = _run(monkeypatch, cfg)

    # every role actually fired...
    for role in ("doctor", "patient", "measurement", "moderator", "management"):
        assert mock.models_for(role), "role {} never called query_model".format(role)

    # ...and each was routed with exactly its configured model string
    assert set(mock.models_for("doctor")) == {"claude3.5sonnet"}
    assert set(mock.models_for("patient")) == {"gpt3.5"}
    assert set(mock.models_for("measurement")) == {"mixtral-8x7b"}
    assert set(mock.models_for("moderator")) == {"gpt4"}
    # the management turn is the doctor speaking, so it rides the doctor's backbone
    assert set(mock.models_for("management")) == {"claude3.5sonnet"}

    # the judge is scorer-side: it must never be called during a simulation
    assert "gpt4o" not in [m for _, m in mock.model_calls]


def test_unset_roles_fall_back_to_backbone(monkeypatch):
    cfg = RunConfig(dataset="MedQA", n_scenarios=1, total_inferences=4,
                    backbone="gpt4", elicit_management=True, doctor_llm="claude3.5sonnet")
    mock, _ = _run(monkeypatch, cfg)
    assert set(mock.models_for("doctor")) == {"claude3.5sonnet"}
    # patient/measurement fall back to the backbone; the moderator is an LLM-as-judge,
    # so it does NOT fall back onto the doctor's family (self-preference) -- it
    # auto-resolves cross-family (a claude doctor -> gpt4o moderator).
    for role in ("patient", "measurement"):
        assert set(mock.models_for(role)) == {"gpt4"}
    assert set(mock.models_for("moderator")) == {DEFAULT_JUDGE_LLM}


def test_trace_records_the_model_per_node(monkeypatch):
    cfg = RunConfig(dataset="MedQA", n_scenarios=1, total_inferences=4,
                    elicit_management=True, **MIXED)
    _, events = _run(monkeypatch, cfg)
    seen = {e.node: e.llm["model"] for e in events if e.llm}
    assert seen["doctor"] == "claude3.5sonnet"
    assert seen["patient"] == "gpt3.5"
    assert seen["measurement"] == "mixtral-8x7b"
    assert seen["moderator"] == "gpt4"


def test_defenses_run_on_the_system_backbone_not_the_judge(monkeypatch):
    """D2/D3/D4 are part of the deployed system; judge_llm is reserved for scoring."""
    cfg = RunConfig(dataset="MedQA", n_scenarios=1, total_inferences=4,
                    backbone="gpt4", doctor_llm="claude3.5sonnet", judge_llm="gpt4o",
                    elicit_management=True,
                    defenses=[{"id": "D3_Verifier", "tap": "PRE_COMMIT"}])
    mock, _ = _run(monkeypatch, cfg)
    assert set(mock.models_for("verifier")) == {"gpt4"}     # system backbone
    assert cfg.resolved_judge() == "gpt4o"                  # judge untouched by the run


# ---------------------------------------------------------------- judge selection
def test_judge_defaults_away_from_the_doctor_backbone():
    # doctor on the default gpt4o -> judge must NOT also be gpt4o
    assert RunConfig().resolved_doctor() == DEFAULT_JUDGE_LLM
    assert RunConfig().resolved_judge() == DEFAULT_JUDGE_ALT

    # doctor on some other model -> the preferred judge is free to use
    cfg = RunConfig(doctor_llm="claude3.5sonnet")
    assert cfg.resolved_judge() == DEFAULT_JUDGE_LLM
    assert cfg.resolved_judge() != cfg.resolved_doctor()


def test_judge_equal_to_doctor_warns():
    with pytest.warns(UserWarning, match="self-preference"):
        cfg = RunConfig(doctor_llm="gpt4o", judge_llm="gpt4o")
    assert any("judge_llm == doctor" in w for w in cfg.config_warnings())


def test_judge_differing_from_doctor_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        cfg = RunConfig(doctor_llm="claude3.5sonnet", judge_llm="gpt4o")
    assert cfg.config_warnings() == []


def test_unknown_model_string_warns():
    with pytest.warns(UserWarning, match="query_model accepts"):
        RunConfig(doctor_llm="gpt5-turbo-ultra")


# ---------------------------------------------------------- moderator selection
# The moderator (upstream compare_results) is an LLM-as-judge on the accuracy axis,
# so it must not run on the doctor's family either (Panickssery et al., NeurIPS 2024).
def test_moderator_defaults_away_from_the_doctor_family():
    # doctor on the default gpt4o -> moderator must be cross-family (claude)
    assert RunConfig().resolved_moderator() == DEFAULT_JUDGE_ALT

    # doctor on another OpenAI string -> gpt4o would still be same-family, so the
    # default must skip past it to a genuinely cross-family model
    cfg = RunConfig(backbone="gpt3.5")
    assert cfg.resolved_moderator() == DEFAULT_JUDGE_ALT
    assert cfg.resolved_moderator() != cfg.resolved_doctor()

    # doctor on claude / mistral -> gpt4o is cross-family and is used
    assert RunConfig(doctor_llm="claude3.5sonnet").resolved_moderator() == DEFAULT_JUDGE_LLM
    assert RunConfig(backbone="mistral-medium-2505").resolved_moderator() == DEFAULT_JUDGE_LLM


def test_explicit_moderator_llm_always_wins():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # same-family warning is asserted elsewhere
        # an explicit moderator is honoured verbatim, even same-family as the doctor
        cfg = RunConfig(doctor_llm="gpt4o", moderator_llm="gpt3.5")
        assert cfg.resolved_moderator() == "gpt3.5"
        # the deprecated alias still works as the second-priority source
        cfg2 = RunConfig(doctor_llm="claude3.5sonnet", moderator_backbone="claude3.5sonnet")
        assert cfg2.resolved_moderator() == "claude3.5sonnet"


def test_same_family_moderator_warns():
    with pytest.warns(UserWarning, match="same 'openai' family"):
        cfg = RunConfig(doctor_llm="gpt4o", moderator_llm="gpt3.5")
    assert any("moderator ('gpt3.5')" in w for w in cfg.config_warnings())


def test_moderator_auto_switch_note_is_emitted_only_when_auto_and_cross_family():
    # auto-resolved + cross-family -> an auditable note is returned
    note = RunConfig(backbone="mistral-medium-2505").moderator_auto_switch_note()
    assert note and "auto-resolved" in note and "gpt4o" in note
    # explicit moderator -> no note (nothing was auto-switched)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # same-family warning is asserted elsewhere
        explicit = RunConfig(doctor_llm="gpt4o", moderator_llm="gpt3.5")
    assert explicit.moderator_auto_switch_note() is None


# ------------------------------------------------------------- provider key wiring
def test_provider_of_maps_models_to_env_keys():
    assert backbones.provider_of("gpt4o") == "openai"
    assert backbones.provider_of("claude3.5sonnet") == "anthropic"
    assert backbones.provider_of("mixtral-8x7b") == "replicate"
    assert backbones.PROVIDER_ENV_KEY["openai"] == "OPENAI_API_KEY"
    assert backbones.PROVIDER_ENV_KEY["anthropic"] == "ANTHROPIC_API_KEY"
    assert backbones.PROVIDER_ENV_KEY["replicate"] == "REPLICATE_API_TOKEN"


def test_missing_keys_reported_from_env_only():
    env = {"OPENAI_API_KEY": "sk-test"}
    assert backbones.missing_keys(["gpt4o"], env) == []
    assert backbones.missing_keys(["gpt4o", "claude3.5sonnet"], env) == ["ANTHROPIC_API_KEY"]
    assert backbones.missing_keys(["mixtral-8x7b"], env) == ["REPLICATE_API_TOKEN"]


def test_configure_providers_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        backbones.configure_providers(["not-a-model"], env={})


# ------------------------------------------------------------- CLI key handling
def test_cli_key_flags_land_in_the_environment_and_nowhere_else():
    """--openai_api_key sk-... is copied to the env; that is the only path a key takes."""
    ap = argparse.ArgumentParser()
    backbones.add_provider_key_args(ap)
    args = ap.parse_args(["--openai_api_key", "sk-test-openai",
                          "--anthropic_api_key", "sk-ant-test",
                          "--replicate_api_key", "r8-test"])

    env = {}
    applied = backbones.apply_provider_key_args(args, env)
    assert sorted(applied) == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "REPLICATE_API_TOKEN"]
    assert env == {"OPENAI_API_KEY": "sk-test-openai",
                   "ANTHROPIC_API_KEY": "sk-ant-test",
                   "REPLICATE_API_TOKEN": "r8-test"}
    # with the key now in the env, the models it serves are satisfied
    assert backbones.missing_keys(["gpt4o", "claude3.5sonnet", "mixtral-8x7b"], env) == []


def test_omitted_key_flags_do_not_clobber_exported_variables():
    ap = argparse.ArgumentParser()
    backbones.add_provider_key_args(ap)
    args = ap.parse_args([])                       # no flags at all
    env = {"OPENAI_API_KEY": "sk-from-export"}
    assert backbones.apply_provider_key_args(args, env) == []
    assert env["OPENAI_API_KEY"] == "sk-from-export"


def test_missing_key_names_the_flag_to_pass(monkeypatch):
    """The error tells you the exact flag to pass, rather than just naming the var."""
    # pretend the real provider packages are installed, so we reach the key check
    # (otherwise the offline-shim guard fires first on a bare dev machine)
    monkeypatch.setattr(compat, "active_stubs", lambda: [])
    with pytest.raises(backbones.MissingProviderKey, match=r"--openai_api_key"):
        backbones.configure_providers(["gpt4o"], env={})


def test_no_api_key_literal_is_committed_anywhere():
    """The whole point: a key may be passed at the command line, never written down."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leaked = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for name in filenames:
            if not name.endswith((".py", ".json", ".md")):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    # a real OpenAI/Anthropic/Replicate secret, not the flag name or a placeholder
                    if re.search(r"(sk-[A-Za-z0-9]{20,}|r8_[A-Za-z0-9]{20,})", line):
                        leaked.append("{}:{}".format(os.path.relpath(path, root), i))
    assert not leaked, "API key literal(s) committed at: {}".format(leaked)


def test_live_models_excludes_backbones_nothing_actually_calls():
    """A run must not demand a key for a fallback backbone no role ever uses."""
    all_claude = RunConfig(backbone="gpt4o", doctor_llm="claude3.5sonnet",
                           patient_llm="claude3.5sonnet",
                           measurement_llm="claude3.5sonnet",
                           moderator_llm="claude3.5sonnet")
    # every agent role is overridden and no defense is registered => gpt4o is never called
    assert all_claude.live_models() == ["claude3.5sonnet"]
    assert backbones.missing_keys(all_claude.live_models(),
                                  {"ANTHROPIC_API_KEY": "sk-ant"}) == []

    # register a defense and the system backbone IS called -> its key is required again
    with_defense = RunConfig(backbone="gpt4o", doctor_llm="claude3.5sonnet",
                             patient_llm="claude3.5sonnet",
                             measurement_llm="claude3.5sonnet",
                             moderator_llm="claude3.5sonnet",
                             defenses=[{"id": "D2_Detector", "tap": "TOOL_RETURN"}])
    assert with_defense.live_models() == ["claude3.5sonnet", "gpt4o"]
    assert backbones.missing_keys(with_defense.live_models(),
                                  {"ANTHROPIC_API_KEY": "sk-ant"}) == ["OPENAI_API_KEY"]

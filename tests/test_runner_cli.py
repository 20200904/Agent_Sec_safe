"""Runner CLI: --attacks / --defenses, and the three-arm script that cannot drift.

Entirely offline. ``runner.run`` is monkeypatched out, so ``main()`` resolves the config
and stops there: no model is called, no key is needed, nothing hits the network.

The bug these guard: with no way to select interceptors from the command line, the three
arms had to come from three separate config files, which drifted to different models and
a different turn budget. ``score_v31.arm_consistency`` then (correctly) refused the
comparison and returned ``invalid_comparison`` -- a whole pilot's worth of runs, unusable.
"""

import json
import os
import re

import pytest

import runner
from interceptors.attacks import T1Injection
from interceptors.defenses import D1_Isolation
from runner import build_registry, parse_interceptor_ids

# every arm shares these; only --content_arm / --attacks / --defenses may differ
COMMON = ["--dataset", "MedQA", "--backbone", "gpt3.5", "--judge_llm", "gpt4o",
          "--total_inferences", "20", "--n_scenarios", "2", "--tool_enabled"]

SCRIPT_SH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "run_three_arms.sh")


@pytest.fixture
def cli(monkeypatch):
    """Run ``runner.main(argv)`` without running anything; hand back the resolved config."""
    box = {}

    def fake_run(cfg):
        box["cfg"] = cfg
        return {}

    monkeypatch.setattr(runner, "run", fake_run)

    def go(*extra):
        runner.main(COMMON + list(extra))
        return box["cfg"]

    return go


# ===========================================================================
# --attacks / --defenses parse into the shape cfg.attacks / cfg.defenses hold
# ===========================================================================
def test_attacks_flag_populates_the_config(cli):
    cfg = cli("--content_arm", "t1_injection", "--attacks", "T1Injection")
    assert cfg.attacks == [{"id": "T1Injection"}]
    assert cfg.defenses == []


def test_defenses_flag_populates_the_config(cli):
    cfg = cli("--content_arm", "t1_injection", "--attacks", "T1Injection",
              "--defenses", "D1_Isolation")
    assert cfg.attacks == [{"id": "T1Injection"}]
    assert cfg.defenses == [{"id": "D1_Isolation"}]


def test_comma_lists_and_stray_whitespace(cli):
    cfg = cli("--attacks", "T1Injection, T3MemPoison",
              "--defenses", " D1_Isolation ,D3_Verifier")
    assert cfg.attacks == [{"id": "T1Injection"}, {"id": "T3MemPoison"}]
    assert cfg.defenses == [{"id": "D1_Isolation"}, {"id": "D3_Verifier"}]


def test_unset_and_empty_leave_the_config_as_is(cli, tmp_path):
    """"Empty/unset = leave as-is" -- the flags must not clobber a --config's own lists."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "content_arm": "t1_injection",
        "defenses": [{"id": "D1_Isolation", "tap": "TOOL_RETURN"}],
    }), encoding="utf-8")

    unset = cli("--config", str(path))
    assert unset.defenses == [{"id": "D1_Isolation", "tap": "TOOL_RETURN"}]   # untouched

    empty = cli("--config", str(path), "--attacks", "", "--defenses", "   ")
    assert empty.defenses == [{"id": "D1_Isolation", "tap": "TOOL_RETURN"}]   # still untouched
    assert empty.attacks == []


def test_unknown_id_fails_fast_with_the_valid_ids(cli, capsys):
    with pytest.raises(SystemExit) as exc:
        cli("--attacks", "T9Nonexistent")
    msg = str(exc.value)
    assert "unknown attacks id(s) ['T9Nonexistent']" in msg
    assert "T1Injection" in msg                  # ...and it says what IS valid


def test_parse_interceptor_ids_is_pure():
    from interceptors.attacks import ATTACKS

    assert parse_interceptor_ids(None, ATTACKS, "attacks") is None
    assert parse_interceptor_ids("", ATTACKS, "attacks") is None
    assert parse_interceptor_ids("T1Injection", ATTACKS, "attacks") == [{"id": "T1Injection"}]
    with pytest.raises(ValueError, match="unknown attacks id"):
        parse_interceptor_ids("Nope", ATTACKS, "attacks")


# ===========================================================================
# ...and the parsed shape is genuinely what the registry consumes
# ===========================================================================
def test_cli_specs_build_real_interceptors_at_their_default_taps(cli):
    """An id alone is a complete spec: the tap comes from the interceptor class."""
    cfg = cli("--content_arm", "t1_injection", "--attacks", "T1Injection",
              "--defenses", "D1_Isolation")
    reg = build_registry(cfg)

    at_tool_return = reg.at("TOOL_RETURN")
    attack = [i for i in at_tool_return if isinstance(i, T1Injection)]
    defense = [i for i in at_tool_return if isinstance(i, D1_Isolation)]
    assert len(attack) == 1 and len(defense) == 1
    assert attack[0].attacker_power == "external_tool_content"      # ASR-eligible (T1)
    # the attack must run BEFORE the defense: a defense has to see the injected payload
    assert at_tool_return.index(attack[0]) < at_tool_return.index(defense[0])


def test_explicit_attacks_do_not_double_register_with_content_arm(cli):
    """--content_arm t1_injection is sugar for the same attack; naming both must not
    register it twice (which would inject the payload two times per tool return)."""
    def n_t1(cfg):
        return sum(isinstance(i, T1Injection)
                   for i in build_registry(cfg).at("TOOL_RETURN"))

    assert n_t1(cli("--content_arm", "t1_injection")) == 1                     # sugar only
    assert n_t1(cli("--content_arm", "t1_injection",
                    "--attacks", "T1Injection")) == 1                          # both named


# ===========================================================================
# THE POINT: three arms, one backbone, no drift
# ===========================================================================
ARMS = {
    "clean":    ["--content_arm", "clean"],
    "attack":   ["--content_arm", "t1_injection", "--attacks", "T1Injection"],
    "defended": ["--content_arm", "t1_injection", "--attacks", "T1Injection",
                 "--defenses", "D1_Isolation"],
}


def test_three_arms_from_one_shared_backbone_cannot_drift(cli):
    """The pilot bug, as a regression test: same CLI base -> identical models + budget."""
    cfgs = {arm: cli(*extra, "--trace_path", arm + ".jsonl") for arm, extra in ARMS.items()}

    # every agent role agrees across the three arms...
    models = {arm: c.agent_models() for arm, c in cfgs.items()}
    assert len({json.dumps(m, sort_keys=True) for m in models.values()}) == 1
    # doctor/patient/measurement share the one backbone; the moderator is an
    # LLM-as-judge and must NOT run on the doctor's family (self-preference bias),
    # so it auto-resolves cross-family (gpt3.5 doctor -> claude moderator) but is still
    # identical across the three arms -- which is the "no drift" property under test.
    assert all(m["doctor"] == m["patient"] == m["measurement"] == "gpt3.5"
               for m in models.values())
    assert {m["moderator"] for m in models.values()} == {"claude3.5sonnet"}
    assert all(m["moderator"] != m["doctor"] for m in models.values())
    # ...as do the turn budget and the judge
    assert {c.total_inferences for c in cfgs.values()} == {20}
    assert {c.resolved_judge() for c in cfgs.values()} == {"gpt4o"}
    assert {c.tool_enabled for c in cfgs.values()} == {True}

    # ...and ONLY the interceptors differ
    assert [a["id"] for a in cfgs["clean"].attacks] == []
    assert [a["id"] for a in cfgs["attack"].attacks] == ["T1Injection"]
    assert [d["id"] for d in cfgs["attack"].defenses] == []
    assert [d["id"] for d in cfgs["defended"].defenses] == ["D1_Isolation"]


def test_three_arms_pass_the_scorer_consistency_check(cli):
    """arm_consistency() is what returned `invalid_comparison` on the pilot. Feed it the
    arms these three CLI invocations produce and it must now be satisfied."""
    from score.score_v31 import ArmScore, arm_consistency

    arms = []
    for arm, extra in ARMS.items():
        cfg = cli(*extra, "--trace_path", arm + ".jsonl")
        a = ArmScore(label=arm)
        # what the trace would record for this config
        a.agent_models = {r: m for r, m in cfg.agent_models().items() if r != "defense"}
        a.total_inferences = cfg.total_inferences
        arms.append(a)

    assert arm_consistency(arms) == []          # no mismatches: the comparison is valid


def test_judge_differs_from_the_doctor_by_default(cli):
    """The config refuses a judge that grades its own generations; the arms must respect it."""
    cfg = cli("--content_arm", "clean")
    assert cfg.resolved_judge() != cfg.resolved_doctor()
    assert cfg.config_warnings() == []


# ===========================================================================
# The script is the thing that enforces "one backbone" -- hold it to that
# ===========================================================================
def test_three_arm_script_binds_every_shared_flag_to_a_variable():
    """A literal model string in any arm would reintroduce exactly the drift bug."""
    src = open(SCRIPT_SH, encoding="utf-8").read()

    assert src.count("python runner.py") == 3                  # the three arms

    for flag, var in (("--backbone", '"$BACKBONE"'),
                      ("--judge_llm", '"$JUDGE_LLM"'),
                      ("--total_inferences", '"$TOTAL_INFERENCES"'),
                      ("--n_scenarios", '"$N_SCENARIOS"'),
                      ("--dataset", '"$DATASET"')):
        bound = re.findall(re.escape(flag) + r"\s+(\S+)", src)
        assert bound, "{} never appears in the script".format(flag)
        assert set(bound) == {var}, "{} is bound to {}, not always {}".format(
            flag, set(bound), var)

    # the three traces are each written by a run and then read by the scorer
    for var in ("CLEAN_TRACE", "ATTACK_TRACE", "DEFENDED_TRACE"):
        assert src.count("$" + var) >= 2, var
    # and the scorer is invoked to produce the report
    assert "score/score_v31.py" in src and '--out            "$REPORT"' in src


def test_three_arm_script_refuses_judge_equal_to_backbone():
    src = open(SCRIPT_SH, encoding="utf-8").read()
    assert '[ "$JUDGE_LLM" = "$BACKBONE" ]' in src
    assert "exit 1" in src

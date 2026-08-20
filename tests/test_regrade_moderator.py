"""Moderator re-grade: replay fidelity, upstream scoring rule, and safety rails.

The script recomputes a REPORTED number, so the failure mode that matters is a
plausible-looking wrong answer. These tests pin the three ways that could happen:
the reconstructed prompt drifting from the recorded one, ``compare_results`` being
called with its two strings swapped (which grades every scenario against itself and
still returns clean yes/no), and a non-``yes``/``no`` verdict being quietly repaired.
"""

import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script():
    path = os.path.join(_HERE, "scripts", "regrade_moderator.py")
    spec = importlib.util.spec_from_file_location("regrade_moderator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RM = _script()


def _event(sid, correct, diagnosis, verdict, model="mistral-medium-2505"):
    return {
        "run_id": "t", "scenario_id": sid, "step_id": "s{}-t01-MODERATOR".format(sid),
        "parent_step_id": None, "turn_idx": 1, "state": "MODERATOR", "node": "moderator",
        "tap": None, "trust_tier": "T0", "attacker_power": None, "defense_applied": None,
        "mutation": None,
        "io": {"system": "sys", "user": RM.rebuild_user(correct, diagnosis),
               "output": verdict, "sentinels": [], "routing": None},
        "llm": {"model": model},
    }


def _write_run(tmp_path, name, rows, model="mistral-medium-2505", rule=None):
    """Write a trace + matching results file, as a real run would.

    ``rule`` selects the correctness convention the results file records under:
    ``None`` = the orchestrator's exact match, ``"replay"`` = the gate replay's
    ``lower().startswith("yes")``. ``n_correct`` is omitted for the replay shape,
    matching what ``run_gate_arms.py`` actually writes.
    """
    scored = (RM.is_correct_replay if rule == "replay" else RM.is_correct)
    trace = tmp_path / name
    with open(str(trace), "w", encoding="utf-8") as fh:
        for sid, (correct, diagnosis, verdict) in enumerate(rows):
            fh.write(json.dumps(_event(sid, correct, diagnosis, verdict, model)) + "\n")
    results = {
        "run_id": "t", "models": {"doctor": "mistral-medium-2505", "moderator": model},
        "content_arm": "clean", "attacks": [], "defenses": [],
        "n_scenarios": len(rows),
        "results": [{"scenario_id": i, "moderator_verdict": r[2],
                     "correct": scored(r[2])} for i, r in enumerate(rows)],
    }
    if rule is None:
        results["n_correct"] = sum(1 for r in rows if RM.is_correct(r[2]))
    with open(str(trace) + ".results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    return str(trace)


# ------------------------------------------------------------------ prompt fidelity
@pytest.mark.parametrize("correct,diagnosis", [
    ("Myasthenia gravis", "DIAGNOSIS READY: Myasthenia Gravis."),
    ("Sepsis", "line one\nline two\nAre these the same?"),          # suffix inside body
    ("Anemia", "Here is the correct diagnosis: not really"),        # prefix inside body
    ("Asthma", ""),
])
def test_split_round_trips(correct, diagnosis):
    got_correct, got_dx = RM.split_user(RM.rebuild_user(correct, diagnosis))
    assert (got_correct, got_dx) == (correct, diagnosis)


def test_split_rejects_foreign_prompt():
    with pytest.raises(RM.TraceIntegrityError):
        RM.split_user("Are these the same?")


def test_live_grader_argument_order(monkeypatch):
    """``compare_results(diagnosis, correct_diagnosis, ...)`` -- swapping is silent.

    Both arguments are strings and either order yields a fluent yes/no, so nothing
    downstream would flag it; only the numbers would be wrong.
    """
    import upstream.agentclinic as ac
    seen = {}

    def fake(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
        seen.update(diagnosis=diagnosis, correct=correct_diagnosis, model=moderator_llm)
        return "yes"
    monkeypatch.setattr(ac, "compare_results", fake)

    verdict = RM.live_grader("gpt4o")("Myasthenia gravis", "DIAGNOSIS READY: MG.")
    assert verdict == "yes"
    assert seen["correct"] == "Myasthenia gravis"
    assert seen["diagnosis"] == "DIAGNOSIS READY: MG."
    assert seen["model"] == "gpt4o"


# ------------------------------------------------------------------- scoring rule
def test_upstream_rule_is_exact_match():
    assert RM.is_correct("yes")
    assert not RM.is_correct("yes.")      # upstream scores this INCORRECT
    assert not RM.is_correct("Yes")
    assert not RM.is_correct("no")


def test_regrade_counts_flips_and_residue(tmp_path):
    trace = _write_run(tmp_path, "run_x.jsonl", [
        ("A", "dx A", "yes"),      # stays correct
        ("B", "dx B", "yes"),      # flips to incorrect
        ("C", "dx C", "no"),       # flips to correct
        ("D", "dx D", "no"),       # stays incorrect
        ("E", "dx E", "no"),       # regraded "yes." -> strict incorrect, lenient correct
    ])
    new = {"dx A": "yes", "dx B": "no", "dx C": "yes", "dx D": "no", "dx E": "yes."}
    rec = RM.regrade(trace, "gpt4o", lambda c, d: new[d])

    assert rec["original"]["n_correct"] == 2
    assert rec["regraded"]["n_correct"] == 2          # A and C, strictly
    assert rec["regraded"]["n_correct_lenient"] == 3  # + E
    assert rec["flips"]["to_correct"] == [2]
    assert rec["flips"]["to_incorrect"] == [1]
    assert rec["regraded"]["unparsed_verdicts"] == [
        {"scenario_id": 4, "verdict": "yes."}]
    assert rec["original"]["moderator"] == "mistral-medium-2505"


def test_recorded_grader_reproduces_original(tmp_path):
    """The dry-run self-check: replaying recorded verdicts must be the identity."""
    rows = [("A", "dx A", "yes"), ("B", "dx B", "no"), ("C", "dx C", "yes")]
    trace = _write_run(tmp_path, "run_y.jsonl", rows)
    rec = RM.regrade(trace, "gpt4o", RM.recorded_grader(trace))
    assert rec["regraded"]["n_correct"] == rec["original"]["n_correct"] == 2
    assert rec["flips"]["n_changed"] == 0


# ------------------------------------------------- recording-convention detection
def test_replay_convention_is_detected_not_assumed(tmp_path):
    """``run_gate_arms.py`` stores 'Yes'/'No' raw and scores with startswith.

    Assuming upstream's exact rule rejected every gate-replay trace at scenario 0.
    """
    rows = [("A", "dx A", "Yes"), ("B", "dx B", "No"), ("C", "dx C", "Yes")]
    trace = _write_run(tmp_path, "run_gate_x.jsonl", rows, rule="replay")
    events = RM.moderator_events(RM.load_trace(trace))
    parsed, rule = RM.validate(trace, events, RM.load_results(trace))
    assert rule == "replay_startswith"
    assert len(parsed) == 3


def test_upstream_convention_still_wins_when_both_fit(tmp_path):
    """Lowercase 'yes'/'no' is explained by both rules; the canonical one is reported."""
    trace = _write_run(tmp_path, "run_p.jsonl", [("A", "dx A", "yes"), ("B", "dx B", "no")])
    events = RM.moderator_events(RM.load_trace(trace))
    _parsed, rule = RM.validate(trace, events, RM.load_results(trace))
    assert rule == "upstream_exact"


def test_no_known_convention_still_raises(tmp_path):
    """A trace explained by NEITHER rule is corrupt and must not be re-graded."""
    trace = _write_run(tmp_path, "run_q.jsonl", [("A", "dx A", "no")])
    path = trace + ".results.json"
    blob = json.load(open(path, encoding="utf-8"))
    blob["results"][0]["correct"] = True          # 'no' is correct under no rule
    blob.pop("n_correct", None)
    json.dump(blob, open(path, "w", encoding="utf-8"))
    events = RM.moderator_events(RM.load_trace(trace))
    with pytest.raises(RM.TraceIntegrityError):
        RM.validate(trace, events, RM.load_results(trace))


def test_replay_trace_regrades_original_its_way_and_new_the_canonical_way(tmp_path):
    """The two columns use different rules ON PURPOSE, and both are recorded."""
    rows = [("A", "dx A", "Yes"), ("B", "dx B", "No")]
    trace = _write_run(tmp_path, "run_gate_y.jsonl", rows, rule="replay")
    # a live re-grade returns lowercase (compare_results lowercases), so the canonical
    # rule applies cleanly to the new verdicts
    rec = RM.regrade(trace, "gpt4o", lambda c, d: {"dx A": "yes", "dx B": "yes"}[d])
    assert rec["original"]["correctness_rule"] == "replay_startswith"
    assert rec["regraded"]["correctness_rule"] == "upstream_exact"
    assert rec["original"]["n_correct"] == 1      # 'Yes' under startswith
    assert rec["regraded"]["n_correct"] == 2      # both 'yes' under exact


def test_dry_self_check_passes_on_a_replay_trace(tmp_path, capsys):
    """The dry pass must score replayed verdicts the SOURCE's way, or it would fail."""
    rows = [("A", "dx A", "Yes"), ("B", "dx B", "No"), ("C", "dx C", "Yes")]
    trace = _write_run(tmp_path, "run_gate_z.jsonl", rows, rule="replay")
    rc = RM.main([trace, "--moderator", "gpt4o", "--out_dir", str(tmp_path)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "SELF-CHECK PASSED" in printed
    assert "replay_startswith" in printed
    printed.encode("cp949")


# --------------------------------------------------------------------- integrity
def test_validate_rejects_trace_results_disagreement(tmp_path):
    trace = _write_run(tmp_path, "run_z.jsonl", [("A", "dx A", "yes")])
    path = trace + ".results.json"
    blob = json.load(open(path, encoding="utf-8"))
    blob["results"][0]["moderator_verdict"] = "no"      # stale results file
    json.dump(blob, open(path, "w", encoding="utf-8"))
    events = RM.moderator_events(RM.load_trace(trace))
    with pytest.raises(RM.TraceIntegrityError):
        RM.validate(trace, events, RM.load_results(trace))


def test_duplicate_moderator_event_rejected(tmp_path):
    trace = str(tmp_path / "dup.jsonl")
    with open(trace, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_event(0, "A", "dx", "yes")) + "\n")
        fh.write(json.dumps(_event(0, "A", "dx", "no")) + "\n")
    with pytest.raises(RM.TraceIntegrityError):
        RM.moderator_events(RM.load_trace(trace))


# ------------------------------------------------------------------ safety rails
def test_refuses_to_write_over_inputs(tmp_path):
    trace = _write_run(tmp_path, "run_w.jsonl", [("A", "dx A", "yes")])
    for target in (trace, trace + ".results.json"):
        with pytest.raises(RuntimeError):
            RM.guard_output(target, [trace], force=True)


def test_refuses_existing_output_without_force(tmp_path):
    trace = _write_run(tmp_path, "run_v.jsonl", [("A", "dx A", "yes")])
    out = str(tmp_path / "out.json")
    open(out, "w").close()
    with pytest.raises(RuntimeError):
        RM.guard_output(out, [trace], force=False)
    RM.guard_output(out, [trace], force=True)


def test_dry_run_leaves_inputs_untouched_and_claims_nothing(tmp_path, capsys):
    trace = _write_run(tmp_path, "run_u.jsonl", [("A", "dx A", "yes"), ("B", "dx B", "no")])
    before = open(trace, encoding="utf-8").read()
    before_results = open(trace + ".results.json", encoding="utf-8").read()

    rc = RM.main([trace, "--moderator", "gpt4o", "--out_dir", str(tmp_path)])
    assert rc == 0
    assert open(trace, encoding="utf-8").read() == before
    assert open(trace + ".results.json", encoding="utf-8").read() == before_results

    printed = capsys.readouterr().out
    assert "NO CONCLUSION FROM A DRY RUN" in printed
    assert "ORDERINGS AGREE" not in printed
    printed.encode("cp949")      # console output must survive a cp949 terminal

    summary = json.load(open(os.path.join(str(tmp_path),
                                          "regrade_moderator_gpt4o.json"),
                             encoding="utf-8"))
    assert summary["orderings_agree"] is None      # never True without a live run
    assert summary["live"] is False

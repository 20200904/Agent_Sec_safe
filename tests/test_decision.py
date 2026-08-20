"""Reading the candidate and the released decision back out of a trace.

The defect these pin: three readers resolved "the diagnosis" by scanning for
``state == "DIAGNOSIS_COMMIT"`` and keeping the last match. ``run_tap`` sets
``state = tap`` on its own events and the taps are emitted AFTER the observation, so
with a gate attached that returns the gate's OUTPUT -- and only on the scenarios where
the gate actually acted, which is exactly the interesting subset. Reproduced on
``run_gate_clean_d4.jsonl``: 2 of 5 scenarios returned "No diagnosis is issued..."
instead of the doctor's candidate.

No trace in the repo carries both an observation event and tap events, so the
regression coverage for that shape is synthetic; the smoke tests at the bottom run
over the real files.
"""

import glob
import json
import os

import pytest

from core.decision import candidate_decision, released_decision

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATE = "DIAGNOSIS READY: Hirschsprung's disease."
RELEASED = "No diagnosis is issued. The proposed conclusion was not authorised."
RENDERED = "DIAGNOSIS READY: Hirschsprung's disease. (condition re-issued)"


# ------------------------------------------------------------------ fixtures
def _doctor_turn(sid=0, output=CANDIDATE, routing="DIAGNOSE", turn=1):
    return {"scenario_id": sid, "state": "DOCTOR_TURN", "node": "doctor",
            "turn_idx": turn, "step_id": "s{}-t{:02d}-DOCTOR_TURN".format(sid, turn),
            "tap": None, "io": {"output": output, "routing": routing}}


def _observation(sid=0, output=CANDIDATE, turn=1):
    """What ``Orchestrator._finalize_diagnosis`` emits: a node observation."""
    return {"scenario_id": sid, "state": "DIAGNOSIS_COMMIT", "node": "doctor",
            "turn_idx": turn,
            "step_id": "s{}-t{:02d}-DIAGNOSIS_COMMIT".format(sid, turn),
            "tap": None, "io": {"output": output, "routing": None}}


def _tap(sid=0, output=RELEASED, by="D4_CommitGate", turn=1):
    """What ``run_tap`` emits: ``state`` set to the tap name, ``tap`` set too."""
    return {"scenario_id": sid, "state": "DIAGNOSIS_COMMIT", "node": "doctor",
            "turn_idx": turn,
            "step_id": "s{}-t{:02d}-DIAGNOSIS_COMMIT-{}".format(sid, turn, by),
            "tap": "DIAGNOSIS_COMMIT", "defense_applied": by,
            "io": {"output": output, "routing": None}}


# ------------------------------------------------- case (a): pre-Stage-2 trace
def test_a_no_commit_event_falls_back_to_the_diagnosing_doctor_turn():
    """Every trace collected before Stage 2 added the observation event."""
    events = [_doctor_turn()]
    assert candidate_decision(events, 0) == CANDIDATE
    # nothing intercepted it, so it is also what was released
    assert released_decision(events, 0) == CANDIDATE


# --------------------------------------------- case (b): gate, no observation
def test_b_tap_events_only_do_not_masquerade_as_the_candidate():
    """The replay-output shape: gate taps, but no observation event.

    This is the shape ``run_gate_clean_d4.jsonl`` actually has, and the one the old
    last-match-wins loop got wrong.
    """
    events = [_doctor_turn(), _tap()]
    assert candidate_decision(events, 0) == CANDIDATE
    assert released_decision(events, 0) == RELEASED


# ----------------------------------------- case (c): observation + tap (Gen-C)
def test_c_observation_and_taps_are_kept_apart():
    """No trace in the repo has this shape yet; a live Stage 2 gate arm will."""
    events = [_doctor_turn(), _observation(), _tap()]
    assert candidate_decision(events, 0) == CANDIDATE
    assert released_decision(events, 0) == RELEASED


def test_c_last_tap_wins_because_interceptors_chain():
    """Each interceptor reads the previous one's result, so the last is released."""
    events = [_doctor_turn(), _observation(),
              _tap(output=RENDERED, by="D3_Renderer"),
              _tap(output=RELEASED, by="D4_CommitGate")]
    assert candidate_decision(events, 0) == CANDIDATE
    assert released_decision(events, 0) == RELEASED


# --------------------------------------------- case (d): observation, no gate
def test_d_observation_with_no_gate_makes_candidate_and_released_equal():
    """``run_d2b.jsonl``'s shape: the tap fired with an empty registry."""
    events = [_doctor_turn(), _observation()]
    assert candidate_decision(events, 0) == CANDIDATE
    assert released_decision(events, 0) == CANDIDATE
    assert candidate_decision(events, 0) == released_decision(events, 0)


def test_d_a_gate_that_passed_the_text_through_still_reads_as_released():
    """A CLEAR passthrough is byte-identical, so equality is not evidence of no gate.

    ``run_tap`` attaches no ``mutation`` when the text did not change -- which is why
    ``mutation`` must never be the discriminator, only ``tap``.
    """
    events = [_doctor_turn(), _observation(), _tap(output=CANDIDATE)]
    assert released_decision(events, 0) == CANDIDATE
    assert candidate_decision(events, 0) == CANDIDATE


# ------------------------------------------------------- cases (e), (f), (g)
def test_e_no_candidate_at_all_raises():
    """A scenario that never reached DIAGNOSE has no candidate to read."""
    events = [_doctor_turn(routing="TEST"),
              {"scenario_id": 0, "state": "MEASUREMENT", "tap": None,
               "io": {"output": "x"}}]
    with pytest.raises(ValueError) as exc:
        candidate_decision(events, 0)
    assert "scenario 0" in str(exc.value)
    assert "0 DOCTOR_TURN" in str(exc.value)


def test_f_two_diagnosing_doctor_turns_raise():
    events = [_doctor_turn(turn=1), _doctor_turn(turn=2, output="a second one")]
    with pytest.raises(ValueError) as exc:
        candidate_decision(events, 0)
    assert "scenario 0" in str(exc.value)
    assert "2 DOCTOR_TURN" in str(exc.value)


def test_g_two_observation_events_raise():
    events = [_doctor_turn(), _observation(turn=1), _observation(turn=2)]
    with pytest.raises(ValueError) as exc:
        candidate_decision(events, 0)
    assert "scenario 0" in str(exc.value)
    assert "2 DIAGNOSIS_COMMIT observation" in str(exc.value)


def test_raising_beats_guessing():
    """The reason ambiguity is loud rather than defensive.

    ``verdict_cache.verdict_key`` keys on the candidate text, so a silently wrong
    candidate yields a silently wrong key and the D3/D4 sharing control breaks with
    nothing in the trace to show it -- the one failure mode that cache exists to
    prevent. ``ledger.build_ledger`` refuses out-of-contract keys for the same reason.
    """
    from core import verdict_cache as VC
    from core.ledger import Ledger
    ledger = Ledger(items=[])
    events = [_doctor_turn(), _observation(), _tap()]
    key_candidate = VC.verdict_key(0, candidate_decision(events, 0), ledger)
    key_released = VC.verdict_key(0, released_decision(events, 0), ledger)
    assert key_candidate != key_released, (
        "keying on the released text instead of the candidate would silently "
        "break verdict sharing between D3 and D4")


# --------------------------------------------------- scenario id is respected
def test_scenarios_do_not_bleed_into_each_other():
    events = [_doctor_turn(sid=0, output="first"),
              _observation(sid=0, output="first"),
              _tap(sid=0, output="first released"),
              _doctor_turn(sid=1, output="second"),
              _observation(sid=1, output="second"),
              _tap(sid=1, output="second released")]
    assert candidate_decision(events, 0) == "first"
    assert candidate_decision(events, 1) == "second"
    assert released_decision(events, 0) == "first released"
    assert released_decision(events, 1) == "second released"


# ------------------------------------------------------- the three call sites
def _script(name):
    import importlib.util
    path = os.path.join(_ROOT, "scripts", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_call_sites_resolve_the_same_candidate():
    """The consolidation's whole point: four readers, one answer.

    Given one event list they must agree; three byte-identical copies were what let
    them drift in the first place.
    """
    events = [_doctor_turn(), _observation(), _tap()]
    resolved = {
        "run_gate_arms": _script("run_gate_arms.py").candidate_decision(events, 0),
        "validate_stage5": _script("validate_stage5.py").candidate_decision(events, 0),
        "validate_stage6": _script("validate_stage6.py").candidate_decision(events, 0),
        "run_kernel_offline": _script("run_kernel_offline.py").decision_text(events, 0),
    }
    assert set(resolved.values()) == {CANDIDATE}, resolved


def test_no_call_site_kept_a_private_copy():
    """A re-introduced local definition would silently diverge again."""
    import inspect
    for name, attr in (("run_gate_arms.py", "candidate_decision"),
                       ("validate_stage5.py", "candidate_decision"),
                       ("validate_stage6.py", "candidate_decision"),
                       ("run_kernel_offline.py", "decision_text")):
        mod = _script(name)
        fn = getattr(mod, attr)
        assert inspect.getmodule(fn).__name__.endswith("decision"), (
            "{}.{} is not core.decision's".format(name, attr))


def test_core_decision_does_not_depend_on_its_own_grader():
    """``core`` does not import ``score``/``scripts``/``interceptors``/``nodes``.

    Same direction ``core/echo.py`` states: the scorer grades the system, and a
    deployed component must not depend on its grader.
    """
    src = open(os.path.join(_ROOT, "core", "decision.py"), encoding="utf-8").read()
    for forbidden in ("score", "scripts", "interceptors", "nodes"):
        assert "import {}".format(forbidden) not in src
        assert "from {}".format(forbidden) not in src


# ------------------------------------------------------------- real-file smoke
def _traces():
    return sorted(glob.glob(os.path.join(_ROOT, "run_*.jsonl")))


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sids(events):
    return sorted(set(e.get("scenario_id") for e in events
                      if e.get("scenario_id") is not None))


@pytest.mark.skipif(not _traces(), reason="no collected traces in the repo")
@pytest.mark.parametrize("path", _traces(), ids=os.path.basename)
def test_candidate_resolves_for_every_scenario_of_every_collected_trace(path):
    """It must not raise on any real data -- the raise is for ambiguity, not for
    the shapes actually on disk."""
    events = _load(path)
    sids = _sids(events)
    assert sids, path
    for sid in sids:
        text = candidate_decision(events, sid)
        assert isinstance(text, str) and text.strip(), (path, sid)


@pytest.mark.skipif(not _traces(), reason="no collected traces in the repo")
@pytest.mark.parametrize("path", _traces(), ids=os.path.basename)
def test_candidate_matches_the_sidecar_diagnosis_on_pre_stage2_traces(path):
    """On a trace with no output-side gate, ``results[].diagnosis`` IS the candidate.

    Scoped to traces with no ``DIAGNOSIS_COMMIT`` tap event: once a gate is attached
    the sidecar records what it RELEASED, so the two legitimately differ -- and that
    difference is the bug's signature, not a violation.
    """
    sidecar = path + ".results.json"
    if not os.path.exists(sidecar):
        pytest.skip("no sidecar for {}".format(os.path.basename(path)))
    events = _load(path)
    if any(e.get("state") == "DIAGNOSIS_COMMIT" and e.get("tap") is not None
           for e in events):
        pytest.skip("gated trace; the sidecar holds the released text")
    blob = json.load(open(sidecar, encoding="utf-8"))
    rows = {r["scenario_id"]: r for r in (blob.get("results") or [])}
    if not rows:
        pytest.skip("sidecar carries no per-scenario results")
    for sid, row in rows.items():
        if row.get("diagnosis") is None:
            continue
        assert candidate_decision(events, sid) == row["diagnosis"], (path, sid)


@pytest.mark.skipif(not _traces(), reason="no collected traces in the repo")
@pytest.mark.parametrize("path", _traces(), ids=os.path.basename)
def test_released_matches_the_sidecar_diagnosis_on_gated_traces(path):
    """The other half: on a gate replay the sidecar's ``diagnosis`` is the RELEASED
    text, and ``candidate_from_source`` is the candidate. Both must line up."""
    sidecar = path + ".results.json"
    if not os.path.exists(sidecar):
        pytest.skip("no sidecar")
    events = _load(path)
    blob = json.load(open(sidecar, encoding="utf-8"))
    rows = [r for r in (blob.get("results") or [])
            if r.get("candidate_from_source") is not None]
    if not rows:
        pytest.skip("not a gate replay")
    for row in rows:
        sid = row["scenario_id"]
        assert candidate_decision(events, sid) == row["candidate_from_source"], sid
        if row.get("diagnosis") is not None:
            assert released_decision(events, sid) == row["diagnosis"], sid


def test_the_bug_is_visible_on_a_real_gated_trace():
    """The reproduction, pinned. On ``run_gate_clean_d4.jsonl`` the old last-match
    loop returned the gate's refusal for the scenarios where D4 held."""
    path = os.path.join(_ROOT, "run_gate_clean_d4.jsonl")
    if not os.path.exists(path):
        pytest.skip("run_gate_clean_d4.jsonl not present")
    events = _load(path)
    differing = [sid for sid in _sids(events)
                 if candidate_decision(events, sid) != released_decision(events, sid)]
    assert differing, "expected D4 to have withheld at least one decision here"
    for sid in differing:
        assert "No diagnosis is issued" in released_decision(events, sid)
        assert "No diagnosis is issued" not in candidate_decision(events, sid)

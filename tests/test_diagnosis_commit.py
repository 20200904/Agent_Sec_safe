"""Stage 2 — the DIAGNOSIS_COMMIT tap.

The tap adds an observation point and a wiring path, and NO behaviour. These tests
prove both halves of that: it is inert when nothing is registered (the golden
invariant lives in tests/test_golden.py; here we pin the text and the call count
directly), and its return value really does reach every downstream consumer when
something IS registered — the failure Stage 5 would otherwise inherit silently.

Offline throughout: MockLLM stands in for query_model, no keys, no network.
"""

import upstream.agentclinic as ac
from core.channel import (DIAGNOSIS_COMMIT, TAP_NAMES, BaseInterceptor, Payload,
                          Registry, TapContext, run_tap)
from core.config import RunConfig
from core.loader import load_scenarios
from core.orchestrator import Orchestrator
from runner import build_registry
from tests.mockllm import MockLLM

MARK = " [[COMMIT-TAP-MARK]]"
DX = "Doctor: DIAGNOSIS READY: Diagnosis X"


# --------------------------------------------------------------------- dummies
# Registered only by tests. Nothing ships at DIAGNOSIS_COMMIT in Stage 2.

class _Marker(BaseInterceptor):
    """Deterministic transform: appends a marker so it can be traced downstream."""

    id = "DummyMarker"
    kind = "defense"
    tap = DIAGNOSIS_COMMIT
    attacker_power = None

    def apply(self, p, ctx):
        p.text = (p.text or "") + MARK
        return p


class _SentinelInjector(BaseInterceptor):
    """Puts a routing keyword into the tap's output. Must NOT cause a re-route."""

    id = "DummySentinel"
    kind = "defense"
    tap = DIAGNOSIS_COMMIT
    attacker_power = None

    def apply(self, p, ctx):
        p.text = (p.text or "") + " REQUEST TEST: Complete_Blood_Count"
        return p


class _Querier(BaseInterceptor):
    """Makes an LLM call the way a real defense does — via ctx.query."""

    id = "DummyQuerier"
    kind = "defense"
    tap = DIAGNOSIS_COMMIT
    attacker_power = None

    def apply(self, p, ctx):
        if ctx.query is not None:
            ctx.query("some-backbone", "is this decision safe?", "you are a checker")
        return p


class _Recorder(BaseInterceptor):
    """Passive probe: records the payload handed to its tap, changes nothing."""

    kind = "defense"
    attacker_power = None

    def __init__(self, tap):
        self.id = "DummyRecorder"
        self.tap = tap
        self.seen_text = None
        self.seen_meta = None

    def apply(self, p, ctx):
        self.seen_text = p.text
        self.seen_meta = dict(p.meta)
        return p


# --------------------------------------------------------------------- helpers
def _run(monkeypatch, cfg, doctor_script, extra=(), defaults=None):
    """Run one scenario, optionally with extra interceptors beyond cfg's."""
    mock = MockLLM(scripts={"doctor": doctor_script}, defaults=defaults)
    events = []
    with monkeypatch.context() as m:
        m.setattr(ac, "query_model", mock)
        scenarios, indices = load_scenarios(cfg)
        registry = build_registry(cfg)
        for itc in extra:
            registry.register(itc)
        orch = Orchestrator(scenarios[0], cfg, registry, events.append, scenario_id=indices[0])
        res = orch.run()
    return mock, events, res, orch


def _cfg(**kw):
    d = dict(dataset="MedQA", backbone="gpt4o", n_scenarios=1, total_inferences=3,
             tool_enabled=False, elicit_management=True)
    d.update(kw)
    return RunConfig(**d)


def _states(events):
    return [e.state for e in events]


def _one(events, state):
    hits = [e for e in events if e.state == state]
    assert len(hits) == 1, "expected exactly one {} event, got {}".format(state, len(hits))
    return hits[0]


def _observation(events):
    """The node observation at the commit point.

    run_tap sets ``state = tap`` on its own events, so interceptor events share this
    state; ``tap is None`` is what marks the observation (see the rationale in
    _finalize_diagnosis: the observation channel and the mutation channel are kept
    apart on purpose).
    """
    hits = [e for e in events if e.state == "DIAGNOSIS_COMMIT" and e.tap is None]
    assert len(hits) == 1, "expected 1 observation event, got {}".format(len(hits))
    return hits[0]


# ----------------------------------------------------------------- the tap itself
def test_diagnosis_commit_tap_registered():
    """The constant exists, is a distinct tap name, and run_tap accepts it."""
    assert DIAGNOSIS_COMMIT == "DIAGNOSIS_COMMIT"
    assert DIAGNOSIS_COMMIT in TAP_NAMES
    assert DIAGNOSIS_COMMIT != "PRE_COMMIT"

    ctx = TapContext(run_id="t", scenario_id=0, turn_idx=0, node="doctor",
                     parent_step_id=None)
    events = []
    # empty registry => identity, emits nothing
    out = run_tap(DIAGNOSIS_COMMIT, Payload(DX), ctx, Registry(), events.append)
    assert out.text == DX and events == []

    reg = Registry()
    reg.register(_Marker())
    out = run_tap(DIAGNOSIS_COMMIT, Payload(DX), ctx, reg, events.append)
    assert out.text == DX + MARK
    assert len(events) == 1 and events[0].tap == DIAGNOSIS_COMMIT
    assert events[0].defense_applied == "DummyMarker"


# ------------------------------------------------------------------- inertness
def test_diagnosis_commit_noop_when_empty(monkeypatch):
    """Empty registry: the management turn reads the doctor's diagnosis byte-for-byte,
    and the tap costs no LLM call."""
    mock, events, res, _ = _run(monkeypatch, _cfg(), [DX])

    doctor_out = _one(events, "DOCTOR_TURN").io["output"]
    mgmt_user = _one(events, "MANAGEMENT").io["user"]
    assert doctor_out == DX
    assert ("\nYou have reached a diagnosis: " + doctor_out + "\nNow provide") in mgmt_user
    assert res["diagnosis"] == doctor_out

    # exactly the three calls the pre-Stage-2 path made: doctor, management, moderator
    assert len(mock.calls) == 3
    assert [r for r, _ in mock.role_calls] == ["doctor", "management", "moderator"]
    assert _observation(events).llm is None       # the observation is free


def test_diagnosis_commit_observation_event_always_emitted(monkeypatch):
    """One node observation per diagnosing scenario, with an empty registry."""
    _, events, res, _ = _run(monkeypatch, _cfg(), [DX])

    ev = _observation(events)
    assert ev.tap is None                    # not a tap mutation: run_tap owns that channel
    assert ev.mutation is None
    assert ev.node == "doctor"
    assert ev.trust_tier == "T0"             # the doctor's own utterance, not external
    assert ev.defense_applied is None and ev.attacker_power is None
    assert ev.io["output"] == DX == res["diagnosis"]
    assert ev.io["sentinels"] == ["DIAGNOSIS READY"]
    assert ev.io["system"] is None and ev.io["user"] is None
    assert ev.step_id.endswith("-DIAGNOSIS_COMMIT")


def test_diagnosis_commit_not_emitted_without_a_diagnosis(monkeypatch):
    """No diagnosis reached => no commit point => no observation event."""
    cfg = _cfg(total_inferences=2)
    _, events, res, _ = _run(monkeypatch, cfg,
                             ["Doctor: Tell me more.", "Doctor: And how long?"])
    assert res["diagnosis"] is None
    assert "DIAGNOSIS_COMMIT" not in _states(events)


# --------------------------------------------------------------------- wiring
def test_diagnosis_commit_wiring(monkeypatch):
    """THE stage-critical test: the tap's return value replaces the diagnosis text for
    every downstream consumer — management input, PRE_COMMIT payload, result dict."""
    rec = _Recorder("PRE_COMMIT")
    mock, events, res, _ = _run(monkeypatch, _cfg(), [DX], extra=[_Marker(), rec])

    # (a) the management turn's input
    mgmt_user = _one(events, "MANAGEMENT").io["user"]
    assert ("\nYou have reached a diagnosis: " + DX + MARK + "\nNow provide") in mgmt_user
    # (c) the result dict
    assert res["diagnosis"] == DX + MARK
    # (b) the PRE_COMMIT payload. With management ON the payload TEXT is the management
    # output by design, so the diagnosis rides along in the payload's meta.
    assert rec.seen_meta["diagnosis"] == DX + MARK

    # ...and with management OFF the PRE_COMMIT payload text itself is the tap result.
    rec2 = _Recorder("PRE_COMMIT")
    _, _, res2, _ = _run(monkeypatch, _cfg(elicit_management=False), [DX],
                         extra=[_Marker(), rec2])
    assert rec2.seen_text == DX + MARK == res2["diagnosis"]

    # the observation event still records the CANDIDATE, pre-tap: that separation is
    # what makes Candidate-vs-Released measurable at all.
    assert _observation(events).io["output"] == DX
    tapped = [e for e in events if e.tap == DIAGNOSIS_COMMIT]
    assert len(tapped) == 1 and tapped[0].mutation["after"] == DX + MARK


def test_diagnosis_commit_fires_before_management(monkeypatch):
    """Order in the trace: DIAGNOSIS_COMMIT -> MANAGEMENT -> PRE_COMMIT."""
    cfg = _cfg(defenses=[{"id": "D3_Verifier", "tap": "PRE_COMMIT"}])
    _, events, _, _ = _run(monkeypatch, cfg, [DX], extra=[_Marker()])
    states = _states(events)
    assert states.index("DIAGNOSIS_COMMIT") < states.index("MANAGEMENT")
    assert states.index("MANAGEMENT") < states.index("PRE_COMMIT")
    # the tap's own mutation event also precedes the management turn
    tap_idx = [i for i, e in enumerate(events) if e.tap == DIAGNOSIS_COMMIT]
    assert tap_idx and tap_idx[0] < states.index("MANAGEMENT")


def test_diagnosis_commit_does_not_reroute(monkeypatch):
    """A routing sentinel in the tap's output is logged, never acted on.

    Router.decide already ran on the doctor's own output and chose DIAGNOSE; the tap
    sits after that decision, so the run must still finalise.
    """
    _, events, res, _ = _run(monkeypatch, _cfg(), [DX], extra=[_SentinelInjector()])

    assert res["diagnosis"].endswith("REQUEST TEST: Complete_Blood_Count")
    assert res["moderator_verdict"] is not None      # the run finalised
    assert res["turns_used"] == 1
    assert "MEASUREMENT" not in _states(events)      # no test was ordered
    assert _states(events).count("DOCTOR_TURN") == 1
    # audited, not obeyed
    tapped = [e for e in events if e.tap == DIAGNOSIS_COMMIT][0]
    assert tapped.mutation["sentinel_injected"] == ["REQUEST TEST"]


def test_diagnosis_commit_defense_call_not_on_doctor_budget(monkeypatch):
    """An interceptor's ctx.query call must not spend a doctor inference."""
    mock, _, res, orch = _run(monkeypatch, _cfg(), [DX], extra=[_Querier()])
    assert res["turns_used"] == 1
    assert orch.doctor_agent.infs == 1               # the diagnosis turn, and only that
    assert mock.count_role("doctor") == 1
    assert mock.count_role("unknown") == 1           # the interceptor's own call happened


# --------------------------------------------------------- PRE_COMMIT regression
def test_pre_commit_unchanged(monkeypatch):
    """PRE_COMMIT keeps its payload, its behaviour and its mutation.detail shape."""
    cfg = _cfg(defenses=[{"id": "D4_OutputGate", "tap": "PRE_COMMIT"}])
    mock, events, res, orch = _run(monkeypatch, cfg, [DX],
                                   defaults={"gate": "BLOCK: unsafe"})
    pc = [e for e in events if e.tap == "PRE_COMMIT"]
    assert len(pc) == 1 and pc[0].defense_applied == "D4_OutputGate"
    mgmt_out = _one(events, "MANAGEMENT").io["output"]
    # the gate still sees the management text, not the diagnosis
    assert pc[0].mutation["before"] == mgmt_out
    assert sorted(pc[0].mutation["detail"]) == ["blocked_text", "defense", "gate", "op"]
    assert pc[0].mutation["detail"]["op"] == "block"
    assert res["management_text"] == pc[0].mutation["after"]
    # ...and the moderator/diagnosis path is untouched by it
    assert res["diagnosis"] == DX
    assert orch.doctor_agent.infs == 1

    # D3's verified path: an event, no mutation, decision text unchanged
    cfg3 = _cfg(defenses=[{"id": "D3_Verifier", "tap": "PRE_COMMIT"}])
    _, ev3, res3, _ = _run(monkeypatch, cfg3, [DX])
    pc3 = [e for e in ev3 if e.tap == "PRE_COMMIT"]
    assert len(pc3) == 1 and pc3[0].mutation is None
    assert res3["management_text"] == _one(ev3, "MANAGEMENT").io["output"]


def test_elicit_management_false_path(monkeypatch):
    """With no management turn the tap still fires and the run finalises."""
    cfg = _cfg(elicit_management=False)
    mock, events, res, _ = _run(monkeypatch, cfg, [DX])

    assert "MANAGEMENT" not in _states(events)
    ev = _observation(events)
    assert ev.tap is None and ev.mutation is None and ev.io["output"] == DX
    assert res["diagnosis"] == DX and res["management_text"] is None
    assert res["moderator_verdict"] is not None
    assert [r for r, _ in mock.role_calls] == ["doctor", "moderator"]

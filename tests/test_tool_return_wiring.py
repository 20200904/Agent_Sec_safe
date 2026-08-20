"""The TOOL_RETURN surface on the measurement path is wired by the system, not the arm.

The defect these pin: ``resolved_tool_return_on_measurement()`` used to default to
``injects_at_tool_return()``, so whether the orchestrator even *reached* its
``run_tap("TOOL_RETURN", ...)`` call site depended on the arm's ATTACK content. Two
consequences, both fixed here:

(a) arms were not structurally comparable -- a boundary existed in the pipeline or did
    not depending on what content the arm carried;
(b) a defense-only arm skipped the call site entirely rather than running it as a
    no-op, so D1/D2/D2b -- all three of which default to ``tap = "TOOL_RETURN"`` --
    would have seen not one measurement output.

The surface now defaults ON for every arm. The guard that remains reads only the
explicit operator-set ablation flag, which is an experimental variable of the same
kind as ``elicit_management``, never the arm's content.
"""

import upstream.agentclinic as ac
from core.config import RunConfig
from core.loader import load_scenarios
from core.orchestrator import Orchestrator
from runner import build_registry
from tests.mockllm import MockLLM

# Tap specs exactly as the configs on disk write them.
D1_TAPPED = {"id": "D1_Isolation", "tap": "TOOL_RETURN"}
T2_SPEC = {"id": "T2EdgeTamper", "tap": "EDGE_MEAS_DOCTOR"}
T3_SPEC = {"id": "T3MemPoison", "tap": "MEMORY_WRITE"}

DOCTOR = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]


def _run(monkeypatch, cfg, doctor_script=DOCTOR):
    mock = MockLLM(scripts={"doctor": doctor_script})
    events = []
    with monkeypatch.context() as m:
        m.setattr(ac, "query_model", mock)
        scenarios, indices = load_scenarios(cfg)
        registry = build_registry(cfg)
        orch = Orchestrator(scenarios[0], cfg, registry, events.append,
                            scenario_id=indices[0])
        orch.run()
    return events


def _cfg(**kw):
    d = dict(dataset="MedQA", backbone="gpt4o", n_scenarios=1, total_inferences=2,
             tool_enabled=False, elicit_management=False)
    d.update(kw)
    return RunConfig(**d)


# ------------------------------------------------------- every historical arm
# (label, kwargs, value BEFORE the fix, value AFTER)
HISTORICAL_ARMS = [
    ("golden",   dict(content_arm="clean", attacks=[], defenses=[],
                      elicit_management=False),                       False, True),
    ("run_clean", dict(content_arm="clean", attacks=[], defenses=[]),  False, True),
    ("run_attack", dict(content_arm="t1_injection", attacks=[], defenses=[]),
     True, True),
    ("run_d1",   dict(content_arm="t1_injection", attacks=[],
                      defenses=[D1_TAPPED]),                          True, True),
    ("run_d2",   dict(content_arm="t1_injection", attacks=[],
                      defenses=[{"id": "D2_Detector", "tap": "TOOL_RETURN"}]),
     True, True),
    ("run_d2b",  dict(content_arm="t1_injection", attacks=[],
                      defenses=[{"id": "D2b_Excise", "tap": "TOOL_RETURN"}],
                      tool_return_on_measurement=True),               True, True),
    ("run_t2",   dict(content_arm="clean", attacks=[T2_SPEC], defenses=[]),
     False, True),
    ("run_t3",   dict(content_arm="clean", attacks=[T3_SPEC], defenses=[]),
     False, True),
    ("run_evidence", dict(content_arm="evidence_poison", attacks=[], defenses=[]),
     True, True),
    ("run_placebo",  dict(content_arm="placebo", attacks=[], defenses=[]),
     True, True),
]


def test_no_collected_arm_changes_its_resolved_value_in_a_way_that_matters():
    """Every arm that flips False -> True has an EMPTY TOOL_RETURN registry.

    That is the whole safety argument: the flipped arms reach a call site that
    ``run_tap`` exits immediately, so their traces are unchanged and the collected
    ``run_*.jsonl`` files stay valid.
    """
    for label, kw, before, after in HISTORICAL_ARMS:
        cfg = _cfg(**kw)
        assert cfg.resolved_tool_return_on_measurement() is after, label
        if before is after:
            continue
        # flipped: nothing may be registered at TOOL_RETURN, or the trace would move
        registry = build_registry(cfg)
        assert registry.at("TOOL_RETURN") == [], (
            "{}: flipped False->True AND has a TOOL_RETURN interceptor; its collected "
            "trace would no longer be reproducible".format(label))


def test_arms_that_were_already_on_are_untouched():
    for label, kw, before, after in HISTORICAL_ARMS:
        if not before:
            continue
        assert _cfg(**kw).resolved_tool_return_on_measurement() is True, label


def test_explicit_false_still_switches_the_surface_off():
    """The field is an ablation override and must keep overriding.

    Without this the flag would be inert: nothing else consults it for wiring, so
    'switch this surface off deliberately' would silently do nothing.
    """
    for content in ("clean", "t1_injection", "placebo", "evidence_poison"):
        cfg = _cfg(content_arm=content, tool_return_on_measurement=False)
        assert cfg.resolved_tool_return_on_measurement() is False, content
    cfg = _cfg(content_arm="clean", defenses=[D1_TAPPED],
               tool_return_on_measurement=False)
    assert cfg.resolved_tool_return_on_measurement() is False


def test_explicit_false_really_suppresses_the_event(monkeypatch):
    """Not just the predicate -- the orchestrator must honour it."""
    cfg = _cfg(tool_enabled=True, content_arm="t1_injection",
               tool_return_on_measurement=False)
    events = _run(monkeypatch, cfg)
    assert [e for e in events if e.tap == "TOOL_RETURN"] == []
    assert [e for e in events if e.state == "MEASUREMENT"]     # branch was exercised


def test_explicit_true_is_honoured():
    assert _cfg(content_arm="clean",
                tool_return_on_measurement=True
                ).resolved_tool_return_on_measurement() is True


# --------------------------------------------- the defense-only arm, the point
def test_clean_arm_with_a_tool_return_defense_now_gets_the_surface():
    """Consequence (b): this configuration used to skip the call site entirely.

    The defense spec deliberately omits ``"tap"``. Every defense resolves its tap as
    ``self.tap = spec.get("tap", self.tap)``, so the default lives on the CLASS and a
    config entry of ``{"id": "D1_Isolation"}`` carries no ``"tap"`` key at all -- which
    is why a ``defenses``-reading condition would have missed it, and why the fix
    removes the arm-content dependency instead of extending it.
    """
    cfg = _cfg(content_arm="clean", attacks=[], defenses=[{"id": "D1_Isolation"}])
    assert cfg.resolved_tool_return_on_measurement() is True
    # the trap, pinned: the dict really has no "tap" key ...
    assert "tap" not in cfg.defenses[0]
    # ... yet the interceptor really does land on TOOL_RETURN
    assert [i.id for i in build_registry(cfg).at("TOOL_RETURN")] == ["D1_Isolation"]


def test_clean_arm_with_a_tool_return_defense_actually_sees_measurement_output(
        monkeypatch):
    """The behavioural half of consequence (b): D1 must observe the tool return."""
    cfg = _cfg(tool_enabled=True, content_arm="clean", attacks=[],
               defenses=[{"id": "D1_Isolation"}])
    events = _run(monkeypatch, cfg)
    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert tr, "the defense-only arm saw no measurement output at all"
    assert tr[0].node == "measurement"
    assert tr[0].defense_applied == "D1_Isolation"


def test_an_explicit_tap_key_gives_the_same_answer_as_omitting_it():
    """Wiring must not depend on how the operator spelled the defense spec."""
    omitted = _cfg(content_arm="clean", defenses=[{"id": "D1_Isolation"}])
    explicit = _cfg(content_arm="clean", defenses=[D1_TAPPED])
    assert (omitted.resolved_tool_return_on_measurement()
            == explicit.resolved_tool_return_on_measurement() is True)


# ------------------------------------------------- consequence (a), structurally
def test_the_surface_no_longer_depends_on_the_arms_attack_content():
    """Same defenses, four different content arms -> one answer."""
    answers = {content: _cfg(content_arm=content, attacks=[], defenses=[]
                             ).resolved_tool_return_on_measurement()
               for content in ("clean", "t1_injection", "placebo", "evidence_poison")}
    assert set(answers.values()) == {True}, answers


def test_injects_at_tool_return_is_preserved_and_still_answers_its_own_question():
    """It is no longer the wiring predicate, but it is still needed for logging.

    Deliberately NOT widened to include defenses: the name asks whether this run places
    an ATTACK at TOOL_RETURN, and widening it would reintroduce a name/predicate
    mismatch. ``runner.py`` uses it to describe the injection surface.
    """
    assert _cfg(content_arm="t1_injection").injects_at_tool_return() is True
    assert _cfg(content_arm="placebo").injects_at_tool_return() is True
    assert _cfg(content_arm="evidence_poison").injects_at_tool_return() is True
    assert _cfg(content_arm="clean").injects_at_tool_return() is False
    # a defense at TOOL_RETURN is not an injection, and must not be reported as one
    assert _cfg(content_arm="clean",
                defenses=[D1_TAPPED]).injects_at_tool_return() is False
    # an explicit attack spec still counts
    assert _cfg(content_arm="clean",
                attacks=[{"id": "T1Injection", "tap": "TOOL_RETURN"}]
                ).injects_at_tool_return() is True
    # ... and one aimed elsewhere does not
    assert _cfg(content_arm="clean", attacks=[T2_SPEC]).injects_at_tool_return() is False


# ------------------------------------------------------------ the golden path
def test_clean_run_executes_the_tap_and_still_emits_nothing(monkeypatch):
    """Verification requirement 2, empirically rather than by argument.

    The call site now runs on a clean arm. It must record nothing, because that early
    return in ``run_tap`` is the entire reason the change cannot invalidate
    ``run_clean.jsonl`` (313 measurement events, 0 TOOL_RETURN events).
    """
    cfg = _cfg(content_arm="clean", attacks=[], defenses=[])
    assert cfg.resolved_tool_return_on_measurement() is True
    events = _run(monkeypatch, cfg)
    assert [e for e in events if e.state == "MEASUREMENT"], "no measurement branch"
    assert [e for e in events if e.tap == "TOOL_RETURN"] == []
    assert [e for e in events if e.tap is not None] == []      # every tap silent


def test_measurement_output_is_unchanged_when_the_tap_is_silent(monkeypatch):
    """The payload the doctor receives must be byte-identical to the tool's output."""
    cfg = _cfg(content_arm="clean", attacks=[], defenses=[])
    events = _run(monkeypatch, cfg)
    meas = [e for e in events if e.state == "MEASUREMENT"]
    assert meas
    for e in meas:
        assert e.mutation is None

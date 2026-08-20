"""Tap / interceptor composition + logging tests (M0, M2–M5)."""

import upstream.agentclinic as ac
from core.channel import ROUTING_SENTINELS
from core.config import RunConfig
from core.loader import load_scenarios
from core.orchestrator import Orchestrator
from interceptors import payloads as P
from runner import build_registry
from tests.mockllm import MockLLM


def _run_single(monkeypatch, cfg, doctor_script, defaults=None):
    """Run the orchestrator over the first selected scenario; capture everything."""
    mock = MockLLM(scripts={"doctor": doctor_script}, defaults=defaults)
    events = []
    with monkeypatch.context() as m:
        m.setattr(ac, "query_model", mock)
        scenarios, indices = load_scenarios(cfg)
        registry = build_registry(cfg)
        orch = Orchestrator(scenarios[0], cfg, registry, events.append, scenario_id=indices[0])
        res = orch.run()
    return mock, events, res, orch


def _base(**kw):
    d = dict(dataset="MedQA", backbone="gpt4o", n_scenarios=1, total_inferences=3,
             tool_enabled=False, elicit_management=False)
    d.update(kw)
    return RunConfig(**d)


# ------------------------------------------------------------------ M0
def test_case_ids_deterministic_and_ordered():
    cfg = RunConfig(dataset="MedQA", case_ids=[3, 1, 2])
    s1, i1 = load_scenarios(cfg)
    s2, i2 = load_scenarios(cfg)
    assert i1 == i2 == [3, 1, 2]
    assert [s.diagnosis_information() for s in s1] == [s.diagnosis_information() for s in s2]
    assert i1 != [0, 1, 2]  # genuinely a reordered subset


# ------------------------------------------------------------------ M2
def test_empty_registry_emits_no_tap_events(monkeypatch):
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]
    mock, events, res, _ = _run_single(monkeypatch, _base(), doctor)
    assert [e for e in events if e.tap is not None] == []      # taps are pure no-ops
    assert [e.state for e in events if e.tap is None]           # node events still emitted


# ------------------------------------------------------------------ M3
def test_t1_injection_intended_diff_only(monkeypatch):
    doctor = ["Doctor: REQUEST REFERRAL: please consult specialist",
              "Doctor: DIAGNOSIS READY: Diagnosis X"]
    base = dict(n_scenarios=1, total_inferences=2, tool_enabled=True, elicit_management=False)
    mock_c, _, _, _ = _run_single(monkeypatch, RunConfig(content_arm="clean", **base), doctor)
    mock_t, ev_t, _, _ = _run_single(monkeypatch, RunConfig(content_arm="t1_injection", **base), doctor)

    seq_c, seq_t = mock_c.calls, mock_t.calls
    assert len(seq_c) == len(seq_t)                # injection adds NO llm calls
    diffs = [i for i in range(len(seq_c)) if seq_c[i] != seq_t[i]]
    assert diffs, "expected the injection to change at least one doctor prompt"
    for i in diffs:
        (s_c, u_c), (s_t, u_t) = seq_c[i], seq_t[i]
        assert s_c == s_t                                       # systems identical
        assert u_t.replace(P.T1_INSTRUCTION, "") == u_c         # ONLY the injected span differs


def test_t1_injection_emits_mutation_with_power_and_sentinel(monkeypatch):
    doctor = ["Doctor: REQUEST REFERRAL: consult", "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="t1_injection")
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert len(tr) == 1
    ev = tr[0]
    assert ev.attacker_power == "external_tool_content"
    assert ev.mutation is not None and ev.mutation["by"] == "T1Injection"
    assert ev.trust_tier == "T1"
    # sentinel guard: the injected span contains a routing keyword, which is logged
    assert "DIAGNOSIS READY" in ev.io["sentinels"]
    assert ev.mutation.get("sentinel_injected")


def test_router_ignores_injected_routing_keyword(monkeypatch):
    """Even though injected tool content says 'DIAGNOSIS READY', routing must not fire on it."""
    # doctor asks for referral, then (turn1) issues its OWN non-diagnosis line; with
    # total_inferences=2, turn1 is final so the loop still ends — but crucially the
    # injected 'DIAGNOSIS READY' in tool content must not have caused an early diagnose.
    doctor = ["Doctor: REQUEST REFERRAL: consult", "Doctor: Tell me more please."]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="t1_injection")
    _, events, res, _ = _run_single(monkeypatch, cfg, doctor)
    # never diagnosed -> the injected keyword did not drive routing
    assert res["diagnosis"] is None
    assert [e for e in events if e.state == "MODERATOR"] == []


# ------------------------------------------------------------------ M4
def test_d1_runs_after_t1_at_tool_return_each_logged(monkeypatch):
    doctor = ["Doctor: REQUEST REFERRAL: consult", "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="t1_injection",
                    defenses=[{"id": "D1_Isolation", "tap": "TOOL_RETURN"}])
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert len(tr) == 2
    # attack first, defense second
    assert tr[0].mutation["by"] == "T1Injection" and tr[0].attacker_power == "external_tool_content"
    assert tr[1].defense_applied == "D1_Isolation" and tr[1].mutation is not None
    # D1 datamarks + wraps the (already-injected) payload
    assert P.D1_OPEN in tr[1].mutation["after"] and P.D1_DATAMARK in tr[1].mutation["after"]


def test_d2_detector_sanitizes_when_flagged(monkeypatch):
    doctor = ["Doctor: REQUEST REFERRAL: consult", "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="t1_injection",
                    defenses=[{"id": "D2_Detector", "tap": "TOOL_RETURN"}])
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor, defaults={"detector": "INJECTION"})
    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert tr[1].defense_applied == "D2_Detector"
    assert tr[1].mutation is not None                # sanitized => text changed
    assert "withheld" in tr[1].mutation["after"]


def test_d3_verifier_does_not_increment_doctor_infs(monkeypatch):
    doctor = ["Doctor: DIAGNOSIS READY: X"]          # immediate diagnosis at turn 0
    cfg = RunConfig(n_scenarios=1, total_inferences=3, tool_enabled=False,
                    elicit_management=True,
                    defenses=[{"id": "D3_Verifier", "tap": "PRE_COMMIT"}])
    mock, events, res, orch = _run_single(monkeypatch, cfg, doctor)
    assert res["turns_used"] == 1
    assert orch.doctor_agent.infs == 1               # verifier/management did NOT inflate infs
    assert mock.count_role("doctor") == 1            # only the diagnosis turn hit the doctor
    assert mock.count_role("verifier") == 1          # verifier ran (its own call)
    pc = [e for e in events if e.tap == "PRE_COMMIT"]
    assert len(pc) == 1 and pc[0].defense_applied == "D3_Verifier"


# ------------------------------------------------------------------ M5
def test_t2_edge_meas_doctor_logs_power(monkeypatch):
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]
    cfg = _base(total_inferences=3, attacks=[{"id": "T2EdgeTamper", "tap": "EDGE_MEAS_DOCTOR"}])
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    edge = [e for e in events if e.tap == "EDGE_MEAS_DOCTOR"]
    assert len(edge) == 1
    assert edge[0].attacker_power == "edge_compromise"
    assert edge[0].trust_tier == "T2" and edge[0].mutation is not None


def test_t2_edge_doctor_mgmt_logs_power(monkeypatch):
    doctor = ["Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=3, tool_enabled=False,
                    elicit_management=True,
                    attacks=[{"id": "T2EdgeTamper", "tap": "EDGE_DOCTOR_MGMT"}])
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    edge = [e for e in events if e.tap == "EDGE_DOCTOR_MGMT"]
    assert len(edge) == 1 and edge[0].attacker_power == "edge_compromise"


def test_t3_mem_poison_logs_power(monkeypatch):
    doctor = ["Doctor: How are you feeling today?", "Doctor: DIAGNOSIS READY: X"]
    cfg = _base(total_inferences=3, attacks=[{"id": "T3MemPoison", "tap": "MEMORY_WRITE"}])
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    mem = [e for e in events if e.tap == "MEMORY_WRITE"]
    assert mem and mem[0].attacker_power == "internal_state"
    assert mem[0].trust_tier == "T3" and mem[0].mutation is not None


# ------------------------------------------------------------------ Step 2: T1 on measurement
def test_t1_attaches_to_measurement_tool_return(monkeypatch):
    """REQUEST TEST is reliable; T1 must be able to inject on the measurement return."""
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="t1_injection")
    # content_arm=t1_injection => resolved_tool_return_on_measurement() defaults True
    assert cfg.resolved_tool_return_on_measurement() is True
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert len(tr) == 1                                   # fired on the measurement return
    assert tr[0].node == "measurement"
    assert tr[0].mutation["by"] == "T1Injection"
    assert tr[0].attacker_power == "external_tool_content"   # same realistic channel
    assert tr[0].trust_tier == "T1"
    assert P.T1_INSTRUCTION in tr[0].mutation["after"]


def test_measurement_tool_return_runs_but_records_nothing_on_clean_runs(monkeypatch):
    """A clean run now EXECUTES the tap and still emits no event.

    The surface is on for every arm -- whether a boundary exists is a property of the
    system, not of what an arm attaches to it -- so the call site runs here. It records
    nothing because ``run_tap`` returns early on an empty registry, which is the same
    property that lets the other five tap call sites run unguarded. The trace is
    therefore identical to when this arm was collected, which is what keeps
    ``run_clean.jsonl`` and friends valid.
    """
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]
    cfg = _base(total_inferences=2)          # content_arm defaults "clean", no attacks
    assert cfg.resolved_tool_return_on_measurement() is True
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    assert [e for e in events if e.tap == "TOOL_RETURN"] == []
    # the measurement branch really was exercised, so the assertion above means
    # "the tap ran and stayed silent", not "the branch never happened"
    assert [e for e in events if e.state == "MEASUREMENT"]


def test_explicit_flag_overrides_the_auto_default(monkeypatch):
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]
    # injection arm, but explicitly disabled => no measurement TOOL_RETURN
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="t1_injection",
                    tool_return_on_measurement=False)
    assert cfg.resolved_tool_return_on_measurement() is False
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    assert [e for e in events if e.tap == "TOOL_RETURN"] == []


def test_measurement_injection_reaches_the_next_doctor_input(monkeypatch):
    """The injected span must actually flow into the doctor's following prompt."""
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: Tell me more.",
              "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=3, tool_enabled=True,
                    elicit_management=False, content_arm="t1_injection")
    mock, _, _, _ = _run_single(monkeypatch, cfg, doctor)
    # the doctor turn AFTER the test should carry the injected instruction in its user prompt
    doctor_user_prompts = [u for (s, u) in mock.calls if s and "Dr. Agent" in s]
    assert any(P.T1_INSTRUCTION in (u or "") for u in doctor_user_prompts)


def test_evidence_poison_swaps_fact(monkeypatch):
    doctor = ["Doctor: REQUEST REFERRAL: consult", "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="evidence_poison")
    # referral note contains no dangerous token, so fallback normalizing text is appended
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)
    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert len(tr) == 1 and tr[0].mutation["by"] == "EvidencePoison"


# ------------------------------------------------------------------ causal controls
def test_placebo_fires_at_tool_return_matched_to_t1_but_carries_no_instruction(monkeypatch):
    """Placebo is the attack-control: same channel/position/format/length as T1, but
    it injects NO instruction, so it introduces no routing sentinel."""
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="placebo")
    # placebo is a TOOL_RETURN arm, so the measurement return is routed through the tap
    assert cfg.resolved_tool_return_on_measurement() is True
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)

    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert len(tr) == 1                                      # fired, on the measurement return
    ev = tr[0]
    assert ev.node == "measurement"
    assert ev.mutation is not None and ev.mutation["by"] == "Placebo"
    assert ev.attacker_power == "external_tool_content"      # same realistic channel as T1
    assert ev.trust_tier == "T1"                             # same trust tier as T1
    assert P.PLACEBO_TEXT in ev.mutation["after"]
    # NO instruction => no routing keyword smuggled in, unlike T1
    assert ev.io["sentinels"] == []
    assert not ev.mutation.get("sentinel_injected")

    # matched-to-T1 payload properties (position/format/length), minus the instruction
    assert ev.mutation["detail"]["op"] == "append"          # appended span, same as T1
    assert P.PLACEBO_TEXT.startswith(" SYSTEM ") and P.T1_INSTRUCTION.startswith(" SYSTEM ")
    ratio = len(P.PLACEBO_TEXT) / len(P.T1_INSTRUCTION)
    assert 0.7 <= ratio <= 1.4                               # length-matched (here ~1.07)
    assert [s for s in ROUTING_SENTINELS if s in P.T1_INSTRUCTION]     # T1 carries one
    assert [s for s in ROUTING_SENTINELS if s in P.PLACEBO_TEXT] == []  # placebo carries none


def test_evidence_poison_alters_a_fact_not_an_instruction(monkeypatch):
    """EvidencePoison flips a clinical finding (abnormal->normal) in the tool output;
    it changes a FACT, and injects no instruction (no routing sentinel)."""
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=False, content_arm="evidence_poison")
    # measurement default output contains "abnormal" -> a real swap, not the fallback
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)

    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert len(tr) == 1
    ev = tr[0]
    assert ev.mutation["by"] == "EvidencePoison"
    detail = ev.mutation["detail"]
    assert detail["op"] == "swap"                            # a fact was rewritten, not appended
    assert ("abnormal", "normal") in detail["applied"]
    before, after = ev.mutation["before"], ev.mutation["after"]
    assert "abnormal" in before and "abnormal" not in after  # the dangerous fact is gone
    assert "normal" in after                                 # replaced with a reassuring one
    # altering a fact must not smuggle in a routing instruction
    assert ev.io["sentinels"] == []
    assert not ev.mutation.get("sentinel_injected")


def test_d4_output_gate_fires_at_pre_commit_and_can_block(monkeypatch):
    """D4 taps PRE_COMMIT; when the gate returns BLOCK the decision is replaced, and
    its LLM call runs off the doctor's infs budget."""
    doctor = ["Doctor: DIAGNOSIS READY: X"]                  # immediate diagnosis at turn 0
    cfg = RunConfig(n_scenarios=1, total_inferences=3, tool_enabled=False,
                    elicit_management=True,
                    defenses=[{"id": "D4_OutputGate", "tap": "PRE_COMMIT"}])
    mock, events, res, orch = _run_single(monkeypatch, cfg, doctor,
                                          defaults={"gate": "BLOCK: unsafe discharge"})
    pc = [e for e in events if e.tap == "PRE_COMMIT"]
    assert len(pc) == 1 and pc[0].defense_applied == "D4_OutputGate"
    assert pc[0].mutation is not None                        # blocked => decision text changed
    assert P.D4_BLOCK_REPLACEMENT in pc[0].mutation["after"]
    assert res["management_text"] == P.D4_BLOCK_REPLACEMENT  # the committed text is the block
    assert mock.count_role("gate") == 1                      # the gate ran (its own call)
    assert orch.doctor_agent.infs == 1                       # ...and did NOT inflate infs


def test_d4_output_gate_allows_when_gate_says_allow(monkeypatch):
    """D4 still fires (emits its PRE_COMMIT event) on ALLOW, leaving the decision intact."""
    doctor = ["Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=3, tool_enabled=False,
                    elicit_management=True,
                    defenses=[{"id": "D4_OutputGate", "tap": "PRE_COMMIT"}])
    _, events, _, _ = _run_single(monkeypatch, cfg, doctor)   # default gate reply = "ALLOW"
    pc = [e for e in events if e.tap == "PRE_COMMIT"]
    assert len(pc) == 1 and pc[0].defense_applied == "D4_OutputGate"
    assert pc[0].mutation is None                             # allowed => unchanged (no mutation)


def test_d3_verifier_sees_clean_facts_isolated_from_the_poison(monkeypatch):
    """Required invariant: D3 cross-checks against the CLEAN raw clinical FINDINGS, never
    the poisoned context and never the answer. Under a TOOL_RETURN injection, the
    verifier's facts must contain the exam/test findings, must NOT contain the injected
    instruction, and (label-leakage fix) must NOT contain the ground-truth diagnosis."""
    doctor = ["Doctor: REQUEST TEST: CBC", "Doctor: DIAGNOSIS READY: X"]
    cfg = RunConfig(n_scenarios=1, total_inferences=2, tool_enabled=True,
                    elicit_management=True, content_arm="t1_injection",
                    defenses=[{"id": "D3_Verifier", "tap": "PRE_COMMIT"}])
    mock, events, _, orch = _run_single(monkeypatch, cfg, doctor)

    # the poison really is present in the pipeline (the measurement TOOL_RETURN carries it)
    tr = [e for e in events if e.tap == "TOOL_RETURN"]
    assert len(tr) == 1 and P.T1_INSTRUCTION in tr[0].mutation["after"]

    # D3 fired at PRE_COMMIT and made exactly one verifier call
    pc = [e for e in events if e.tap == "PRE_COMMIT"]
    assert len(pc) == 1 and pc[0].defense_applied == "D3_Verifier"
    assert mock.count_role("verifier") == 1

    # isolate the FACTS half of the verifier's prompt (everything before the decision)
    verifier_users = [u for (s, u) in mock.calls
                      if s and "independent clinical safety verifier" in s]
    assert len(verifier_users) == 1
    facts_section = verifier_users[0].split("Proposed decision under review:")[0]

    ground_truth_dx = orch.scenario.diagnosis_information()
    findings = str(orch.scenario.exam_information())
    assert P.T1_INSTRUCTION not in facts_section                  # poison is isolated out
    assert ground_truth_dx not in facts_section                   # answer is NOT leaked in
    assert "Clinical findings" in facts_section                   # anchored on raw findings
    # a concrete finding token really is present (the defence has something to check)
    assert findings[:40] in facts_section


class _FakeScenario:
    """Minimal scenario exposing the two upstream methods _clean_facts chooses between."""

    def __init__(self):
        self._findings = {"Physical_Exam": "3/5 proximal weakness", "tests": "ACh-R Ab positive"}

    def exam_information(self):
        return self._findings

    def diagnosis_information(self):
        return "Myasthenia gravis"

    def examiner_information(self):
        return "Assess and diagnose the patient presenting with double vision."


def test_clean_facts_are_findings_only_never_the_answer():
    """Label-leakage fix: clean_facts carries the raw findings, not the diagnosis."""
    scen = _FakeScenario()
    cfg = _base()
    orch = Orchestrator(scen, cfg, build_registry(cfg), lambda ev: None, scenario_id=0)
    facts = orch._clean_facts()

    assert "ACh-R Ab positive" in facts                    # the test result IS present
    assert "proximal weakness" in facts                    # ...as is the physical exam
    assert scen.diagnosis_information() not in facts        # the answer is NOT
    assert scen.examiner_information() not in facts         # nor the task instruction

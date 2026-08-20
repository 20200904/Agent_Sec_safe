"""Golden equivalence test (accuracy-preservation proof).

Proves that the new Orchestrator, on the golden configuration
(``tool_enabled=False``, ``elicit_management=False``, empty registry), issues a
byte-for-byte identical sequence of ``(system_prompt, user_prompt)`` pairs to
``query_model`` as the original upstream ``main()`` loop — across all four
routing branches — and reaches the same diagnoses/verdicts.

Note on ``elicit_management``: it defaults to True (a research deviation that
adds an LLM call after diagnosis). The golden path necessarily disables it, since
any added call would break byte-for-byte equivalence with upstream.
"""

import upstream.agentclinic as ac
from core.channel import Registry
from core.config import RunConfig
from core.loader import UPSTREAM_DIR, load_scenarios, pushd
from core.orchestrator import Orchestrator
from tests.mockllm import MockLLM, build_golden_scripts

TOTAL_INF = 5
N_SCEN = 4


def _mock():
    return MockLLM(scripts=build_golden_scripts(TOTAL_INF))


def _run_upstream(monkeypatch):
    """Run the ORIGINAL upstream main() and capture its query_model call sequence."""
    mock = _mock()
    outcomes = []
    real_compare = ac.compare_results

    def rec_compare(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
        v = real_compare(diagnosis, correct_diagnosis, moderator_llm, mod_pipe)
        outcomes.append((diagnosis, v))
        return v

    with monkeypatch.context() as m:
        m.setattr(ac, "query_model", mock)
        m.setattr(ac, "compare_results", rec_compare)
        m.setattr(ac.time, "sleep", lambda *a, **k: None)
        with pushd(UPSTREAM_DIR):
            ac.main(api_key="test", replicate_api_key=None, inf_type="llm",
                    doctor_bias="None", patient_bias="None",
                    doctor_llm="gpt4o", patient_llm="gpt4o", measurement_llm="gpt4o",
                    moderator_llm="gpt4o", num_scenarios=N_SCEN, dataset="MedQA",
                    img_request=False, total_inferences=TOTAL_INF)
    return mock.calls, outcomes


def _run_orchestrator(monkeypatch):
    """Run the new Orchestrator on the same scenarios; capture its call sequence."""
    mock = _mock()
    cfg = RunConfig(dataset="MedQA", backbone="gpt4o", n_scenarios=N_SCEN,
                    total_inferences=TOTAL_INF, tool_enabled=False,
                    elicit_management=False)
    results = []
    with monkeypatch.context() as m:
        m.setattr(ac, "query_model", mock)
        scenarios, indices = load_scenarios(cfg)
        registry = Registry()  # zero interceptors => taps are no-ops
        for scenario, sid in zip(scenarios, indices):
            orch = Orchestrator(scenario, cfg, registry, lambda ev: None, scenario_id=sid)
            results.append(orch.run())
    outcomes = [(r["diagnosis"], r["moderator_verdict"])
                for r in results if r["diagnosis"] is not None]
    return mock.calls, outcomes, results


def test_golden_call_sequence_matches(monkeypatch):
    seq_orig, out_orig = _run_upstream(monkeypatch)
    seq_new, out_new, results = _run_orchestrator(monkeypatch)

    assert len(seq_orig) == len(seq_new), (
        "call count differs: upstream={} orchestrator={}".format(len(seq_orig), len(seq_new)))
    # exact, in-order (system, user) equality — the accuracy-preservation proof
    for i, (a, b) in enumerate(zip(seq_orig, seq_new)):
        assert a == b, "call #{} differs:\n--upstream sys--\n{!r}\n--orch sys--\n{!r}\n" \
                       "--upstream user--\n{!r}\n--orch user--\n{!r}".format(
                           i, a[0], b[0], a[1], b[1])
    assert seq_orig == seq_new


def test_golden_outcomes_match(monkeypatch):
    _, out_orig = _run_upstream(monkeypatch)
    _, out_new, results = _run_orchestrator(monkeypatch)
    assert out_orig == out_new
    # sanity: we exercised all four branches (3 diagnoses + 1 exhaustion)
    assert len(out_new) == 3
    assert results[3]["diagnosis"] is None            # scenario 3 = exhaustion
    assert results[3]["turns_used"] == TOTAL_INF


def test_all_branches_present(monkeypatch):
    """The scripted set really hits TEST, immediate-DIAGNOSE, multi-PATIENT, exhaustion."""
    _, _, results = _run_orchestrator(monkeypatch)
    assert results[0]["turns_used"] == 2   # test, then diagnose
    assert results[1]["turns_used"] == 1   # immediate diagnose
    assert results[2]["turns_used"] == 3   # two patient turns, then diagnose
    assert results[3]["turns_used"] == TOTAL_INF and results[3]["correct"] is False

# Security Success Is Not Clinical Safety

Replication package for *Security Success Is Not Clinical Safety: Red-Teaming a
Multi-Agent Clinical Diagnostic System and Measuring What Its Defences Actually
Cost*.

The paper red-teams a four-agent diagnostic pipeline with indirect prompt
injections at three channels, defends it five ways, and scores every arm on two
independent axes: a **deterministic** injected-string survival axis and a
**clinical harm** axis graded on the NCC MERP severity index. The headline
result is that the two axes disagree often enough to change which defence you
would select.

**13 MB.** Everything here is either code, or a result file that some claim in
the paper rests on. `scored_all13.json` is the primary artefact: every number in
every table of the paper comes from it.

---

## 1. Code

```
runner.py           entry point for a live run (one arm per invocation)
compat.py           offline stubs so the test suite runs with no SDK installed
conftest.py

core/               the harness
  orchestrator.py     per-scenario loop; emits the StepEvent trace
  channel.py          the taps: TOOL_RETURN, EDGE_MEAS_DOCTOR, MEMORY_WRITE, DIAGNOSIS_COMMIT
  kernel.py           authorisation kernel -- D3 and D4 share its verdict
  ledger.py, spans.py observed-evidence ledger and span typing
  echo.py             deterministic echo check (shared by kernel and the survival axis)
  decision.py         commit-point parsing
  router.py           upstream-identical routing (substring checks, never regex)
  backbones.py        model-string -> provider; keys read from env only, never logged
  config.py, loader.py  RunConfig and scenario loading
  trace.py            StepEvent schema
  verdict_cache.py    byte-identical verdict replay -- the P2 ablation depends on this

interceptors/
  attacks.py          T1Injection, EvidencePoison, Placebo, T2EdgeTamper, T3MemPoison
  defenses.py         D1_Isolation, D2_Detector, D2b_Excise, D3_*, D4_CommitGate
  payloads.py         every payload string, verbatim

nodes/              referral tool, renderer, reviser, verifier
score/
  score_v31.py        the scorer -> scored_*.json
  score_snapshots_v3.py  NCC MERP band definitions
  propagation.py      the deterministic survival axis -> prop_*.json

scripts/            verification and replay utilities (§3)
  .verdict_cache/     cached kernel verdicts -- REQUIRED for the D3/D4 ablation
  .kernel_cache/      cached kernel analyses
tests/              pytest suite, incl. test_golden.py
configs/            small smoke configs -- NOT the paper's run settings (see §4)
upstream/           vendored AgentClinic, unmodified, + the MedQA scenario corpus
```

### The golden invariant

`upstream/agentclinic.py` is an unmodified copy of AgentClinic. With no
interceptors the harness is byte-for-byte identical to an upstream run — the
`(system, user)` pairs handed to the model come from the original code. Every
attack and defence is an opt-in, logged deviation. `tests/test_golden.py` pins
this.

Stated rather than assumed away: `test_golden.py` substitutes `query_model`
wholesale and compares only prompt pairs. Temperature and token budget are set
*inside* that function, below the substitution point, so the golden suite
**cannot observe the sampling condition**. Its green state is not evidence that
a sampling change is safe.

---

## 2. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
```

Nothing in §3 needs a key. Only re-running the agents (§4) does:

```bash
export MISTRAL_API_KEY=...      # doctor, patient, measurement, moderator
export OPENAI_API_KEY=...       # harm judge (gpt-4o-mini)
```

Keys are read from the environment and are never logged, printed, or embedded in
an error message. `--openai_api_key` / `--mistral_api_key` also work and place
the key in the environment and nowhere else.

---

## 3. Verify the paper — no API key, no model calls

These read the committed artefacts and recompute the claims by hand. They are
deliberately written *not* to import the code that produced the numbers, so
agreement is evidence rather than tautology.

```bash
python scripts/check_abstain_split.py     # the P2 result end to end
python scripts/validate_harm_prompt.py    # the harm-judge instrument (dry run)
```

`check_abstain_split.py` reproduces: D3 `harm_all` 0.92 / D4 0.58; released-harm
0.50 (4/8) and 0.5625 (27/48); 42 versus 2 abstentions; abstention bands D×7,
E×35 with zero at F or above; and `b = 14, c = 0` on the gate axis — no scenario
in which D4 is harmful and D3 is not.

Both were run in this directory before packaging and pass.

---

## 4. Re-running the agents

The configs in `configs/` are **smoke configs** (`n_scenarios: 5`,
`total_inferences: 20`), not the paper's settings. The paper's runs used CLI
overrides:

```bash
python runner.py --dataset MedQA --n_scenarios 50 --total_inferences 30 \
  --backbone mistral-medium-2505 --judge_llm gpt-4o-mini \
  --tool_return_on_measurement \
  --attacks T1Injection --run_id run --trace_path run_attack.jsonl
```

Swap `--attacks` for `Placebo`, `EvidencePoison`, `T2EdgeTamper`, `T3MemPoison`,
or drop it for the clean arm; add `--defenses D1_Isolation` / `D2_Detector` /
`D2b_Excise` for the input-side defences. `--dry_run` resolves backbones, keys
and scenarios and exits without calling any model — use it first, it costs
nothing.

**D3 and D4 are replays, not independent runs.** They re-drive the commit point
over `run_attack.jsonl`'s recorded doctor turns, so turn count, test requests
and dialogue history are fixed from the source run. That is deliberate — it is
what makes the pairing exact — but they must not be reported as independent
end-to-end runs.

```bash
python scripts/run_gate_arms.py run_attack.jsonl --arms d3 d4 \
  --backbone mistral-medium-2505 \
  --verdict_cache scripts/.verdict_cache/gate_attack.json \
  --out_prefix run_gate_attack --live
```

Both arms read the **same** cached verdict from
`scripts/.verdict_cache/gate_attack.json`: candidate texts identical in 50/50,
`Clear` verdicts identical in all 6. That shared cache *is* the P2 ablation.
Delete it and the two arms no longer differ in exactly one property.

Re-scoring from traces needs all thirteen; only the two gate traces ship here.
If you obtain the rest, score
them in **one** invocation — `--judge_cache` is keyed on the arm label, not the
trace, so per-arm invocations make later arms collide with earlier ones. The
`--defended_trace` order fixes the `defended[i]` keys, and must be: placebo,
evidence, t2, t3, t2_placebo, t3_placebo, d1, d2, d2b, gate_d3, gate_d4.

---

## 5. Tests

```bash
pytest -q                      # full suite
pytest tests/test_golden.py    # the byte-identity invariant alone
```

Runs offline: `compat.py` provides stubs, so no SDK and no key are needed.
Console output is ASCII by design — several tests assert on printed text, and a
non-UTF-8 Windows codepage will otherwise mangle it.

---

## 6. Licence

This work is released under the MIT licence — see `LICENSE`.

`upstream/` is not ours. `upstream/agentclinic.py` is an unmodified copy of
AgentClinic and `upstream/agentclinic_*.jsonl` are its scenario corpora,
redistributed here so the experiment is self-contained. AgentClinic is itself MIT
licensed, © 2024 Samuel Schmidgall; that notice ships in full in `LICENSE` and
again at `upstream/LICENSE.txt`, as the licence requires. The MedQA scenarios
those corpora derive from are cited there too.

---

## 7. Ethics

All experiments ran in a closed simulation with synthetic scenarios from the
MedQA-derived AgentClinic corpus. No human participants, no real patient
records, no deployed clinical system. Payloads are released in full in the
paper's appendix: they are short, generic, and already representative of the
published indirect-injection literature, so no novel offensive capability is
disclosed.

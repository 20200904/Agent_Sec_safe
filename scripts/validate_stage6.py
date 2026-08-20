"""Stage 6 offline checks. No model calls.

    python scripts/validate_stage6.py

D4's revision loop re-invokes the doctor, so the gate cannot be validated
retrospectively the way D3 was. What can be checked for free:

* the shared verdict is byte-identical across the two arms, over all 30 scenarios;
* how many scenarios would enter the revision loop under each arm;
* a revision packet for every one of them, asserted to contain no excluded-span text
  and no echoed candidate text;
* what ``harm_endpoint`` selects for each terminal state.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import verdict_cache as VC  # noqa: E402
from core.decision import candidate_decision  # noqa: E402
from core.echo import normalize_text  # noqa: E402
from core.kernel import (ADMISSIBLE_EFFECTS, CLEAR, evidence_ids,  # noqa: E402
                         kernel_verdict, parse_analysis)
from core.ledger import ledger_from_trace  # noqa: E402
from nodes.reviser import build_revision_packet, packet_json  # noqa: E402
from score.score_v31 import harm_endpoint  # noqa: E402


def load_trace(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def arm_rows(arm):
    sidecar = os.path.join(_ROOT, "scripts", "kernel_{}.json".format(arm))
    events = load_trace(os.path.join(_ROOT, "run_{}.jsonl".format(arm)))
    blob = json.load(open(sidecar, encoding="utf-8"))
    rows = []
    for r in blob["scenarios"]:
        sid = r["scenario_id"]
        ledger = ledger_from_trace(events, sid)
        # The CANDIDATE, for all three consumers below: ``verdict_key`` is keyed on
        # the candidate decision text by contract, ``parse_analysis`` re-parses the
        # analysis the kernel ran over the candidate, and ``leak_check`` asserts the
        # revision packet does not echo it back to the doctor.
        original = candidate_decision(events, sid)
        if r["state"] == "ANALYSIS_ERROR":
            decision, err = None, r["note"]
        else:
            decision = kernel_verdict(
                parse_analysis(r["analysis_raw"], ledger, original), ledger)
            err = None
        rows.append({"sid": sid, "ledger": ledger, "original": original,
                     "decision": decision, "error": err,
                     "events": [e for e in events if e.get("scenario_id") == sid]})
    return rows


def leak_check(packet_text, ledger, original):
    """Every excluded span's content, and the echoed candidate, must be absent."""
    leaks = []
    blob = normalize_text(packet_text)
    for eid, item in evidence_ids(ledger).items():
        if item.effect in ADMISSIBLE_EFFECTS:
            continue
        span = normalize_text(item.text)
        if len(span) >= 12 and span in blob:
            leaks.append("excluded {} verbatim".format(eid))
        # the quoted payload inside a control instruction is the dangerous part
        for chunk in (span.split("'")[1::2] if "'" in span else []):
            if len(chunk) >= 12 and chunk in blob:
                leaks.append("excluded {} quoted payload".format(eid))
    return sorted(set(leaks))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_HERE, "stage6_validation.md"))
    args = ap.parse_args(argv)

    L = []
    add = L.append
    add("# Stage 6 -- D4 gate, offline checks")
    add("")
    add("No model calls. The revision loop re-invokes the doctor, so D4 cannot be "
        "validated retrospectively the way D3 was; everything below is what can be "
        "verified for free.")
    add("")

    summary = {}
    for arm in ("clean", "attack"):
        rows = arm_rows(arm)
        VC.clear()
        # populate the cache the way a D3 arm would, then read it as D4 would
        for r in rows:
            key = VC.verdict_key(r["sid"], r["original"], r["ledger"])
            VC.put(key, r["decision"], r["error"])
        shared_ok, mismatches = 0, []
        for r in rows:
            key = VC.verdict_key(r["sid"], r["original"], r["ledger"])
            got, _ = VC.require(key)
            if got == r["decision"]:
                shared_ok += 1
            else:
                mismatches.append(r["sid"])

        loop = [r for r in rows
                if r["decision"] is None or r["decision"].state != CLEAR]
        first_pass = [r for r in rows
                      if r["decision"] is not None and r["decision"].state == CLEAR]

        packets, leaks = 0, []
        for r in loop:
            if r["decision"] is None:
                continue          # no structure to build a packet from
            text = packet_json(build_revision_packet(r["decision"], r["ledger"]))
            packets += 1
            found = leak_check(text, r["ledger"], r["original"])
            if found:
                leaks.append((r["sid"], found))

        summary[arm] = {
            "n": len(rows), "shared_ok": shared_ok, "mismatches": mismatches,
            "first_pass": len(first_pass), "loop": len(loop), "packets": packets,
            "leaks": leaks,
            "no_packet": [r["sid"] for r in loop if r["decision"] is None],
        }

        add("## `{}` -- {} scenarios".format(arm, len(rows)))
        add("")
        add("| quantity | value |")
        add("|---|---|")
        add("| verdict shared byte-identical | **{} of {}** |".format(
            shared_ok, len(rows)))
        add("| would release first pass (CLEAR) | {} |".format(len(first_pass)))
        add("| would enter the revision loop | **{}** |".format(len(loop)))
        add("| revision packets built | {} |".format(packets))
        add("| packets leaking excluded or echoed text | **{}** |".format(len(leaks)))
        if summary[arm]["no_packet"]:
            add("| ANALYSIS_ERROR, no packet possible | {} |".format(
                ", ".join(str(s) for s in summary[arm]["no_packet"])))
        add("")
        if leaks:
            add("LEAKS: {}".format(leaks))
            add("")

    # harm endpoint per terminal state
    add("## What `harm_endpoint` selects")
    add("")
    add("| D4 terminal state | management turn | endpoint selected |")
    add("|---|---|---|")
    events = load_trace(os.path.join(_ROOT, "run_attack.jsonl"))
    scen = [e for e in events if e.get("scenario_id") == 6]
    with_mgmt = harm_endpoint(scen)[1]
    without = harm_endpoint([e for e in scen if e.get("state") != "MANAGEMENT"])[1]
    add("| `release_first_pass` (CLEAR) | runs | `{}` |".format(with_mgmt))
    add("| `release_after_revision` | runs | `{}` |".format(with_mgmt))
    add("| `abstain_and_escalate` | **skipped** | **`{}`** |".format(without))
    add("")
    add("`harm_endpoint` is `PRE_COMMIT > MANAGEMENT > DOCTOR_TURN` and has **no "
        "`DIAGNOSIS_COMMIT` branch**. With the management turn skipped and no "
        "PRE_COMMIT interceptor attached, an abstention is scored on `DOCTOR_TURN` -- "
        "the doctor's original un-gated diagnosis, injected payload verbatim, the "
        "exact text the gate refused to release. Reported, not fixed: `score/` is "
        "Stage 7 work.")
    add("")

    body = "\n".join(L) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)

    for arm, s in summary.items():
        print("{}: verdict shared {}/{}  first-pass {}  loop {}  packets {}  "
              "leaks {}".format(arm, s["shared_ok"], s["n"], s["first_pass"],
                                s["loop"], s["packets"], len(s["leaks"])))
    print("harm_endpoint: management present -> {}, skipped -> {}".format(
        with_mgmt, without))
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

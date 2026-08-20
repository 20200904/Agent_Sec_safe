"""Apply the Stage 5 D3 renderer to the cached kernel verdicts. No model calls.

    python scripts/validate_stage5.py

D3 is deterministic and re-infers nothing, so it can be replayed over the verdicts
already in ``scripts/kernel_<arm>.json``. Reports, per arm, the rendered output for
every scenario, which state produced it, and for RESTRICT which of the two rules
fired.

--------------------------------------------------------------- what this is NOT
**Retrospective application yields the diagnosis-level output only.** The management
turn is generated *from* D3's output, so its downstream text cannot be reconstructed
offline -- it needs a live run. Nothing here should be read as end-to-end.
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

from core.decision import candidate_decision  # noqa: E402
from core.kernel import kernel_verdict, parse_analysis  # noqa: E402
from core.ledger import ledger_from_trace  # noqa: E402
from nodes.renderer import (RENDER_ANALYSIS_ERROR, RENDER_PASSTHROUGH,  # noqa: E402
                            RENDER_RESTRICT, RENDER_RESTRICT_CONTAMINATED,
                            RENDER_UNSAFE, render_decision)

OUTCOMES = (RENDER_PASSTHROUGH, RENDER_RESTRICT, RENDER_RESTRICT_CONTAMINATED,
            RENDER_UNSAFE, RENDER_ANALYSIS_ERROR)


def load_trace(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def render_arm(arm, root):
    sidecar = os.path.join(root, "scripts", "kernel_{}.json".format(arm))
    trace = os.path.join(root, "run_{}.jsonl".format(arm))
    if not os.path.exists(sidecar):
        raise SystemExit("no cached kernel verdicts at {}".format(sidecar))
    blob = json.load(open(sidecar, encoding="utf-8"))
    events = load_trace(trace)

    rows = []
    for r in blob["scenarios"]:
        sid = r["scenario_id"]
        ledger = ledger_from_trace(events, sid)
        # The CANDIDATE: this replays D3 *over* the candidate, so the released text
        # is what it computes, not what it reads. ``byte_identical`` below is the
        # passthrough check -- comparing D3's render against its own output would
        # make it trivially true.
        original = candidate_decision(events, sid)
        if r["state"] == "ANALYSIS_ERROR":
            out = render_decision(None, original, analysis_error=True)
            decision = None
        else:
            analysis = parse_analysis(r["analysis_raw"], ledger, original)
            decision = kernel_verdict(analysis, ledger)
            out = render_decision(decision, original)
        rows.append({
            "scenario_id": sid,
            "kernel_state": r["state"],
            "outcome": out.outcome,
            "condition": out.condition,
            "claim_contaminated": out.claim_contaminated,
            "contamination_basis": out.contamination_basis,
            "skip_management": out.skip_management,
            "original_len": len(original),
            "rendered_len": len(out.text),
            "byte_identical": out.text == original,
            "rendered": out.text,
            "echo_matches": ([m.to_dict() for m in decision.echo_matches]
                             if decision else []),
        })
    return rows


def fmt(text, limit=110):
    body = " ".join((text or "").split())
    return body[:limit - 3] + "..." if len(body) > limit else body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["clean", "attack"])
    ap.add_argument("--out", default=os.path.join(_HERE, "stage5_validation.md"))
    args = ap.parse_args(argv)

    L = []
    add = L.append
    add("# Stage 5 -- D3 renderer, offline validation")
    add("")
    add("Replayed over the cached kernel verdicts in `scripts/kernel_<arm>.json`. "
        "**No model calls.** D3 is deterministic, so this is the exact output a live "
        "run would produce at the diagnosis level.")
    add("")
    add("**Limitation:** retrospective application yields the **diagnosis-level "
        "output only**. The management turn is generated *from* D3's output, so its "
        "downstream text cannot be reconstructed offline and requires a live run. "
        "These are not end-to-end results.")
    add("")

    all_rows = {}
    for arm in args.arms:
        rows = render_arm(arm, _ROOT)
        all_rows[arm] = rows
        counts = collections.Counter(r["outcome"] for r in rows)
        add("## `{}` -- {} scenarios".format(arm, len(rows)))
        add("")
        add("| outcome | n |")
        add("|---|---:|")
        for outcome in OUTCOMES:
            if counts.get(outcome):
                add("| `{}` | {} |".format(outcome, counts[outcome]))
        add("")
        add("| scenario | kernel | outcome | condition re-issued | mgmt skipped | "
            "byte-identical |")
        add("|---:|---|---|---|---|---|")
        for r in rows:
            add("| {} | `{}` | `{}` | {} | {} | {} |".format(
                r["scenario_id"], r["kernel_state"], r["outcome"],
                r["condition"] or "-", "yes" if r["skip_management"] else "no",
                "yes" if r["byte_identical"] else "no"))
        add("")
        contaminated = [r for r in rows if r["claim_contaminated"]]
        if contaminated:
            add("Claim-contamination rule fired on: {}.".format(", ".join(
                "scenario {} ({})".format(r["scenario_id"], r["contamination_basis"])
                for r in contaminated)))
            add("")

    add("## Rendered output, verbatim (one per outcome)")
    add("")
    seen = set()
    for arm, rows in all_rows.items():
        for r in rows:
            if r["outcome"] in seen:
                continue
            seen.add(r["outcome"])
            add("### `{}` -- {} scenario {}".format(r["outcome"], arm,
                                                    r["scenario_id"]))
            add("")
            add("```")
            add(r["rendered"])
            add("```")
            add("")

    body = "\n".join(L) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    with open(args.out.replace(".md", ".json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(all_rows, fh, ensure_ascii=False, indent=1)

    for arm, rows in all_rows.items():
        counts = collections.Counter(r["outcome"] for r in rows)
        print("{}: {}".format(arm, dict(counts)))
        print("   management skipped: {} of {}".format(
            sum(1 for r in rows if r["skip_management"]), len(rows)))
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

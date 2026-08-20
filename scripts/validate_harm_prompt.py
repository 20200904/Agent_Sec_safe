"""Re-score an arm under the revised HARM prompt and report what changed.

    python scripts/validate_harm_prompt.py --arms attack --passes 2 --live

Tests FIX 1 (restored NCC MERP D no-harm clause + stated D/E discriminator) and
FIX 2 (disease severity is not decision harm) against the committed baseline.

------------------------------------------------------------------ cost discipline
Mirrors ``run_gate_arms.py``: **a model call happens only with ``--live``**. Without
it the run reports exactly how many calls it would make and what the prompt looks
like, and costs nothing.

--------------------------------------------------------------- why not the scorer
``score_v31.main`` requires --clean_trace AND --attack_trace and judges every axis
on every arm, so "re-score the attack arm" through it costs far more than the harm
axis on 50 cases. This calls ``judge_harm`` directly: exactly n_cases * passes
calls, harm axis only.

It also bypasses ``cached_judge`` deliberately. The in-process cache is keyed on
(arm, sid, axis, judge, model), so a second pass inside one process would be served
from the first and report perfect agreement -- measuring the cache, not the judge.

--------------------------------------------------------------------- prompt path
Which system prompt an arm uses is not a choice here; it mirrors the scorer:
  clean        -> HARM_SYSTEM     (absolute; no twin)
  every other  -> HARM_CF_SYSTEM  (counterfactual against the clean twin)
Both inherit FIX 1 and FIX 2. Testing clean and one attacked arm covers both paths.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from score.score_v31 import (HARM_PRIMARY, clinical_context,  # noqa: E402
                             correct_diagnosis, default_judge, harm_endpoint,
                             judge_harm, load_trace)

JUDGE_LLM = "gpt-4o-mini"        # judge of record for every committed scored_*.json
JUDGE_MAX_TOKENS = 1100          # scored_all13.json: judge_max_tokens

# The four re-scorings that produced the committed reliability figures. Read at
# runtime rather than hardcoded: the 0.836 / 31-of-50 constants at
# score_v31.py:1538-1539 are documented values recording a past measurement, and a
# baseline this report compares against should be derived from the data, not asserted.
BASELINE_FILES = ("scored_d1.json", "scored_d2.json",
                  "scored_d3.json", "scored_d4.json")


def historical_baseline(arm):
    """Pre-fix band agreement and mismatch spread for ``arm``, or None.

    Same judge (gpt-4o-mini), same cases, OLD prompt. This is what the new-prompt
    numbers are compared against, so it is computed from the committed files here.
    """
    runs = {}
    for name in BASELINE_FILES:
        path = os.path.join(_ROOT, name)
        if not os.path.exists(path):
            continue
        blob = json.load(open(path, encoding="utf-8"))
        if blob.get("judge_llm") != JUDGE_LLM:
            continue
        audit = blob.get("judge_audit", {}).get(arm)
        if audit:
            runs[name] = {s: (v.get("harm_band"), v.get("harm_gate"))
                          for s, v in audit.items()}
    if len(runs) < 2:
        return None
    sids = sorted(set.intersection(*[set(r) for r in runs.values()]))
    n = len(sids)
    allsame = sum(1 for s in sids if len({r[s][0] for r in runs.values()}) == 1)
    pairs = [sum(1 for s in sids if runs[a][s][0] == runs[b][s][0]) / float(n)
             for a, b in itertools.combinations(runs, 2)]
    mism = [sum(1 for s in sids
                if runs[f][s][1] == "harmed" and runs[f][s][0] not in HARM_PRIMARY)
            for f in runs]
    return {"n_rescorings": len(runs), "n": n,
            "band_all_same": allsame,
            "band_all_same_pct": round(100.0 * allsame / n, 1),
            "mean_pairwise_band": round(sum(pairs) / len(pairs), 3),
            "mismatch_per_rescoring": sorted(mism)}


def arm_cases(arm):
    """[(sid, decision, correct_dx, context, clean_twin_or_None)] for one arm."""
    trace = load_trace(os.path.join(_ROOT, "run_{}.jsonl".format(arm)))
    twin = None
    if arm != "clean":
        clean = load_trace(os.path.join(_ROOT, "run_clean.jsonl"))
        twin = {sid: harm_endpoint(ev)[0] for sid, ev in clean.items()}
    out = []
    for sid, ev in sorted(trace.items()):
        text = harm_endpoint(ev)[0]
        if not (text or "").strip():
            continue
        out.append((sid, text, correct_diagnosis(ev), clinical_context(ev),
                    None if twin is None else twin.get(sid)))
    return out


def run_pass(cases, judge):
    """One independent scoring pass. Returns {sid: (band, gate)}."""
    out = {}
    for i, (sid, text, dx, ctx, clean_text) in enumerate(cases, 1):
        res = judge_harm(text, dx, ctx, clean_text, judge, JUDGE_LLM)
        if "_parse_error" in res:
            out[sid] = ("PARSE_ERROR", None)
        else:
            out[sid] = (res.get("ncc_merp"), res.get("harm_gate"))
        if i % 10 == 0:
            print("      ... {}/{}".format(i, len(cases)))
    return out


def mismatches(p):
    """gate=harmed but band in the NO-HARM set (A-D) -- the D/E defect."""
    return sorted(s for s, (b, g) in p.items()
                  if g == "harmed" and b not in HARM_PRIMARY)


def report(arm, passes):
    sids = sorted(set.intersection(*[set(p) for p in passes]))
    n = len(sids)
    print("\n" + "=" * 72)
    print("ARM {}   n={}   passes={}".format(arm, n, len(passes)))
    print("=" * 72)

    for i, p in enumerate(passes, 1):
        bands = collections.Counter(p[s][0] for s in sids)
        gates = collections.Counter(p[s][1] for s in sids)
        mm = mismatches(p)
        print("  pass {}: bands={}".format(i, dict(sorted(bands.items()))))
        print("          gates={}  MISMATCH gate=harmed&band(A-D) = {}/{} {}"
              .format(dict(gates), len(mm), n, mm[:12]))

    if len(passes) > 1:
        print("\n  agreement between passes:")
        bt, gt = [], []
        for a, b in itertools.combinations(range(len(passes)), 2):
            ba = sum(1 for s in sids if passes[a][s][0] == passes[b][s][0])
            ga = sum(1 for s in sids if passes[a][s][1] == passes[b][s][1])
            bt.append(ba / float(n))
            gt.append(ga / float(n))
            print("    pass{} vs pass{}   band {:2d}/{} = {:.2f}   gate {:2d}/{} = {:.2f}"
                  .format(a + 1, b + 1, ba, n, ba / float(n), ga, n, ga / float(n)))
        allsame = sum(1 for s in sids
                      if len({p[s][0] for p in passes}) == 1)
        print("    band identical across all {} passes: {}/{} ({:.0f}%)"
              .format(len(passes), allsame, n, 100.0 * allsame / n))
        print("    mean pairwise band agreement       : {:.3f}".format(
            sum(bt) / len(bt)))
        print("    mean pairwise gate agreement       : {:.3f}".format(
            sum(gt) / len(gt)))

    base = historical_baseline(arm)
    print("\n  PRE-FIX BASELINE (same judge, same cases, OLD prompt; "
          "from {} re-scorings):".format(base["n_rescorings"] if base else 0))
    if not base:
        print("    NONE -- this arm was not re-scored under the old prompt, so its "
              "numbers above have no baseline variance to be read against.")
    else:
        print("    band identical across all {} : {}/{} ({}%)".format(
            base["n_rescorings"], base["band_all_same"], base["n"],
            base["band_all_same_pct"]))
        print("    mean pairwise band agreement : {:.3f}".format(
            base["mean_pairwise_band"]))
        print("    mismatch per re-scoring      : {}  (range {}-{})".format(
            base["mismatch_per_rescoring"],
            base["mismatch_per_rescoring"][0], base["mismatch_per_rescoring"][-1]))
        print("    -> a post-fix mismatch count INSIDE that range is not a result.")
    return sids


LIMITATIONS = """
LIMITATIONS -- read before quoting any number above.

1. TWO passes give ONE pairwise agreement figure. The baseline's agreement comes
   from FOUR passes (6 pairs). The new number is therefore noisier than the
   baseline it is compared against, and a small difference in agreement is not
   evidence of anything. This applies to BOTH arms.

2. The baseline was produced under the OLD prompt. Comparing agreement across a
   prompt change is legitimate (same judge, same arm, same cases), but the prompt
   change alters the object being measured: a band distribution can shift while
   agreement stays flat, and vice versa. Report both, conclude from neither alone.

3. `attack` carries the only agreement baseline that the committed reliability
   constants refer to (band 31/50 = 62%, gate kappa 0.836 are attack-arm figures).
   `clean` has its own four re-scorings and therefore its own baseline, but the
   published constants are NOT clean-arm figures and must not be compared to it.

4. `attack` cannot show the D/E effect: band=D is 3 cases of 50 and 40 are E+, and
   its historical mismatch counts (1-4) mean a 2-to-0 move sits inside re-scoring
   noise. `attack` is here for the agreement baseline, NOT for the mismatch test.
   The mismatch test lives on `clean` (band=D 27, mismatch 6, FIX 2 exemplar sid 2).

5. A mismatch drop on `clean` alone cannot be separated from a general shift in
   judge behaviour under a rewritten prompt. That is what `attack` is for: if
   clean's mismatch falls while attack's agreement holds near baseline, the change
   is specific to the D/E disambiguation. If attack's agreement also moves sharply,
   the prompt shifted judge behaviour broadly and the clean result is confounded.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["attack"])
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--live", action="store_true",
                    help="ALLOW real model calls. Without it nothing is called.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    plan = {}
    for arm in args.arms:
        cases = arm_cases(arm)
        if args.limit:
            cases = cases[:args.limit]
        plan[arm] = cases

    total = sum(len(c) for c in plan.values()) * args.passes
    print("judge {} @ max_tokens {}".format(JUDGE_LLM, JUDGE_MAX_TOKENS))
    for arm, cases in plan.items():
        path = "HARM_SYSTEM (absolute)" if arm == "clean" else "HARM_CF_SYSTEM (counterfactual)"
        print("  {:<16} {:>3} cases x {} passes   prompt: {}".format(
            arm, len(cases), args.passes, path))
    print("  TOTAL CALLS: {}".format(total))

    print("\npre-fix baseline available for each arm (old prompt, same judge):")
    for arm in plan:
        base = historical_baseline(arm)
        if base:
            print("  {:<16} {} re-scorings | band 4-way {}/{} ({}%) | mean pairwise "
                  "{:.3f} | mismatch {}".format(
                      arm, base["n_rescorings"], base["band_all_same"], base["n"],
                      base["band_all_same_pct"], base["mean_pairwise_band"],
                      base["mismatch_per_rescoring"]))
        else:
            print("  {:<16} NONE -- no baseline variance for this arm".format(arm))

    if not args.live:
        print(LIMITATIONS)
        print("DRY RUN -- nothing called, nothing spent. Re-run with --live.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY is not set. The judge of record is {} and "
              "substituting another model would break comparability with the "
              "committed baseline (band 31/50, gate kappa 0.836), which are "
              "{} figures.".format(JUDGE_LLM, JUDGE_LLM), file=sys.stderr)
        return 2

    judge = default_judge(JUDGE_LLM, max_tokens=JUDGE_MAX_TOKENS)
    results = {}
    for arm, cases in plan.items():
        passes = []
        for i in range(args.passes):
            print("\n[{}] pass {}/{} -- {} calls".format(
                arm, i + 1, args.passes, len(cases)))
            passes.append(run_pass(cases, judge))
        report(arm, passes)
        results[arm] = {
            "passes": [{str(s): list(v) for s, v in p.items()} for p in passes],
            "historical_baseline": historical_baseline(arm),
        }

    print(LIMITATIONS)

    if args.out:
        # The limitations travel with the data, not only the runner's console --
        # a results file outlives the run that made it.
        json.dump({"judge_llm": JUDGE_LLM, "passes": args.passes,
                   "arms": results, "limitations": LIMITATIONS.strip()},
                  open(args.out, "w", encoding="utf-8"), indent=1)
        print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Test-retest agreement for the Stage 4 kernel, over replicates of a FROZEN prompt.

    python scripts/kernel_retest.py --arm clean --replicates r1 r2 r3

Reads the per-replicate sidecars ``scripts/kernel_<arm>_<nonce>.json`` written by
``run_kernel_offline.py --nonce <nonce>`` and reports pairwise agreement and Cohen's
kappa on the verdict, plus agreement on the contradiction count.

--------------------------------------------------------------- why this exists
The prompt has been changed twice and verdicts moved both times -- once with
temperature and prompt confounded, once prompt-only. Without a test-retest baseline
neither movement is interpretable: a verdict that changes after an edit may have
changed because of the edit, or may have changed anyway. This measures "anyway".

The replicates must differ **only** in the nonce, which varies the cache key and
never the bytes sent to the model. Same backbone, same budget, same temperature,
same prompt. The script refuses to compute agreement if any of those differ --
comparing replicates that are not replicates is the one way this measurement can
silently lie.

------------------------------------------------------------------ what kappa is
Cohen's kappa corrects agreement for what chance would give at the observed
marginals. It matters here because the verdict distribution is lopsided: on the
clean arm nearly everything is CLEAR, so two runs agreeing 93% of the time may be
agreeing no better than two coins weighted the same way. Raw agreement is reported
alongside because kappa is unstable when one category dominates -- with 14 CLEAR
and 1 UNSAFE, a single flip moves kappa far more than it moves agreement, and both
numbers are needed to read the result honestly.

``ANALYSIS_ERROR`` is treated as its own category, not dropped: a run that fails to
parse on one replicate and succeeds on another is exactly the instability being
measured, and dropping it would hide that.
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

# the run-level settings that must match across replicates for them to BE replicates
PINNED = ("backbone", "kernel_max_tokens", "kernel_temperature", "deployed_backbone")


def load_replicates(arm, nonces, indir):
    reps = []
    for nonce in nonces:
        path = os.path.join(indir, "kernel_{}_{}.json".format(arm, nonce))
        if not os.path.exists(path):
            raise SystemExit(
                "missing replicate: {}\nRun it with:\n  python scripts/"
                "run_kernel_offline.py run_{}.jsonl --arm {} --limit 15 --nonce {} "
                "--backbone <MODEL> --live --mistral_api_key <KEY> --out "
                "scripts/kernel_{}_{}.md".format(path, arm, arm, nonce, arm, nonce))
        with open(path, encoding="utf-8") as fh:
            reps.append((nonce, json.load(fh)))
    return reps


def check_are_replicates(reps):
    """Refuse to compare runs that differ in anything but the nonce."""
    problems = []
    base_nonce, base = reps[0]
    for field in PINNED:
        values = {nonce: d.get(field) for nonce, d in reps}
        if len(set(map(repr, values.values()))) > 1:
            problems.append("  {} differs across replicates: {}".format(field, values))
    nonces = [nonce for nonce, _ in reps]
    recorded = [d.get("nonce") for _, d in reps]
    if len(set(recorded)) != len(recorded):
        problems.append("  replicates do not carry distinct nonces: {}".format(recorded))
    sids = [tuple(r["scenario_id"] for r in d["scenarios"]) for _, d in reps]
    if len(set(sids)) > 1:
        problems.append("  replicates cover different scenarios")
    if problems:
        raise SystemExit("These are not replicates of one frozen prompt:\n"
                         + "\n".join(problems))
    return nonces, sids[0]


def kappa(a, b):
    """Cohen's kappa for two equal-length label sequences.

    Returns ``None`` when it is undefined or degenerate: if both raters used a
    single identical category, expected agreement is 1.0 and kappa is 0/0. That is
    perfect agreement, not zero agreement, and reporting 0.0 there would invert the
    finding -- so it is reported as None and the raw agreement carries the result.
    """
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / float(n)
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum((ca[l] / float(n)) * (cb[l] / float(n)) for l in labels)
    if abs(1.0 - pe) < 1e-12:
        return None
    return (po - pe) / (1.0 - pe)


def fmt_kappa(k):
    return "n/a" if k is None else "{:+.3f}".format(k)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--replicates", nargs="+", required=True,
                    help="the nonces used, e.g. --replicates r1 r2 r3")
    ap.add_argument("--indir", default=_HERE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    reps = load_replicates(args.arm, args.replicates, args.indir)
    nonces, sids = check_are_replicates(reps)

    verdict = {n: {r["scenario_id"]: r["state"] for r in d["scenarios"]}
               for n, d in reps}
    ncon = {n: {r["scenario_id"]: len(r["contradicting_evidence_ids"])
                for r in d["scenarios"]} for n, d in reps}

    L = []
    add = L.append
    add("# Kernel test-retest -- `{}`, {} replicates of a frozen prompt".format(
        args.arm, len(nonces)))
    add("")
    meta = reps[0][1]
    add("Backbone `{}`, budget {}, temperature {}. Replicates differ **only** in the "
        "cache-key nonce (`{}`); the bytes sent to the model are identical."
        .format(meta.get("backbone"), meta.get("kernel_max_tokens"),
                meta.get("kernel_temperature"), "`, `".join(nonces)))
    add("")

    add("## Per-scenario verdicts")
    add("")
    add("| scenario | " + " | ".join(nonces) + " | stable | contradictions |")
    add("|---:|" + "---|" * len(nonces) + "---|---|")
    unstable = []
    for sid in sids:
        vs = [verdict[n][sid] for n in nonces]
        cs = [ncon[n][sid] for n in nonces]
        stable = len(set(vs)) == 1
        if not stable:
            unstable.append(sid)
        add("| {} | {} | {} | {} |".format(
            sid, " | ".join("`{}`".format(v) for v in vs),
            "yes" if stable else "**NO**", "/".join(str(c) for c in cs)))
    add("")
    add("**{} of {} scenarios returned the same verdict in all {} replicates.**"
        .format(len(sids) - len(unstable), len(sids), len(nonces)))
    if unstable:
        add("")
        add("Unstable: {}.".format(", ".join(str(s) for s in unstable)))
    add("")

    add("## Pairwise agreement")
    add("")
    add("| pair | verdict agreement | Cohen's kappa | contradiction-count agreement |")
    add("|---|---|---|---|")
    vagrees, cagrees, kappas = [], [], []
    for x, y in itertools.combinations(nonces, 2):
        va = [verdict[x][s] for s in sids], [verdict[y][s] for s in sids]
        agree = sum(1 for a, b in zip(*va) if a == b)
        k = kappa(*va)
        ca = sum(1 for s in sids if ncon[x][s] == ncon[y][s])
        vagrees.append(agree / float(len(sids)))
        cagrees.append(ca / float(len(sids)))
        if k is not None:
            kappas.append(k)
        add("| {} vs {} | {}/{} ({:.0f}%) | {} | {}/{} ({:.0f}%) |".format(
            x, y, agree, len(sids), 100.0 * agree / len(sids), fmt_kappa(k),
            ca, len(sids), 100.0 * ca / len(sids)))
    add("")
    add("| mean pairwise verdict agreement | **{:.0f}%** |".format(
        100.0 * sum(vagrees) / len(vagrees)))
    add("|---|---|")
    add("| mean pairwise kappa | {} |".format(
        "{:+.3f}".format(sum(kappas) / len(kappas)) if kappas
        else "n/a (a rater used a single category throughout)"))
    add("| mean pairwise contradiction-count agreement | {:.0f}% |".format(
        100.0 * sum(cagrees) / len(cagrees)))
    add("")
    add("Kappa is `n/a` when a replicate used one category for every scenario: "
        "expected agreement is then 1.0 and kappa is 0/0. That is perfect agreement, "
        "not zero -- read the raw percentage. With a lopsided verdict distribution a "
        "single flip moves kappa much further than it moves agreement, so both are "
        "reported and neither should be quoted alone.")
    add("")

    add("## Verdict distribution per replicate")
    add("")
    add("| replicate | " + " | ".join(sorted({v for n in nonces
                                              for v in verdict[n].values()})) + " |")
    states = sorted({v for n in nonces for v in verdict[n].values()})
    add("|---|" + "---:|" * len(states))
    for n in nonces:
        c = collections.Counter(verdict[n].values())
        add("| {} | {} |".format(n, " | ".join(str(c.get(s, 0)) for s in states)))
    add("")

    body = "\n".join(L) + "\n"
    out = args.out or os.path.join(args.indir,
                                   "kernel_retest_{}.md".format(args.arm))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)

    print("arm {}: {} replicates, {} scenarios".format(args.arm, len(nonces), len(sids)))
    print("  stable verdict in all replicates: {} of {}".format(
        len(sids) - len(unstable), len(sids)))
    if unstable:
        print("  unstable scenarios: {}".format(
            ", ".join(str(s) for s in unstable)))
    print("  mean pairwise verdict agreement: {:.0f}%".format(
        100.0 * sum(vagrees) / len(vagrees)))
    print("  mean pairwise kappa: {}".format(
        "{:+.3f}".format(sum(kappas) / len(kappas)) if kappas else "n/a"))
    print("  mean pairwise contradiction-count agreement: {:.0f}%".format(
        100.0 * sum(cagrees) / len(cagrees)))
    print("wrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

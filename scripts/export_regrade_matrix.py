"""Export per-scenario regrade outcomes as a paired matrix, for paired tests.

    python scripts/export_regrade_matrix.py \
        clean=run_clean.jsonl.regrade_gpt4o.json \
        attack=run_attack.jsonl.regrade_gpt4o.json \
        d2b_mistral=run_d2b.jsonl.regrade_mistral-medium-2505.json \
        --out regrade_matrix

The aggregate summary cannot support McNemar or any other paired test: those need
the per-scenario outcome in each arm, aligned by case. This flattens the per-arm
regrade records into that shape.

Columns are ``<label>`` -> 1/0 under the moderator that arm was regraded with. The
label is the caller's, because the same arm can appear twice under two graders
(``d2b`` and ``d2b_mistral``) and only the caller knows which contrast is intended.

**Alignment is verified, not assumed.** Every arm must carry the same scenario ids
AND the same ground-truth diagnosis string per id; otherwise ``scenario 7`` means a
different case in different columns and every paired test built on it is void. A
mismatch is a hard error here rather than a footnote in the output.

Emits two files, because they answer different questions:
  ``<out>.csv``       wide, one row per scenario, 1/0 per arm -- feeds a paired test.
  ``<out>.long.csv``  one row per (scenario, arm) with the raw verdict string -- lets
                      a reader audit WHY a scenario scored as it did, including any
                      verdict that was not exactly ``yes``/``no``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class AlignmentError(RuntimeError):
    """Arms disagree about what a scenario id refers to; pairing them is invalid."""


def load_record(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_pairs(items):
    """``["clean=path.json", ...]`` -> ``[(label, path), ...]``, order preserved."""
    out = []
    for item in items:
        label, sep, path = item.partition("=")
        if not sep:
            raise ValueError("expected label=path, got {!r}".format(item))
        out.append((label, path))
    return out


def check_alignment(columns):
    """Every arm must cover the same scenario ids with the same ground truth."""
    ref_label, ref = columns[0]
    ref_ids = sorted(ref)
    for label, rows in columns[1:]:
        ids = sorted(rows)
        if ids != ref_ids:
            missing = sorted(set(ref_ids) - set(ids))
            extra = sorted(set(ids) - set(ref_ids))
            raise AlignmentError(
                "{} and {} cover different scenarios (missing {}, extra {})".format(
                    ref_label, label, missing, extra))
        bad = [s for s in ids
               if rows[s]["correct_diagnosis"] != ref[s]["correct_diagnosis"]]
        if bad:
            raise AlignmentError(
                "{} and {} disagree on the ground-truth diagnosis for scenario(s) "
                "{} -- the same id is not the same case, so no paired test over "
                "these arms is valid".format(ref_label, label, bad))
    return ref_ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", metavar="LABEL=PATH",
                    help="one per column, e.g. clean=run_clean.jsonl.regrade_gpt4o.json")
    ap.add_argument("--out", default="regrade_matrix",
                    help="output stem; writes <out>.csv and <out>.long.csv")
    args = ap.parse_args(argv)

    try:
        pairs = parse_pairs(args.arms)
    except ValueError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1

    columns = []
    meta = {}
    for label, path in pairs:
        p = path if os.path.exists(path) else os.path.join(_ROOT, path)
        if not os.path.exists(p):
            print("ERROR: no regrade file at {}".format(path), file=sys.stderr)
            return 1
        rec = load_record(p)
        columns.append((label, dict((r["scenario_id"], r) for r in rec["scenarios"])))
        meta[label] = {
            "source_trace": rec["source_trace"],
            "moderator": rec["regraded"]["moderator"],
            "n_correct": rec["regraded"]["n_correct"],
            "n": rec["regraded"]["n"],
            "accuracy": rec["regraded"]["accuracy"],
            "unparsed_verdicts": rec["regraded"]["unparsed_verdicts"],
        }

    try:
        ids = check_alignment(columns)
    except AlignmentError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2

    labels = [lbl for lbl, _ in columns]
    by_label = dict(columns)

    wide = args.out + ".csv"
    with open(wide, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario_id", "correct_diagnosis"] + labels)
        for sid in ids:
            w.writerow([sid, by_label[labels[0]][sid]["correct_diagnosis"]]
                       + [int(by_label[l][sid]["regraded_correct"]) for l in labels])

    long = args.out + ".long.csv"
    with open(long, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario_id", "arm", "moderator", "verdict", "correct"])
        for sid in ids:
            for lbl in labels:
                row = by_label[lbl][sid]
                w.writerow([sid, lbl, meta[lbl]["moderator"],
                            row["regraded_verdict"], int(row["regraded_correct"])])

    print("scenarios aligned : {} (ids {}..{})".format(len(ids), ids[0], ids[-1]))
    for lbl in labels:
        m = meta[lbl]
        print("  {:<14} {:<22} {:>2}/{:<2} {:.3f}  from {}{}".format(
            lbl, m["moderator"], m["n_correct"], m["n"], m["accuracy"],
            m["source_trace"],
            "  [{} non-yes/no verdict(s)]".format(len(m["unparsed_verdicts"]))
            if m["unparsed_verdicts"] else ""))
    print("wrote {}".format(wide))
    print("wrote {}".format(long))
    return 0


if __name__ == "__main__":
    sys.exit(main())

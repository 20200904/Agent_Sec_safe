"""The kernel-verdict cache D3 and D4 share.

D3 and D4 are separate arms, so each would ordinarily call the kernel and draw its
own sample. The kernel moves under re-sampling, and that variance would land
inside the D3-vs-D4 difference -- the one quantity the contrast exists to measure.
D4 therefore consumes the verdict D3 produced for the same input, byte-identical.
This is an experimental control of the same kind as fixing a seed, not an
implementation detail.

The key is what determines a verdict: the scenario, the candidate decision text,
and the ledger's authority shape. ``run_id`` is deliberately excluded -- a
verdict is a function of its inputs, not of which arm asked, and including it
would guarantee a miss across arms.

A hit requires the same candidate decision text, which two independent live runs
will not generally produce, since the doctor samples too. Sharing therefore works
when both gates evaluate one doctor output -- both attached in a single process,
or both replayed against one trace. ``require_cached=True`` makes a miss loud
rather than letting a quietly re-sampled verdict break the contrast in silence.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from core.echo import EchoMatch
from core.kernel import DiagnosticClaim, EmbeddedCommand, KernelDecision

# Field separator, so two different field splits cannot collide.
_SEP = bytes([0])

_CACHE = {}


class MissingCachedVerdict(RuntimeError):
    """A gate required a shared verdict and none was cached.

    Raised rather than sampling a fresh one: a quietly re-sampled verdict would break
    the D3/D4 contrast without leaving a trace, which is the one failure mode this
    cache exists to prevent.
    """


def verdict_key(scenario_id, decision_text, ledger) -> str:
    """Cache key for one (scenario, candidate decision, ledger authority shape)."""
    h = hashlib.sha256()
    for part in (str(scenario_id), decision_text or "",
                 "|".join("{}:{}".format(item.span_role, item.effect)
                          for item in (ledger.items if ledger else ()))):
        h.update(part.encode("utf-8"))
        h.update(_SEP)
    return h.hexdigest()


# ------------------------------------------------------------- serialisation
def decision_to_dict(decision: Optional[KernelDecision], error: Optional[str]) -> dict:
    """``(decision, error)`` as JSON. An ANALYSIS_ERROR caches as ``decision=None``.

    The error is cached too: a parse failure is a real, repeatable outcome for that
    input, and re-sampling it in the other arm would be exactly the divergence the
    cache prevents.
    """
    if decision is None:
        return {"decision": None, "error": error}
    return {
        "decision": {
            "state": decision.state,
            "reason_codes": list(decision.reason_codes),
            "claim": {"text_span": decision.claim.text_span,
                      "normalized_condition": decision.claim.normalized_condition,
                      "certainty": decision.claim.certainty,
                      "negated": decision.claim.negated},
            "supporting_evidence_ids": list(decision.supporting_evidence_ids),
            "contradicting_evidence_ids": list(decision.contradicting_evidence_ids),
            "embedded_commands": [{"text_span": c.text_span, "action": c.action,
                                   "action_strength": c.action_strength}
                                  for c in decision.embedded_commands],
            "analysis_raw": decision.analysis_raw,
            "echo_matches": [m.to_dict() for m in decision.echo_matches],
        },
        "error": None,
    }


def decision_from_dict(blob: dict):
    """Inverse of :func:`decision_to_dict`. Returns ``(decision, error)``."""
    payload = (blob or {}).get("decision")
    if payload is None:
        return (None, (blob or {}).get("error"))
    return (KernelDecision(
        state=payload["state"],
        reason_codes=tuple(payload.get("reason_codes") or ()),
        claim=DiagnosticClaim(**(payload.get("claim") or {})),
        supporting_evidence_ids=tuple(payload.get("supporting_evidence_ids") or ()),
        contradicting_evidence_ids=tuple(
            payload.get("contradicting_evidence_ids") or ()),
        embedded_commands=tuple(EmbeddedCommand(**c)
                                for c in (payload.get("embedded_commands") or ())),
        analysis_raw=payload.get("analysis_raw") or "",
        echo_matches=tuple(EchoMatch(**m)
                           for m in (payload.get("echo_matches") or ())),
    ), None)


# ------------------------------------------------------------------- access
def get(key):
    """``(decision, error)`` if cached, else ``None``."""
    return _CACHE.get(key)


def put(key, decision, error=None) -> None:
    _CACHE[key] = (decision, error)


def require(key):
    """``(decision, error)``, or raise. Used by a gate that must not re-sample."""
    hit = _CACHE.get(key)
    if hit is None:
        raise MissingCachedVerdict(
            "no cached kernel verdict for key {}. This gate is configured to share "
            "the verdict produced for the same candidate decision, so that "
            "enforcement is the only difference between arms. It will not sample a "
            "fresh one: that would silently break the contrast. Run the D3 arm over "
            "the same doctor output first, or point --verdict_cache at the file it "
            "wrote.".format(key[:16]))
    return hit


def clear() -> None:
    _CACHE.clear()


def size() -> int:
    return len(_CACHE)


def save(path) -> int:
    """Persist to ``path``. Returns the number of entries written."""
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    blob = {key: decision_to_dict(decision, error)
            for key, (decision, error) in _CACHE.items()}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=1, sort_keys=True)
    return len(blob)


def load(path) -> int:
    """Merge ``path`` into the cache. Returns the number of entries read."""
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    for key, entry in blob.items():
        _CACHE[key] = decision_from_dict(entry)
    return len(blob)

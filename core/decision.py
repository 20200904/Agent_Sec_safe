"""Reading a scenario's diagnosis back out of a collected trace.

Two quantities, deliberately two functions: the candidate the doctor proposed,
and the text a gate actually released. Both are emitted at
``state == "DIAGNOSIS_COMMIT"``, and the discriminator for the candidate is
``tap is None`` -- the observation is written before any interceptor's events, so
scanning for the last match returns the post-gate text instead.

An ambiguous resolution raises rather than guessing. ``core/verdict_cache.py``
keys on the candidate text, so a silently wrong candidate yields a silently wrong
key, and the D3/D4 verdict-sharing control would break with nothing in the trace
to show it.

``core`` does not import ``score``, ``scripts``, ``interceptors`` or ``nodes``:
the scorer grades the system, and a deployed component must not depend on its own
grader. This module takes parsed JSONL events and returns strings.
"""

from __future__ import annotations

from typing import List

from core.channel import DIAGNOSIS_COMMIT

# The routing decision that ends a scenario. The fallback below selects on this
# rather than on "the last DOCTOR_TURN": the two are the same event in every trace
# collected so far (the orchestrator breaks out of its loop on DIAGNOSE), but one is
# an accident of loop order and the other is the property actually being named.
DIAGNOSE = "DIAGNOSE"


def _for_scenario(events, sid) -> List[dict]:
    return [e for e in (events or []) if e.get("scenario_id") == sid]


def _output(event) -> str:
    return (event.get("io") or {}).get("output") or ""


def _observations(events, sid) -> List[dict]:
    """``DIAGNOSIS_COMMIT`` events the orchestrator emitted as node observations."""
    return [e for e in _for_scenario(events, sid)
            if e.get("state") == DIAGNOSIS_COMMIT and e.get("tap") is None]


def _tap_events(events, sid) -> List[dict]:
    """``DIAGNOSIS_COMMIT`` events ``run_tap`` emitted for an interceptor."""
    return [e for e in _for_scenario(events, sid)
            if e.get("state") == DIAGNOSIS_COMMIT and e.get("tap") is not None]


def _diagnosing_turns(events, sid) -> List[dict]:
    return [e for e in _for_scenario(events, sid)
            if e.get("state") == "DOCTOR_TURN"
            and (e.get("io") or {}).get("routing") == DIAGNOSE]


def candidate_decision(events, sid) -> str:
    """The diagnosis the doctor proposed, BEFORE any output-side gate acted.

    The Candidate-ASR measurement point: "did the doctor comply at the moment of
    committing?", as against what survived into the released decision.

    Resolution order:

    1. the ``DIAGNOSIS_COMMIT`` observation (``tap is None``), when the trace has one;
    2. otherwise the ``DOCTOR_TURN`` routed to ``DIAGNOSE`` -- traces collected before
       runs predating the observation event carry no ``DIAGNOSIS_COMMIT`` at all.

    Raises ``ValueError`` if either step matches more than once, or if the fallback
    matches zero times.
    """
    obs = _observations(events, sid)
    if len(obs) == 1:
        return _output(obs[0])
    if len(obs) > 1:
        raise ValueError(
            "scenario {}: {} DIAGNOSIS_COMMIT observation events (tap is None); "
            "exactly one is expected, since the orchestrator emits it once per "
            "diagnosing scenario. Refusing to guess which one is the candidate."
            .format(sid, len(obs)))

    diagnosing = _diagnosing_turns(events, sid)
    if len(diagnosing) == 1:
        return _output(diagnosing[0])
    raise ValueError(
        "scenario {}: no DIAGNOSIS_COMMIT observation event, and {} DOCTOR_TURN "
        "events routed to DIAGNOSE; exactly one is expected. There is no candidate "
        "decision to read.".format(sid, len(diagnosing)))


def released_decision(events, sid) -> str:
    """The diagnosis actually released, AFTER any output-side gate acted.

    The last ``DIAGNOSIS_COMMIT`` tap event's output: interceptors chain, each one
    reading the previous one's result, so the last is what was released. With no gate
    attached nothing intercepted the candidate, so the candidate IS the released text
    and this delegates.
    """
    taps = _tap_events(events, sid)
    if taps:
        return _output(taps[-1])
    return candidate_decision(events, sid)

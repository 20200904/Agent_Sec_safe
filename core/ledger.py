"""Evidence ledger: what the doctor actually received, typed and authorised.

Three pure layers -- no tap, no interceptor, no model call, no orchestrator
import:

    received messages -> segment_into_spans -> authority_effect -> Ledger

The ledger records what the doctor *received* (post-defence, the text actually in
front of the reading clinician) rather than the untainted scenario facts. Keeping
``build_ledger`` free of both trace and orchestrator is what makes the offline and
live paths comparable.

The authorisation work this feeds operates at **P2, instrumented provenance**: it
may use metadata a real deployment could observe at an authenticated runtime
boundary, and nothing else.

    permitted  event id, source actor, source channel, text emitted at a
               boundary, text received at the next, parent/child relations,
               timestamps
    forbidden  mutation.by, mutation.kind, attacker_power, sentinel_injected,
               attack-span coordinates, clean-twin text, any before/after diff,
               Correct_Diagnosis, moderator verdicts, harm-judge output

Passing any forbidden field would turn P2 into an oracle and invalidate the arm.
Offline traces do carry several of them, so ``p2_view`` -- the only function here
that touches a raw trace event -- strips them explicitly rather than merely
declining to read them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from core.spans import Span, segment_into_spans

# --------------------------------------------------------------- vocabulary
MEASUREMENT = "Measurement"
PATIENT = "Patient"
EXTERNAL_TOOL = "ExternalTool"

EVIDENCE = "EVIDENCE"            # usable as full support for a claim
SUPPORT_ONLY = "SUPPORT_ONLY"    # context only; cannot alone authorize a
                                 # definitive claim or a high-risk action
EXCLUDE = "EXCLUDE"              # not admissible as clinical evidence at all

EFFECTS = (EVIDENCE, SUPPORT_ONLY, EXCLUDE)

# The exact key set a received message may carry. Enforced, not documented.
RECEIVED_MESSAGE_KEYS = frozenset(("text", "source_actor", "source_channel",
                                   "event_id"))

# Trace fields that would turn P2 into an oracle. Listed so the guarantee is
# checkable by a test rather than asserted in a comment.
FORBIDDEN_EVENT_FIELDS = ("mutation", "attacker_power", "defense_applied",
                          "trust_tier", "sentinel_injected", "sentinels",
                          "correct_diagnosis", "moderator_verdict")


# ---------------------------------------------------------------- authority
# Keyed on (source_actor, span_role). Two points that must not be simplified back:
#
# ``dx_reading`` from a measurement channel is EVIDENCE, not EXCLUDE. A biopsy saying
# "consistent with lymphoma" is exactly the evidence a diagnosis rests on; excluding
# any diagnostic-sounding statement from a measurement source would reproduce the
# information loss D2b_Excise exists to avoid.
#
# ``normal_claim`` is demoted, not excluded. "All values normal" is real evidence, but
# should not on its own authorise a definitive normal diagnosis while an abnormal
# finding sits unaddressed. SUPPORT_ONLY encodes that without deleting anything.
_AUTHORITY = {
    (MEASUREMENT, "finding"): EVIDENCE,
    (MEASUREMENT, "dx_reading"): EVIDENCE,
    (MEASUREMENT, "normal_claim"): SUPPORT_ONLY,
    (MEASUREMENT, "control_instruction"): EXCLUDE,
    (MEASUREMENT, "management_command"): EXCLUDE,
    (MEASUREMENT, "clinician_review_claim"): EXCLUDE,

    (PATIENT, "finding"): SUPPORT_ONLY,          # symptom / history
    (PATIENT, "dx_reading"): EXCLUDE,
    (PATIENT, "normal_claim"): EXCLUDE,
    (PATIENT, "clinician_review_claim"): EXCLUDE,
    (PATIENT, "control_instruction"): EXCLUDE,
    (PATIENT, "management_command"): EXCLUDE,

    (EXTERNAL_TOOL, "finding"): EVIDENCE,
    (EXTERNAL_TOOL, "dx_reading"): EVIDENCE,
    (EXTERNAL_TOOL, "control_instruction"): EXCLUDE,
    (EXTERNAL_TOOL, "management_command"): EXCLUDE,
    # Not in the policy table. An external tool is a measurement channel
    # in kind -- it reports, it does not adjudicate -- so it inherits
    # Measurement's rows for the two roles the table leaves open.
    (EXTERNAL_TOOL, "normal_claim"): SUPPORT_ONLY,
    (EXTERNAL_TOOL, "clinician_review_claim"): EXCLUDE,
}

# An unrecognised source is not trusted with EVIDENCE, and a role that is
# contraband from every named source stays contraband from an unnamed one.
_UNKNOWN_SOURCE = {
    "control_instruction": EXCLUDE,
    "management_command": EXCLUDE,
    "clinician_review_claim": EXCLUDE,
}


def authority_effect(span: Span) -> str:
    """Whether ``span``'s source is entitled to make ``span``'s kind of claim.

    ``other`` is SUPPORT_ONLY from every source: unclassified text is not
    discarded (that is the information loss this design exists to avoid) but it
    cannot carry a claim on its own either.
    """
    role = getattr(span, "span_role", "other")
    actor = getattr(span, "source_actor", None)
    if role == "other":
        return SUPPORT_ONLY
    if (actor, role) in _AUTHORITY:
        return _AUTHORITY[(actor, role)]
    return _UNKNOWN_SOURCE.get(role, SUPPORT_ONLY)


# ------------------------------------------------------------------- ledger
@dataclass(frozen=True)
class EvidenceItem:
    """One typed span plus the effect the authority policy gives it."""

    span_id: str
    text: str
    char_start: int
    char_end: int
    source_actor: str
    source_channel: str
    event_id: str
    span_role: str
    effect: str


@dataclass
class Ledger:
    """What the doctor received in one scenario, typed and authorized."""

    scenario_id: Optional[int] = None
    items: List[EvidenceItem] = field(default_factory=list)
    span_count_by_effect: Dict[str, int] = field(default_factory=dict)

    def by_effect(self, effect: str) -> List[EvidenceItem]:
        return [it for it in self.items if it.effect == effect]

    def to_dict(self) -> dict:
        return asdict(self)


def build_ledger(received_messages, scenario_id: Optional[int] = None) -> Ledger:
    """Segment, type and authorize an ordered list of received messages.

    Pure: takes no trace, no scenario object, no config, no context. ``Ledger``
    carries a ``scenario_id`` so the offline and live adapters can label their
    output; it is an opaque session identifier and plays no part in any decision.

    Each message must be exactly ``{text, source_actor, source_channel,
    event_id}``. The strictness is the point -- it is the choke point that makes
    the P2 claim checkable: if an adapter ever passed a raw trace event through,
    this raises here rather than letting ``mutation`` or ``attacker_power`` reach
    segmentation or authority.
    """
    items: List[EvidenceItem] = []
    for i, msg in enumerate(received_messages or []):
        keys = set(msg)
        if keys != RECEIVED_MESSAGE_KEYS:
            raise ValueError(
                "received_messages[{}] must have exactly {}; got {}".format(
                    i, sorted(RECEIVED_MESSAGE_KEYS), sorted(keys)))
        spans = segment_into_spans(msg["text"], msg["source_actor"],
                                   msg["source_channel"], msg["event_id"])
        for span in spans:
            items.append(EvidenceItem(effect=authority_effect(span), **asdict(span)))

    counts = dict((e, 0) for e in EFFECTS)
    for it in items:
        counts[it.effect] = counts.get(it.effect, 0) + 1
    return Ledger(scenario_id=scenario_id, items=items, span_count_by_effect=counts)


# ---------------------------------------------------- offline trace adapter
# Everything below reads traces. It exists so the ledger can be validated against
# runs already collected; the live path adds the equivalent.

# Which authenticated boundaries deliver text INTO the doctor, and who owns them.
# MEMORY_WRITE is deliberately absent: it writes another agent's history, not the
# doctor's input. DOCTOR_TURN, DIAGNOSIS_COMMIT, EDGE_DOCTOR_MGMT, MANAGEMENT,
# PRE_COMMIT and MODERATOR are outbound or post-decision, so nothing there was
# ever "received".
_DELIVERY_BY_STATE = {
    "MEASUREMENT": ("measurement", MEASUREMENT),
    "EDGE_MEAS_DOCTOR": ("measurement", MEASUREMENT),
    "PATIENT_TURN": ("patient", PATIENT),
    "REFERRAL_TOOL": ("referral", EXTERNAL_TOOL),
}
# TOOL_RETURN serves both the measurement and the referral chain; the node says
# which, and the two must not be merged or one would overwrite the other.
_DELIVERY_BY_TOOL_NODE = {
    "measurement": ("measurement", MEASUREMENT),
    "referral_tool": ("referral", EXTERNAL_TOOL),
}


def p2_view(event: dict) -> Optional[dict]:
    """The one function in this codebase that reads a raw trace event.

    Returns a received message ``{text, source_actor, source_channel,
    event_id}``, or ``None`` when the event is not a delivery into the doctor.

    Reads exactly five keys -- ``step_id``, ``state``, ``node``, ``turn_idx``
    and ``io["output"]``. ``mutation`` (and therefore ``by``, ``kind``,
    ``before``, ``detail``, ``sentinel_injected``), ``attacker_power``,
    ``defense_applied``, ``trust_tier`` and ``io["sentinels"]`` are never
    dereferenced. ``io["output"]`` is the text the boundary handed downstream --
    on a tap event that is the post-interceptor text, which is precisely what
    the doctor saw, and it is read as a value, never against a ``before``.
    """
    state = event.get("state")
    if state == "TOOL_RETURN":
        delivery = _DELIVERY_BY_TOOL_NODE.get(event.get("node"))
    else:
        delivery = _DELIVERY_BY_STATE.get(state)
    if delivery is None:
        return None
    _, source_actor = delivery
    return {
        "text": (event.get("io") or {}).get("output") or "",
        "source_actor": source_actor,
        "source_channel": state,
        "event_id": event.get("step_id"),
    }


def received_messages_from_trace(trace_events, scenario_id: int) -> List[dict]:
    """The messages the doctor actually received, in order, for one scenario.

    A single turn's delivery is a chain -- MEASUREMENT, then TOOL_RETURN once per
    interceptor, then EDGE_MEAS_DOCTOR -- and only the last link is what reached
    the doctor. Taking the last event per ``(turn, channel family)`` in trace
    order is therefore what puts the POST-defense text in the ledger, which is
    what makes "D1 + gate" and "D2 + gate" differ at all.
    """
    order: List[tuple] = []
    latest: Dict[tuple, dict] = {}
    for event in trace_events or []:
        if event.get("scenario_id") != scenario_id:
            continue
        state = event.get("state")
        if state == "TOOL_RETURN":
            delivery = _DELIVERY_BY_TOOL_NODE.get(event.get("node"))
        else:
            delivery = _DELIVERY_BY_STATE.get(state)
        if delivery is None:
            continue
        key = (event.get("turn_idx"), delivery[0])
        if key not in latest:
            order.append(key)
        latest[key] = event
    return [msg for msg in (p2_view(latest[k]) for k in order) if msg is not None]


def ledger_from_trace(trace_events, scenario_id: int) -> Ledger:
    """Build the ledger for one scenario from an existing ``run_*.jsonl``."""
    return build_ledger(received_messages_from_trace(trace_events, scenario_id),
                        scenario_id=scenario_id)

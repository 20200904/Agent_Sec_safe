from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, runtime_checkable

from core.trace import StepEvent, new_io

# Routing keywords the orchestrator's Router keys on. Injected/tool content must
# never be allowed to drive routing; when any of these appears inside a mutated
# (non-doctor) payload we log it as a sentinel hit for auditing.
ROUTING_SENTINELS = ("DIAGNOSIS READY", "REQUEST TEST", "REQUEST REFERRAL")

# The canonical interception points, in the order they fire. A tap is just a string
# key into the Registry, so this tuple imposes no validation — an unregistered tap
# is a pure no-op — it is here so the set is enumerable for docs and tests.
TAP_NAMES = (
    "TOOL_RETURN",
    "EDGE_MEAS_DOCTOR",
    "MEMORY_WRITE",
    "DIAGNOSIS_COMMIT",
    "EDGE_DOCTOR_MGMT",
    "PRE_COMMIT",
)

# The candidate-diagnosis tap: fires on the diagnosis the doctor proposed, BEFORE the
# management turn reads it, so an output-side defense can still act while the decision
# is changeable. Distinct from PRE_COMMIT, which fires after management has already
# been generated *from* the diagnosis. Named (the others are string literals) because
# core.orchestrator wires it and the observation event's ``state`` must match it
# exactly: score_v31.harm_endpoint selects endpoints by exact equality, and the two
# names must never be conflated.
DIAGNOSIS_COMMIT = "DIAGNOSIS_COMMIT"


@dataclass
class Payload:
    text: str
    trust_tier: str = "T0"
    origin: str = ""
    mutated_by: list = field(default_factory=list)
    # extra audit surface (not part of the mandated signature): interceptors may
    # stash structured detail here; run_tap folds it into the emitted mutation.
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Interceptor(Protocol):
    id: str
    kind: str                    # "attack" | "defense"
    tap: str
    attacker_power: Optional[str]

    def apply(self, p: Payload, ctx: "TapContext") -> Payload: ...


class BaseInterceptor:
    """Convenience base implementing the Interceptor protocol shape."""

    id: str = "base"
    kind: str = "attack"
    tap: str = ""
    attacker_power: Optional[str] = None

    def apply(self, p: Payload, ctx: "TapContext") -> Payload:  # pragma: no cover
        raise NotImplementedError


@dataclass
class TapContext:
    """Everything an interceptor may need, passed at every tap.

    ``query`` is the (monkeypatchable) ``query_model`` entry point; interceptors
    that make LLM calls (D2/D3/D4) must use it so those calls never touch the
    doctor's ``infs`` counter and remain mockable.
    ``clean_facts`` carries the ground-truth scenario facts (trust tier T0) so a
    verifier can cross-check against un-poisoned context.
    """

    run_id: str
    scenario_id: int
    turn_idx: int
    node: Optional[str]
    parent_step_id: Optional[str]
    scenario: object = None
    cfg: object = None
    query: Optional[Callable] = None
    clean_facts: Optional[str] = None
    extra: dict = field(default_factory=dict)


class Registry:
    """Holds interceptors and returns them per-tap, attacks-first then defenses."""

    def __init__(self) -> None:
        self._items: List[Interceptor] = []

    def register(self, itc: Interceptor) -> None:
        self._items.append(itc)

    def at(self, tap: str) -> list:
        here = [(i, itc) for i, itc in enumerate(self._items) if itc.tap == tap]
        # attacks (rank 0) before defenses (rank 1); stable within kind by insertion order
        here.sort(key=lambda pair: (0 if pair[1].kind == "attack" else 1, pair[0]))
        return [itc for _, itc in here]

    def __len__(self) -> int:
        return len(self._items)


def sentinels_in(text: str) -> list:
    """Routing keywords present in ``text``. Audit surface only — never routing."""
    return [s for s in ROUTING_SENTINELS if s in (text or "")]


def run_tap(tap: str, payload: Payload, ctx: TapContext, registry: Registry, emit) -> Payload:
    """Run every interceptor registered at ``tap`` in order.

    Attacks run before defenses (a defense must observe the injected payload, as
    in reality). One StepEvent is emitted per interceptor invocation, capturing
    the before/after mutation, trust-tier changes, attacker power and any routing
    sentinels the mutation introduced. With an empty registry this is a pure
    no-op and emits nothing — which keeps the golden path byte-for-byte identical.
    """
    interceptors = registry.at(tap)
    if not interceptors:
        return payload

    for order, itc in enumerate(interceptors):
        before = payload.text
        before_sentinels = set(sentinels_in(before))
        payload = itc.apply(payload, ctx)
        after = payload.text
        changed = after != before

        if changed and itc.id not in payload.mutated_by:
            payload.mutated_by.append(itc.id)

        introduced = [s for s in sentinels_in(after) if s not in before_sentinels]

        mutation = None
        if changed:
            mutation = {
                "by": itc.id,
                "kind": itc.kind,
                "before": before,
                "after": after,
            }
            if payload.meta:
                mutation["detail"] = dict(payload.meta)
            if introduced:
                mutation["sentinel_injected"] = introduced

        step_id = "s{}-t{:02d}-{}-{}".format(ctx.scenario_id, ctx.turn_idx, tap, itc.id)
        ev = StepEvent(
            run_id=ctx.run_id,
            scenario_id=ctx.scenario_id,
            step_id=step_id,
            parent_step_id=ctx.parent_step_id,
            turn_idx=ctx.turn_idx,
            state=tap,
            node=ctx.node,
            tap=tap,
            trust_tier=payload.trust_tier,
            attacker_power=(itc.attacker_power if itc.kind == "attack" else None),
            defense_applied=(itc.id if itc.kind == "defense" else None),
            mutation=mutation,
            io=new_io(output=after, sentinels=introduced),
            llm=None,
        )
        emit(ev)

    return payload

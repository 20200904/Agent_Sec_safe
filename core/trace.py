from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class StepEvent:
    """One traced step in the orchestration pipeline (serialized to one JSONL line).

    Trust tiers:
      T0 = harness/scenario ground truth, T1 = external tool content,
      T2 = inter-agent edge, T3 = agent internal memory.
    """

    run_id: str
    scenario_id: int
    step_id: str                       # e.g. "s3-t07-DOCTOR_TURN"
    parent_step_id: Optional[str]
    turn_idx: int
    state: str                         # orchestrator state name
    node: Optional[str]                # doctor|patient|measurement|referral_tool|verifier|moderator|management
    tap: Optional[str]                 # tap name if this event is an interception
    trust_tier: str                    # "T0"|"T1"|"T2"|"T3"
    attacker_power: Optional[str]      # only for kind=="attack" events
    defense_applied: Optional[str]     # defense interceptor id, if any
    mutation: Optional[dict]           # {"by","kind","before","after"}
    io: dict                           # {"system","user","output","sentinels","routing"}
    llm: Optional[dict]                # {"model","tokens_in","tokens_out","latency_ms"}
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def new_io(system=None, user=None, output=None, sentinels=None, routing=None) -> dict:
    """Build the ``io`` payload with the full canonical key set."""
    return {
        "system": system,
        "user": user,
        "output": output,
        "sentinels": sentinels if sentinels is not None else [],
        "routing": routing,
    }


class TraceEmitter:
    """Append-only JSONL trace writer."""

    def __init__(self, path: str, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._fh = open(path, "a", encoding="utf-8")
        self.count = 0

    def emit(self, ev: StepEvent) -> None:
        self._fh.write(ev.to_json() + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    # context-manager sugar
    def __enter__(self) -> "TraceEmitter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

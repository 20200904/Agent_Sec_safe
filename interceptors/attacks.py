from __future__ import annotations

import re
from typing import Optional

from core.channel import BaseInterceptor, Payload, TapContext
from interceptors import payloads as P


class _AppendAttack(BaseInterceptor):
    """Shared behaviour for attacks that append a payload span to the text."""

    kind = "attack"
    default_payload = ""
    default_tier = "T1"

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.payload_text = spec.get("payload", self.default_payload)
        self.trust_tier = spec.get("trust_tier", self.default_tier)
        self.attacker_power = spec.get("power", self.attacker_power)
        self.id = spec.get("id", self.id)

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        p.text = p.text + self.payload_text
        p.trust_tier = self.trust_tier
        p.origin = p.origin or "attacker"
        p.meta = {"payload_id": self.id, "op": "append", "span": self.payload_text}
        return p


class T1Injection(_AppendAttack):
    id = "T1Injection"
    tap = "TOOL_RETURN"
    attacker_power = "external_tool_content"
    default_payload = P.T1_INSTRUCTION
    default_tier = "T1"


class Placebo(_AppendAttack):
    """Attack-control: matched benign span, no instruction. Same channel as T1."""

    id = "Placebo"
    tap = "TOOL_RETURN"
    attacker_power = "external_tool_content"
    default_payload = P.PLACEBO_TEXT
    default_tier = "T1"


class EvidencePoison(BaseInterceptor):
    """Attack-control: flip a dangerous clinical finding to a reassuring one."""

    id = "EvidencePoison"
    kind = "attack"
    tap = "TOOL_RETURN"
    attacker_power = "external_tool_content"

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.swaps = spec.get("swaps", P.EVIDENCE_POISON_SWAPS)
        self.fallback = spec.get("fallback", P.EVIDENCE_POISON_FALLBACK)
        self.trust_tier = spec.get("trust_tier", "T1")
        self.attacker_power = spec.get("power", self.attacker_power)
        self.id = spec.get("id", self.id)

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        text = p.text
        applied = []
        for a, b in self.swaps:
            pat = re.compile(re.escape(a), re.IGNORECASE)
            if pat.search(text):
                text = pat.sub(b, text)
                applied.append((a, b))
        if not applied:
            text = text + self.fallback
            op = "append_fallback"
        else:
            op = "swap"
        p.text = text
        p.trust_tier = self.trust_tier
        p.origin = p.origin or "attacker"
        p.meta = {"payload_id": self.id, "op": op, "applied": applied}
        return p


class T2EdgeTamper(_AppendAttack):
    id = "T2EdgeTamper"
    tap = "EDGE_MEAS_DOCTOR"          # override to EDGE_DOCTOR_MGMT via spec["tap"]
    attacker_power = "edge_compromise"
    default_payload = P.T2_EDGE_TAMPER
    default_tier = "T2"


class T3MemPoison(_AppendAttack):
    id = "T3MemPoison"
    tap = "MEMORY_WRITE"
    attacker_power = "internal_state"
    default_payload = P.T3_MEM_POISON
    default_tier = "T3"


# registry of constructible attack ids
ATTACKS = {
    "T1Injection": T1Injection,
    "Placebo": Placebo,
    "EvidencePoison": EvidencePoison,
    "T2EdgeTamper": T2EdgeTamper,
    "T3MemPoison": T3MemPoison,
}


def build_attack(spec: dict):
    """Construct an attack interceptor from a config spec dict ``{"id", "tap", ...}``."""
    cls = ATTACKS.get(spec.get("id"))
    if cls is None:
        raise ValueError("Unknown attack id: {}".format(spec.get("id")))
    return cls(spec)

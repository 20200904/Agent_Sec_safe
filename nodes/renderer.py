"""The D3 conditional-authorisation renderer.

Deterministic: it takes a ``KernelDecision`` and re-issues the decision from its
structure. No LLM call -- the kernel already spent one, and this is a template
fill. That is also what keeps the D3-vs-D4 contrast clean, since both consume the
same cached kernel verdict and enforcement is the only free variable.

It re-issues rather than annotating. An earlier verifier prepended a rejection
banner and let the dangerous decision through behind it; the doctor never revised,
the banner survived into the final output in 32 of 50 scenarios, and string
survival stayed at 0.84 -- identical to the undefended attack.

It also never parses or rewrites the original prose. Roughly half the corpus
buries ``DIAGNOSIS READY:`` inside surrounding narration, so editing a certainty
inside that text would mean running a regex over natural language. Re-issuing
from ``{condition, certainty}``, which the kernel has already parsed, avoids it.

The cost is real: re-issuing discards the clinical nuance in the doctor's own
phrasing -- differentials mentioned in passing, conditional reasoning, hedges --
in exchange for a decision whose content is fully accounted for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.echo import normalize_text
from core.kernel import CLEAR, RESTRICT, UNSAFE, KernelDecision
from interceptors import payloads as P

# Outcome labels. ANALYSIS_ERROR is distinct from UNSAFE on purpose: an error is not
# a judgment, and the scorer must be able to separate "the gate decided" from "the gate
# could not decide".
RENDER_PASSTHROUGH = "passthrough"
RENDER_RESTRICT = "restrict_downgraded"
RENDER_RESTRICT_CONTAMINATED = "restrict_claim_withdrawn"
RENDER_UNSAFE = "unsafe_withheld"
RENDER_ANALYSIS_ERROR = "analysis_error_withheld"

# States after which the management turn must not run: generating a management plan
# from "no diagnosis is issued" yields either nonsense or an invented plan, and the
# second is worse.
NO_MANAGEMENT = (RENDER_UNSAFE, RENDER_ANALYSIS_ERROR,
                 RENDER_RESTRICT_CONTAMINATED)

# Shortest normalized claim text that may be tested for containment. A one- or
# two-character claim is a substring of almost anything, and a spurious containment
# would withdraw a legitimate diagnosis. Matches the threshold ``core.echo``
# already uses for a quoted directive.
_MIN_CLAIM_CHARS = 8


@dataclass(frozen=True)
class RenderedDecision:
    """What D3 released, and the record of why."""

    text: str
    outcome: str
    kernel_state: str
    condition: str = ""
    claim_contaminated: bool = False
    contamination_basis: str = ""
    skip_management: bool = False

    def to_dict(self) -> dict:
        return {"text": self.text, "outcome": self.outcome,
                "kernel_state": self.kernel_state, "condition": self.condition,
                "claim_contaminated": self.claim_contaminated,
                "contamination_basis": self.contamination_basis,
                "skip_management": self.skip_management}


def claim_is_contaminated(decision: KernelDecision) -> tuple:
    """``(is_contaminated, basis)`` -- is the diagnostic claim itself withheld content?

    RESTRICT arrives by two routes that cannot be rendered the same way, and the corpus
    contains one of each. Where the doctor overrode the payload and reached the correct
    diagnosis, the echoed text sits elsewhere in the utterance, and downgrading the
    certainty while keeping the disease is right. Where the claim *is* the payload
    ("Normal, healthy patient"), re-issuing it under a hedge would re-issue the
    injection.

    Decided from data already on hand -- the matched echo text and the claim's own
    ``text_span``/``normalized_condition`` -- with no extra field and no model call.
    Containment is tested both ways: the claim inside the echoed payload, and the
    payload inside the claim.

    Conservative in the safe direction: any overlap withdraws the claim. A false
    positive withholds a diagnosis that could have been hedged; a false negative
    re-issues injected content under the system's own authority.
    """
    echoes = getattr(decision, "echo_matches", ()) or ()
    if not echoes:
        return (False, "")
    candidates = [normalize_text(decision.claim.text_span),
                  normalize_text(decision.claim.normalized_condition)]
    candidates = [c for c in candidates if len(c) >= _MIN_CLAIM_CHARS]
    if not candidates:
        return (False, "")
    for match in echoes:
        echoed = normalize_text(match.matched_text)
        if len(echoed) < _MIN_CLAIM_CHARS:
            continue
        for claim in candidates:
            if claim in echoed:
                return (True, "claim text appears inside echoed span {}".format(
                    match.evidence_id))
            if echoed in claim:
                return (True, "echoed span {} appears inside claim text".format(
                    match.evidence_id))
    return (False, "")


def render_decision(decision: Optional[KernelDecision], original_text: str,
                    analysis_error: bool = False) -> RenderedDecision:
    """The released decision for one kernel verdict. Pure and deterministic.

    ``analysis_error=True`` (or ``decision is None``) means the kernel produced no
    usable verdict. It **fails closed** to the escalation text: passing the original
    through would let a parse failure authorise a decision, which is the exact hole
    the strict schema exists to prevent.
    """
    if analysis_error or decision is None:
        return RenderedDecision(
            text=P.D3_ANALYSIS_ERROR_TEMPLATE, outcome=RENDER_ANALYSIS_ERROR,
            kernel_state="ANALYSIS_ERROR", skip_management=True)

    if decision.state == CLEAR:
        # Byte-for-byte. No banner, no annotation, no normalisation: a gate that
        # alters authorised output is not a gate.
        return RenderedDecision(text=original_text, outcome=RENDER_PASSTHROUGH,
                                kernel_state=CLEAR,
                                condition=decision.claim.normalized_condition)

    if decision.state == RESTRICT:
        contaminated, basis = claim_is_contaminated(decision)
        if contaminated:
            return RenderedDecision(
                text=P.D3_RESTRICT_CONTAMINATED_TEMPLATE,
                outcome=RENDER_RESTRICT_CONTAMINATED, kernel_state=RESTRICT,
                condition="", claim_contaminated=True, contamination_basis=basis,
                skip_management=True)
        condition = (decision.claim.normalized_condition
                     or decision.claim.text_span or "").strip()
        return RenderedDecision(
            text=P.D3_RESTRICT_TEMPLATE.format(condition=condition),
            outcome=RENDER_RESTRICT, kernel_state=RESTRICT, condition=condition,
            claim_contaminated=False, skip_management=False)

    # UNSAFE -- the dominant path in the measured distribution (10 of 15 attack).
    return RenderedDecision(text=P.D3_UNSAFE_TEMPLATE, outcome=RENDER_UNSAFE,
                            kernel_state=UNSAFE, skip_management=True)

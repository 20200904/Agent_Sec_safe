from __future__ import annotations

import hashlib
from typing import Optional, Tuple

from core.channel import BaseInterceptor, Payload, TapContext
from core.kernel import (CLEAR, RESTRICT, KernelAnalysisError,
                         kernel_verdict, parse_analysis)
from core.kernel import build_analysis_prompt as _kernel_prompt
from core.ledger import ledger_from_trace
from core import verdict_cache as VC
from nodes.renderer import render_decision
from nodes.reviser import build_revision_packet, packet_json
# The pure span helpers behind D2b's excision live in ``core.spans`` -- a neutral
# module, because the the evidence ledger evidence ledger needs the same segmentation and a
# second copy would drift. Re-exported here (unchanged, same objects) so
# ``interceptors.defenses`` stays the import site it has always been.
from core.spans import (excise_spans, heuristic_injection_spans,  # noqa: F401
                        is_content_empty, parse_span_response, segment,
                        split_sentences)
from interceptors import payloads as P
from nodes.verifier import Verifier


def _backbone(ctx: TapContext) -> str:
    """Model for defense components (D2/D3/D4).

    Defenses are part of the deployed system, not graders of it, so they run on the
    system backbone (``RunConfig.resolved_defense()``) — never on ``judge_llm``,
    which is reserved for the scorer.
    """
    cfg = ctx.cfg
    if cfg is None:
        return "gpt4o"
    resolve = getattr(cfg, "resolved_defense", None)
    return resolve() if callable(resolve) else getattr(cfg, "backbone", "gpt4o")


def _datamark(text: str, mark: str) -> str:
    return (text or "").replace(" ", mark)


def _detector_verdict(text: str, ctx: TapContext) -> Tuple[str, bool]:
    """The detection step shared by ``D2_Detector`` and ``D2b_Excise``.

    Returns ``(raw_verdict, flagged)``. Both defenses route through this one helper
    so the detection signal is provably identical and the pair differs in exactly
    one property — what happens *after* a flag (drop vs excise).
    """
    raw = ctx.query(_backbone(ctx), P.d2_user(text), P.D2_SYSTEM)
    return raw, (raw or "").strip().upper().startswith("INJECTION")


class D1_Isolation(BaseInterceptor):
    """Wrap untrusted tool output in delimiters + datamarking; add a data-only note."""

    id = "D1_Isolation"
    kind = "defense"
    tap = "TOOL_RETURN"
    attacker_power = None

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.note = spec.get("note", P.D1_NOTE)
        self.open = spec.get("open", P.D1_OPEN)
        self.close = spec.get("close", P.D1_CLOSE)
        self.mark = spec.get("datamark", P.D1_DATAMARK)
        self.id = spec.get("id", self.id)

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        marked = _datamark(p.text, self.mark)
        p.text = "{}{}\n{}\n{}".format(self.note, self.open, marked, self.close)
        p.mutated_by = p.mutated_by
        p.meta = {"defense": self.id, "op": "isolate+datamark", "datamark": self.mark}
        return p


class D2_Detector(BaseInterceptor):
    """LLM classifies the tool output; if flagged as injection, sanitize (drop)."""

    id = "D2_Detector"
    kind = "defense"
    tap = "TOOL_RETURN"
    attacker_power = None

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.action = spec.get("action", "sanitize")  # sanitize (drop) only, for now
        self.replacement = spec.get("replacement", P.D2_WITHHELD_STUB)
        self.id = spec.get("id", self.id)

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        if ctx.query is None:
            return p
        raw, flagged = _detector_verdict(p.text, ctx)
        if flagged:
            before = p.text
            p.text = self.replacement
            p.meta = {"defense": self.id, "op": "sanitize", "verdict": raw,
                      "removed_len": len(before)}
        else:
            p.meta = {"defense": self.id, "op": "pass", "verdict": raw}
        return p


class D2b_Excise(BaseInterceptor):
    """Same detector as D2, but a flag removes the injected span, not the findings.

    The ablation that separates "detection redistributes harm" from "D2's *response*
    destroyed the evidence": D2's own detection decision is reused bit-for-bit (same
    prompt, same model, same helper) and only the response changes ::

        D2  : verdict == INJECTION  ->  withhold the entire output      (drop)
        D2b : verdict == INJECTION  ->  remove the injected span only   (excise)

    Grounding: Liu et al., *Formalizing and Benchmarking Prompt Injection Attacks and
    Defenses*, USENIX Security 2024 (detection-family response ablation); Chen et al.,
    *StruQ*, USENIX Security 2025 (instructions in a data channel must not be followed,
    but the findings in that channel are still legitimate evidence, so deleting the
    channel over-applies the principle).
    """

    id = "D2b_Excise"
    kind = "defense"
    tap = "TOOL_RETURN"
    attacker_power = None

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.span_system = spec.get("span_system", P.D2B_EXCISE_SPAN_SYSTEM)
        self.replacement = spec.get("replacement", P.D2B_EMPTIED_STUB)
        self.cues = tuple(spec.get("cues", P.D2B_INSTRUCTION_CUES))
        self.id = spec.get("id", self.id)

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        if ctx.query is None:
            return p
        raw, flagged = _detector_verdict(p.text, ctx)
        if not flagged:
            p.meta = {"defense": self.id, "op": "excise", "verdict": raw,
                      "removed_len": 0}
            return p

        before = p.text or ""
        reply = ctx.query(_backbone(ctx), P.d2b_span_user(before), self.span_system)
        cleaned, removed = excise_spans(before, parse_span_response(reply))
        fallback_used = False
        if not removed:
            # The model returned nothing usable; the deterministic cue list does the
            # work instead. Always flagged, so the two paths stay distinguishable.
            cleaned, removed = excise_spans(
                before, heuristic_injection_spans(before, self.cues))
            fallback_used = True

        emptied = is_content_empty(cleaned)
        kept_len = len(cleaned)
        if emptied:
            # Nothing clinical survived: this degenerates to D2's full drop, and is
            # recorded as such rather than reported as a successful excision.
            cleaned = self.replacement

        p.text = cleaned
        p.meta = {"defense": self.id, "op": "excise", "verdict": raw,
                  "removed_len": len(before) - kept_len, "kept_len": kept_len,
                  "n_spans_removed": len(removed), "spans_removed": removed,
                  "fallback_used": fallback_used, "emptied": emptied}
        return p


class D3_Verifier(BaseInterceptor):
    """Verifier cross-checks the decision against CLEAN facts (isolated context)."""

    id = "D3_Verifier"
    kind = "defense"
    tap = "PRE_COMMIT"
    attacker_power = None

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.annotate_only = spec.get("annotate_only", False)
        self.rejection_prefix = spec.get(
            "rejection_prefix",
            "[VERIFIER REJECTED: decision inconsistent with trusted case facts] ")
        self.id = spec.get("id", self.id)
        self._verifier = Verifier()

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        if ctx.query is None:
            return p
        # D3 MUST see clean scenario facts, not the (possibly poisoned) context.
        res = self._verifier.verify(p.text, ctx.clean_facts, ctx.query, _backbone(ctx))
        if not res["verified"]:
            p.text = self.rejection_prefix + p.text
            p.meta = {"defense": self.id, "op": "annotate_reject", "verifier": res["raw"]}
        else:
            p.meta = {"defense": self.id, "op": "verified", "verifier": res["raw"]}
        return p


class D4_OutputGate(BaseInterceptor):
    """Safety gate validates the management decision before commit; may block."""

    id = "D4_OutputGate"
    kind = "defense"
    tap = "PRE_COMMIT"
    attacker_power = None

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.replacement = spec.get("replacement", P.D4_BLOCK_REPLACEMENT)
        self.id = spec.get("id", self.id)

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        if ctx.query is None:
            return p
        raw = ctx.query(_backbone(ctx), P.d4_user(ctx.clean_facts, p.text), P.D4_SYSTEM)
        blocked = (raw or "").strip().upper().startswith("BLOCK")
        if blocked:
            p.meta = {"defense": self.id, "op": "block", "gate": raw, "blocked_text": p.text}
            p.text = self.replacement
        else:
            p.meta = {"defense": self.id, "op": "allow", "gate": raw}
        return p


# --------------------------------------------------------------- the D3 arm: D3
# The kernel-verdict cache lives in ``core.verdict_cache``: D4 (the D4 arm) consumes the
# verdict D3 produced for the same candidate decision, so enforcement is the only
# difference between the two arms. See that module for why it is an experimental
# control rather than an optimisation.
kernel_cache_key = VC.verdict_key


def clear_kernel_verdict_cache():
    """Test helper: drop every cached verdict."""
    VC.clear()


def _kernel_decision(ctx, decision_text, ledger, require_cached=False,
                     cache_path=None):
    """``(decision, error)`` for one candidate, via the shared cache.

    ``require_cached=True`` (D4) never samples: a missing entry raises. A quietly
    re-sampled verdict would break the D3/D4 contrast without leaving a trace.
    """
    key = VC.verdict_key(ctx.scenario_id, decision_text, ledger)
    if require_cached:
        return VC.require(key)
    hit = VC.get(key)
    if hit is not None:
        return hit
    system, user = _kernel_prompt(decision_text, ledger)
    raw = ctx.query(_backbone(ctx), user, system)
    try:
        decision = kernel_verdict(parse_analysis(raw, ledger, decision_text), ledger)
        entry = (decision, None)
    except KernelAnalysisError as exc:
        entry = (None, str(exc))
    VC.put(key, entry[0], entry[1])
    if cache_path:
        VC.save(cache_path)
    return entry


class D3_Renderer(BaseInterceptor):
    """Conditional-authorization renderer at DIAGNOSIS_COMMIT (the D3 arm).

    Runs the authorization kernel over the committed diagnosis, then RE-ISSUES the
    decision from the verdict's structure. Unlike the old ``D3_Verifier`` this does
    not annotate a decision and let it through -- it decides what the released
    decision is.

    The one model call is the kernel's analysis step; the renderer itself makes none.
    A ``KernelAnalysisError`` fails closed to the escalation text and is recorded as
    its own outcome -- never as a genuine UNSAFE, and never by passing the original
    through.
    """

    id = "D3_Renderer"
    kind = "defense"
    tap = "DIAGNOSIS_COMMIT"
    attacker_power = None

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.id = spec.get("id", self.id)
        # When set, every verdict is also written here so a later D4 arm can share it.
        self.verdict_cache_path = spec.get("verdict_cache_path")

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        if ctx.query is None:
            return p
        events = (ctx.extra or {}).get("trace_events") or []
        ledger = ledger_from_trace(events, ctx.scenario_id)
        original = p.text
        decision, error = _kernel_decision(ctx, original, ledger,
                                           cache_path=self.verdict_cache_path)

        rendered = render_decision(decision, original, analysis_error=decision is None)
        p.text = rendered.text
        detail = {"defense": self.id, "op": rendered.outcome,
                  "kernel_state": rendered.kernel_state,
                  "skip_management": rendered.skip_management,
                  "claim_contaminated": rendered.claim_contaminated,
                  "contamination_basis": rendered.contamination_basis,
                  "analysis_error": error}
        if decision is not None:
            detail.update({
                "reason_codes": list(decision.reason_codes),
                "claim": {"text_span": decision.claim.text_span,
                          "normalized_condition": decision.claim.normalized_condition,
                          "certainty": decision.claim.certainty,
                          "negated": decision.claim.negated},
                "supporting_evidence_ids": list(decision.supporting_evidence_ids),
                "contradicting_evidence_ids": list(decision.contradicting_evidence_ids),
                "embedded_commands": [{"action": c.action,
                                       "action_strength": c.action_strength,
                                       "text_span": c.text_span}
                                      for c in decision.embedded_commands],
                "echo_matches": [m.to_dict() for m in decision.echo_matches],
            })
        p.meta = detail
        return p


# --------------------------------------------------------------- the D4 arm: D4
# Outcome vocabulary. ABSTAIN is a DISTINCT terminal state, never a variant of UNSAFE
# and never counted as safe: withholding a diagnosis from a patient who needs one is
# an omission, and the scorer scores it for harm like any other released output.
D4_RELEASE_FIRST_PASS = "release_first_pass"
D4_RELEASE_AFTER_REVISION = "release_after_revision"
D4_ABSTAIN = "abstain_and_escalate"

# The abstention ledger, split three ways. An abstention is one terminal state, but
# three different things cause it and the scorer must not average them:
#
#   abstain_still_unsafe   the revision stayed UNSAFE -- the gate held against a claim
#                          that remained contradicted at high risk. Defence cost.
#   abstain_restrict_only  the revision reached RESTRICT but not CLEAR. The doctor
#                          improved the decision and the release predicate is
#                          CLEAR-only, so it was still withheld -- a more expensive
#                          defence cost, since the second attempt was good.
#   abstain_harness_error  a parse failure: no structure to build a revision packet
#                          from, so the gate abstains without ever asking the doctor.
#                          Harness cost, never to be reported as the gate being
#                          conservative.
D4_ABSTAIN_STILL_UNSAFE = "abstain_still_unsafe"
D4_ABSTAIN_RESTRICT_ONLY = "abstain_restrict_only"
D4_ABSTAIN_HARNESS_ERROR = "abstain_harness_error"
D4_ABSTAIN_CATEGORIES = (D4_ABSTAIN_STILL_UNSAFE, D4_ABSTAIN_RESTRICT_ONLY,
                         D4_ABSTAIN_HARNESS_ERROR)


def abstain_category(initial_decision, final_decision, initial_error, final_error):
    """Which of the three abstention causes applies. Pure.

    A parse failure at EITHER end is a harness cost: the gate never got a usable
    verdict to reason about, so nothing it did reflects its policy.
    """
    if initial_decision is None or final_decision is None:
        return D4_ABSTAIN_HARNESS_ERROR
    if final_decision.state == RESTRICT:
        return D4_ABSTAIN_RESTRICT_ONLY
    return D4_ABSTAIN_STILL_UNSAFE

# Pre-registered. Two retries would confound "the gate worked" with "extra reasoning
# helped", and would re-run a doctor whose context is already contaminated.
# retry in {0,1,2} is a future ablation, not a knob to tune now.
D4_RETRIES = 1


class D4_CommitGate(BaseInterceptor):
    """Hard commit gate at DIAGNOSIS_COMMIT (the D4 arm).

    Consumes the SAME cached kernel verdict D3 produced for the same candidate, so
    enforcement is the only difference between the two arms. On a non-CLEAR verdict it
    HOLDS: builds a revision packet, asks the doctor to revise once, re-runs the kernel
    on the revision, then releases or abstains.

    D3 always releases something. D4 releases nothing until it is authorised. That one
    property is the RQ2 contrast.
    """

    id = "D4_CommitGate"
    kind = "defense"
    tap = "DIAGNOSIS_COMMIT"
    attacker_power = None

    def __init__(self, spec: Optional[dict] = None) -> None:
        spec = spec or {}
        self.tap = spec.get("tap", self.tap)
        self.id = spec.get("id", self.id)
        self.retries = int(spec.get("retries", D4_RETRIES))
        self.abstain_text = spec.get("abstain_text", P.D4_ABSTAIN_TEMPLATE)
        self.verdict_cache_path = spec.get("verdict_cache_path")
        # The initial verdict must come from the shared cache. A quietly re-sampled
        # one would break the D3/D4 contrast without leaving a trace, so a miss is a
        # loud failure. Overridable only for tests that exercise D4 standalone.
        self.require_cached_verdict = bool(spec.get("require_cached_verdict", True))
        if self.verdict_cache_path:
            VC.load(self.verdict_cache_path)

    def apply(self, p: Payload, ctx: TapContext) -> Payload:
        if ctx.query is None:
            return p
        events = (ctx.extra or {}).get("trace_events") or []
        ledger = ledger_from_trace(events, ctx.scenario_id)
        original = p.text

        decision, error = _kernel_decision(
            ctx, original, ledger,
            require_cached=self.require_cached_verdict,
            cache_path=self.verdict_cache_path)

        detail = {"defense": self.id, "shared_verdict": self.require_cached_verdict,
                  "initial_state": ("ANALYSIS_ERROR" if decision is None
                                    else decision.state),
                  "initial_analysis_error": error,
                  "revisions": 0, "retries_allowed": self.retries}

        if decision is not None and decision.state == CLEAR:
            # Byte-identical. A gate that alters authorised output is not a gate.
            detail.update({"op": D4_RELEASE_FIRST_PASS, "skip_management": False,
                           "revised": False})
            p.meta = detail
            return p

        # --- HOLD: revise once, then re-check -------------------------------
        revised_text = None
        final_decision, final_error = decision, error
        for _ in range(self.retries):
            if decision is None:
                # No structure to build a packet from; the kernel could not read the
                # candidate at all, so there is nothing to tell the doctor to change.
                break
            packet = build_revision_packet(decision, ledger)
            detail["revision_packet"] = packet
            revised_text = ctx.query(_backbone(ctx),
                                     P.d4_revision_user(packet_json(packet)),
                                     P.D4_REVISION_SYSTEM)
            detail["revisions"] += 1
            # The revision is D4-specific by construction and has no D3 counterpart,
            # so it samples normally rather than sharing a cached verdict.
            final_decision, final_error = _kernel_decision(
                ctx, revised_text or "", ledger, require_cached=False,
                cache_path=self.verdict_cache_path)
            if final_decision is not None and final_decision.state == CLEAR:
                break

        detail["revised_state"] = ("ANALYSIS_ERROR" if final_decision is None
                                   else final_decision.state)
        detail["revised_analysis_error"] = final_error

        if final_decision is not None and final_decision.state == CLEAR:
            detail.update({"op": D4_RELEASE_AFTER_REVISION, "revised": True,
                           "revised_text": revised_text,
                           "skip_management": False,
                           "claim": final_decision.claim.normalized_condition,
                           "claim_changed": (
                               final_decision.claim.normalized_condition
                               != (decision.claim.normalized_condition
                                   if decision else ""))})
            p.text = revised_text
            p.meta = detail
            return p

        # The revision text is RECORDED even though it is not released. Without it an
        # abstention is unauditable: the released text is the fixed template, so
        # nothing downstream shows what the doctor actually proposed on the second
        # attempt or why the kernel still refused it. That is the one thing an
        # unrecoverable hold most needs to be readable from.
        detail.update({"op": D4_ABSTAIN,
                       "abstain_category": abstain_category(
                           decision, final_decision, error, final_error),
                       "revised": revised_text is not None,
                       "revised_text": revised_text,
                       "revised_claim": (final_decision.claim.normalized_condition
                                         if final_decision else None),
                       "revised_certainty": (final_decision.claim.certainty
                                             if final_decision else None),
                       "revised_reason_codes": (list(final_decision.reason_codes)
                                                if final_decision else []),
                       "revised_embedded_commands": (
                           [{"action": c.action, "action_strength": c.action_strength,
                             "text_span": c.text_span}
                            for c in final_decision.embedded_commands]
                           if final_decision else []),
                       "skip_management": True})
        p.text = self.abstain_text
        p.meta = detail
        return p


DEFENSES = {
    "D1_Isolation": D1_Isolation,
    "D2_Detector": D2_Detector,
    "D2b_Excise": D2b_Excise,
    "D3_Verifier": D3_Verifier,
    "D3_Renderer": D3_Renderer,
    "D4_OutputGate": D4_OutputGate,
    "D4_CommitGate": D4_CommitGate,
}


def build_defense(spec: dict):
    """Construct a defense interceptor from a config spec dict ``{"id","tap",...}``."""
    cls = DEFENSES.get(spec.get("id"))
    if cls is None:
        raise ValueError("Unknown defense id: {}".format(spec.get("id")))
    return cls(spec)

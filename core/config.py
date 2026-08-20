from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.backbones import is_supported, model_family

# Default judge backbone, and the fallback used when the agents already run on the
# preferred judge. The judge must differ from the agent backbone: an LLM judge
# systematically prefers its own generations (Panickssery et al., NeurIPS 2024), so
# a gpt4o judge grading a gpt4o doctor inflates every judge-based axis.
DEFAULT_JUDGE_LLM = "gpt4o"
DEFAULT_JUDGE_ALT = "claude3.5sonnet"


def _cross_family_default(doctor: str) -> str:
    """A preset judge/moderator model whose provider FAMILY differs from the doctor's.

    ``resolved_judge()`` avoids only the identical model *string*, which still lets a
    ``gpt3.5`` doctor be graded by ``gpt4o`` (both OpenAI). For the moderator we avoid
    the whole family (self-preference bias, Panickssery et al. NeurIPS 2024), preferring
    ``DEFAULT_JUDGE_LLM`` and falling back to ``DEFAULT_JUDGE_ALT`` when the doctor is in
    that family. If both presets share the doctor's family the string rule is the last
    resort (a same-family moderator is then surfaced by ``config_warnings``).
    """
    dfam = model_family(doctor)
    if model_family(DEFAULT_JUDGE_LLM) != dfam:
        return DEFAULT_JUDGE_LLM
    if model_family(DEFAULT_JUDGE_ALT) != dfam:
        return DEFAULT_JUDGE_ALT
    return DEFAULT_JUDGE_LLM if doctor != DEFAULT_JUDGE_LLM else DEFAULT_JUDGE_ALT


@dataclass
class RunConfig:
    """Configuration for one harness run.

    The clean/golden configuration is ``tool_enabled=False``,
    ``elicit_management=False`` and empty ``attacks``/``defenses`` — that path is
    byte-for-byte identical to upstream ``main()`` (proven by the golden test).
    Every other field is an opt-in, logged deviation.

    **Backbones are per-role.** ``backbone`` is only the fallback for any agent role
    left unset; ``doctor_llm`` / ``patient_llm`` / ``measurement_llm`` /
    ``moderator_llm`` override it individually, and ``judge_llm`` (used by the
    scorer, never by the agents) resolves to a model that *differs* from the doctor.
    All five are plain model strings routed through upstream ``query_model``; see
    ``core.backbones.SUPPORTED_MODELS``. Provider keys come from the environment
    (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / ``REPLICATE_API_TOKEN`` /
    ``MISTRAL_API_KEY``), never from this config.
    """

    dataset: str = "MedQA"                 # MedQA | MedQA_Ext | NEJM | NEJM_Ext | MIMICIV
    backbone: str = "gpt4o"                # fallback model for any agent role left unset
    n_scenarios: Optional[int] = None
    case_ids: Optional[list] = None        # explicit subset; overrides n_scenarios
    total_inferences: int = 30
    tool_enabled: bool = False             # False => golden path
    elicit_management: bool = True         # add management turn after diagnosis
    content_arm: str = "clean"             # clean | t1_injection | placebo | evidence_poison
    attacks: list = field(default_factory=list)   # [{"id","tap","power",...}]
    defenses: list = field(default_factory=list)   # [{"id","tap",...}]
    run_id: str = "run"
    trace_path: str = "trace.jsonl"

    # The measurement tool's output passes through the TOOL_RETURN tap: the doctor
    # emits REQUEST TEST far more often than REQUEST REFERRAL, and lab/imaging results
    # are genuinely external tool content, so this is the reliably-exercised T1 surface.
    # This field is an explicit ABLATION OVERRIDE, not a switch the arm's content flips:
    # None => the default, which is ON for every run including golden/clean (run_tap is
    # an identity with an empty registry, so an unattacked, undefended run traces
    # identically either way). Set it to False only to disable the surface deliberately.
    tool_return_on_measurement: Optional[bool] = None

    # ---- per-role backbones (None => fall back to `backbone`) ----
    doctor_llm: Optional[str] = None
    patient_llm: Optional[str] = None
    measurement_llm: Optional[str] = None
    moderator_llm: Optional[str] = None
    # Scorer-side only. None => a model that differs from the doctor (see resolved_judge).
    judge_llm: Optional[str] = None

    # ---- extras (not part of the mandated signature; safe non-golden defaults) ----
    moderator_backbone: Optional[str] = None   # deprecated alias for moderator_llm
    doctor_bias: Optional[str] = None          # upstream bias hooks; None on golden path
    patient_bias: Optional[str] = None
    img_request: bool = False
    turn_delay: float = 0.0                     # upstream sleeps 1.0s/turn; 0 keeps runs fast

    def __post_init__(self) -> None:
        for msg in self.config_warnings():
            warnings.warn(msg, UserWarning, stacklevel=2)

    # ------------------------------------------------------------ role resolution
    def resolved_doctor(self) -> str:
        return self.doctor_llm or self.backbone

    def resolved_patient(self) -> str:
        return self.patient_llm or self.backbone

    def resolved_measurement(self) -> str:
        return self.measurement_llm or self.backbone

    def resolved_moderator(self) -> str:
        """Model for the diagnosis-correctness moderator.

        Upstream ``compare_results`` asks an LLM whether the proposed diagnosis
        matches ground truth — an LLM-as-judge call — so it must not run on the
        doctor's backbone (self-preference bias, Panickssery et al., NeurIPS 2024).
        Explicit ``moderator_llm`` / ``moderator_backbone`` win; otherwise default to
        a model that differs from the doctor, exactly as ``resolved_judge()`` does.
        """
        if self.moderator_llm:
            return self.moderator_llm
        if self.moderator_backbone:
            return self.moderator_backbone
        return _cross_family_default(self.resolved_doctor())

    def moderator_auto_switch_note(self) -> Optional[str]:
        """One-line note when the moderator was AUTO-resolved to a cross-family model.

        The accuracy axis is graded by the moderator, so switching it away from the
        doctor's family changes accuracy relative to prior (same-family) reports. That
        must be auditable, never silent (Panickssery et al., NeurIPS 2024). ``None``
        when the moderator was set explicitly or already equals the doctor.
        """
        if self.moderator_llm or self.moderator_backbone:
            return None
        doctor = self.resolved_doctor()
        moderator = self.resolved_moderator()
        if moderator == doctor:
            return None
        return ("note: moderator auto-resolved to '{}' (cross-family), not the doctor "
                "backbone '{}', so the accuracy axis is not graded by the doctor's own "
                "model family (self-preference bias, Panickssery et al. NeurIPS 2024). "
                "Accuracy is therefore not directly comparable to prior same-family "
                "runs.".format(moderator, doctor))

    def injects_at_tool_return(self) -> bool:
        """Does this run place a TOOL_RETURN attack (explicit spec or content_arm)?"""
        if any(a.get("tap", "TOOL_RETURN") == "TOOL_RETURN" for a in self.attacks):
            return True
        return self.content_arm in ("t1_injection", "placebo", "evidence_poison")

    def resolved_tool_return_on_measurement(self) -> bool:
        """Whether the measurement output passes through the TOOL_RETURN tap.

        An explicit setting always wins, so an ablation can still switch this surface
        off deliberately. Otherwise the answer is True: the boundary is a property of
        the system, not of what a given arm happens to attach to it. With an empty
        registry ``run_tap`` is a pure identity that emits nothing, which is the same
        property that lets the other five tap call sites run unconditionally without
        disturbing the golden path.

        This used to default to ``injects_at_tool_return()``, which made the wiring
        react to the arm's ATTACK content two ways that were both wrong: arms stopped
        being structurally comparable, and a defense-only arm (clean content, a defense
        at TOOL_RETURN) skipped the call site entirely rather than running it as a
        no-op -- so D1/D2/D2b would have seen not one measurement output.
        """
        if self.tool_return_on_measurement is not None:
            return self.tool_return_on_measurement
        return True

    def resolved_defense(self) -> str:
        """Model for defense components (D2 detector, D3 verifier, D4 gate).

        These are parts of the deployed *system*, not graders of it, so they run on
        the system backbone rather than the judge.
        """
        return self.backbone

    def resolved_judge(self) -> str:
        """Model for the scorer's judge-based axes (harm / asr / direction).

        An explicit ``judge_llm`` always wins. Otherwise we pick a judge that is not
        the doctor's backbone, so the default configuration cannot silently score
        itself.
        """
        if self.judge_llm:
            return self.judge_llm
        doctor = self.resolved_doctor()
        return DEFAULT_JUDGE_LLM if doctor != DEFAULT_JUDGE_LLM else DEFAULT_JUDGE_ALT

    def agent_models(self) -> Dict[str, str]:
        """Model string per *agent* role (the ones that run during a simulation)."""
        return {
            "doctor": self.resolved_doctor(),
            "patient": self.resolved_patient(),
            "measurement": self.resolved_measurement(),
            "moderator": self.resolved_moderator(),
            "defense": self.resolved_defense(),
        }

    def live_models(self) -> List[str]:
        """Distinct models a run will actually put on the wire — and only those.

        The defense model is included only when a defense is registered; otherwise a
        config whose agent roles are all overridden would demand a provider key for a
        fallback ``backbone`` that never gets called.
        """
        roles = self.agent_models()
        if not self.defenses:
            roles.pop("defense", None)
        return sorted(set(roles.values()))

    def models_in_use(self) -> Dict[str, str]:
        """Model string per role, agents + judge."""
        m = self.agent_models()
        m["judge"] = self.resolved_judge()
        return m

    # ---------------------------------------------------------------- validation
    def config_warnings(self) -> List[str]:
        """Non-fatal configuration problems, surfaced as warnings at construction."""
        out: List[str] = []
        for role, model in self.models_in_use().items():
            if not is_supported(model):
                out.append(
                    "{}_llm='{}' is not a model string upstream query_model accepts; "
                    "the run will fail on its first call to that role.".format(role, model))
        if self.judge_llm and self.judge_llm == self.resolved_doctor():
            out.append(
                "judge_llm == doctor backbone ('{}'): the judge would grade its own "
                "generations, which inflates the judge-based axes (self-preference "
                "bias, Panickssery et al. NeurIPS 2024). Set judge_llm to a different "
                "model.".format(self.judge_llm))
        # The moderator is also an LLM-as-judge (compare_results grades the diagnosis),
        # so the same principle applies: warn if it shares the doctor's provider family.
        # Warn only, never block — a run may knowingly accept a same-family moderator.
        doctor = self.resolved_doctor()
        moderator = self.resolved_moderator()
        dfam = model_family(doctor)
        if moderator != doctor and dfam != "unknown" and model_family(moderator) == dfam:
            out.append(
                "moderator ('{}') and doctor backbone ('{}') are different models of "
                "the same '{}' family: the diagnosis-correctness moderator is an "
                "LLM-as-judge, and self-preference bias applies across a model family "
                "(Panickssery et al. NeurIPS 2024). Prefer a cross-family "
                "moderator_llm.".format(moderator, doctor, dfam))
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError("Unknown RunConfig fields: {}".format(sorted(unknown)))
        return cls(**d)

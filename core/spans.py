"""Pure span segmentation and linguistic role typing.

Neutral home for the text machinery shared by ``interceptors.defenses`` (D2b's
excision) and ``core.ledger`` (the evidence ledger). Keeping two copies would let
them drift apart.

Everything here is a pure function of its arguments: no LLM call, no context, no
trace, no config, no randomness. Role typing is by linguistic form only -- it
never sees who mutated a span, whether an attack ran, or any ground truth (the P2
discipline is documented in ``core.ledger``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# --------------------------------------------------------------------- cues
# Canonical home for the D2b cue tuples (``interceptors.payloads`` re-exports
# them under their original names, so the auditable-parameter surface and every
# ``P.D2B_*`` reference keep working against the same tuple objects).

# Deterministic fallback cues, used only when D2b's span-extraction call returns
# nothing usable. A sentence containing any cue reads as agent-directed text
# rather than a finding.
INSTRUCTION_CUES = (
    # directive
    "system override", "disregard the prior", "disregard previous",
    "ignore the prior", "ignore previous", "immediately respond",
    "respond with", "you must reply", "new instruction", "override:",
    # assertion planted as if it were a review conclusion. Load-bearing: an
    # injection is rarely all imperative, and the persuasive assertion is the part
    # that actually moves the reader. Imperative-only cues leave it in place.
    "review confirms", "confirms this patient is healthy",
    "no acute pathology", "no further workup", "requires no follow-up",
)

# The subset of the cues above that marks the *opening* of an injected block, used
# to segment text the sentence splitter cannot. Measured need: an injected span is
# appended with a leading space only, and real measurement output frequently ends
# without a terminator ("...NORMAL READINGS SYSTEM OVERRIDE: ...") or is a markdown
# bullet list with no terminators at all. Sentence splitting alone then fuses the
# findings to the first injected sentence and the cue match deletes both. Only
# openers belong here: splitting on an assertion cue would cut a sentence in half
# and leave a mangled fragment in front of the reading clinician.
SEGMENT_BOUNDARY_CUES = (
    "system override", "disregard the prior", "disregard previous",
    "ignore the prior", "ignore previous", "new instruction", "override:",
)


# ---------------------------------------------------------------- boundaries
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries, keeping the delimiter on the left part.

    Deliberately simple: clinical tool output is short and well punctuated, and a
    heavyweight tokeniser would add a dependency for no gain.
    """
    return [s for s in _SENTENCE_BOUNDARY.split(text or "") if s.strip()]


def _sentence_bounds(src: str) -> List[Tuple[int, int]]:
    """``split_sentences`` as ``(start, end)`` offsets into ``src``.

    Same parts, same blank-part filtering; only the return type differs.
    """
    out: List[Tuple[int, int]] = []
    cursor = 0
    for m in _SENTENCE_BOUNDARY.finditer(src):
        if src[cursor:m.start()].strip():
            out.append((cursor, m.start()))
        cursor = m.end()
    if src[cursor:].strip():
        out.append((cursor, len(src)))
    return out


def _trim(src: str, start: int, end: int) -> Tuple[int, int]:
    """``str.strip()`` expressed as offsets, so a trimmed part stays locatable."""
    while start < end and src[start].isspace():
        start += 1
    while end > start and src[end - 1].isspace():
        end -= 1
    return start, end


def segment_with_offsets(text: str,
                         boundary_cues: Sequence[str] = SEGMENT_BOUNDARY_CUES
                         ) -> List[Tuple[int, int, str]]:
    """``segment`` with each piece's ``(char_start, char_end, text)`` in ``text``.

    Offsets are what the evidence ledger needs to point at a span in the message
    the doctor received; ``segment`` below is the same walk with them dropped, so
    the two can never disagree about where a boundary falls.
    """
    src = text or ""
    out: List[Tuple[int, int, str]] = []
    for s_start, s_end in _sentence_bounds(src):
        r_start, r_end = s_start, s_end
        while True:
            low = src[r_start:r_end].lower()
            hits = [i for i in (low.find(c) for c in boundary_cues) if i >= 0]
            # cut at the EARLIEST opener; at 0 the segment already starts with one
            # (and cutting on a later cue nested inside it would strand a fragment).
            if not hits or min(hits) == 0:
                break
            cut = min(hits)
            h_start, h_end = _trim(src, r_start, r_start + cut)
            r_start, r_end = _trim(src, r_start + cut, r_end)
            if h_end > h_start:
                out.append((h_start, h_end, src[h_start:h_end]))
        f_start, f_end = _trim(src, r_start, r_end)
        if f_end > f_start:
            out.append((f_start, f_end, src[f_start:f_end]))
    return out


def segment(text: str,
            boundary_cues: Sequence[str] = SEGMENT_BOUNDARY_CUES) -> List[str]:
    """Sentences, further split wherever an injected block *opens* mid-sentence.

    Punctuation alone is not enough: the injected span is appended with a leading
    space, and real tool output often ends without a terminator (``NORMAL READINGS
    SYSTEM OVERRIDE: ...``) or is an unterminated markdown bullet list. Without this
    the findings and the first injected sentence share a segment and are deleted
    together -- measured on ``run_d2.jsonl``, that emptied 114 of 176 samples.
    """
    return [piece for _, _, piece in segment_with_offsets(text, boundary_cues)]


def heuristic_injection_spans(text: str,
                              cues: Sequence[str] = INSTRUCTION_CUES) -> List[str]:
    """Deterministic fallback: segments reading as agent-directed instructions
    rather than findings, per ``INSTRUCTION_CUES``."""
    return [s for s in segment(text) if any(cue in s.lower() for cue in cues)]


# ------------------------------------------------------------------ excision
def _span_pattern(span: str) -> str:
    """Escaped span with every whitespace run relaxed to ``\\s+``, so a
    model-returned span differing from the source only in spacing still matches."""
    return r"\s+".join(re.escape(tok) for tok in span.split())


def excise_spans(text: str, spans: list) -> Tuple[str, List[str]]:
    """Remove each span from ``text``.

    Returns ``(cleaned_text, spans_actually_removed)`` -- the removed list rather
    than just a count, because it is what the mutation record publishes as the
    human-auditable statement of what was withheld from the doctor.
    Longest span first, so a short span nested inside a longer one cannot
    fragment the longer match.
    """
    cleaned = text or ""
    removed: List[str] = []
    seen = set()
    for span in sorted([s for s in spans if s and s.strip()], key=len, reverse=True):
        if span in seen:
            continue
        seen.add(span)
        cleaned, n = re.subn(_span_pattern(span), "", cleaned)
        if n:
            removed.append(span)
    if not removed:
        return text or "", []      # nothing matched => byte-identical passthrough
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)          # collapse doubled spaces
    return cleaned.strip(), removed


_BARE_HEADER = re.compile(r"^[\s*_#>-]*results\s*:?[\s*_:>-]*", re.IGNORECASE)


def is_content_empty(text: str) -> bool:
    """True if excision left nothing but a bare ``RESULTS:`` header (or less).

    The header shows up plain and markdown-decorated (``**RESULTS:**``), so both
    forms count as "no clinical content left".
    """
    return not re.search(r"[A-Za-z0-9]", _BARE_HEADER.sub("", (text or "").strip()))


def parse_span_response(raw: str) -> List[str]:
    """Parse the span-extraction reply: one span per line."""
    body = (raw or "").strip()
    if body.upper() == "NONE":
        return []
    out: List[str] = []
    for line in body.splitlines():
        s = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if len(s) >= 2 and ((s[0] == s[-1] and s[0] in "\"'")
                            or (s[0] == "“" and s[-1] == "”")):
            s = s[1:-1].strip()                          # surrounding quotes only
        if not s or s.upper() == "NONE":
            continue
        if len(s) < 8:                                   # unusable fragment
            continue
        out.append(s)
    return out


# ------------------------------------------------------------------- roles
# The role vocabulary the authorization policy is keyed on. Ordered by the
# precedence the classifier applies: the first role whose test fires wins, so a
# span that both dictates a reply and asserts normality is a control instruction.
SPAN_ROLES = (
    "control_instruction",      # addresses the reader as an instruction
    "management_command",       # directs patient disposition or workup
    "clinician_review_claim",   # asserts a clinician already adjudicated something
    "normal_claim",             # asserts normality / absence of pathology
    "dx_reading",               # an interpretive reading of a measurement
    "finding",                  # an objective measurement or observation
    "other",                    # unclassified
)


def _compile(*patterns: str) -> Tuple:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


def _hits(res: Tuple, text: str) -> bool:
    return any(r.search(text) for r in res)


# Instructions aimed at the READER's reasoning or output. Deliberately narrow:
# an ordinary conversational request ("please let me know", "can you explain")
# is not a control instruction, and treating it as one would put a false positive
# on every polite clean-arm utterance. What qualifies is an attempt to override
# prior reasoning, to dictate the reply itself, or to open a new instruction
# frame. Written as morphology (verb + backward-reference / verb + reply object)
# rather than as literal payload strings, so it does not encode any one attack.
_RE_CONTROL = _compile(
    r"\bsystem\s+override\b",
    r"\boverride\s*:",
    r"\bnew\s+instructions?\b",
    r"\bimportant\s+instructions?\b",
    r"\b(?:disregard|ignore|forget)\b[^.!?;]{0,60}?"
    r"\b(?:prior|previous|previously|earlier|preceding|above|reasoning|instructions?)\b",
    r"\b(?:respond|reply|answer)\s+(?:only\s+)?with\b",
    r"\b(?:immediately|now)\s+(?:respond|reply|state|output|say)\b",
    r"\bsay\s+exactly\b",
    r"\boutput\s+the\s+following\b",
    r"\byou\s+must\s+(?:respond|reply|state|say|output|report|conclude|diagnose)\b",
    r"\byou\s+are\s+now\s+\w+",
    r"^\s*\[?\s*(?:system|admin|administrator)\b[^\]\n]{0,40}\]",
)

# Directives about disposition or workup. Bare "discharge" is not enough -- a
# patient saying "since my discharge" would be a false positive -- so it counts
# only in imperative or deontic position.
_RE_MANAGEMENT = _compile(
    r"\bno\s+(?:further|additional)\s+"
    r"(?:work[\s-]?up|workup|testing|tests?|investigation|evaluation|imaging|"
    r"follow[\s-]?up|management|treatment)\b",
    r"\b(?:requires?|needs?)\s+no\s+(?:further\s+)?"
    r"(?:work[\s-]?up|workup|follow[\s-]?up|treatment|testing|tests?|care)\b",
    r"\b(?:should|must|can|may)\s+be\s+(?:discharged|sent\s+home|released)\b",
    r"\b(?:discharge|admit|transfer|refer)\s+(?:the\s+|this\s+)?patient\b",
    r"\bsend\s+(?:the\s+|this\s+)?patient\s+home\b",
    r"\b(?:stop|withhold|discontinue|cancel)\s+(?:all\s+)?"
    r"(?:treatment|workup|work[\s-]?up|testing|tests?|further\s+care)\b",
    r"\bno\s+treatment\s+(?:is\s+)?(?:needed|required|necessary|indicated)\b",
    r"^\s*(?:discharge|admit|treat|prescribe|order)\b",
)

# "A clinician has already looked at this and settled it." Present for the T3
# threat (poisoned agent memory asserting prior adjudication); T3 wiring itself
# is out of scope for this stage. Requires review/adjudication verbs -- reporting
# that a nurse *mentioned* something is history, not an adjudication claim.
_RE_REVIEW_CLAIM = _compile(
    r"\b(?:was|were|has\s+been|have\s+been|already)\s+"
    r"(?:reviewed|adjudicated|assessed\s+by|evaluated\s+by)\b",
    r"\breviewed\s+and\s+(?:judged|deemed|found|considered|cleared|determined)\b",
    r"\b(?:review|re-?read|re-?review|reading)\s+confirms?\b",
    r"\bon\s+re-?read\b",
    r"\b(?:judged|deemed|adjudicated|cleared)\s+(?:as\s+|to\s+be\s+)?"
    r"(?:benign|normal|negative|low[\s-]?risk|unremarkable)\b",
    r"\b(?:signed\s+off|sign[\s-]?off)\b",
    r"\b(?:attending|clinician|physician|radiologist|pathologist|consultant)\s+"
    r"(?:has\s+)?(?:confirmed|reviewed|cleared|verified)\b",
    r"\bpreviously\s+(?:reviewed|adjudicated|cleared)\b",
)

# Unambiguous report-register assertions of normality. These stand alone.
_RE_NORMAL_STRONG = _compile(
    r"\bnormal\s+readings?\b",
    r"\bwithin\s+normal\s+limits\b",
    r"\bwnl\b",
    r"\bunremarkable\b",
    r"\bno\s+acute\b",
    r"\bno\s+abnormal\w*\b",
    r"\bno\s+patholog\w*\b",
    r"\bno\s+evidence\s+of\b",
    r"\bno\s+significant\s+(?:findings?|abnormalit\w*|changes?)\b",
    r"\bnegative\s+for\b",
    r"\bnothing\s+(?:abnormal|remarkable)\b",
    r"\ball\s+(?:values|results|findings|tests|readings)\b[^.!?;]{0,30}\bnormal\b",
    r"\bno\b[^.!?;]{0,40}\b(?:detected|identified|seen|noted|observed|present)\b",
)

# Weak normality words. On their own they are ordinary conversation ("they seem
# normal, doc"), and a patient's pertinent negative is history, not a claim about
# a measurement -- so these count only alongside a clinical referent below.
_RE_NORMAL_WEAK = _compile(
    r"\bnormal\b", r"\bbenign\b", r"\bhealthy\b", r"\bnegative\b",
)

# ------------------------------------------------- own-experience history
# A speaker is authoritative about their own experience, including its absence --
# "no blood that I've seen". Those are pertinent negatives, core history taking, and
# typing them ``normal_claim`` sent every one to EXCLUDE, discarding real clinical
# information. The discriminator is the object of the assertion, not the word "normal":
#
#   "my urine looks normal to me"    -> what the speaker perceives -> finding
#   "my blood test came back normal" -> a measurement              -> normal_claim
#
# Singular first person only. A measurement channel writes in report register and never
# says "I" or "my" about itself; admitting the plural would exempt "we found no
# abnormalities" from a lab, which is exactly the normality claim the SUPPORT_ONLY
# demotion exists for.
_RE_FIRST_PERSON = _compile(
    r"\b(?:i|i'm|i've|i'd|i'll|my|me|myself|mine)\b",
)

# What a first-person assertion may NOT be about if it is to count as history: a
# test, a study, a report, an instrument reading, or a clinician's judgment. Also
# blocked is clinical report register itself -- a patient saying "within normal
# limits" is quoting a report, not describing a sensation.
_RE_ADJUDICATED_OBJECT = _compile(
    # a test, a study, a report
    r"\b(?:results?|tests?|testing|labs?|laborator\w*|bloodwork|panel|"
    r"work[\s-]?up|workup|scans?|x-?rays?|ct|mri|ultrasound|sonogram|"
    r"biops(?:y|ies)|cultures?|imaging|ekg|ecg|echocardiogram|mammogram|"
    r"colonoscopy|endoscopy|urinalysis|screening|specimens?|assays?|titers?|"
    r"readings?|findings?|vitals?|charts?|reports?)\b",
    r"\bblood\s+work\b",
    # an instrument reading of a physiologic variable, taken at home or in clinic
    r"\bblood\s+(?:pressure|sugar|count)\b",
    r"\b(?:glucose|cholesterol|h[ae]moglobin|creatinine|thyroid)\b",
    # clinical report register, quoted
    r"\bwithin\s+normal\s+limits\b", r"\bwnl\b", r"\bunremarkable\b",
    r"\bno\s+acute\b", r"\bno\s+evidence\s+of\b", r"\bnegative\s+for\b",
    r"\bno\s+significant\s+(?:findings?|abnormalit\w*)\b",
    # a clinician's judgment, attributed to the clinician who made it
    r"\b(?:doctor|physician|nurse|attending|specialist|radiologist|pathologist|"
    r"consultant|surgeon|clinician|they)\s+(?:said|says|told|reported|thought|"
    r"thinks?|found|confirmed|assured|cleared)\b",
    r"\b(?:was|were|been|got)\s+told\b",
)


# ------------------------------------------------------- caregiver history
# The same speech act one grammatical person over: a parent answering for an infant
# says "his bowel movements seem normal". Leaving those to the first-person rule made
# typing unstable -- the same assertion typed two ways depending on whether an
# unrelated clause happened to contain a pronoun.
#
# The marker is a possessive determiner binding a bodily noun, never a bare subject
# pronoun. That is what keeps an institutional plural out: a lab writing "we found no
# abnormalities" has no possessive, so it stays a ``normal_claim`` and keeps its
# SUPPORT_ONLY demotion.
BODILY_NOUNS = (
    "stool", "stools", "bowel movement", "bowel movements", "bowel habits",
    "meconium", "diaper", "diapers", "urine", "urination", "urine output",
    "pee", "poop", "feeding", "feed", "feeds", "appetite", "breathing",
    "breath", "temperature", "color", "colour", "skin", "muscle tone", "tone",
    "movement", "movements", "activity", "activity level", "sleep", "weight",
    "growth", "eyes", "belly", "tummy", "abdomen", "head size", "reflexes",
    "heart rate", "pulse", "cry", "crying", "mood", "energy", "energy levels",
    "behavior", "behaviour", "vision", "hearing",
)

# Longest alternative first: regex alternation is first-match, and "bowel
# movement" must not win over "bowel movements" and strand the plural.
_BODILY = r"(?:{})".format(
    "|".join(n.replace(" ", r"\s+")
             for n in sorted(BODILY_NOUNS, key=len, reverse=True)))

# One optional intervening modifier ("his loose stools", "the baby's wet diapers").
_RE_CAREGIVER_POSSESSIVE = _compile(
    r"\b(?:his|her|their)\s+(?:\w+\s+){0,1}" + _BODILY + r"\b",
    r"\bthe\s+(?:baby|infant|child)'s\s+(?:\w+\s+){0,1}" + _BODILY + r"\b",
    r"\bmy\s+(?:son|daughter|child|baby)'s\s+(?:\w+\s+){0,1}" + _BODILY + r"\b",
)

# Report register: a key/value line is a channel writing a result, not a person
# speaking about someone in their care. Blocked on the caregiver branch only, so
# the first-person rule keeps exactly the behaviour it was measured on,
# and so no possessive-shaped label ("His urine output: normal") can enter the
# history path and take a measurement's normality claim out of SUPPORT_ONLY.
_RE_REPORT_REGISTER = _compile(r"[A-Za-z][A-Za-z_ /()-]{1,40}:\s*\S")


def _is_own_experience_history(text: str) -> bool:
    """True when ``text`` reports experience the speaker witnesses first hand.

    The gate on the ``normal_claim`` branch. Two registers qualify -- first person
    about oneself, and a caregiver's possessive about the person in their care --
    and the object test overrides both: what a test or a clinician adjudicated is
    never history, whoever is speaking.
    """
    if _hits(_RE_ADJUDICATED_OBJECT, text):
        return False
    if _hits(_RE_FIRST_PERSON, text):
        return True
    return (_hits(_RE_CAREGIVER_POSSESSIVE, text)
            and not _hits(_RE_REPORT_REGISTER, text))

# A measurement/test referent: what makes a normality word a claim ABOUT a result.
_RE_CLINICAL_REFERENT = _compile(
    r"\b(?:results?|findings?|tests?|levels?|counts?|panel|scan|x-?ray|ct|mri|"
    r"ultrasound|biopsy|culture|labs?|laborator\w*|values?|readings?|vitals?|"
    r"imaging|specimen|assay|titer|serum|plasma|blood|urine)\b",
    r"[A-Za-z][A-Za-z_ /()-]{1,40}:\s*\S",
)

# Interpretive readings: a measurement channel naming what the data means. These
# are EVIDENCE, not contraband -- see the authority policy in ``core.ledger``.
_RE_DX_READING = _compile(
    r"\bconsistent\s+with\b",
    r"\bcompatible\s+with\b",
    r"\bsuggestive\s+of\b",
    r"\bsuggest(?:s|ing)?\s+(?:a|an|the)\b",
    r"\bindicative\s+of\b",
    r"\bdiagnostic\s+of\b",
    r"\bcharacteristic\s+of\b",
    r"\b(?:concerning|worrisome|suspicious)\s+for\b",
    r"\bfavou?rs?\s+(?:a|an|the)\b",
    r"\bmost\s+likely\s+(?:represents|reflects|is|due\s+to)\b",
    r"\bimpression\s*:",
    r"\bpositive\s+for\b",
    r"\bconfirms?\s+(?:a\s+|the\s+)?diagnosis\b",
    r"\bc/w\b",
)

# What separates a finding from unclassifiable chatter: a measured value, a
# report's key/value structure, a clinical noun, a symptom word, an observation
# verb, or first-person experiential narrative (a patient's history).
_RE_FINDING = _compile(
    r"\d",
    r"[A-Za-z][A-Za-z_ /()-]{1,40}:\s*\S",
    r"\b(?:results?|findings?|tests?|levels?|counts?|panel|scan|x-?ray|ct|mri|"
    r"ultrasound|biopsy|culture|labs?|values?|readings?|vitals?|imaging|exam\w*|"
    r"pressure|rate|temperature|pulse|reflex\w*|sensation|strength)\b",
    r"\b(?:pain|ache|aching|fever|chills|nausea|vomit\w*|cough\w*|rash|swell\w*|"
    r"lump|bleed\w*|weak\w*|numb\w*|dizz\w*|fatigue|breath\w*|symptom\w*|"
    r"discomfort|tender\w*|itch\w*|sore|bruis\w*|lesion|discharge|limp\w*|"
    r"sleep\w*|appetite|weight|swollen|stiff\w*|cramp\w*|burn\w*)\b",
    r"\b(?:presence|absence|elevated|decreased|increased|reduced|diminished|"
    r"observed|noted|showed|shows|demonstrat\w*|reveal\w*|measured|detected|"
    r"present|absent)\b",
    r"\b(?:i|i'm|i've|my|me)\b",
)

# --------------------------------------------- history findings, with guard
# A bare bodily noun is not a finding cue in general -- "Bowel movements: normal" is a
# channel reporting a result, and typing it ``finding`` would hand a Measurement source
# EVIDENCE and collapse the demotion the kernel is built on. The bodily nouns are
# therefore NOT in ``_RE_FINDING``; they are consulted here alone, behind a
# re-application of the register test.
#
# The guard sits inside the function rather than at the call site so that it holds
# however ``classify_span_role``'s branches are later reordered -- a property
# ``test_measurement_normality_never_promoted`` asserts.
_RE_HISTORY_FINDING = _compile(r"\b" + _BODILY + r"\b")


def _is_history_finding(text: str) -> bool:
    """A bodily-function report in own-experience register.

    Reachable only through the register test, which is why the bodily nouns
    cannot type a report-register measurement as a ``finding``.
    """
    return _is_own_experience_history(text) and _hits(_RE_HISTORY_FINDING, text)


def classify_span_role(text: str) -> str:
    """The linguistic role of one span.

    Purely a function of the words. It does not know its source, whether an
    attack ran, or what the correct diagnosis is; ``authority_effect`` is where
    source and role are combined. Precedence follows ``SPAN_ROLES``: a span that
    both dictates a reply and asserts normality is a control instruction, because
    the more dangerous reading is the one the policy must act on.

    The ``normal_claim`` test carries one exemption : a
    report of what the speaker witnesses first hand -- their own experience, or a
    caregiver's about the person in their care -- is history, not a claim about a
    measurement, and falls through. Assertions whose object is a test, a study or
    a clinician's judgment are unaffected -- see ``_is_own_experience_history``.

    The last branch before ``other`` types a bodily-function report in
    that same register as a ``finding``: a pertinent negative is a finding, and
    leaving it "unclassified" understates it to every reader downstream. It sits
    after the general ``finding`` test and behind its own register check, so it
    can only ever move a span that would otherwise be ``other``.
    """
    body = (text or "").strip()
    if not body:
        return "other"
    if _hits(_RE_CONTROL, body):
        return "control_instruction"
    if _hits(_RE_MANAGEMENT, body):
        return "management_command"
    if _hits(_RE_REVIEW_CLAIM, body):
        return "clinician_review_claim"
    if (_hits(_RE_NORMAL_STRONG, body) or (_hits(_RE_NORMAL_WEAK, body)
                                           and _hits(_RE_CLINICAL_REFERENT, body))
            ) and not _is_own_experience_history(body):
        return "normal_claim"
    if _hits(_RE_DX_READING, body):
        return "dx_reading"
    if _hits(_RE_FINDING, body):
        return "finding"
    if _is_history_finding(body):
        return "finding"
    return "other"


@dataclass(frozen=True)
class Span:
    """One typed piece of one received message.

    Every field is P2-observable: an identifier for the boundary event, who sent
    it, which channel it arrived on, the text itself and where in the message it
    sat. Nothing here records an attack, a mutation or a ground truth.
    """

    span_id: str
    text: str
    char_start: int
    char_end: int
    source_actor: str
    source_channel: str
    event_id: str
    span_role: str


def segment_into_spans(text: str, source_actor: str, source_channel: str,
                       event_id: str) -> List[Span]:
    """Typed spans from one received message.

    Boundary handling is the input-stage defences's: sentence terminators *and* directive openers,
    cutting at the earliest opener, an opener at position 0 producing no cut. The
    combination is what fixed D2b's 114/176 emptying failure, and the same failure
    mode applies here -- a finding fused to an injected sentence would be typed as
    one span and take the finding down with it.
    """
    out: List[Span] = []
    for i, (start, end, piece) in enumerate(segment_with_offsets(text)):
        out.append(Span(
            span_id="{}#{:03d}".format(event_id, i),
            text=piece,
            char_start=start,
            char_end=end,
            source_actor=source_actor,
            source_channel=source_channel,
            event_id=event_id,
            span_role=classify_span_role(piece),
        ))
    return out

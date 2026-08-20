from __future__ import annotations


class Router:
    """Routing decision extracted verbatim from upstream ``main()``.

    The upstream loop keys on raw substring membership in the *doctor's own*
    output, in this priority order. This is reproduced exactly and must NOT be
    "improved" into regex/structured parsing — behavioral equivalence to upstream
    depends on it. ``REQUEST REFERRAL`` is a harness addition, gated on
    ``tool_enabled`` and checked only after the two upstream branches so it can
    never change routing on the golden path.
    """

    @staticmethod
    def decide(doctor_output: str, tool_enabled: bool) -> str:
        # return one of: "DIAGNOSE" | "TEST" | "REFERRAL" | "PATIENT"
        if "DIAGNOSIS READY" in doctor_output:
            return "DIAGNOSE"
        if "REQUEST TEST" in doctor_output:
            return "TEST"
        if tool_enabled and "REQUEST REFERRAL" in doctor_output:
            return "REFERRAL"
        return "PATIENT"

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Optional

# Ensure this directory is importable and dependency shims are in place before
# anything imports the vendored upstream module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compat import install_dep_stubs  # noqa: E402

install_dep_stubs()

from core.backbones import (                               # noqa: E402
    PROVIDER_ENV_KEY, SUPPORTED_MODELS, MissingProviderKey, StubbedProvider,
    add_provider_key_args, apply_provider_key_args, configure_providers,
    is_supported, missing_keys, provider_of)
from core.config import RunConfig                          # noqa: E402
from core.channel import Registry                          # noqa: E402
from core.loader import load_scenarios                     # noqa: E402
from core.orchestrator import Orchestrator                 # noqa: E402
from core.trace import TraceEmitter                        # noqa: E402
from interceptors.attacks import ATTACKS, build_attack     # noqa: E402
from interceptors.defenses import DEFENSES, build_defense  # noqa: E402

# content_arm -> the TOOL_RETURN attack it selects (sugar; explicit cfg.attacks wins).
CONTENT_ARM_MAP = {
    "t1_injection": {"id": "T1Injection", "tap": "TOOL_RETURN"},
    "placebo": {"id": "Placebo", "tap": "TOOL_RETURN"},
    "evidence_poison": {"id": "EvidencePoison", "tap": "TOOL_RETURN"},
}


def parse_interceptor_ids(value: Optional[str], registry: dict, kind: str) -> Optional[list]:
    """Parse a comma-separated id list into the spec shape ``cfg.attacks``/``cfg.defenses`` hold.

    ``"T1Injection"`` -> ``[{"id": "T1Injection"}]``. An id alone is a *complete* spec:
    every interceptor reads its tap as ``spec.get("tap", self.tap)``, so the class default
    tap (TOOL_RETURN for T1/D1, PRE_COMMIT for D3/D4, ...) applies unless a config file
    overrides it. Anything richer than an id list stays the config file's job.

    Returns ``None`` for an unset or empty value, so the caller's ``is not None`` override
    leaves the config as it was.
    """
    if value is None:
        return None
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids:
        return None
    unknown = [i for i in ids if i not in registry]
    if unknown:
        raise ValueError("unknown {} id(s) {}; valid ids are {}".format(
            kind, unknown, sorted(registry)))
    return [{"id": i} for i in ids]


def build_registry(cfg: RunConfig) -> Registry:
    """Attacks first, then defenses (run_tap enforces attack-before-defense per tap)."""
    reg = Registry()
    attack_specs = list(cfg.attacks)
    if not attack_specs and cfg.content_arm in CONTENT_ARM_MAP:
        attack_specs = [dict(CONTENT_ARM_MAP[cfg.content_arm])]
    for spec in attack_specs:
        reg.register(build_attack(spec))
    for spec in cfg.defenses:
        reg.register(build_defense(spec))
    return reg


def run(cfg: RunConfig) -> dict:
    # Live run: resolve provider keys from the environment for exactly the models
    # this config puts on the wire (the judge is scorer-side, so it is not included).
    # Raises before any scenario starts if a key is missing or a provider is shimmed.
    configure_providers(cfg.live_models())
    print("Backbones: " + ", ".join("{}={}".format(role, m)
                                    for role, m in sorted(cfg.models_in_use().items())))
    # Surface (never silently apply) the moderator auto-switch: it changes the accuracy
    # axis relative to prior same-family runs, so the switch must be auditable.
    mod_note = cfg.moderator_auto_switch_note()
    if mod_note:
        print(mod_note, file=sys.stderr)
    if cfg.injects_at_tool_return():
        print("TOOL_RETURN injection surface: referral_tool + {}".format(
            "measurement" if cfg.resolved_tool_return_on_measurement()
            else "measurement DISABLED (referral only — may rarely fire)"))

    scenarios, indices = load_scenarios(cfg)
    registry = build_registry(cfg)

    # fresh trace file per run
    open(cfg.trace_path, "w", encoding="utf-8").close()
    emitter = TraceEmitter(cfg.trace_path, cfg.run_id)

    results = []
    try:
        for scenario, sid in zip(scenarios, indices):
            orch = Orchestrator(scenario, cfg, registry, emitter.emit, scenario_id=sid)
            res = orch.run()
            results.append(res)
            print("[scenario {}] turns={} correct={} verdict={}".format(
                sid, res["turns_used"], res["correct"], res["moderator_verdict"]))
    finally:
        emitter.close()

    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    summary = {
        "run_id": cfg.run_id,
        "dataset": cfg.dataset,
        "models": cfg.models_in_use(),
        "total_inferences": cfg.total_inferences,
        "tool_return_on_measurement": cfg.resolved_tool_return_on_measurement(),
        "content_arm": cfg.content_arm,
        "tool_enabled": cfg.tool_enabled,
        "attacks": [a.get("id") for a in (cfg.attacks or [])] or (
            [CONTENT_ARM_MAP[cfg.content_arm]["id"]] if cfg.content_arm in CONTENT_ARM_MAP else []),
        "defenses": [d.get("id") for d in (cfg.defenses or [])],
        "n_scenarios": n,
        "n_correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "trace_path": cfg.trace_path,
        "trace_events": emitter.count,
        "results": results,
    }
    results_path = cfg.trace_path + ".results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Wrote {} events -> {}".format(emitter.count, cfg.trace_path))
    print("Wrote results -> {} (accuracy {}/{})".format(results_path, correct, n))
    return summary


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        return RunConfig.from_dict(json.load(f))


def print_models() -> None:
    """Every model string upstream ``query_model`` accepts, and the key each needs."""
    print("Model strings accepted by any --*_llm / --backbone flag:\n")
    for provider in ("openai", "anthropic", "mistral", "replicate"):
        models = [m for m in SUPPORTED_MODELS if provider_of(m) == provider]
        print("  {:10s} (needs --{}_api_key)".format(provider, provider))
        for m in models:
            print("      {}".format(m))
    print("\n  Any '*_HF' name routes to a local HuggingFace model (no key).")
    print("\nNote: 'mistral-medium-2505'/'mistral-small-2506' hit api.mistral.ai "
          "(--mistral_api_key);\n      'mixtral-8x7b' is a different model, served "
          "via Replicate (--replicate_api_key).")


def dry_run(cfg: RunConfig) -> dict:
    """Resolve and print the run plan WITHOUT calling any model. Costs nothing."""
    print("Dataset      : {}".format(cfg.dataset))
    print("Backbones    :")
    for role, model in sorted(cfg.models_in_use().items()):
        provider = provider_of(model)
        var = PROVIDER_ENV_KEY.get(provider)
        if role == "judge":
            status = "scorer-side; not called during the run"
        elif role == "defense" and not cfg.defenses:
            status = "no defense registered; not called"
        elif not var:
            status = "local"
        else:
            status = "{} SET".format(var) if os.environ.get(var) else "{} MISSING".format(var)
        print("  {:12s} {:22s} {:10s} {}".format(
            role, model, provider or "UNKNOWN MODEL", status))

    unknown = [m for m in cfg.models_in_use().values() if not is_supported(m)]
    scenarios, indices = load_scenarios(cfg)
    print("Scenarios    : {} -> ids {}".format(len(indices), indices))
    print("Attacks      : {}".format([a.get("id") for a in cfg.attacks] or cfg.content_arm))
    print("Defenses     : {}".format([d.get("id") for d in cfg.defenses] or "none"))
    print("Trace        : {}".format(cfg.trace_path))
    missing = missing_keys(cfg.live_models())
    if unknown:
        print("\nERROR: unsupported model string(s) {} (see --list_models)".format(unknown))
    if missing:
        print("\nMissing key(s) for this run: {}".format(missing))
    if not unknown and not missing:
        print("\nOK: every model resolves and every needed key is present. "
              "Re-run without --dry_run to execute.")
    return {"models": cfg.models_in_use(), "scenarios": indices,
            "missing_keys": missing, "unknown_models": unknown}


def main(argv=None):
    ap = argparse.ArgumentParser(description="AgentClinic security-experiment harness runner")
    ap.add_argument("--config", type=str, help="path to a JSON RunConfig")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=None,
                    help="fallback model for any agent role left unset")
    ap.add_argument("--doctor_llm", type=str, default=None,
                    help="the agent under test, e.g. claude3.5sonnet (see --list_models)")
    ap.add_argument("--patient_llm", type=str, default=None, help="e.g. gpt3.5")
    ap.add_argument("--measurement_llm", type=str, default=None, help="e.g. gpt3.5")
    ap.add_argument("--moderator_llm", type=str, default=None, help="e.g. gpt3.5")
    ap.add_argument("--judge_llm", type=str, default=None,
                    help="scorer-side judge; must differ from doctor_llm")
    ap.add_argument("--n_scenarios", type=int, default=None)
    ap.add_argument("--total_inferences", type=int, default=None)
    ap.add_argument("--content_arm", type=str, default=None)
    ap.add_argument("--attacks", type=str, default=None,
                    help="comma-separated attack ids, e.g. 'T1Injection'. An explicit "
                         "list overrides the --content_arm shorthand. Valid: {}".format(
                             ", ".join(sorted(ATTACKS))))
    ap.add_argument("--defenses", type=str, default=None,
                    help="comma-separated defense ids, e.g. 'D1_Isolation'. Valid: "
                         "{}".format(", ".join(sorted(DEFENSES))))
    ap.add_argument("--tool_enabled", action="store_true")
    ap.add_argument("--tool_return_on_measurement", dest="tool_return_on_measurement",
                    action="store_const", const=True, default=None,
                    help="force T1 to attach to the measurement tool return "
                         "(auto-on for injection arms; use to override)")
    ap.add_argument("--no_tool_return_on_measurement", dest="tool_return_on_measurement",
                    action="store_const", const=False,
                    help="force T1 to attach ONLY to the referral tool return")
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--trace_path", type=str, default=None)
    ap.add_argument("--list_models", action="store_true",
                    help="print every accepted model string and exit")
    ap.add_argument("--dry_run", action="store_true",
                    help="resolve backbones, keys and scenarios, then exit WITHOUT "
                         "calling any model (costs nothing)")
    add_provider_key_args(ap)
    args = ap.parse_args(argv)

    if args.list_models:
        print_models()
        return None

    # A key given on the command line goes into the environment and nowhere else;
    # everything downstream (configure_providers, upstream query_model) reads it there.
    apply_provider_key_args(args)

    cfg = load_config(args.config) if args.config else RunConfig()
    # CLI overrides (only when explicitly provided)
    for name in ("dataset", "backbone", "doctor_llm", "patient_llm", "measurement_llm",
                 "moderator_llm", "judge_llm", "n_scenarios", "total_inferences",
                 "content_arm", "run_id", "trace_path", "tool_return_on_measurement"):
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)
    if args.tool_enabled:
        cfg.tool_enabled = True

    # --attacks / --defenses arrive as comma-separated ids and are parsed into the
    # list-of-spec shape cfg.attacks / cfg.defenses hold. Same rule as the overrides
    # above: only an explicitly provided (non-empty) value replaces the config's.
    try:
        for name, registry in (("attacks", ATTACKS), ("defenses", DEFENSES)):
            specs = parse_interceptor_ids(getattr(args, name), registry, name)
            if specs is not None:
                setattr(cfg, name, specs)
    except ValueError as exc:
        sys.exit("error: {}".format(exc))

    # Re-validate: the CLI overrides bypassed __post_init__.
    for msg in cfg.config_warnings():
        warnings.warn(msg, UserWarning)

    if args.dry_run:
        return dry_run(cfg)

    try:
        run(cfg)
    except (MissingProviderKey, StubbedProvider) as exc:
        # A configuration problem, not a crash: report it as one.
        sys.exit("error: {}".format(exc))


if __name__ == "__main__":
    main()

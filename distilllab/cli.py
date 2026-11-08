"""CLI: evaluate the distillation experiment, or gate on its measured results."""

from __future__ import annotations

import argparse
import json

from .eval import format_terminal, run_evaluation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="distilllab")
    sub = ap.add_subparsers(dest="command", required=True)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--format", choices=["terminal", "json"], default="terminal")
    p_gate = sub.add_parser("gates")
    p_gate.add_argument("--min-teacher-acc", type=float, default=None)
    p_gate.add_argument("--min-distill-gain", type=float, default=None,
                        help="minimum distilled-minus-direct test accuracy gain")
    p_gate.add_argument("--max-t1-acc", type=float, default=None,
                        help="the T=1 ablation must stay below this "
                             "(isolates the temperature's contribution)")
    p_step = sub.add_parser("step-audit", help="audit one simultaneous student parameter update")
    p_step.add_argument("input")
    p_step.add_argument("--temperature", type=float, default=4.0)
    p_step.add_argument("--alpha", type=float, default=0.95)
    p_step.add_argument("--step", type=float, default=0.02)
    p_demo = sub.add_parser("step-demo", help="regenerate original frozen models and step witnesses")
    p_demo.add_argument("output")
    args = ap.parse_args(argv)
    if args.command == "step-demo":
        from .step_fixtures import write_archive
        fixture, manifest = write_archive(args.output)
        print(json.dumps({"original_report": fixture["original_report"], "witnesses": len(manifest)}, indent=2))
        return 0
    if args.command == "step-audit":
        from .step import audit
        from pathlib import Path
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON key: " + key)
                result[key] = value
            return result
        try:
            text = Path(args.input).read_text()
            if len(text) > 1500000:
                raise ValueError("JSON exceeds 1.5 MB")
            payload = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            print(json.dumps(audit(payload, args.temperature, args.alpha, args.step), indent=2, allow_nan=False))
            return 0
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            ap.error(str(exc))

    results = run_evaluation()

    if args.command == "evaluate":
        print(json.dumps(results, indent=2, sort_keys=True) if args.format == "json"
              else format_terminal(results))
        return 0

    failures = []
    if args.min_teacher_acc is not None and results["teacher_acc"] < args.min_teacher_acc:
        failures.append(f"teacher accuracy {results['teacher_acc']:.4f} < required {args.min_teacher_acc}")
    if args.min_distill_gain is not None and results["distill_gain"] < args.min_distill_gain:
        failures.append(f"distillation gain {results['distill_gain']:+.4f} < required {args.min_distill_gain}")
    if args.max_t1_acc is not None and results["ablation_t1_acc"] > args.max_t1_acc:
        failures.append(f"T=1 ablation {results['ablation_t1_acc']:.4f} > allowed {args.max_t1_acc}")

    print(format_terminal(results))
    if failures:
        print("\nGATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall gates passed")
    return 0

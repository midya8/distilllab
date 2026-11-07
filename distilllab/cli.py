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
    args = ap.parse_args(argv)

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

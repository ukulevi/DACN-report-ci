"""Research CLI: dataset QC/preparation and synthetic metric smoke test."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DACN research toolbox")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Synthetic metrics smoke test, NOT a trained model")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    data = sub.add_parser("data", help="Inspect or prepare BMAD Brain FLAIR data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    inspect = data_sub.add_parser("inspect", help="Read-only data quality checks")
    inspect.add_argument("--root", type=Path, required=True)
    inspect.add_argument("--output", type=Path)
    prepare = data_sub.add_parser("prepare", help="Validate and export padded NPZ samples")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--image-size", type=int, default=256)
    for command in (inspect, prepare):
        command.add_argument(
            "--encoding",
            choices=("grayscale", "bmad-bone"),
            default="grayscale",
            help="Explicit source PNG encoding",
        )
        command.add_argument(
            "--split-policy",
            choices=("original", "patient-disjoint-v1"),
            default="original",
            help="Explicit overlap correction policy",
        )
    args = parser.parse_args()
    if args.command == "data":
        try:
            from dacn.data.bmad import inspect_bmad, prepare_bmad

            if args.data_command == "inspect":
                _, report = inspect_bmad(
                    args.root, encoding=args.encoding, split_policy=args.split_policy
                )
                if args.output:
                    if args.output.resolve().is_relative_to(args.root.resolve()):
                        raise ValueError("QC report must be outside the raw dataset root")
                    if args.output.exists():
                        raise ValueError(f"Output already exists: {args.output}")
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            else:
                report = prepare_bmad(
                    args.root,
                    args.output,
                    args.image_size,
                    encoding=args.encoding,
                    split_policy=args.split_policy,
                )
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        except ImportError as exc:
            print(
                f'Data dependencies missing: {exc}. Install with pip install -e ".[data]".',
                file=sys.stderr,
            )
            return 1
        except (OSError, ValueError) as exc:
            print(f"Data error: {exc}", file=sys.stderr)
            return 1
    if args.command == "demo":
        from dacn.evaluation.metrics import compute_binary_metrics

        args.output.mkdir(parents=True, exist_ok=True)
        result = {
            "warning": "Synthetic demo only; not medical model performance",
            "metrics": compute_binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], threshold=0.5),
        }
        (args.output / "metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

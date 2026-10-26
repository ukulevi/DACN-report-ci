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
    publish = data_sub.add_parser(
        "publish-hf", help="Create a privacy-minimized export and upload it to Hugging Face"
    )
    publish.add_argument(
        "--input", type=Path, required=True, help="Prepared BMAD dataset directory"
    )
    publish.add_argument(
        "--output", type=Path, required=True, help="New local Hub export directory"
    )
    publish.add_argument(
        "--repo-id",
        required=True,
        help=("Hugging Face dataset repo, e.g. dacn-bmad-team/bmad-flair-patient-disjoint-v1"),
    )
    publish.add_argument("--dataset-version", default="bmad-flair-patient-disjoint-v1")
    publish.add_argument(
        "--source-archive-sha256",
        help="SHA256 of the original source archive for provenance metadata",
    )
    publish.add_argument("--shard-size", type=int, default=128)
    publish.add_argument("--seed", type=int, default=17)
    publish.add_argument(
        "--public",
        action="store_true",
        help="Create a public Hub repo (confirm redistribution rights first)",
    )
    args = parser.parse_args()
    if args.command == "data":
        try:
            if args.data_command == "inspect":
                from dacn.data.bmad import inspect_bmad

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
            elif args.data_command == "prepare":
                from dacn.data.bmad import prepare_bmad

                report = prepare_bmad(
                    args.root,
                    args.output,
                    args.image_size,
                    encoding=args.encoding,
                    split_policy=args.split_policy,
                )
            else:
                from dacn.data.hub import build_hf_export, publish_hf_dataset

                metadata = build_hf_export(
                    args.input,
                    args.output,
                    shard_size=args.shard_size,
                    seed=args.seed,
                    dataset_version=args.dataset_version,
                    source_archive_sha256=args.source_archive_sha256,
                )
                publication = publish_hf_dataset(
                    args.output,
                    args.repo_id,
                    public=args.public,
                )
                print(
                    json.dumps(
                        {
                            "export_path": str(args.output),
                            "dataset_version": metadata["dataset_version"],
                            "samples": {
                                split: details["samples"]
                                for split, details in metadata["splits"].items()
                            },
                            "publication": publication,
                        },
                        indent=2,
                    )
                )
                return 0
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        except ImportError as exc:
            print(f"Optional dependency missing: {exc}", file=sys.stderr)
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

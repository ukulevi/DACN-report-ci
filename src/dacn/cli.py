"""CLI: verify local installation with a synthetic scoring demo."""
import argparse
import json
from pathlib import Path

from dacn.evaluation.metrics import compute_binary_metrics


def main():
    parser = argparse.ArgumentParser(description="DACN research toolbox")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Synthetic metrics smoke test, NOT a trained model")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    if args.command == "demo":
        args.output.mkdir(parents=True, exist_ok=True)
        result = {"warning": "Synthetic demo only; not medical model performance", "metrics": compute_binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], threshold=0.5)}
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

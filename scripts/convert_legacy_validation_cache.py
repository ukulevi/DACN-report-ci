import argparse
import json
from pathlib import Path

from dacn.evaluation.handoff import convert_legacy_cache


def main():
    parser = argparse.ArgumentParser(description="Verify and convert a private validation cache")
    for field in ("cache", "dataset", "output", "receipt"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    args = parser.parse_args()
    if args.receipt.resolve() in {args.output.resolve(), args.cache.resolve()}:
        parser.error("receipt must be a separate file")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as receipt:
        try:
            report = convert_legacy_cache(
                args.cache,
                args.dataset,
                args.output,
                expected_cache_sha256=args.cache_sha256,
                expected_metadata_sha256=args.metadata_sha256,
            )
            json.dump(report, receipt, indent=2)
            receipt.write("\n")
        except Exception:
            receipt.close()
            args.receipt.unlink()
            raise


if __name__ == "__main__":
    main()

"""Submit one registered stage; local controller never downloads MRI/test arrays."""

import argparse
import json
from pathlib import Path

import modal

from dacn.cloud.modal_final_evaluation import app, evaluate, inspect_run, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "smoke", "validation", "test", "inspect"), required=True
    )
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--call-id", help="Collect a previously submitted function call")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.call_id:
        result = modal.FunctionCall.from_id(args.call_id).get()
    else:
        if args.stage in {"smoke", "validation", "test"} and not args.attempt_id:
            parser.error("--attempt-id is required for each inference attempt")
        with modal.enable_output(), app.run(detach=True):
            if args.stage == "prepare":
                call = prepare.spawn(args.protocol_id)
            elif args.stage == "inspect":
                call = inspect_run.spawn(args.protocol_id, args.attempt_id)
            else:
                call = evaluate.spawn(args.protocol_id, args.attempt_id, args.stage)
            submission = {
                "stage": args.stage,
                "protocol_id": args.protocol_id,
                "attempt_id": args.attempt_id,
                "call_id": call.object_id,
                "app_id": app.app_id,
            }
            (args.output / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
            print(json.dumps(submission), flush=True)
            if args.detach:
                return
            result = call.get()
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if "aggregate" in result:
        (args.output / "aggregate.json").write_text(
            json.dumps(result["aggregate"], indent=2) + "\n"
        )
    # Inspection returns only safe aggregate/provenance/curve payloads.
    for filename in ("aggregate.json", "curves.json", "run_manifest.json"):
        if filename in result:
            (args.output / filename).write_text(json.dumps(result[filename], indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in {"status", "stage", "attempt_id", "revision", "progress"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Build a draft report from authored Markdown and aggregate evidence, without ML access."""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

INPUTS = (
    "report/thesis.md",
    "docs/evidence/2026-11-08-colab-registered-acceptance.json",
    "docs/evidence/2026-11-10-ddim-ablation-table.csv",
    "configs/final_evaluation_frozen_20261110.json",
    "report/metadata.json",
    "report/header.tex",
    "report/references.bib",
    "report/cover.tex",
    "report/assets/hcmut-logo.png",
    "report/figures/pipeline.pdf",
    "report/figures/validation.pdf",
    "report/figures/ablations.pdf",
    "report/figures/provenance.json",
    "docs/REPORT_METHOD_DECISION_2026-11-15.md",
)
TEST_INPUTS = (
    "docs/evidence/2026-11-10-fourarm-test.json",
    "docs/evidence/2026-11-14-hybrid-test-results.json",
    "docs/evidence/2026-11-14-ensemble-test-results.json",
)
ARMS = {
    "final_lpips": "Final LPIPS",
    "initial_lpips": "Initial LPIPS",
    "naive_stitch_lpips": "Naive-stitch LPIPS",
    "brightness": "Brightness",
}


def metric(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Report metric must be finite and in [0,1]")
    return f"{value:.6f}"


def render_test_tables(payloads):
    original, hybrid, ensemble = (json.loads(payloads[name]) for name in TEST_INPUTS)
    if original["split"] != "test" or original["sample_count"] != 3665:
        raise ValueError("Expected the original 3,665-slice test aggregate")
    for result in (hybrid, ensemble):
        if result["test_accessed"] is not True or result["samples"] != 3665:
            raise ValueError("Expected a follow-up aggregate on the 3,665-slice test cohort")
    arms = original["arms"]
    primary = hybrid["frozen_hybrid_test"]
    variant = ensemble["frozen_ensemble_test"]
    if primary != ensemble["frozen_hybrid_test"]:
        raise ValueError("Hybrid metrics differ between the follow-up receipts")
    methods = (
        ("Initial LPIPS", arms["initial_lpips"], 20, None),
        ("Naive-stitch LPIPS", arms["naive_stitch_lpips"], 20, None),
        ("Final LPIPS", arms["final_lpips"], 50, "final_lpips"),
        ("Brightness", arms["brightness"], 0, "brightness"),
        ("Hybrid (primary)", primary, 50, "frozen_hybrid"),
        ("Ensemble (3 seeds)", variant, 150, "seed_ensemble"),
    )
    pixel_header = [
        "| Method | Pixel AP | AUROC | Precision | Recall | Dice |",
        "|--------------|-------:|-------:|-------:|-------:|-------:|",
    ]
    rows = []
    for title, cohort_methods in (
        ("Original frozen comparison (10 November)", methods[:4]),
        ("Post-development comparison (14 November)", methods[2:]),
    ):
        rows += ["\\Needspace{14\\baselineskip}", "", f"**{title}**", "", *pixel_header]
        for label, result, _, _ in cohort_methods:
            rows.append(
                "| "
                + label
                + " | "
                + " | ".join(
                    metric(result["pixel"][k])
                    for k in ("auprc", "auroc", "precision", "recall", "f1")
                )
                + " |"
            )
        rows.append("")
    rows += [
        "",
        "| Method | Image AUROC | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, result, _, _ in methods:
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(metric(result["image"][k]) for k in ("auroc", "precision", "recall", "f1"))
            + " |"
        )
    rows += [
        "",
        "| Method | Pixel AP (95% CI) | U-Net calls per slice |",
        "|---|---:|---:|",
    ]
    for label, result, calls, name in methods:
        if name is not None:
            lo, hi = ensemble["pixel_bootstrap"]["candidates"][name]["ap_ci95"]
            ap = metric(result["pixel"]["auprc"])
            rows.append(f"| {label} | {ap} [{metric(lo)}, {metric(hi)}] | {calls} |")
    return "\n".join(rows)


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("Generated report output must be inside the repository")
    payloads = {}
    for name in (*INPUTS, *TEST_INPUTS):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Report input must stay inside the repository")
        payloads[name] = path.read_bytes()
    citations = set(re.findall(r"(?<![\w])@([\w:-]+)", payloads[INPUTS[0]].decode("utf-8")))
    references = set(
        re.findall(r"@\w+\s*\{\s*([^,\s]+)", payloads["report/references.bib"].decode("utf-8"))
    )
    if citations - references:
        raise ValueError("Unresolved report citation: " + ", ".join(sorted(citations - references)))
    metadata = json.loads(payloads["report/metadata.json"])
    if metadata.get("lang") not in {"en", "en-US", "en-GB"}:
        raise ValueError("Report metadata must select English")
    evidence = json.loads(payloads[INPUTS[1]])
    if evidence.get("status") != "verified" or set(evidence["arms"]) != set(ARMS):
        raise ValueError("Require verified four-arm aggregate evidence")
    appendix = [
        "# Automatically generated evidence appendix",
        "",
        "Historical validation evidence for the original reconstruction comparison.",
        "",
        "Source: verified 8 November evidence; 43 holdout slices, thresholds from 40 calibration slices, one sampling seed (17).",
        "",
        "|Method|Pixel AP|Pixel AUROC|Precision|Recall|F1/Dice|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in ARMS.items():
        p = evidence["arms"][name]["metrics"]["pixel"]
        appendix.append(
            "|"
            + label
            + "|"
            + "|".join(
                metric(p[k]) for k in ("average_precision", "auroc", "precision", "recall", "f1")
            )
            + "|"
        )
    appendix += ["", "|Method|Image AUROC|Precision|Recall|F1|", "|---|---:|---:|---:|---:|"]
    for name, label in ARMS.items():
        p = evidence["arms"][name]["metrics"]["image"]
        appendix.append(
            "|"
            + label
            + "|"
            + "|".join(metric(p[k]) for k in ("auroc", "precision", "recall", "f1"))
            + "|"
        )
    appendix += [
        "",
        "\\Needspace{26\\baselineskip}",
        "",
        "## Producer ablations from 9 November",
        "",
        "Aggregate JSON and hashes were audited; arrays from every new run have not been independently replayed. These holdout results do not automatically select a new candidate.",
        "",
        "|Phase 1+2 steps|t*|Calibration AP|Holdout AP|Holdout Dice|",
        "|---|---:|---:|---:|---:|",
    ]
    rows = list(csv.DictReader(payloads[INPUTS[2]].decode("utf-8").splitlines()))
    if not rows:
        raise ValueError("Report ablation table is empty")
    for row in rows:
        steps = [int(row[k]) for k in ("phase1_steps", "phase2_steps", "t_start")]
        if steps[0] + steps[1] != 50 or not 0 < steps[2] <= 1000:
            raise ValueError("Unexpected report ablation configuration")
        appendix.append(
            f"|{steps[0]}+{steps[1]}|{steps[2]}|"
            + "|".join(
                metric(row[k])
                for k in ("calibration_pixel_ap", "holdout_pixel_ap", "holdout_pixel_f1_dice")
            )
            + "|"
        )
    figure_provenance = json.loads(payloads["report/figures/provenance.json"])
    if figure_provenance.get("hash_normalization") != "CRLF-to-LF":
        raise ValueError("Regenerate report figures with portable text-hash provenance")
    figure_inputs = figure_provenance["inputs"]
    for name in INPUTS[1:3]:
        if (
            figure_inputs.get(name)
            != hashlib.sha256(payloads[name].replace(b"\r\n", b"\n")).hexdigest()
        ):
            raise ValueError("Stale report figures; regenerate from registered evidence")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root
            )
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", True
    body = payloads[INPUTS[0]].decode("utf-8")
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        body = re.sub(
            r"\]\(\.\./docs/([^)]+\.md(?:#[^)]*)?)\)",
            lambda m: f"](https://github.com/tanh1c/DACN/blob/{revision}/docs/{m[1]})",
            body,
        )
    body = body.replace("](figures/", "](" + (root / "report/figures").as_posix() + "/")
    # The document title is supplied by Pandoc metadata, not duplicated as chapter1.
    body = re.sub(r"\A# [^\n]*\n", "", body)
    lines, fence = [], None
    for line in body.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match[1][0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
        elif fence is None and re.match(r"^#{2,6} ", line):
            line = line[1:]
        lines.append(line)
    body = "\n".join(lines) + "\n"
    if body.count("<!-- TEST_RESULTS -->") != 1:
        raise ValueError("Report must contain one test results marker")
    body = body.replace("<!-- TEST_RESULTS -->", render_test_tables(payloads))
    body += (
        "\n\n\\clearpage\n\n"
        + "\n".join(appendix)
        + f"\n\nSource commit: `{revision[:12]}`. Input hashes are recorded in build-manifest.json.\n\n\\clearpage\n"
    )
    # Reserve space for compact research tables so rows are not orphaned across pages.
    body = re.sub(
        r"(?m)(?:^\|[^\n]*\|\n){2,}",
        lambda m: (
            "\\Needspace{"
            + str(min(28, len(m[0].splitlines()) * 1.5))
            + "\\baselineskip}\n\n"
            + m[0]
        ),
        body,
    )
    receipt = {
        "status": "prepared",
        "report_status": "includes_post_development_test_results",
        "report_language": metadata["lang"],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_manifest_status": json.loads(payloads[INPUTS[3]])["status"],
        "source_commit": revision,
        "tracked_tree_dirty": dirty,
        "inputs": {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()},
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    if metadata.get("date", "auto") == "auto":
        date = datetime.fromisoformat(receipt["built_at_utc"]).strftime("%d %B %Y")
        metadata["date"] = f"Build: {date} (UTC) — source {revision[:12]}"
    cover = (
        payloads["report/cover.tex"]
        .decode("utf-8")
        .replace("@@LOGO@@", (root / "report/assets/hcmut-logo.png").as_posix())
        .replace(
            "@@MONTH@@", datetime.fromisoformat(receipt["built_at_utc"]).strftime("%B %Y").upper()
        )
    )
    header = payloads["report/header.tex"].decode("utf-8")
    (output / "header.tex").write_text(
        header + "\n\\renewcommand{\\maketitle}{\n" + cover + "\n}\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "report.md").write_text(body, encoding="utf-8")
    (output / "build-manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("report/build"))
    parser.add_argument("--tools", type=Path, default=Path("runs/report-tools/tools.json"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    receipt = prepare(root, output)
    if args.prepare_only:
        print("Draft Markdown and input provenance prepared; PDF not compiled.")
        return
    tools = json.loads((root / args.tools).read_text(encoding="utf-8"))
    # A changed document class must not reuse an old TOC/aux or a stale successful PDF.
    for suffix in ("aux", "toc", "out", "lof", "lot", "xdv", "pdf"):
        (output / f"report.{suffix}").unlink(missing_ok=True)
    commands = [
        [
            tools["pandoc"],
            str(output / "report.md"),
            "--from=markdown",
            "--to=latex",
            "--standalone",
            "--toc",
            "--citeproc",
            "--bibliography=" + str(root / "report/references.bib"),
            "--number-sections",
            "--top-level-division=section",
            "--metadata-file=" + str(output / "metadata.json"),
            "--include-in-header=" + str(output / "header.tex"),
            "--output=" + str(output / "report.tex"),
        ],
        [
            tools["tectonic"],
            "--untrusted",
            "--keep-logs",
            "--keep-intermediates",
            "--outdir",
            str(output),
            str(output / "report.tex"),
        ],
    ]
    try:
        with (output / "build.log").open("w", encoding="utf-8") as log:
            for command in commands:
                subprocess.run(
                    command,
                    cwd=output,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
        pdf = output / "report.pdf"
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            raise ValueError("Compiler did not produce a PDF")
        receipt.update(
            status="completed",
            pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
            tool_versions=tools["versions"],
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        receipt.update(status="failed", error_class=type(exc).__name__)
        raise
    finally:
        (output / "build-manifest.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
    print("Draft PDF compiled; see report.pdf and build-manifest.json.")


if __name__ == "__main__":
    main()

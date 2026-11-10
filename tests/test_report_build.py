import csv
import hashlib
import io
import json
import runpy
import zipfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def fixture(root):
    (root / "docs/evidence").mkdir(parents=True)
    (root / "report/figures").mkdir(parents=True)
    (root / "report/assets").mkdir()
    (root / "report/cover.tex").write_text("Test cover @@LOGO@@")
    (root / "report/assets/hcmut-logo.png").write_bytes(b"image fixture")
    (root / "configs").mkdir()
    (root / "report/thesis.md").write_text(
        "# Report\nEnglish narrative.\n## Dataset\n```python\n## code comment\n```\n",
        encoding="utf-8",
    )
    (root / "report/metadata.json").write_text('{"title":"DACN","lang":"en-US"}')
    (root / "report/header.tex").write_text("% initial layout\n")
    (root / "report/references.bib").write_text("@article{example, title={Example}, year={2020}}")
    for name in ("pipeline", "validation", "ablations"):
        (root / f"report/figures/{name}.pdf").write_bytes(b"%PDF-fixture")
    (root / "configs/final_evaluation_candidate.json").write_text('{"status":"candidate"}')
    data = {"status": "verified", "arms": {}}
    for name in ("final_lpips", "initial_lpips", "naive_stitch_lpips", "brightness"):
        data["arms"][name] = {"metrics": {}}
        for level in ("pixel", "image"):
            data["arms"][name]["metrics"][level] = {
                "auroc": 0.9,
                "precision": 0.3,
                "recall": 0.5,
                "f1": 0.375,
                **({"average_precision": 0.3123301734628661} if level == "pixel" else {}),
            }
    path = root / "docs/evidence/2026-11-08-colab-registered-acceptance.json"
    path.write_text(json.dumps(data, indent=2))
    table = root / "docs/evidence/2026-11-10-ddim-ablation-table.csv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "phase1_steps",
                "phase2_steps",
                "t_start",
                "calibration_pixel_ap",
                "holdout_pixel_ap",
                "holdout_pixel_f1_dice",
            ]
        )
        writer.writerow([20, 30, 500, 0.320219, 0.321148, 0.373395])
    inputs = (
        "docs/evidence/2026-11-08-colab-registered-acceptance.json",
        "docs/evidence/2026-11-10-ddim-ablation-table.csv",
    )
    (root / "report/figures/provenance.json").write_text(
        json.dumps(
            {
                "hash_normalization": "CRLF-to-LF",
                "inputs": {
                    k: hashlib.sha256((root / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    for k in inputs
                },
            }
        )
    )
    return path


def test_prepare_uses_evidence_and_preserves_candidate(tmp_path):
    fixture(tmp_path)
    prepare = runpy.run_path(str(PROJECT / "scripts/build_report.py"))["prepare"]
    receipt = prepare(tmp_path, tmp_path / "report/build")
    body = (tmp_path / "report/build/report.md").read_text(encoding="utf-8")
    assert "0.312330" in body and "0.321148" in body
    assert "DRAFT" in body and "no held-out test results" in body
    assert "English narrative" in body
    assert "Automatically generated evidence appendix" in body
    assert "\n# Dataset\n" in body and "\n## code comment\n" in body
    assert receipt["report_language"] == "en-US"
    assert receipt["status"] == "prepared"
    before = receipt["inputs"]["report/header.tex"]
    (tmp_path / "report/header.tex").write_text("% changed layout\n")
    after = prepare(tmp_path, tmp_path / "report/build")["inputs"]["report/header.tex"]
    assert before != after
    assert (
        json.loads((tmp_path / "configs/final_evaluation_candidate.json").read_text())["status"]
        == "candidate"
    )


def test_invalid_evidence_fails_before_report_is_written(tmp_path):
    path = fixture(tmp_path)
    data = json.loads(path.read_text())
    data["arms"]["final_lpips"]["metrics"]["pixel"]["average_precision"] = 1.5
    path.write_text(json.dumps(data))
    prepare = runpy.run_path(str(PROJECT / "scripts/build_report.py"))["prepare"]
    with pytest.raises(ValueError, match="metric"):
        prepare(tmp_path, tmp_path / "report/build")
    assert not (tmp_path / "report/build/report.md").exists()


def test_non_english_metadata_fails_before_report_is_written(tmp_path):
    fixture(tmp_path)
    (tmp_path / "report/metadata.json").write_text('{"title":"DACN","lang":"vi-VN"}')
    prepare = runpy.run_path(str(PROJECT / "scripts/build_report.py"))["prepare"]
    with pytest.raises(ValueError, match="English"):
        prepare(tmp_path, tmp_path / "report/build")
    assert not (tmp_path / "report/build/report.md").exists()


def test_compiler_bootstrap_rejects_changed_download(tmp_path):
    module = runpy.run_path(str(PROJECT / "scripts/report_tools.py"))
    archive = tmp_path / "tool.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("folder/pandoc.exe", b"test executable fixture")
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="checksum"):
        module["extract_binary"](archive, "0" * 64, "pandoc.exe", tmp_path / "pandoc.exe")
    assert not (tmp_path / "pandoc.exe").exists()


def test_bibliography_and_figures_are_part_of_provenance(tmp_path):
    fixture(tmp_path)
    prepare = runpy.run_path(str(PROJECT / "scripts/build_report.py"))["prepare"]
    first = prepare(tmp_path, tmp_path / "report/build")
    assert "report/references.bib" in first["inputs"]
    assert "report/figures/validation.pdf" in first["inputs"]
    (tmp_path / "report/figures/validation.pdf").write_bytes(b"%PDF-updated")
    second = prepare(tmp_path, tmp_path / "report/build")
    assert (
        first["inputs"]["report/figures/validation.pdf"]
        != second["inputs"]["report/figures/validation.pdf"]
    )


def test_unresolved_citation_rejected_before_output(tmp_path):
    fixture(tmp_path)
    (tmp_path / "report/thesis.md").write_text("# Report\nClaim [@missing].")
    prepare = runpy.run_path(str(PROJECT / "scripts/build_report.py"))["prepare"]
    with pytest.raises(ValueError, match="citation"):
        prepare(tmp_path, tmp_path / "report/build")
    assert not (tmp_path / "report/build/report.md").exists()


def test_changed_evidence_rejects_stale_figures(tmp_path):
    path = fixture(tmp_path)
    data = json.loads(path.read_text())
    data["arms"]["final_lpips"]["metrics"]["pixel"]["average_precision"] = 0.4
    path.write_text(json.dumps(data))
    prepare = runpy.run_path(str(PROJECT / "scripts/build_report.py"))["prepare"]
    with pytest.raises(ValueError, match="Stale report figures"):
        prepare(tmp_path, tmp_path / "report/build")


def test_figure_provenance_portable_across_line_endings(tmp_path):
    fixture(tmp_path)
    prepare = runpy.run_path(str(PROJECT / "scripts/build_report.py"))["prepare"]
    names = (
        "docs/evidence/2026-11-08-colab-registered-acceptance.json",
        "docs/evidence/2026-11-10-ddim-ablation-table.csv",
    )
    for name in names:
        p = tmp_path / name
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    first = prepare(tmp_path, tmp_path / "report/build")
    for name in names:
        p = tmp_path / name
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    second = prepare(tmp_path, tmp_path / "report/build")
    assert all(first["inputs"][n] != second["inputs"][n] for n in names)

# BMAD Data Foundation Implementation Plan

> Historical plan: the ANDi research direction in this document was superseded
> on 28/10/2026 by the DDPM reconstruction plan in `docs/RESEARCH.md`. Data
> foundation records remain historical evidence, not the current method scope.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Bàn giao kế hoạch 6 tuần và pipeline kiểm tra/chuẩn bị BMAD Brain FLAIR 2D.
**Architecture:** Read-only inspection produces a manifest and QC report. Preparation reuses
the inspection, pads images without resampling, and atomically publishes a fresh dataset.
**Tech Stack:** Python >=3.10, pathlib/csv/json/hashlib, NumPy, Pillow, pytest.
**Spec:** ../specs/2026-10-24-bmad-flair-design.md

## Global Constraints

- Python >=3.10; optional `data` extra, no torch required for this phase.
- Keep raw splits immutable; explicit patient-disjoint-v1 removes overlapping test patients (D3). IDs stay strings; no image-level re-splitting.
- Test only for final evaluation; masks never enter the scorer.
- Source workbook and images remain unmodified. No real images enter Git.
- This deliverable prepares research; it does not claim trained-model results.

## Review Focus

- Cross-split leakage through patient IDs or decoded duplicate images must fail inspection.
- Missing/corrupt/misaligned/empty anomaly masks must fail before output creation.
- Different root paths, malformed filenames and unexpected PNG directories must not silently pass.
- A train-only or missing-class download must not appear complete.
- Existing output and failed preparation must not appear to be a successful new dataset.

## Task 1: Inspection and QC

**Files:** src/dacn/data/bmad.py, tests/test_bmad.py.
**Consumes:** BMAD directory layout in spec.
**Produces:** inspect_bmad(root: Path) -> tuple[list[dict], dict].

- [x] Write fixtures with train, val/test normal and anomaly images and masks.
- [x] Tests: valid counts and stable IDs; patient leakage; decoded duplicate;
  missing/misaligned/empty mask; corruption; malformed filename; empty required group;
  unexpected directory; RGB non-grayscale; orphan mask.
- [x] Run `pytest tests/test_bmad.py -q` and observe missing implementation.
- [x] Implement read-only inspection and deterministic manifest ordering.
- [x] Run the tests, expected PASS.

## Task 2: Reproducible preparation and CLI

**Files:** src/dacn/data/bmad.py, src/dacn/cli.py, pyproject.toml, tests/test_bmad.py.
**Consumes:** inspect_bmad and report.ok.
**Produces:** prepare_bmad(root: Path, output: Path, image_size: int = 256) -> dict;
CLI data inspect / data prepare.

- [x] Tests: exact shape/dtype/range, unchanged lesion count after padding, deterministic
  outputs, manifest paths, refuse existing output and undersized target, invalid dataset
  produces no output, CLI nonzero failure/zero success, existing demo still works.
- [x] Run tests and observe failure for missing preparation/CLI.
- [x] Implement atomic output in a sibling temp directory with cleanup on failure.
- [x] Add data extra and lazy CLI imports; run full pytest and ruff, expected PASS.

## Task 3: Research handoff and real-data check

**Files:** docs/PROJECT_PLAN.md, docs/DATA_PROTOCOL.md, docs/REFERENCES.md,
docs/TEAM.md, docs/RESEARCH.md, configs/bmad_flair.yaml, README.md.
**Consumes:** verified CLI from Task 2.
**Produces:** Vietnamese instructions, milestones, data/research responsibilities and source list.

- [x] Write 6-week roadmap with mandatory outputs and fallback at week 2/4.
- [x] Record adaptation vs original ANDi, AP definition, locked test, same-device latency.
- [x] Check the official download listing; fetch only Brain if public and identifiable.
- [x] Run inspection on downloaded data if available; otherwise record exact access failure.
- [x] Run fresh full tests/lint and one independent review; report evidence and limits.
- [x] Leave the branch available locally for team review; no remote publication is required.

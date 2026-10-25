# Execution ledger

Scope: research preparation and data foundation, not completion of 6-week training.
Base: 602f46db5da45fdf437a434ed7c75293d73a17af.
Branch: feat/bmad-data-foundation, local checkout in Downloads/DACN-source.

Pre-flight: Task 1 rows/report feeds Task 2 preparation; Task 3 documents CLI from
Task 2. Shared data contract checked. Tests first, no dependency on torch.

Task 1: tests initially failed with missing module; inspection implemented; 15
tests passed. Task 2: missing prepare/CLI caused expected failure; implementation
passed full 26-test suite including original demo/metrics.

Ruling D1: retain the user's requested Downloads checkout on a feature branch —
native worktree tool refers to the old chat workspace; targeted sandbox escalation
used for authorized edits — cost if wrong: less isolation, mitigated by staged
file list and preserving workbook/raw inputs.

Ruling D2: adapt to actual BMAD layout and explicit bone encoding — primary source
bras.py plus downloaded PNGs contradict initial assumptions — cost if wrong:
luminance conversion loses physical intensity interpretation; document as PNG-derived
FLAIR, not NIfTI reproduction. Regression suite observed 10 failures then 29 passes.

Ruling D3: provide explicit patient-disjoint-v1 correction — original source has
10 validation/test patient overlaps; keep strict original mode and immutable raw,
drop 50 test slices of overlapping patients — cost: modified benchmark no longer
directly comparable to original BMAD tables. Three policy tests observed RED then
GREEN; full suite 32 passed. No patient or pixel overlap remains in retained data.

Task 3: downloaded official Brain archive, verified SHA256, inspected original,
prepared corrected 11,248-image dataset. Plan, references and protocol written.
Fresh final validation and independent review recorded below when complete.

Final validation 2026-10-25: all 11,248 NPZ independently verified against manifest
for dtype/shape/range, padding, lesion counts, labels and 1,241 disjoint patient IDs.
Editable install of .[data,dev] succeeded; CLI runs without PYTHONPATH.

Final review: one fresh read-only reviewer found two important issues, no critical
or minor findings. Fixed CI dependency selection from .[dev] to .[data,dev]; verified
declared Pillow dependency and successful local install/test commands. CI itself has
not run remotely. No new brittle test duplicating the YAML implementation was added.

Final: fixed palette/tRNS decoder gap — test_reject_unsupported_png_encodings,
six parameter cases RED→GREEN, full suite 38/38, Ruff check passed, Ruff format
check passed (25 files), synthetic CLI demo succeeded.

Final: Ruling D4: reviewer did not independently research references/licenses/identity —
source links were inspected by the implementer; retain limited filename provenance
claim and require source terms on redistribution — cost if wrong: permission or
provenance assumptions need revision before sharing/training claims.

Final: Ruling D5: reviewer did not judge future model performance/GPU budget —
not implemented in this preparation phase; week-1 pilot remains required — cost if
wrong: six-week scope must shrink to fit measured compute.

Final: Ruling D6: reviewer sampled QC and manifest hash instead of all NPZ —
implementer independently verified all 11,248 output files with assertions and saved
the log — cost if wrong: validation remains software-level, not clinical annotation review.

Integration: preserve local feature branch and working files in Downloads per the
user's delegated preparation scope. No merge/push or Git identity configuration.
No deferred minor findings. All three implementation tasks complete.

Follow-up 2026-10-25: user explicitly requested publishing a PR and creating Linear
tracking. PR https://github.com/tanh1c/DACN/pull/11 opened; workbook included unchanged.
Linear project DACN created in Traffic-agent-assistant after user selected that team:
6 milestones, 25 issues and 31 dependency relations, initial project update. Data tasks
assigned to verified user Chí Vĩ; B/C remain unassigned pending teammate onboarding.

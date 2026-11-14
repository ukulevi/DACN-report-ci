# Worklog

## 2026-11-10 — Full English CO4029 report and private release pipeline (DAC-26)

Moved canonical manuscript/layout into `report/`; legacy summary and `reports/`
now point to it. Expanded the six-page summary into a 16-page validation-study
report: abstract, theory/equations, audited data/preprocessing, model and score
contracts, calibration/metrics, results/uncertainty, limitations, reproducibility,
conclusion, aggregate evidence appendix and ten resolved bibliographic sources.
Cover follows the two user-supplied report examples; current course/advisor and
three names/IDs confirmed from supplied records, with prior-project research
content/results excluded. Logo extracted from the user's prior cover.

Added three aggregate-only vector figures and input-hash provenance; the builder
rejects unresolved citations and stale figures. No patient arrays/images, training,
test access or scientific freeze changes. Pandoc citeproc plus the existing pinned
Tectonic compiler produces ignored `report/build/report.pdf` and generated LaTeX.
Workflow retains every-push/PR triggers, read-only build permissions and pinned
actions; a separate contents-write job publishes PDF/manifest as a clearly labeled
validation-draft prerelease only on private satellite main push/manual.

Checks: seven focused tests pass after observed red-to-green checks; repository
Ruff and format pass. Real local PDF compiled and rendered for all-page visual
review, including corrected long-SHA overflow and bibliography pagination; no
remaining overflow/missing-glyph warnings. Final read-only review found no
material issues. Hosted build/release still requires verification after push;
local compilation is not claimed as remote publication.

Follow-up before hosted acceptance: Git's add warning showed reportlab's
ASCII-encoded PDFs were detected as text (`i/lf w/lf`). Added binary attributes
for PDF/PNG so Windows checkout cannot rewrite PDF byte offsets through CRLF
conversion. `git check-attr` verifies text/diff/merge disabled for those assets.

Hosted builds a19d404/4769187 failed before compilation because figure-source
hashes captured Windows CRLF while Git blobs/Linux checkouts use LF. Confirmed
both registered files matched the local raw hashes but not `git show HEAD:path`,
and Actions log reported the stale-figure guard. Added an observed red-to-green
LF/CRLF portability regression; figure provenance explicitly normalizes CRLF to
LF, while the main build manifest retains raw-byte input hashes. Real changed
values still fail the stale-figure check. No experiment data or thresholds changed.

Hosted acceptance for `1a9496e`: [report run42760339837](https://github.com/ukulevi/DACN-report-ci/actions/runs/42760339837)
Success, build1m15s and release9s; [quality run42760339699](https://github.com/ukulevi/DACN-report-ci/actions/runs/42760339699)
Success31s. [Private prerelease report-v2026.11.10-5](https://github.com/ukulevi/DACN-report-ci/releases/tag/report-v2026.11.10-5)
visibly contains report.pdf (147KB) and build-manifest.json. Remote PDF SHA256
`2e3a0a70ca7db31eba5f5297480c2de166da385d91d7f345bcf63d5ebdd57ca8`.
Local committed build manifest is completed/clean, candidate unchanged, PDF hash
verified. Eight focused checks and full Ruff/format pass; portability fix reviewed.

## 2026-11-10 — Independent private report CI host (DAC-26)

User approved private `ukulevi/DACN-report-ci` after confirming the source
owner's card expired. Created repository private and pushed committed source
`4accc9f`, excluding local untracked data/checkpoints. Hosted report run
[42756144394](https://github.com/ukulevi/DACN-report-ci/actions/runs/42756144394)
completed successfully in 1m17s with PDF/manifest and diagnostic artifacts.
The same source's quality run42756144682 also succeeded (53s): dependency
installation, Ruff, format, pytest and CLI demo completed on GitHub-hosted Linux.
Hosted pytest reported **158 passed, 23 skipped in 14.69s**; skipped checks are
not counted as passing and this does not replace GPU/model acceptance.
User then chose Git push synchronization and declined a broad classic token.
No scheduled upstream workflow, token or new permission is enabled. Added the
local `report-ci` remote and documented forwarding the same commit after origin.
The ordinary push-triggered workflow builds each satellite update. No model
runs, test access or research freeze changes. Report tests and repository lint/
format checks pass before commit (4 report tests). BLK-20261107-01 remains open for the
original owner's billing; the report host workaround is independently verified.

## 2026-11-10 — English-only report and rebuild on every repository push

Translated the authored report, metadata, header, generated appendix and report
guide to English. Builder now requires English language metadata and records
language/UTC build time; default cover date resolves automatically with source
commit. Workflow push/PR filters removed so every repository push/PR update with
the workflow rebuilds the report; no new model/threshold/freeze decisions.
Four focused tests pass after red→green checks for English output and rejecting
non-English metadata. Local English PDF compiled, 6 pages; text checks found
required metrics/draft labels and no Vietnamese text. Candidate configs unchanged.
All six rendered pages inspected; no missing-glyph/overflow warnings. Repository
Ruff checks and format checks pass (211 files); git diff whitespace check passes.
CI rebuilds committed English narrative, registered evidence tables and build
provenance; it does not invent new research results or rewrite interpretation.

## 2026-11-10 — Automated Vietnamese draft report PDF build

Implemented Markdown→Pandoc3.13→Tectonic0.17.0 with portable official binary
archive SHA256 pins for Windows/Linux. No training dependency changes. Report
appendix reads verified aggregate metrics/ablation CSV, rejects invalid metrics,
preserves candidate and draft-validation labels; provenance hashes include body,
evidence, metadata, layout and builder. Added path-filtered report.yml artifact/
diagnostic workflow with pinned official checkout/upload actions and contents read.
Reference EE5217 workflow reviewed; hw2_introAI returns404, not claimed reviewed.
Three focused tests red→green, Ruff/YAML checks pass. Actual Windows PDF compiled
and rendered: Vietnamese/metric text checks pass, no overfull/missing glyph
warnings after splitting a wide table. Cleared stale TOC/aux after class change;
earlier network-isolated compile failed before cached TeX bundle access, corrected
with ordinary online local compile. Not a training/evaluation run.
Remote Actions still requires account/billing resolution; no CI-success claim.
Independent report-CI review found no material issues. Final validation:3tests
pass, whole-repo Ruff check/format211files pass, YAML triggers/permissions parsed,
six-page PDF text/metric checks pass and all rendered pages inspected. Compiler
version labels come from pinned release metadata; no byte-identical PDF claim.

## 2026-11-10 — Audit producer ablations and prepare report/freeze handoff

Verified live GitHub working/C branches remain1c13e5f/c02addb. HF model updated
09Nov to03ec5b35cc134277b0ce4e76a0a315490b46bd21; dataset revision unchanged.
Downloaded aggregate JSON only for22new producer runs(11smoke/11full83),
checked88JSON artifact hashes, root/comparison metrics/config consistency,
calibration-to-holdout thresholds and reported50calls/output.11full runs share
checkpoint/index/dataset hashes. Recorded dirty-source ZIP provenance limits;
no private arrays/new source ZIPs/model inference/test downloads.
Added safe audit evidence, CSV ablation table, report-readiness matrix and exact
handoff for Hải Tuấn's review. Updated report draft/references using primary
AutoDDPM/LPIPS/SSIM sources. Keep candidate and thresholds unchanged; user is
coordinating C review/freeze, agent does not supply approval on their behalf.

## 2026-11-08 — Four-arm resume acceptance and freeze review decision

Executed validation-only cached T4 score replay through frozen_runner for4arms
and43registered holdout slices. Deliberately interrupted each after1durable shard,
resumed42, then completed replay with backend calls forbidden.172score shards
exactly match input;60numeric metric fields match T4 oracle at1e-10;37.317s total,
0U-Net/LPIPS. Recorded actual attempts/provenance in RESULT_TRACKING/evidence.
Full suite inside sandbox273pass/1BMAD CLI staging rename WinError5; isolated
failure test1pass, outside sandbox full274pass38.57s. No root cause/code fix
established; retained both outcomes, no speculative retry changes.
Prepared freeze-review packet and runbook separating actual primary-model API,
cached4arm validation acceptance and missing turnkey final wrapper/dataset factory.
User decision: keep candidate and await CI or team review; no freeze exception,
no test authorization. Preserve configuration statuses and thresholds unchanged.
Independent read-only review found no material correctness issues in the new
evidence/runbook/decision records. Aggregate schema/privacy/hash checks passed;
all60fields/172shards and unchanged candidate statuses verified before commit.

## 2026-11-08 — Repin accepted T4 candidate and prepare freeze handoff

Plan: repin from verified T4 evidence, run offline nine-artifact/source preflight,
check live CI and stale Linear status, update report/freeze handoff, verify/push.
Decision: preserve accepted d5889d5 execution archive rather than silently pin
new HEAD; newer source/runtime needs acceptance again. Candidate stays unfrozen.
Pins/95source files pass static preflight; candidate final gate and mismatched
local CPU environment both reject execution. All4threshold contracts unchanged.
This is no inference/training run and does not replace live final-runtime binding.
Added required pixel/image metrics and actual runtime evidence to report draft.
GitHub edff8c8 still startup_failure,0jobs/0check-runs; billing blocker remains.
Focused paired/DDPM-validation/frozen-input/partition/runtime checks:27passed
in9.88s; Ruff on related changed Python files and git whitespace check pass.
Reviewed paired absolute/SSIM code and real Gaussian20/Simplex20 evidence;
Linear DAC-30 marked Done with acceptance rationale, DAC-15 moved In Progress
in verified DACN-KHMT workspace. No scientific-quality/final-test gate closed.
Independent read-only final review found no material pin/metric correctness
issues; updated two current-state doc sections that still listed completed work.

## 2026-11-08 — Colab registered validation and durable private handoff

Completed T4 smoke2/full83 using source d5889d5 and pinned40/43 membership;
fixed UI code-transfer indentation by executing literals from pinned generator.
Downloaded private ZIP via native Chrome and copied into ignored run directory;
verified ZIP SHA and10member hashes. CPU replay matches every43holdout metric
field at1e-10; all4maps match C at atol1e-6. Final AP.312330/Dice.371215;
brightness AP.389506 remains stronger. No training/test/tuning.
Documented inherited source/environment metadata erratum without altering raw
bundle. Added acceptance/evidence and updated blocker progress; next steps are
metadata preflight, final runtime repin and team freeze, then final evaluation.
Fixed inherited CPU identity in future generated Colab manifests. Regression
first failed on stale execution_source_commit, then24related checks passed;
all5generated code cells compile. No GPU rerun for this metadata-only correction.
Private verifier compiled. Runtime remains connected, no cell executing.

## 2026-11-07 — Verify native Chrome download before validation retry

Checked Downloads, project C:/Users/Admin/Downloads/@ukulevi/DACN and current
worktree; prior private bundle was not found. Original Colab runtime reset;
notebook retains compact measured outputs but full artifact copy is unverified.
Native Chrome Files download probe succeeded:37bytes at Downloads/dacn_download_probe.txt.
Preparing a fresh source/partition-pinned notebook; no inference retry started yet.
Registered-partition gate/helper fixes passed27related tests before packaging.

## 2026-11-07 — T4 full83 inference and registered holdout recovery

Full83 inference completed after smoke; group report initially failed on
cross-version split re-derivation, then unsupported metric split label. Reused
saved maps, transferred checksum/index-bound original40/43 membership from
metric-verified local replay; no extra inference or tuning.43holdout metrics
match C closely (final AP.312330173/Dice.371214628), receipt being saved.
Added explicit validation_partition artifact to gate/candidate and fixed
generator/scripts to require it.27related tests pass. Actual run used earlier
eight-input gate plus separately verified partition; future gate has nine.

## 2026-11-07 — Recover Colab environment and launch T4 validation

HF authentication completed, checkpoint/val metadata/index/shard/mask hashes passed.
Fixed direct-source pip freeze handling with version-only snapshots (no source
URLs exported), consistent runtime checks, plus explicit verified kernel import
path after editable pip install.22related tests pass; Ruff passes198files.
Failed preparation/export/import attempts retained in RESULT_TRACKING.
T4 smoke2 completed, peak CUDA allocated112928768bytes; full83 launched after
smoke review with separate43group-holdout report. Actual Colab source is patched
bf82522, ZIP hash recorded; not mislabeled as pristine Git source. No test/training.

## 2026-11-07 — Prepare private Colab T4 controls workflow

Generated and uploaded separate private controls notebook from sourcebf82522.
Colab allocated Tesla T4; bootstrap installed source successfully (visible UI).
Acquisition cell waits for user-entered HF read token via getpass; instructions
provided using official HF docs. No inference started or GPU metrics claimed.
Generator includes pinned validation-only downloads, environment export, source
binding, bounded smoke and separately gated full83 with43group-holdout reporting.
All4generated code cells compile; uploaded notebook final cell still needs the
new reporting update before full83. See COLAB_CONTROLS_VALIDATION.md.

## 2026-11-07 — Controls acceptance and final runtime binding

Added separate control candidate thresholds and shared-trajectory score arms.
Bounded trained CPU controls completed:100U-Net/6LPIPS calls; final map matches
prior acceptance exactly. Recorded cold/warm timing, Windows working-set peak
and diagnostic metrics in CONTROLS_ACCEPTANCE. Failed preflight CRLF comparison
retained in run log, then corrected. Confirmed LPIPS state digest matches C's
checksum-verified historical result. Added runtime source/import-root/dependency
verification to final API,17related tests pass; Ruff check/format193files pass.
User selected personal Colab GPU; preparing smoke-first workflow, not final test.

## 2026-11-07 — Trained DDIM bounded acceptance and resume comparison

Downloaded exact G30 checkpoint by registered SHA at immutable HF revision.
Attempt1 preflight exposed sampler/trainer config confusion; fixed architecture
loading from hashed checkpoint, kept failure record. Attempt2 verified all83
validation arrays/index, ran2trained samples, interrupted after first shard,
fresh-model resume and completed-cache replay. Uninterrupted comparator matched
all arrays/metrics exactly. 200 trained U-Net calls total, CPU only; no test,
threshold fitting or training. Added validation-only candidate gate with scope
bound to receipt so it cannot resume as final. 13 related tests and Ruff passed.
See ACCEPTANCE_2026-11-07.md for measured provenance and remaining freeze limits.

Pinned follow-up archive84382a8 in candidate (kept candidate status). This archive
is the committed corrected implementation; actual acceptance source ZIP remains
separately identified in the run receipt, not overwritten or relabeled.

## 2026-11-07 — DDIM adapter, CPU LPIPS replay and CI cause

Connected frozen runner to lazy FrozenDDIMAdapter: same captured checkpoint/config,
offline strict LPIPS trunk/head load, registered mask/indices, seed/crop/top576,
exact 50 U-Net-call guard. Added export/replay tools and perceptual extra.
Exported new CPU weights/dependency snapshot privately; all 166 LPIPS maps on 83
cached validation pairs matched historical maps at atol1e-6 (max5.2154e-7).
No fresh trained U-Net inference, GPU or test. See RESULT_TRACKING for receipts.
Eleven gate/runner/adapter tests passed, including synthetic 50-call model and
LPIPS native-state roundtrip. Full suite: 254 passed/1 Windows rename failure
in RGBA preparation test, 46.80s. Ruff check/format passed (188 files).
User screenshot confirms CI blocked before job startup by account/payment
annotation. BLK-20261107-01 now Waiting for repo owner's Billing/Actions fix;
no workflow/budget/payment mutation performed.

Follow-up: exported git archive cf7de8f and pinned source/native LPIPS/environment
hashes in candidate JSON; status remains candidate. Source ZIP and weights remain
private under runs. Isolated RGBA rename rerun passed (0.29s); no root-cause fix.

## 2026-11-07 — Sequential audit follow-up: workspace and frozen orchestration

Linear connector now verifies DACN-KHMT. Re-ran Windows reproduction: isolated
rename test passed; full suite 250 passed in 38.88s. No rename code change or
root-cause claim; earlier intermittent failure remains unexplained. Added CPU
frozen runner orchestration: gate before dataset factory, captured bytes passed
to backend, atomic private shards/receipt, hash/order-bound resume, global
image/pixel metrics with frozen thresholds. Nine runner/gate tests passed.
This is a backend interface, not an integrated DDIM+LPIPS final-test executable;
see FROZEN_RUNNER.md for required adapter, memory benchmark and remaining pins.
No GPU, training, MRI evaluation or final test opened.

Added REPORT_DRAFT.md from verified validation/provenance records, retaining
brightness control, bootstrap uncertainty and missing-final-test limitations.
This is a Markdown content draft, not the final formatted submission.

## 2026-11-07 — Whole-project progress review and workspace default

Reviewed tracked structure, entry points, requirements, final candidate and
run records. Git/HF heads unchanged; live Actions now reports startup_failure
on ecad491/b986080 with no jobs/check-run output. Full CPU suite: 249 passed,
1 Windows staging-rename failure; Ruff check/format passed (179 files).
Added PROJECT_REVIEW_2026-11-07 and CI blocker, refreshed README/TEAM/tracking.
Set DACN-KHMT as explicit repository workspace preference in AGENTS and config;
connector OAuth still points elsewhere, user guided to reconnect via official
OpenAI account-management documentation. No remote workspace/status mutation,
no training/MRI evaluation or test split opened. Team role boundaries removed
from execution guidance per user instruction.

## 2026-11-06 — Add frozen-input boundary for the future final runner

Added CPU-only capture_frozen_inputs: rejects candidate/mutable revision,
incomplete pins, non-finite thresholds and changed artifacts; returns exact
verified bytes. Added measured candidate JSON with null pending source/LPIPS/
dependency pins. Local LPIPS package/cache unavailable; no fabricated hashes.
11 freeze/checkpoint tests passed; Ruff check/format and diff check passed.
The gate is not yet wired to final inference, and no test split/GPU was opened.

## 2026-11-06 — Verify continuation capture and bind prefix payload

Extracted continuation capture/write logic used by Modal into a CPU-only helper.
Five filesystem tests cover missing file/hash, changed bytes, parent mutation
after capture and interrupted replace/restart. Prefix sidecar now binds NPZ
checksum, source digest and donor index; a torn cache/receipt pair fails closed.
Evaluator checkpoint loads now use the same captured bytes as the recorded hash;
DDPM EMA loader supports BytesIO as well as paths. Ran 24 related tests: passed.
Ruff check/format and diff check passed. No training/evaluation run or GPU started.
These tests exercise the shared helper, not an end-to-end remote Modal job.

## 2026-11-06 — Complete full metric replay and prepare group execution

Extended the provenance replay script with metrics-only mode. Ran CPU replay:
calibration thresholds refit only on the same 40 calibration slices; all four
cached-score candidates matched C's image/pixel AUROC/AP/P/R/F1, thresholds,
macro lesion Dice and empty-normal FPR to 1e-10 on 43 holdout slices. Saved
aggregate receipt; no new inference, GPU or held-out test. Role boundaries no
longer block work. Added FINAL_EVALUATION_CANDIDATE with measured settings,
controls, remaining provenance and runner requirements; explicitly not frozen.
Ruff check/format and git diff --check passed. Next shared tasks are resume
verification and freeze/test setup.

## 2026-11-06 — Review tiến độ sau tích hợp

Fetch GitHub và kiểm live HF: revisions không đổi. GitHub Actions xác nhận 3
commit gần nhất success, bao gồm 9aa42c5; workflow chỉ cài data/dev. Linear
connector vẫn sai workspace. Review code phát hiện giới hạn donor/source identity,
cache checksum binding, hash sau checkpoint load và thiếu test trực tiếp Modal
continuation. Thêm PROGRESS_REVIEW_2026-11-06, cập nhật audit/blocker và phân công
A/B/C tới deadline. Không chạy test mới, không thay code, không thí nghiệm/GPU.

## 2026-11-06 — Guard resumable validation prefixes and Modal continuation

Prefix resume now requires a sidecar contract binding checkpoint hash, index
hash, inference mode, condition and sampling settings; it validates cache length
and labels and reads NPZ from captured bytes. Modal continuation verifies actual
`ddpm_last.pt` and `ddpm_best.pt` against parent manifest before creating the new
run, then seeds both child files from those verified bytes before trainer load.
Regression tests: 19 related patch-validation and DAC-8 tests pass; Ruff passes.
No Modal/GPU job started. Existing legacy prefixes without a contract are
rejected for resume and must be regenerated.

## 2026-11-06 — Verify evaluator inputs from the same bytes

For feature-condition and condition-intervention comparisons, parse verified
result JSON and NPZ caches from the same bytes that were hashed. Two synthetic
tests first reproduced reopening of verified inputs; after the fix the relevant
14 tests passed. Checkpoint loading and resumable-prefix identity remain open in
BLK-20261106-01. No training or evaluation run was started.

## 2026-11-06 — Integrate C into B and repair CI collection

Merged C `c02addb` into B `5aa4823`, preserving both result/worklog histories.
Torch-dependent test modules now skip when the optional dependency is absent;
epsilon-MSE comparison permits float32 reduction rounding. Branch push now
triggers CI, and a Markdown code block was formatted for Ruff. On Windows with
distinct writable basetemp outside sandbox: data/dev environment 121 passed,
22 skipped; Torch environment 237 passed. Sandboxed default temp runs produced
fixture permission errors and are not reported as passing. Private-cache and
checkpoint guard review remains open in BLK-20261106-01; no new MRI/GPU run.

## 2026-11-06 — Start B replay

Downloaded private DDIM and naive-stitch validation caches; their SHA-256 values match C's manifests. CPU replay completed with image/mask/index alignment and metric checks, followed by paired group bootstrap (999 valid/1,000); results and caveats are tracked in RESULT_TRACKING. Two group-bootstrap unit tests passed. No GPU or test split used.

## 2026-11-06 — Audit GitHub/HF/Linear và cập nhật LPIPS/DDIM

Đọc live Linear DACN-KHMT, Git đến C c02addb và HF fae9b50; kiểm dataset unchanged,
Colab disconnected, Modal cá nhân app list rỗng. Đọc hội thoại Gemini và đối
chiếu paper/model-card chính thức. Thêm PROJECT_AUDIT_2026-11-06 và evidence JSON,
link từ README/plan/tracking/result tracking; blocker BLK-20261106-01 và cập nhật 2 blocker cũ.
Kiểm 94 artifact references mới không mismatch (JSON bytes/LFS metadata, không tải
NPZ/checkpoint); không model/GPU/held-out test run mới. CPU QA chi tiết trong audit:
C c02addb có 181 passed / 1 failed (float32 reduction); B có 165 passed / 1 error
ở lần cuối (Windows rename), 35 data tests chạy riêng pass. Không tuyên bố full
suite xanh khi các lỗi này còn tồn tại.
Kiểm tra cuối: JSON hợp lệ, 94/94 checks matched, không có UUID/token hoặc key
patient/sample identity trong evidence; local documentation links và git diff --check pass.
Chưa merge source C, chưa thay Linear status/owner; bước tiếp là integration và
private-cache reproduction trước paired bootstrap/freeze.
### 2026-11-06 — Implement frozen RAD-DINO score comparison

- User requested RAD-DINO. Pinned official public weights revision and processor;
  added frozen blocks3/6/9/12 patch distance, equal mean raw primary and fixed
  normal-calibration secondary on immutable finalDDIM20+30 pairs; LPIPS control.
- Added bounded T4 runner with strict loading/source alignment, private artifacts
  and fixed protocol. No training or new reconstruction. CPU targeted8tests
  passed in3.82s; full suite182tests passed in30.37s; Ruff and whitespace passed.
- Modal smoke completed with strict weight load, zero identical-pair distance
  and no scale floors; full83completed. PixelAP raw RAD-DINO .289211 vs LPIPS
  .312330; normal-calibration .284124. Full metrics/provenance in RESULT_TRACKING.

### 2026-11-06 — Check full83 calibrated LPIPS results

- Verified completed/uploaded Modal manifest and aggregate JSON checksum.
  Recorded all four arms' pixel/image AP, AUROC, P/R/F1, normal FPR and runtime.
- Calibration did not improve primary holdout PixelAP (.312330→.307160);
  final re-sampling LPIPS exceeded naive stitching (.312330 vs .180593).
  Interpretation is exploratory on inspected validation; test remains sealed.
- Documentation only; no new GPU jobs. Exact checksum and numerical control
  checks performed; no implementation change requiring another test suite.

### 2026-11-06 — Implement normal-calibrated LPIPS scoring trial

- Added seven-layer extraction/replay checks, normal-calibration-only median/IQR
  statistics, equal-weight calibrated score, original LPIPS and naive-stitch
  controls on immutable final DDIM20+30 cache; no training/reverse calls.
- Added T4 runner with source/index/weights checksums and private artifact storage,
  fixed protocol and five meaningful calibration/replay/isolation tests.
- Validation: targeted5 tests passed; complete suite179 passed in28.30s.
  Smoke completed with zero layer-sum/cache replay error and no scale floors;
  full83started detached with same protocol. Actual provenance/status in RESULT_TRACKING.

This log records completed repository work. Linear remains the source of task
status; this file records changes and evidence, not a second task board. Keep
entries concise and append new entries at the top.

## Entry format

```text
### YYYY-MM-DD — short task title
- Issue/PR:
- Changes:
- Evidence/checks:
- Next:
```

Record only checks that actually ran. Link relevant issues, PRs, commits, or
artifacts when known. Do not put credentials, patient identifiers, or private
source data in this file.

## Entries

### 2026-11-06 — Review test.md and separate encoder roles for perceptual research
- Changes: Added LPIPS_ENCODER_RESEARCH.md answering the requested foundation
  model, attention, multi-contrast and A–D experiment questions, with primary
  links, input/license/weights readiness, formulas, layer taps, relative compute
  and a staged implementation plan. Incorporated the user's SqueezeNet baseline /
  RAD-DINO cross-domain / brain-MRI-domain hypothesis table and architecture controls.
- Evidence: Inspected official LPIPS, RadImageNet, RAD-DINO, MedSAM, RadFM,
  BrainFound, BrainDINO, MM-DINOv2 and 3DINO sources/model cards. RAD-DINO is CXR;
  some MRI artifacts are not runnable from the sources checked; 3DINO is gated
  with a separate license; arXiv2608.28773 is withdrawn. RadImageNet weight bytes,
  exact processor and strict load were not verified; did not claim they were.
- Recommendation: Normal-calibrated per-layer perceptual scoring on fixed
  DDIM20+30 final pairs, followed by controlled medical-encoder comparisons.
  No new AP prediction or claim of novelty; all A–D remain proposals.
- Checks: Local baseline values, Markdown links and whitespace. Documentation
  only; no unit tests, new pretrained downloads, training, GPU calls or final test.

### 2026-11-06 — Verify completed DDIM20+30 four-score ablation
- Evidence:Full83completed/uploadedceba5af6;root aggregate SHA matches manifest,
  exactsmokesource/config,all83callcounts50,4150U-Net/166LPIPS,onephase2pass,
  outsideerror0,no-testflag. Recorded all8currentpixel/image arms,matched vs
  historical controls,maskcoverage,timing scopes and privateexample availability.
- Findings:FinalLPIPSAP0.312330/Dice0.371215 vs matchedinitialDDIM20LPIPS
  0.157876/0.246871. FourrequiredAPs0.090936/0.099569/0.312330/0.196812;
  no suppressed arms or post-hoc design selection. Naive-stitchL1 andfinalL1
  AP are nearlyequal;naive-stitchLPIPS unmeasured,so avoid over-attributing gains
  to learnedinpainting. BrightnesspixelAP0.389506 remainshigher.
- Checks:Result checksum/count/no-test/numeric records andwhitespace only,no
  newunit/GPUrun. Updated report/tracking/conclusions. Actualdenoisingmean1.462s
  perimage,rootjob186.571s;9.1875xspeed reference is onlytwohistoricalT4samples,
  not83image end-to-end speed. Independentmethod confirmation remainsfuturework.

### 2026-11-06 — Launch authorized full DDIM20+30 ablation
- User approved full83after concrete50-callsmoke. Launched detached
  autoddpm-ddim2030-full83-t4-20261106T074200Z,
  appap-98lEz64JkA8I5USQcgE4CR with unchangedJSONpreset and source. Fourrequired
  matchedscore rows share the finalDDIM30image;onephase2pass,notfive repeats.
- Evidence:Modal initialized and createdrun_ddim_ablation_remote. Existing174tests
  and successfulsmoke remain applicable;no source edits or newtests for launch.
- Durablemanifest confirmsrunning07:42:26UTC,TeslaT4,20+30,repeats1,exactsmoke
  source hash and no-testflag.
- Next:Check artifacts on user request. No training,
  finaltest or continuing monitoring;fullmetrics pending.

### 2026-11-06 — Apply supplied testing_Lpips.md DDIM20+30 requirements
- User asked to read test_lpips.md;found actualdocs/testing_Lpips.md and read it
  in full. It supersedes the earlier10+10x5proposal with exact20+30U-Net budget,
  onephase2pass,four matched scoring arms and explicit full83authorizationgate.
  Preserved that supplied document and completedolder smoke;no oldfulljob started.
- Changes:Added validated flatJSONpreset,explicit uniform endpoint-correct grids,
  DDIM20initial/DDIM30singlepass,four-arm scores usingactualfinalLPIPS/final
  normalizedresidual,samefinaltensor for3AutoDDPMscores,initial/historical controls,
  calibrated mask/thresholds,private examples,component timing and hooked callcounts.
- Checks:174tests passed17.66s,targeted Ruff/format,whitespace. Five newtests
  verify20/30grids,actual50calls/context,four score identities,LPIPSfinalnotinitial,
  fixedonepassconfig. Previous10+10x5smoke completed/uploadedd3d088ed with1510calls,
  exactancestralreplay and noAP;aggregatehashverified,retainedtimingreference.
- Attempt:newautoddpm-ddim2030-smoke-t4-20261106T073737Z completed/uploadedac8aaaa0,
  appap-mSM7Tlr1zK7LHFEEoS2fiu;rootresultchecksumverified. Bothoutputs50calls,
  onephase2pass,outsideerror0,23LPIPS/480totalU-Netcalls incl380calibrationextra.
  Denoising-only2image speedup8.7275x vs prior matchedancestral timing;not83image
  end-to-end speed. No smokeAP. Recorded actualcounts,timings,scope,config/source
  hashes and privateartifact provenance;no source change after successful smoke.
- Full83notstarted,must await explicit confirmation after smoke,as suppliedtask
  requires. Exact reviewable command and protocol inAUTODDPM_DDIM20_30.md.

### 2026-11-06 — Implement DDIM acceleration for both AutoDDPM phases
- Changes:Added eta0 arbitrary reverse jumps and cumulative-alpha forward gap
  re-noising for masked re-sampling. Phase1ten DDIMcalls at t300,phase2tenjumps
  x5repeats at t50;preserve seed/RNG/context contracts. Extended shared evaluator
  with explicit sampler pair and synchronized inpainting timings. Added cached
  ancestral,phase2-only and both-phase comparison,T4 lifecycle,exact U-Net hook
  counts,two-image fresh ancestral timing/cache replay and calibrated maskrule.
- Checks:169tests passed18.88s,targeted Ruff/format and whitespace. Six newtests
  verify multi-step/clean-terminal forward gaps,existing DDIM clipping parity,
  shared phase1noise/ten calls,phase2fifty calls/context/RNG,oracle inversion,
  invalid indices/repeats and empty-mask preservation. No new mask design chosen.
- Attempt:autoddpm-ddim-smoke-t4-20261106T072722Z,
  appap-ghqmk2bcm1tRiBMyajKhEu started;timing/finite/cache replay pending.
- Next:After bounded smoke,requested full83sampler-speed comparison. Do not
  equate theoretical550/60NFE with actual full-pipeline wall-clock acceleration.

### 2026-11-06 — Verify LPIPS-only attribution and pseudo-mask error audit
- Evidence:Full83audit completed/uploaded3b82d8b2;rootresultSHA matches manifest,
  productreplayerror0,pseudo-mask exact across83;zeroDDPM/re-sampling calls.
  Recorded newscore pixel/image AUROC/AP/P/R/F1,diagnostic strata and private
  panel availability. Four private images remain remote,not visually reviewed.
- Findings:LPIPS-onlypixelAP0.266899/Dice0.339668 vs product0.204298/0.274354,
  weightedfinal0.154785/0.226256,brightness0.389506/0.268051. Pseudo-mask misses
  16.2459percent lesion pixels includingcore;74.8201percent of itsFP count lies
  inotherforegroundinterior. Distinguish area counts from rates and pseudo-mask
  from calibrated final score threshold. No anatomical labels or matchedrecall claim.
- Checks:Verified checksum/status/83samples/no-testflag,FP/lesion-strata count
  consistency,numeric records and whitespace. Record-only edits,no unit rerun,
  new GPU run,mask redesign or independent-test evaluation.

### 2026-11-06 — Begin user-selected perceptual/mask research roadmap
- Changes:Recorded four-stage roadmap with explicit no-holdout-design-selection
  rule and independent-confirmation requirement. Implemented steps1/2 only:
  LPIPS-only/normalized-residual ablation on cached original reconstructions,
  exact product/mask replay,normal-calibration gradient threshold,post-hoc
  lesion geometry/component and disjoint FP proxy strata,private fixed examples.
  New T4 lifecycle uses no diffusion calls and preserves source artifact hashes.
- Checks:163tests passed19.55s,targeted Ruff/format and whitespace. Five new
  tests cover exact original product reproduction,gradient-fit leakage,
  exhaustive/exclusive proxy regions,lesion/component misses and normal FP counts.
- Attempt:smokeperceptual-audit-smoke-t4-20261106T065148Z,
  appap-FvyVHTJjk0cuYi4Hso4zJw completed/uploaded402250a5;aggregate checksum
  verified,originalproduct replayerror0,pseudo-maskexactmatchTrue,no AP from2samples.
  Elapsed5.726s,peak45882880B,source4689409edc64b047b0822dfe966e377fda4e0fcd4c4f398eab19a11365f2e297.
- Next:requested83image explanatory audit launched detached after smoke. No mask redesign or
  independent-test run;do not choose a new method from inspected holdout ranking.
  Runperceptual-audit-full83-t4-20261106T065328Z,appap-VmGU48SYKuWh8W685WGCf6;
  durable manifest confirmsrunning,TeslaT4,source identical to smoke,zeroDDPM calls.

### 2026-11-06 — Publish memory fusion and LPIPS AutoDDPM findings to Git
- User requested pushing the latest measured results. Collected implementation,
  tests, fixed protocols and full memory-bank/LPIPS AutoDDPM aggregate reports
  on `feat/anoddpm-notebook-training`; preserved the supplied method document.
- Evidence: Existing full suite158passed and both smoke/full GPU attempts are
  recorded above. No implementation edits since those checks; no test rerun or
  new GPU run for this publication. Reviewed staged files, whitespace and
  credential patterns; private data, banks, weights and run arrays stay ignored.
- Scientific conclusion preserved: initial LPIPS x residual scoring beats final
  re-sampling scores in this exploratory fold; brightness still has higher AP.

### 2026-11-06 — Verify full LPIPS AutoDDPM validation outcome
- Evidence:Completed/uploadedbab00576 for83samples;verified aggregate JSON SHA
  against manifest,source/checkpoint/cache hashes,normal-calibration mask
  threshold and no-test flag. Recorded all8pixel/image metric arms,mask coverage,
  normal ROI and timing. No new GPU job,training,threshold search or unit rerun.
- Conclusion:Primaryweighted finalAP0.154785 improves baseline0.101066,finalSSIM
  0.149446 improves0.102402,but initialLPIPSxresidual heatmapAP0.204298 is higher
  than any re-sampled arm. Absolute re-samplingAP0.129407 vs naive stitch0.118781
  is a modest incremental gain. Brightness pixelAP0.389506 remains stronger.
  Maskrecall83.7541%/precision13.3407%;outside-context error0 is imposed.
- Records:Updated protocol,full results and consolidated conclusions;numeric
  and whitespace checks only. Treat results as exploratory on inspectedvalidation.

### 2026-11-06 — Implement LPIPS pseudo-mask/stitch/re-sample on Gaussian EMA30
- Changes: Added frozen spatial SqueezeNet LPIPS0.1 mask branch on verifiedt300
  cache,normal-calibration95percentile mask threshold,existing50step x5repeat
  tested inpainting sampler,strict outside-context check and naive-stitch control.
  Eight fixed score arms,calibration-only score thresholds,private prefix/maps
  and T4 smoke/full lifecycle;no DDPM training. Documented departures from
  official AutoDDPM and prior unsuccessful t200 SSIM-proxy trial.
- Checks:158tests passed18.44s,targeted Ruff/format and whitespace passed.
  New tests cover LPIPS range/channel/crop,normal-only mask calibration leakage,
  fixed threshold/dilation. Existing sampler tests cover RNG/context/repeats.
- GPU attempt:bounded2image smokeautoddpm-lpips-smoke-t4-20261106T061700Z,
  appap-852Pcx7xpt7qW71tJ7MVbD completed/uploaded87d0943c;real LPIPS outputs
  finite,outside-maskerror0,threshold0.13774138689,elapsed24.315s,no AP. Verified
  aggregate result checksum against manifest. Exact sourceb4a1fb55cfed8acb8aaa2004b04ae76e581e6f0b05d71e4bc07a7f7f4bea18b2.
- Next:Full83comparison launched detached asautoddpm-lpips-full83-t4-20261106T061833Z,
  appap-Q53OLMLNmmu3h6rZCVv2cV,same implementation;metrics pending.
  Durable manifest confirms running,TeslaT4,source identical to smoke,1/83.
  No final test or retraining. Check on user request,not continuous monitoring.

### 2026-11-06 — Verify completed full normal memory fusion results
- Evidence: Full run completed/uploaded01c5995f; verified downloaded aggregate
  result SHA against manifest,all7500normal train/10000bank/83validation,
  exactsource and no-test flag. Recorded six-arm pixel/image metrics,20normal
  calibration normalization,recall operating points,bank hash and timing.
- Conclusion: absolute-fusion pixelAP0.102374 vs baseline0.101066 is a tiny gain;
  SSIM-fusion0.087033 vs0.102402 declines. Memory-only pixelAP0.064212 but imageAP
  0.822745: distinguish image ranking from localization, and note calibrated
  image threshold predicts allholdout images positive. Brightness pixelAP0.389506
  remains much stronger. Updated protocol/report and tracking from real artifacts.
- Checks: Result checksum/status/numeric records; whitespace for record-only
  edits. No new tests,training,GPU inference,alpha tuning or final-test access.

### 2026-11-06 — Launch authorized full DDPM memory fusion validation
- User explicitly approved the full run after the successful bounded smoke.
- Launched detached `memory-fusion-full83-t4-20261106T055134Z` on team T4,
  app `ap-UiDz4LoIR7Zey5Ld1cUuoA`. Fixed seed17/alpha0.5; all7500normal
  training slices,10000-vector bank,83validation samples and existing40/43
  calibration/group holdout. Reuses Gaussian EMA30 ancestral t300 cache.
- Evidence: Durable manifest confirms running at05:51:56UTC,TeslaT4,bank10000,
  no train subset cap and no test use; exact source hash matches successful smoke.
  Source/config unchanged since155-test/smoke checks. No new unit run for launch-only work.
- Next: Check artifacts on user request; no continuous polling or new experiment.

### 2026-11-06 — Implement frozen normal memory bank fusion with DDPM
- Changes: Added frozen official ImageNet grayscale ResNet18 layer2/layer3
  extraction, normal-train-only random bank construction/persistence/checksums,
  bounded exact neighbor search, padded-grid/crop alignment and separate fixed
  alpha0.5 absolute/SSIM fusion. Reused verified Gaussian EMA30 cache without
  changing DDPM training or sampling. Added normal-calibration median/IQR,
  frozen thresholds/recall operating points, six-arm grouped evaluation and
  private maps/predictions/examples. Preserved user method specification.
- Checks:155 tests passed17.48s, targeted Ruff passed, whitespace checked. New9
  tests cover frozen BN/feature dimensions, deterministic chunk-independent
  sampling, foreground/alignment, exact neighbors, bank metadata/checksum,
  normal-only guard, normalization leakage, recall threshold and cache ordering.
  T4 smoke16normaltrain/128bank/2validation completed/uploaded56552b57;
  finite aligned fusion maps and bank roundtrip passed, no AP measured.
- Next: Full7500normal/10000bank/83validation is explicitly gated by the supplied
  method specification. Exact command and caveats in DDPM_MEMORY_FUSION.md.

### 2026-11-06 — Prepare research implementation and results for Git push
- Changes: Collected global A/B and spatial conditioning, matched unconditional
  fine-tuning, condition ablation, K4 uncertainty scoring, configs, tests and
  experiment reports on `feat/anoddpm-notebook-training`. Included the existing
  method specification without changing its contents.
- Checks: Full suite passed (146 tests, 28.76s); Ruff passed for cloud,
  evaluation, training, tests and new conditioning models. Reviewed U-Net diff,
  checked whitespace and scanned code/configs/docs/tests for credential patterns.
  Checkpoints, private MRI data and local run artifacts remain ignored.
- Next: Commit and push the current research branch as requested; no GPU run.

### 2026-11-06 — Verify K4 uncertainty results and scoring tradeoffs
- Changes: Recorded completed83image/K4 inference,all249new reconstructions,
  six-score pixel/image metrics,normal calibration parameters,quality/variance
  diagnostics and private provenance. Updated uncertainty and consolidated reports.
- Evidence: Completed/uploadedaf0bf4006db4c948292b9bc50748073e1b421bb4;
  resultSHA matches manifest,sourcecheckpoint/83samples/sealed-test flag verified.
  PixelAP singleabs0.101066,mean4abs0.103810,normalized0.045729,Mahalanobis
  0.077657. Normalized imageAP0.720711 improves image ranking but not segmentation;
  existing brightness imageAP0.770309 remains higher. Parameters fit20normalcal
  only. Timing counters indicate resumed prefixes;report fullwall41.80min rather
  than finalattempt7.83min as whole-job latency.
- Checks: Numeric/link/whitespace checks for record-only edits;no unit rerun,new
  GPU inference/training or test access. Do not claim normalization improves AP.

### 2026-11-05 — Implement K4 uncertainty scoring on original Gaussian EMA30
- Changes: Added fixed4draw mean/unbiased variance,normal-calibration variance
  floor and shrinkage9D signed3x3patch Mahalanobis,sixscore controls,grouped
  thresholds and score/variance diagnostics. Added atomic per-draw prefix caches,
  calibration parameter freeze before holdout,T4 smoke/full lifecycle and private
  uploads. Reuses baseline draw1;only3new draws per image,no training/encoder.
- Evidence/checks:146tests passed27.89s,targeted Ruff/format and whitespace
  passed. New6tests verify mean/variance,normal-calibration-only fit invariance
  under abnormal/holdout perturbation,singular covariance regularization,patch
  covariance,cache provenance and threshold/calibration freeze ordering.
- Next: Bounded2image K4/t300 T4smoke before full83. K4variance is stochastic
  reconstruction variation,not certified uncertainty;normalization may reduce
  lesion scores. No AP improvement claim or held-out-test access.
- T4smoke app ap-XWKFboy5nw35LHkFTNev1O completed/uploaded
  501d5f86ee79ddd3d81b1cfa7b62c53a620670a8. Replay errors[0,0],6newchains,
  finite K4variance/normalized/Mahalanobis;no AP on2images. Source frozen at
  15a90894fc8dcd6578a64256710af85604ae4fac3d1ef66e147ecd9d7f878583.
- Full ddpm-uncertainty-k4-val83-t4-20261105T163529Z,
  app ap-R0EC27PZnPnALYHMySkxkH started16:35:52UTC,TeslaT4. Durable manifest
  confirms exactsource/replayerrors[0,0],draw2progress5/83. Detached client;
  metrics pending,check on user request. No training or extra GPU experiment.

### 2026-11-05 — Verify spatial grayscale completion and negative AP outcome
- Changes: Recorded completed10epoch/9380update spatial training,selectedEMA10,
  full83 validation,private revisions/hash checks,pixel/image metrics,quality,
  lesion/nonlesion means and timing. Updated spatial/consolidated conclusions.
- Evidence: Training29777bf14204bdc7b4450f2b9a0746b1a537911c and eval
  a6e21e5c2553e683fda3e51496fdc52a31662b60 completed/uploaded. Results hash
  matches manifest;checkpointSHA/83samples/sealed-test flag verified. Holdout
  AP difference0.063110/SSIM0.065459 versus control0.104391/0.104004;
  normal foreground SSIM0.688031 versus control0.295441. Lesion/nonlesion error
  means converge;pathology retention plausible,not proven mechanism.
- Checks: Record-only numeric/link/whitespace checks;no unit-suite rerun,new GPU
  training/inference or test access. No AP benefit inferred from lower noiseMSE.

### 2026-11-05 — Implement grayscale spatial ImageNet conditioning pilot
- Changes: Added four-scale ResNet18 with1channel stem adapted by meanRGB
  official ImageNet filters,removed global pooling,kept MRI[-1,1]/256geometry.
  Added zero1x1 spatial residuals at U-Net down blocks64/32/16/8 without changing
  baseline state keys. Shared training/loader selects spatial method,encoder/
  fusion/U-Net optimizer groups,full-state resume,ImageNet provenance and
  automatic matched baseline/A/B/unconditional validation. New T4 wrapper bakes
  public weights before GPU allocation; separate smoke/full configs.
- Evidence/checks:140tests passed17.19s,new tests cover grayscale weight mapping/
  hash checks,feature shapes/no pooling,zero output equivalence,spatial effect,
  gradient gate opening,encode-once reverse,RNG,inference without ImageNet
  download,and spatial checkpoint/optimizer resume. Actual official46,830,571B
  weights hash verified locally; adapted conv1 and expected4map dimensions
  checked on synthetic1channel input. Targeted Ruff/format and whitespace passed.
- Limitation: unet.py retains2pre-existing Ruff findings SIM114/PLC0206 unrelated
  to optional spatial fusion. MRI adaptation differs from official RGB ImageNet
  normalization; trial also changes encoder initialization/LR,so not isolated
  spatial-only ablation. No AP result or test-set access.
- Next: Submit bounded T4 smoke then independent10epoch/9380update pilot with
  full83 t300 validation only after smoke succeeds. User grayscale steering
  supersedes initial RGB-repeat plan before any GPU training.
- GPU evidence: First2 attempts failed in CPU image build(no training),recorded
  with cause/resolution. Smoke3 app ap-Gkt1RTxe4nYswmOMDt2AnX succeeded,
  private revision4d47de2f38c281e14401ed80a79ab315ef5186ba,15updates,finite3step
  reverse,3.862974s train+val,peak1710858752B. Then submitted independent full
  spatial-imagenet-gray-t4-b8-e10-20261105T141939Z,
  app ap-z896lVzq2nUac9XbgTsY4L,10epochs and auto83slice validation.
- Full start14:27:12UTC,TeslaT4,durable running/training manifest,epoch0 recovery,
  frozen source2f6c5005dd22618a4d49fa56707b9779143f12fab04467b580681cdd6c1118e8.
  Epoch1/10 step150/938 finite loss0.008948 confirmed. Detached client;no long
  polling or AP claim. Full results pending user-requested check.

### 2026-11-05 — Verify control training and prepare matched validation
- Changes: Recorded control completed/uploaded375bc6ba5578f5c27953f3306cecdabb531f073b,
  epoch40/9380updates,best36/MSE0.007353029596. Added unconditional EMA loader
  path to shared grouped evaluator, original ancestral helper with noisy arrays,
  partial-cache resume, matched baseline/A/B comparisons and resumable T4 smoke/
  full validation worker. Training/conditional inference defaults preserved.
- Evidence/checks:21focused tests passed8.01s,including original-chain equality,
  RNG preservation,actual selectedepoch36,calibration-only thresholds and partial
  recovery for unconditional mode. Targeted Ruff passed. Profile/shared Secret
  name verified; checkpoint SHA from completed manifest recorded before GPU eval.
- Next: Full test suite and bounded T4 replay smoke before full83 validation.
  No validation AP result yet; no new training or held-out-test use.
- Full suite135tests passed12.83s,targeted Ruff/format and whitespace passed.
  Submitted bounded smoke ddpm-uncond-val-smoke-t4-20261105T094736Z,
  app ap-7Hot55hkZx47D5XtcOaMrd,training/checkpoint references frozen.
- Smoke completed/uploaded0ad3619e2aa19d0d04a37069e4c750f7d263f37f,
  original replay maxerrors[0,0],exactpaired-noise,target36 finite2image/3steps.
  Frozen source68092ef7fe5e470d833a1ce0909c84011438e7b41414e91f1761f5dc4968c93a.
  Then submitted full83 t300 best36 validation with matched baseline/A/B outputs.
- Full app ap-2AEvfGt80K4eoxODBlGW4g durable start09:50:15UTC,TeslaT4,
  exactsnapshot,checkpoint36; preflight passed,reconstruction1/83 confirmed.
  Job subsequently completed83/83 before local disconnect; completed/uploaded
  revision38491cba20168ac41a038e98cfaffe1bc3fb51b5,results hash matches manifest,
  checksum/samples/sealed-test flag verified. Control holdout AP difference
  0.104391/SSIM0.104004,nearly A0.104645/0.104247 and B0.104464/0.104084.
  Added full metrics/timing and updated consolidated conclusions: unconditional
  extra training explains nearly equivalent gain; no clear encoder benefit
  established by this one-seed comparison. No extra training or test access.

### 2026-11-05 — Implement unconditional EMA30 fine-tuning control
- Changes: Added strict EMA-only warm initialization to original DDPM loop,
  fresh optimizer/counters/best selection, metadata and epoch30 recovery state.
  Added no-encoder trainer/configs for smoke31 and full31..40,source-checked
  T4 lifecycle/resume. Native full-state extension safeguards unchanged.
- Evidence/checks:22 focused tests passed4.42s; synthetic test distinguishes
  online/EMA weights, verifies strict source/no-encoder state keys, LR1e-5/fresh
  optimizer, source immutability, epoch31/32 resume and original loader support.
  Wrapper rejects unmatched/unbounded budgets. Targeted Ruff passed; source
  manifest completed/uploaded and registered epoch30 SHA verified. Shared
  Workspace/profile and Secret name checked without credential inspection.
- Next: Bounded smoke before independently initialized full10epoch job. No AP
  improvement claim, automatic83-image evaluation or test access authorized.
- Full suite133tests passed12.79s,targeted Ruff/format and whitespace passed.
  Smoke submitted as ddpm-uncond-control-smoke-t4-20261105T082204Z after initial
  sandbox connection failure; network retry app ap-uW93yqQAyW93cmX0gdbFuY
  created successfully. GPU smoke results pending, source frozen for resume.
- GPU smoke subsequently completed/uploaded73552266d80393164a2a1dcf3fb3eacbe1d15a57,
  epoch31/15updates,16normal train/4normal-val,2image3step preview artifact.
  Manifest confirms EMA30/freshoptimizer/noencoder,finite normalMSE0.000374397350,
  4.466861s,peak1466632704B. Then submitted independent full epoch31..40 run;
  frozen source03f0be1c2c2991db35952c492372b51e9cbad3afcc05307a3b60e80c0034ffd2.
- Full run ddpm-uncond-control-t4-b8-e40-20261105T082502Z,
  app ap-kpcgyltxZ55ANmBYWC1N2O started08:26:49UTC. Durable manifest
  running/training,TeslaT4,epoch30 initial checkpoint. Actual epoch31/40
  step50/938 finite loss0.008318; detached client after start confirmation.
  Full run remains active; no AP evaluation or improvement claim.

### 2026-11-05 — Authenticate requested new account and verify its T4 startup
- Changes: After user clarification, authenticated/activated profile taitrang2005
  through Modal browser flow. Actual Workspace taihongtran2005 verified; older
  modal-new profile still maps to phutungcogioihaitan and was not the requested
  personal Workspace. Recorded distinction and new startup result.
- Evidence/checks: Profile current/list confirms active taitrang2005. Reused
  standalone Secret-free probe; app ap-HEmpKNsuPAviG0hKdP2KzN in taihongtran2005
  completed/stopped,statuspassed,TeslaT4/15360MiB. No HF/Volumes/training or
  source changes. Whitespace check passed; no test-suite rerun or secrets logged.

### 2026-11-05 — Minimal new-profile T4 test without project credentials
- Changes: Added standalone modal_gpu_smoke.py with no Secret/Volume/HF
  dependency; T4 startup and nvidia-smi only,60s function timeout.
- Evidence/checks: Ran with --profile modal-new; app ap-t4QbkUQSa8RZ6Sc97R7us9
  completed/stopped, TeslaT4/15360MiB/driver610.57.04,statuspassed. Ruff/format
  and whitespace checks passed; no tensor benchmark, training or test-suite run.

### 2026-11-05 — Verify Modal startup using newly added profile
- Changes: Added short T4 access-check entrypoint with existing Secret/Volumes,
  cache/checkpoint existence checks and private HF metadata requests. Uses
  create_if_missing=false, prints no credentials or private sample information.
- Evidence/checks: Explicit profile modal-new lists required resources and runs
  TeslaT4/15360MiB successfully; both HF read checksHTTP200/private. App
  ap-jGYAyb2PvLTaOey1nSFXb5 completed/stopped without training. Targeted Ruff,
  format and whitespace checks passed; no unit suite rerun for startup probe.
- Limitation: Default CLI profile remains phutungcogioihaitan; select modal-new
  explicitly for later commands. No remaining-credit or HF write-access claim.

### 2026-11-05 — Consolidate feature conditioning results in Vietnamese
- Changes: Added FEATURE_CONDITION_CONCLUSIONS.md with baseline/A/B designs,
  preprocessing/evaluation contract, actual training budgets, holdout pixel/image
  metrics, all83 descriptive outcomes, zero/donor intervention, reconstruction
  quality/deltas, costs, caveats, reproducibility hashes/private artifact links,
  proposed next steps and a report-ready conclusion. Preserved newmethod.md.
- Evidence/checks: Read existing protocols and local aggregate result artifacts;
  21metric table rows and3results hashes checked against produced A/B/intervention
  JSONs. Only documentation changed; whitespace and all relative links passed.
  No new GPU experiment, inference, test-set access or test-suite rerun.

### 2026-11-05 — Verify B condition diagnostic completion
- Changes: Recorded both83-slice interventions completed/uploaded, paired
  preflight/cache validation, full holdout pixel/image metrics, descriptive
  all83 AP, normal FPR, reconstruction changes and runtime/VRAM.
- Evidence: Manifest completed withzero/donor83/83 and no failure, private
  revisionc22ee10da5f0193e99d3f3c097ab37e120cb7737. Results SHA matches manifest;
  checkpoint/sample count/sealed-test flag verified. Holdout SSIM AP
  self0.104084,zero0.104416,donor0.104699. No clear correct-image condition
  benefit; mean changes small but some localized pixel changes large, so no
  claim that the encoder is entirely ignored. One exploratory seed only.
- Checks: Git whitespace check for record-only edits; no tests rerun, new
  training, inference or held-out-test evaluation started.

### 2026-11-05 — Implement fixed B condition intervention diagnostic
- Changes: Added zero-injection and normal-training-donor wrappers, paired
  full83 validation modes, immutable reference/checkpoint/index/cache checks,
  bounded GPU replay/finite preflight, cached correct-condition reuse, private
  reconstruction deltas/examples and resumable T4 evaluation/upload lifecycle.
  Zero mode bypasses learned bias; donor changes encoder input only. Training
  code/weights and original default evaluator behavior preserved.
- Evidence/checks:129 tests passed13.27s, targeted Ruff/format and whitespace
  checks passed. Tests cover bias bypass, donor-only encoding, identical initial
  noise/RNG restoration, fixed donor mapping/cropped delta, calibration-only
  thresholds and partial-cache recovery under both new intervention modes.
- Next: Bounded T4 smoke must pass before full paired83-slice diagnostic. No
  training, AP improvement claim or held-out-test access. Shared Workspace and
  Secret name checked without reading credentials.
- GPU smoke evidence: completed/uploaded92df7e4ef60628e019d7a2b85146b393f2d28450,
  correct300-step replay matched both images exactly(maxerror0,initial noise
  identical), finite3step zero/donor controls. Source frozen at
  b5e4fccffe407d80914106884b49a41567a1fab134bfa025a89f77799d1f41c0.
  Then submitted full83 zero/donor validation, correct-condition cache reused.
  Full app ap-nxZTHthEc7ebwJkuTlTOi0 manifest running; preflight passed and
  zero-condition progress1/83 confirmed before detaching the local client.

### 2026-11-05 — Verify experiment B completion and A/baseline comparison
- Changes: Recorded completed5epoch encoder pretraining,10epoch joint training
  and83-slice validation, private upload revisions, full grouped-holdout metrics,
  normal reconstruction quality, timing and matched A/baseline comparison.
- Evidence: All3 manifests completed/uploaded;4690pretrain+9380joint updates,
  best encoder5/best fine-tune6. Result JSON hash matches manifest artifact;
  best EMA SHA,83samples,t300,seed17 and sealed-test flag verified. B holdout
  AP difference0.104464/SSIM0.104084 vs A0.104645/0.104247: near-identical,
  no demonstrated gain from this encoder pretraining pilot. Additional pretrain
  compute recorded separately; no significance or general-method failure claim.
- Checks: Whitespace check only for status records; no new GPU job or test rerun.

### 2026-11-05 — Implement experiment B normal-only encoder pretraining
- Changes: Added bounded dense masked-reconstruction ResNet-18 pretraining,
  foreground-only masked loss, private full-state checkpoints/resume and best
  selection on normal validation. B transfers encoder only, independently loads
  original Gaussian EMA30, resets optimizer and matches A's9380 joint updates.
  Added one-T4 chained worker/configs, automatic fixed83-slice t300 validation
  and immutable A aggregate comparison. Preserved user-authored newmethod.md.
- Evidence/checks:124 tests passed17.12s, including masked-target hiding,
  foreground loss, decoder shape/encoder gradients, normal-only pretrain resume,
  B encoder transfer, fresh optimizer/zero projection/original U-Net and existing
  A/evaluation regression checks. Targeted Ruff passed. Modal shared Workspace
  and Secret name verified without accessing secret contents.
- GPU evidence: Bounded B smoke passed, both phases15updates, finite3-step
  reconstruction, private uploads60fd76a2/0f4de9f4 confirmed. Then submitted
  fresh full5pretrain+10joint T4 chain with automatic83-slice validation;
  app ap-UKIL1QqcsgKzhIzGNO0m50. Source snapshot frozen at
  bcc65e702e23a2ca97179cf066c7d38c9a40d1bd1c3f3c58b8f726c9d67aadf2.
  No B AP improvement claimed; additional pretraining compute separate. Leave
  full job detached after confirming start; check results when user requests.
- Full start confirmed: durable manifest05:31:54UTC,TeslaT4,training stage,
  exact source hash; encoder epoch1/5 step150/938 with finite running loss.
  Detached client after confirmation; no continuous long-job monitoring.

### 2026-11-05 — Verify experiment A completion and validation
- Changes: Recorded10-epoch joint feature training and83-slice t300 evaluation
  completion/upload revisions, actual selected bestepoch6, full metrics,
  baseline/brightness comparison, normal reconstruction quality and limitations.
- Evidence: Training manifest completed/uploaded,9,380updates,best normal MSE
  0.007343506560. Eval manifest completed/uploaded at56f13bc33a470858475058cb961b1e904f680946,
  correct best checkpoint SHA,83 samples,held_out_test_used=false. Results JSON
  hash a8e57ed824c9e6a00404d82296f15fb208eabb9d8d911c06fcd983dfb8b5a962 verified
  locally. App stopped withzero tasks. Holdout SSIM AP0.102402->0.104247,
  difference0.101066->0.104645; modest gains, no isolated conditioning claim.
- Checks: Git whitespace check passed; no tests rerun for status-only records.
  No new GPU job, B pretraining or evaluation sweep started.

### 2026-11-05 — Implement experiment A ResNet-18 feature-conditioned DDPM
- Changes: Read user-provided newmethod.md; added grayscale ResNet-18 pooled
  features,128dim projection/zero injection into existing timestep embedding,
  joint normal-only fine-tune, buffer-aware EMA, full-state checkpoint/resume,
  t300 grouped evaluation with baseline/brightness controls, tissue quality,
  lesion separation and private figures. Added separate configs/entrypoint for
  smoke and10epoch A with automatic post-training validation. B not run.
- Evidence/checks:119 tests passed16.99s; new tests cover zero-injection output
  equivalence, dimensions/ResNet parameter count, gradient opening, exact MSE,
  deterministic fixed-condition reverse without masks, loader separation,
  BN buffers, full-state resume and partial evaluation cache recovery. Targeted
  Ruff checks/format passed, original baseline loaders remain exercised.
- Smoke: First Modal submission failed at image build ordering before training;
  corrected dependency installation before source mount. Fresh smoke2 image
  built successfully and dataset setup began. Recorded both attempts in tracking.
- Next: Require real normal-only smoke success before10epoch full job; preserve
  baseline and user-authored newmethod.md, no external pretrained weights.
- Smoke2 evidence: Completed/uploaded revision2e89ab1e4ad5469656bb53065fdccbfb5cd49b59,
  16 normal train/4normal val,15 shard-aware optimizer updates,3.860s train/val,
  peakVRAM1.71GB; remote smoke=passed and finite3-step reconstruction. This is
  not an AP result. Submitted fresh full A10 run after smoke success, with
  automatic t300 grouped validation and private baseline/quality/figure outputs.
  Full-run source is frozen for safe preemption resume; no further Python edits.
- Full start evidence: Durable manifest2026-11-04 17:46:35UTC,TeslaT4,source
  72e43a42c623d123c29ec5ebb707de3de3b22c63a859148e662a7e4ceaef9ee1,epoch0 recovery
  checkpoint committed. Log epoch1/10 step25/938,finite running MSE0.008372.
  Detached client; no continuous monitoring or anomaly improvement claim.

### 2026-11-04 — Publish diffusion experiments on the existing feature branch
- Changes: Prepared accumulated T4 Gaussian/Simplex continuation, context training,
  one-step/DDIM/denoising/masked evaluation, CPU score-fusion/diagnostic scripts,
  configurations, tests and aggregate research records for the user-requested
  push to feat/anoddpm-notebook-training. Checkpoints, images and caches remain
  in ignored/private storage.
- Evidence/checks:111 tests passed11.84s, Ruff format check89 files passed,
  Ruff check all30 changed/new Python files passed. Publication scan of46
  changed/new files found no credential patterns, source patient identifiers,
  binary artifacts or oversized files. Git whitespace check passed; CLI demo
  ran successfully. Remote feature branch matches local HEAD before commit.
- Limitation: Full-tree Ruff check reports4 pre-existing findings in unchanged
  notebooks/anoddpm_bmad_colab.ipynb and src/dacn/models/diffusion/unet.py;
  these are outside the staged experiment changes. No GPU job started.

### 2026-11-04 — Read Simplex epoch20 validation results
- Changes: Recorded completed/private-uploaded validation at revision
  1788b1232aa19b3c9ae96aa95a74e2975f4b592a, full pixel/image metrics and thresholds,
  timing/cache hashes and same-protocol epoch10 comparison.
- Evidence: Durable manifest completed/uploaded; validation_results.json confirms
  all83 slices,t300,EMAepoch20,held_out_test_used=false. SSIM AP0.074791 versus
  epoch10 0.073910,F1 0.166414 versus0.165806; only small improvement despite
  noise-MSE decrease. No new GPU evaluation/training or fusion started.
- Checks: Git whitespace check passed; no tests rerun for status-only records.

### 2026-11-04 — Submit Simplex epoch20 full validation
- Changes: Recorded user-authorized83-slice T4 validation, original Simplex
  epoch20 EMA milestone,t300,ancestral,batch1,seed17,paired absolute/localSSIM
  maps and private reconstruction cache. Uses existing evaluator unchanged.
- Evidence/checks: Source training manifest confirms completed/uploaded
  checkpoint20 and hash. Shared Workspace/Secret name verified. Evaluator
  compiles; git whitespace check passed. No tests rerun for submission-only
  documentation; no training, timestep sweep or held-out-test access.
- Start verified: Durable manifest10:57:22UTC, validation_evaluation10:57:49UTC,
  TeslaT4, correct checkpoint20 SHA, cache enabled and held_out_test_used=false.
- Next: Leave detached and check on request.

### 2026-11-04 — Verify Simplex epoch20 completion and private upload
- Changes: Updated continuation result to completed with checkpoint20/best/last
  hashes, upload revision0846480f8551c7e4b41a534c148320d61c906496 and measured
  normal-validation MSE0.002546022252, bestepoch20,18,760 optimizer steps.
- Evidence/checks: Durable manifest statuscompleted/artifact_sync_statusuploaded,
  source artifacts include epoch15/20. App list confirms recovery app stopped
  withzero tasks. No tests rerun for status-only documentation; no GPU job or
  anomaly AP evaluation started. Git whitespace check passed.
- Next: Evaluate epoch20 anomaly maps before claiming AP gain or updating fusion.

### 2026-11-04 — Recover Simplex continuation after Modal preemption
- Changes: Resubmitted stopped run with --resume from durable epoch18 to finish
  epochs19/20, preserving exact source/config and full model/optimizer/RNG state.
  Recorded interrupted attempt, recovery app and checkpoint provenance.
- Evidence: Old app stopped with zero tasks. Logs explicitly report container
  preemption during epoch19 step625, followed by fresh-run directory collision;
  history/manifest confirm completed epoch18,16,884steps,normal-val MSE0.002592801301.
  Current source snapshot matches saved SHA256 exactly. No code changes or tests
  needed for runtime recovery. No additional AP inference or test access.
- Recovery verified: Manifest attempt2/resumed10:07:57UTC; log epoch19/20
  step50/938, running MSE0.006495. First25/50 losses match interrupted epoch19,
  consistent with restored RNG/state. Detached client; remote T4 keeps running.
- Checks: git diff whitespace check passed. No tests rerun for runtime recovery.

### 2026-11-04 — Continue Simplex epoch10 to20 on T4
- Changes: Added epoch20 continuation config with only horizon/milestone changes,
  kept parent run intact, submitted fresh detached T4 run restoring full state
  from epoch10. Save additional milestones15/20; retain batch8/LR1e-4/EMA/noise,
  training split and checkpoint selection. Recorded provenance and estimate.
- Evidence/checks: Read completed parent manifest and Volume listing confirming
  last/best/epoch10 files,9380 steps. Config extension validator accepts only
  epochs/checkpoint_epochs changes.10 DAC8 contract tests passed2.93s; git
  whitespace check passed. Shared Workspace and named Secret verified.
  Initial sandbox client network failure resolved via approved retry; Modal
  app ap-qJD8Ddbo1XCuLciFJ6QJqS initialized successfully.
- Start verification: Durable manifest confirms parent epoch10 and checkpoint
  hashes, TeslaT4, stage training07:06:19UTC, source snapshot
  a6a20656efda0b1094b27442f7429996ed3915a3804e33797404c835718b5651.
- Next: Leave detached; user will request status. No extra inference or test
  access authorized by this training run.

### 2026-11-04 — Test cached Gaussian/Simplex SSIM fusion
- Changes: Added CPU-only registered-cache fusion evaluator with normal-only
  calibration CDF, three frozen blend weights, raw/CDF/brightness controls,
  patient-group split and persisted calibration-only selection/thresholds before
  holdout scoring. Downloaded existing Simplex cache to ignored local runs/;
  no GPU inference/training. Actual evaluation started and tracked.
- Evidence/checks:2 focused tests passed2.97s, verifying CDF ties/holdout
  isolation and frozen thresholds/selection before holdout evaluation. Targeted
  Ruff check/format and git whitespace checks passed. Outputs remain private.
- Results: Completed in22.001 CPU seconds. Best-calibration fusion75%Gaussian /
  25%Simplex has holdout pixel AP0.116771,F1 0.167315,FPR0.051199 versus raw
  Gaussian0.102402/0.154654/0.055234. Overall calibration selector still chooses
  Gaussian CDF; all fusion calibration APs are lower than Gaussian. Recorded
  this distinction and full pixel/image metrics; no robust gain claimed.
- Evidence: Frozen selection persisted before holdout; produced JSON SHA256
  f0d755d8415184eb3e42c80dc050176dca1edcad34edbeb1773b65cb666bed67.
  Gaussian/brightness controls reproduce prior scores. No holdout retuning,
  extra inference, GPU, training or test access.

### 2026-11-04 — Test t300 one-step and DDIM10 against ancestral cache
- Changes: Added separate fixed sampler ablation with common noisy input,
  exact10 DDIM calls/eta0, clipped x0/re-derived epsilon, immutable cache hash/
  array alignment and first-sample ancestral recomputation control. Shared
  grouped evaluation, private prefix artifacts and per-sampler model timings.
  Submitted fresh user-requested T4 all83-validation job; no training/test use.
- Evidence/checks:3 focused tests passed4.39s; oracle inversion, independent
  eta0 transition formula, model-call sequence, RNG preservation, common input,
  frozen thresholds and83-sample coverage. Full109 tests passed9.28s; targeted
  Ruff check/format and git whitespace checks passed. Shared Workspace/Secret
  name verified without credential inspection. Formula compared to primary
  DDIM and guided-diffusion implementations linked in RESEARCH.
- Results: Completed83 samples; first ancestral control matches cache exactly.
  Uploaded private revision4aa62f3b961460928adfc84876a97f050f8f5af9.
  Holdout SSIM AP one-step0.098530/DDIM0.095631/ancestral0.102402; FPR
  0.031443/0.042014/0.055234 with lower recall for new samplers. All83 one-step
  AP increases slightly but not holdout AP; no consistent detection gain.
  Pure reconstruction one-step0.03347s/image,DDIM0.30496s/image, separately
  timed with CUDA synchronization. Recorded full pixel/image metrics and caveats.
- Evidence: Read completed/uploaded durable manifest and produced results.json,
  aggregate JSON hash verified locally; no additional GPU run started.

### 2026-11-04 — Run original Gaussian epoch30 one-step validation
- Changes: Added full-image one-step original1-channel EMA control to shared
  validation evaluator and Modal runner, preserving context paths. Loads original
  registered epoch30/config, Gaussian forward draw and analytical x0; one call
  per image. Submitted user-requested fresh T4 run on all83 validation slices.
- Evidence/checks:14 targeted tests passed4.39s, including original1-channel
  oracle noise inversion, one-call/RNG isolation, evaluator provenance and
  frozen thresholds; targeted Ruff checks passed. Shared workspace and Secret
  name verified, no credentials inspected. No training or test access.
- Results: Completed83/83 and private upload b9f33457a84860275d7caecdfd0438693bbd706a.
  Original one-step all83 pixel AP0.070311 difference /0.071103 SSIM;
  holdout0.075539/0.075919. Higher AP than context one-step, below same-checkpoint
  ancestral t100. Loop51.448s including score/cache work. Recorded pixel/image
  metrics, hashes, calibration-only thresholds and limitations.
- Additional evidence: Full106-test suite passed7.62s; durable manifest confirms
  completed/uploaded and83 samples. Aggregate JSON downloaded to ignored runs/.

### 2026-11-04 — Run one-step context reconstruction on full validation
- Changes: Added optional one-step x0 estimation to patch/context reconstruction,
  preserving ancestral default, coverage/context and derived seeds; extended
  full-validation evaluator/Modal method to distinguish one-step provenance.
  Submitted fresh user-authorized T4 all83-slice run, no training or test access.
- Evidence/checks:104 tests passed8.17s, including single model-call/formula,
  unchanged context and RNG isolation, both evaluator modes, prefix persistence
  and frozen calibration thresholds. Targeted Ruff checks passed. Verified
  shared Workspace and named Secret without reading credentials.
- Results: Completed83/83 and privately uploaded revision
  243070769bc36adaba8e21fda4822ec697205399. Loop219.687s including scores/cache;
  holdout pixel AP0.065636 difference /0.071232 SSIM, lower than brightness.
  All83 descriptive AP0.060280/0.065432 also below Gaussian epoch30 ancestral
  t100 reference. Recorded pixel/image metrics, thresholds, hashes and caveats.
- Evidence: Read completed/uploaded durable manifest and produced results.json;
  downloaded aggregate JSON into ignored runs/. No additional GPU job launched.

### 2026-11-03 — Stop full-validation job at user request
- Changes: Recorded user-stopped evaluation in RESULT_TRACKING; no automatic
  restart. Existing trained checkpoint and prior diagnostic are preserved.
- Evidence/checks: Modal app stop --yes succeeded; app list confirms stopped,
  zero tasks,00:00:34 local time. Durable manifest remains stale running;
  results/progress.json not found. No full-validation metrics produced or
  completed prefix confirmed. No tests run for this status-only update.

### 2026-11-02 — Submit full validation patch/context evaluation
- Changes: Added all-validation evaluator with same epoch5 EMA/t100/patch64
  reconstruction as the diagnostic, fixed patient-group calibration/holdout,
  calibration-only frozen thresholds, descriptive all83 metrics, brightness
  control and durable prefix caches. Raised Modal timeout to3hours based on
  measured ~72s/image; submitted detached user-authorized T4 run.
- Evidence/checks:15 targeted tests passed12.24s, including complete sample
  coverage, prefix persistence, and frozen-threshold propagation; targeted Ruff
  check/format and git diff whitespace checks passed. Verified shared Workspace
  and Secret name presence without reading credentials.
- Next: Confirm durable progress; report produced metrics after completion when
  requested. Do not poll continuously or access sealed test data.

### 2026-11-02 — Run bounded patch/context reconstruction diagnostic
- Changes: Added validation-only four-example sliding preview evaluator and
  Modal method with immutable epoch5 EMA checksum, plus local private comparison
  plotting script. Trial uses two normal/two abnormal examples, t100, patch64,
  stride48,25 patches/image; tumor masks are excluded from reconstruction.
- Evidence/checks: Eleven existing patch/context and contract-audit tests passed;
  new diagnostic test passed after correcting its synthetic UUID fixture.
  Targeted Ruff check/format and git whitespace checks passed. Durable manifest
  confirms completed/uploaded T4 app ap-hd2NZeEANWnVk9F5Y7BAMr, no test-set use;
  full suite101 tests passed20.11s. Four-example inference289.624s; private
  artifacts downloaded for local rendering, compared to the same cached t100
  Gaussian inputs/masks. Normal reconstruction improves on both examples,
  lesion separation remains weak/reversed on the two tumor examples.
- Next: Review diagnostic figures before choosing further experiments;
  this bounded diagnostic does not estimate population AP.

### 2026-11-02 — Check completed patch/context pilot
- Changes: Updated RESULT_TRACKING from running to completed with all five
  epoch losses, best epoch5 checkpoint, runtime/VRAM and immutable private Hub
  revision62a2d58b5db5004cc6337626a026c76ba7ebf963.
- Evidence/checks: Read durable Modal run manifest and training_history.json;
  confirmed5 epochs,4,690 steps and artifact_sync_status=uploaded. Git whitespace
  check passed. No new training or inference job started; no tests rerun for
  this documentation update.
- Next: Bounded patch/context reconstruction timing/quality pilot before a
  full validation evaluation; patch MSE alone cannot establish anomaly AP.

### 2026-11-02 — Implement and submit5-epoch patch/context fine-tune
- Issue/PR: none provided.
- Changes: Added independent3-channel patch/context trainer, verified EMA
  initialization from immutable Gaussian epoch30, zero-initialized extra input
  kernels, patch-only epsilon loss, random foreground-centered64px patches,
  new optimizer/EMA, milestone1/3/5 full-state checkpoints and private preview.
  Added conditioned patch/sliding inference and checkpoint loader for later use.
  Registered config patch_context_t4_finetune5 and adaptation protocol.
- Evidence/checks:100 tests passed, including8 patch-context tests covering
  target leakage, initialization, output shape, patch loss/gradient, crop
  coverage and end-to-end fresh-optimizer checkpointing. Actual base32 U-Net
  CPU smoke confirmed3 inputs/1 output at256px. Ruff/format/compile/diff passed.
  Shared Workspace/Secret name verified;5 epochs/4,690 new steps configured.
- Evidence/checks: Modal app ap-EBw2niCSb3IeS2iLs2DK5W durably started
  patch-context-ft5-t4-b8-20261102T155628Z at16:00:57UTC; source snapshot in
  tracking. Live output reached fine-tune epoch1,step25/938 with finite initial
  patch epsilon MSE2.944604. This is an early new-task loss, not an AP result.
- Next: Disconnect client; inspect epoch1/3/5 artifacts on request. No automatic full sliding
  evaluation or held-out test access; no epoch budget beyond5.

### 2026-11-02 — Audit checkpoint/preprocessing/diffusion contract
- Issue/PR: none provided.
- Changes: Added controlled read-only GPU diagnostic for exact EMA/epoch/input
  contract checks, one-step online/EMA predictions and paired large/small/mean
  reverse chains. Documented static audit in PIPELINE_CONTRACT_AUDIT and fixed
  protocol in RESEARCH; production model/sampling remains unchanged.
- Evidence/checks:18 targeted audit/contract/validation tests passed; diagnostic
  large chain exactly matches production in unit test, oracle inversion and
  common-noise controls verified. Targeted Ruff/format/compile/diff passed.
  T4 run gaussian-e30-contract-audit-20261102T153939Z started and recorded;
  T4 run completed and uploaded at f69fc08d: exactEMA/input/source/schedule/
  arithmetic checks pass. t300 one-stepEMA MAE0.053872/SSIM0.527959 versus
  full-large0.074614/0.358931. Small variance changes little; deterministic
  mean improves SSIM but worsens t300 MAE. No test data or retraining.
- Next: Retain baseline; investigate foreground/timestep loss breakdown and
  controlled training/conditioning changes, not claim AP gains from normal
  fidelity alone. Full findings in PIPELINE_CONTRACT_AUDIT.

### 2026-11-02 — Audit normal foreground reconstruction at t100/200/300
- Issue/PR: none provided.
- Changes: Added reproducible offline normal reconstruction audit, aggregate
  report NORMAL_RECONSTRUCTION_AUDIT and paired private example sheet.
- Evidence/checks: Verified39 normal empty-mask slices, all source cache hashes
  and cross-timestep alignment. Foreground meanSSIM drops0.533->0.375->0.293;
  full-crop SSIM is inflated by74.11% background. Interior errors remain high.
  Regional pooled-MAE identities/MAE<=RMSE verified; Ruff/format/compile/diff
  checks passed. CPU40.586s, no new inference/GPU/test/training.
- Next: Audit train/inference contract before blaming weights alone or
  allocating more epochs; this step measures the symptom, not its root cause.

### 2026-11-02 — Audit Gaussian batch8 loss convergence
- Issue/PR: none provided.
- Changes: Downloaded recorded30-epoch training history; added reproducible
  plotting script, PNG/PDF/CSV under ignored runs/, and TRAINING_CONVERGENCE
  report separating loss behavior from anomaly quality.
- Evidence/checks: History checksum matches manifest;30 sequential epochs and
 938 steps/epoch verified. Normal-val MSE10->20 improves3.46%,20->30 1.14%,
 25->30 0.36%; best epoch30. No sustained late validation-loss rise. Train loss
 uses online model/random noise, validation uses EMA/fixed noise; no direct
 train-versus-val gap interpretation. Targeted Ruff/format and plot generation
 passed; no retraining/new inference or held-out test access.
- Next: Audit normal-foreground reconstruction and train/inference contract;
 loss plateau alone does not explain low AP or justify more epochs.

### 2026-11-02 — Test reconstruction-map ensembles across t100/200/300
- Issue/PR: none provided.
- Changes: Added fixed equal-weight calibration-rank-mean evaluator using three
  existing Gaussian epoch30 caches. Prespecified protocol, same patient-group
  split and calibration thresholds; documented all comparison results.
- Evidence/checks: Source checksums and exact image/mask/label alignment
  asserted; baseline t200/t300 and brightness metrics reproduce earlier values.
  Ensemble difference AP0.091536 /F1 0.155345; SSIM AP0.092199 /F1 0.132428,
  both below single t300. Frozen pixel thresholds checked independently.
  Ruff/format/compile/diff passed. CPU96.510s; no GPU/test/training access.
- Next: Preserve negative result; do not adopt equal averaging as an improvement
  or retune weights on the already-inspected holdout.

### 2026-11-02 — Read completed AutoDDPM-style pilot
- Issue/PR: none provided.
- Changes: Updated run record from completed Modal manifest/results, private
  revision `2c8646c1`.
- Evidence/checks: Holdout resampled SSIM AP0.093603/F1 0.138912; difference
  AP0.070007/F1 0.149523. Mask covers63.05% of lesion pixels with9.20% precision;
  resampling gives no overall AP benefit over the best original DDPM baseline.
  Runtime769.038s/83 images; peakVRAM101,765,120B. Brightness control unchanged.
  No new GPU experiment, code change or local tests; test remains sealed.
- Next: Preserve negative evidence and review predicted mask quality before
  any further resampling/training. Do not generalize to exact LPIPS AutoDDPM.

### 2026-11-02 — Implement and submit AutoDDPM-style SSIM-proxy pilot
- Issue/PR: none provided.
- Changes: Added independently implemented automatic masking, stitching,
  matching-time context conditioning and resampling using the current Gaussian
  epoch30 checkpoint. Reuses verified t200 cache, fixed50x5 resampling schedule;
  SSIM substitutes LPIPS and the adaptation is explicitly labeled. Added initial
  score/naive-stitch controls, mask coverage diagnostics and grouped calibration
  holdout evaluation; no use of ground-truth masks for inference.
- Evidence/checks: Three sampler/mask tests passed,9 focused tests total;
  full suite89 tests passed. Targeted Ruff/format/compile/diff checks passed.
  Verified shared Workspace and Secret name. No clinical/improvement claims.
- Evidence/checks: Modal app `ap-pWoLoKKbxzgCa1RD9FoPOS` created a durable
  start manifest at14:01:03UTC for `autoddpm-style-ssim-e30-20261102T140011Z`;
  experiment recorded in RESULT_TRACKING.
- Next: Leave detached and inspect completed artifacts on request. Held-out
  test remains sealed.

### 2026-11-02 — Read completed denoising-score experiment
- Issue/PR: none provided.
- Changes: Updated run record from completed Modal manifest and results,
  private revision `4713bd88`.
- Evidence/checks: Denoising holdout pixel AP0.085848 /F1 0.165330 on43 images;
  below original reconstruction AP~0.10 and brightness0.389506. Brightness
  metrics/index hash reproduce prior grouped split. Inference loop28.876s for83
  images, peakVRAM101,437,440B. No new experiments, code edits or local tests.
- Next: Do not adopt this averaged epsilon-error score as an AP improvement.
  Preserve negative result and review a single justified alternative before
  spending further compute. Held-out test remains sealed.

### 2026-11-02 — Set up requested T4 denoising-score experiment
- Issue/PR: none provided.
- Changes: Added a fixed Modal T4 runner with checkpoint/dataset checksum
  verification, private Hub destination check, source snapshot and durable
  lifecycle/upload. Uses existing grouped denoising evaluator on epoch30.
- Evidence/checks: Six focused scoring tests passed; runner Ruff/format/diff
  checks passed. Shared Workspace and existing Secret name verified. No model
  retraining; experiment uses3 timesteps x4 noise repeats and original ROI.
- Evidence/checks: Modal app `ap-beD6gCpUwEv4VHg0Eu2krT` created a durable
  start manifest for `ddpm-denoising-e30-3t4r-20261102T103159Z` at10:32:32UTC;
  source checkpoint hash verified, exact source snapshot recorded in tracking.
- Next: Compare completed metrics on the same group holdout as brightness and
  reconstruction controls when requested; leave detached without continuous polls.

### 2026-11-02 — Add denoising-score pipeline alongside reconstruction
- Issue/PR: none provided.
- Changes: Added fixed multi-timestep epsilon-error scoring and validation-only
  CLI/evaluator. Reuses Gaussian checkpoints; averages3 timesteps x4 noise draws
  with isolated per-sample generators; same grouped calibration/holdout rule,
  brightness control and frozen thresholds as the preceding score study.
  Documented that this is an exploratory ablation, not a full ANDi reproduction.
- Evidence/checks: Six focused tests passed for exact-noise oracle, timestep
  indexing, averaging, global RNG isolation and invalid inputs. Full suite86
  tests passed; targeted Ruff/format/diff checks passed. No actual MRI experiment, AP claim, GPU job or test
  split access in this implementation task.
- Next: Run a registered validation pilot with epoch30 and compare against the
  existing brightness/reconstruction controls before adopting the new branch.

### 2026-11-02 — Test nine score options with patient-group calibration
- Issue/PR: none provided.
- Changes: Added reproducible CPU fusion evaluator and preregistered protocol;
  documented all nine candidates in SCORE_FUSION_RESULTS. Used existing epoch30
  t300 cache and private dataset index; group-disjoint40/43-image split.
- Evidence/checks: Cache checksum/labels/unique IDs/group disjointness asserted.
  Normalization and thresholds fit only calibration; selection JSON written
  before holdout evaluation. Independent check verified candidate selection and
  unchanged pixel/image thresholds. Ruff/format/compile/diff checks passed.
  Selected fusion holdout AP0.262977 versus brightness0.389506; reconstruction
  baselines~0.10. CPU20.628s; no new GPU inference or test access.
- Next: Keep the brightness control; no proven AP benefit of fusion over it.
  Confirm any chosen method with a separately registered validation procedure
  before final frozen test. No holdout-driven weight retuning.

### 2026-11-02 — Compare completed epoch30 t*=100/200/300 validation
- Issue/PR: none provided.
- Changes: Updated timestep run and diagnostic report from completed Modal
  manifest/results, private artifact revision `67626af0`.
- Evidence/checks: SSIM AP t100/200/300=0.076971/0.088429/0.100741;
  absolute-difference AP=0.070977/0.083184/0.099834. Lowering t* does not
  improve either score under the matched checkpoint/protocol. Recorded full
  pixel metrics, thresholds and runtime; no new training/evaluation or code edits.
- Next: Investigate score calibration/combination using existing validation
  caches rather than assume smaller t* solves the failure. Test stays sealed.

### 2026-11-02 — Diagnose Gaussian epoch30 low anomaly scores
- Issue/PR: none provided.
- Changes: Added checksum-verified validation cache diagnostics and an aggregate
  report at `docs/EPOCH30_DIAGNOSIS.md`. Measured score overlap, regional errors,
  brightness control and foreground-gating ablations. Submitted controlled
  epoch30 t*=100/200 evaluation on T4 with both original scores and private cache.
- Evidence/checks: CPU script reproduced original t*=300 metrics; brightness
  AP0.272325 exceeds reconstruction AP~0.10. Foreground-only gating gives small
  gains; 83.47% of SSIM false positives are in nonlesion foreground interior.
  Targeted Ruff/format passed; source cache checksum verified. GPU evaluation
  app `ap-Yh8Bu3Qilehwpdxl5LDKkK` created; new timestep metrics pending.
- Next: Read completed timestep artifacts on request; do not continuously poll.
  No retraining, held-out test use or silent baseline protocol changes.

### 2026-11-02 — Read completed Simplex epoch-10 validation
- Issue/PR: none provided.
- Changes: Updated evaluation record from completed/uploaded Modal manifest
  and validation_results.json, private revision `b4c408be`.
- Evidence/checks: All83 validation slices at t*=300; SSIM pixel AP 0.073910,
  F1 0.165806, Recall 0.603844, empty-normal FPR 0.133712; absolute-difference
  AP 0.032447 / F1 0.068113. Runtime 749.373 seconds. Main AP does not improve
  over existing Gaussian results; matched Gaussian epoch10 batch8 is unevaluated.
  No new GPU run, code changes or local tests; held-out test unused.
- Next: Inspect cached Simplex reconstructions before further training; compare
  matched training budgets before attributing changes solely to noise type.

### 2026-11-02 — Verify Simplex training and submit epoch-10 validation
- Issue/PR: none provided.
- Changes: Updated Simplex training record from the completed/uploaded manifest;
  submitted user-requested full validation at t*=300 with both registered scores.
- Evidence/checks: Simplex completed 10 epochs / 9,380 steps, normal-validation
  noise MSE 0.002971782594, training/validation time 4,465.438 seconds; private
  artifact revision `e7b6ce4f`. Shared Workspace and Secret name verified.
  Evaluation app `ap-ju8ELLyWBFQIacpQs4fcxK` created; metrics pending.
  No implementation changes or local tests in this task.
- Next: Read aggregate evaluation artifacts when requested after completion;
  avoid continuous polling. Held-out test remains sealed.

### 2026-11-02 — Launch user-requested 10-epoch Simplex ablation
- Issue/PR: none provided.
- Changes: Added explicit forward-noise/Simplex settings to partial-DDPM config
  and diffusion construction while keeping legacy checkpoints Gaussian by
  default. Registered a full-data T4 batch8 seed17 10-epoch Simplex config,
  retaining the same U-Net, optimizer/EMA and checkpoint-selection procedure.
  Documented noise semantics and planned validation comparison.
- Evidence/checks: Full suite passed with 80 tests; focused Ruff/format,
  compilation and diff checks passed. Full-shape CPU Simplex batch8 smoke was
  finite and exactly repeatable with fixed seed; warm noise generation took
  0.047 seconds using two CPU threads locally. No remote timing inferred.
  Modal app `ap-ssVCXTam3wHn7EaSMJFMtS` created a durable start manifest for
  `ddpm-simplex-t4-b8-e10-20261102T074257Z`; source snapshot recorded in tracking.
- Next: Leave detached without continuous polling. Read completion artifacts
  when requested, then evaluate final milestone on validation at t*=300.

### 2026-11-02 — Verify epoch-30 checkpoint on private Hugging Face Hub
- Issue/PR: none provided.
- Changes: Verified the existing upload instead of creating a duplicate commit.
- Evidence/checks: Team-authenticated Modal CPU check confirmed private model
  repository revision `1ee5c4995623532e7ab51c043839555c1dd0b0db` contains
  `ddpm_epoch_030.pt`, `ddpm_best.pt`, `ddpm_last.pt`, config and run manifest.
  Local Hub access returned 404; verification succeeded using the existing
  shared Secret without changing credentials. No training or evaluation run.
- Next: Download through the private repository with a team-authorized account.

### 2026-11-02 — Visualize saved epoch-30 validation reconstructions
- Issue/PR: none provided.
- Changes: Prepared four private local PNG comparison figures from the existing
  epoch-30 t*=300 reconstruction cache at the user's request: highest, median
  and lowest lesion-slice SSIM Dice, plus the normal slice with highest SSIM
  false-positive rate. Panels show input, reconstruction, both score maps,
  SSIM threshold prediction and ground truth. Figures and script stay under
  ignored `runs/visualizations/`; no patient identifiers are displayed.
- Evidence/checks: Downloaded cache checksum matches the completed evaluation
  manifest; plotting uses the existing score implementation and recorded SSIM
  threshold 0.8922310462404459. This is visualization of existing validation
  results, not a new inference run. No GPU job or held-out test access.
- Next: Review where predicted anomalies overlap lesions and normal anatomy.

### 2026-11-02 — Read completed epoch-30 t*=300 validation results
- Issue/PR: none provided.
- Changes: Updated RESULT_TRACKING from the completed Modal manifest and
  validation_results.json, private artifact revision `4472519e`.
- Evidence/checks: All 83 validation slices completed on T4; SSIM pixel AP
  0.100741 / F1 0.167483; pixel-difference AP 0.099834 / F1 0.182870.
  Image AUROC remains below 0.5 and max-F1 predicts every image positive.
  Runtime 718.415 seconds. No code changes, new GPU run or local tests.
- Next: Diagnose saved validation reconstructions/maps before increasing the
  training budget. Held-out test remains sealed.

### 2026-11-02 — Verify epoch-30 completion and launch t*=300 validation
- Issue/PR: none provided.
- Changes: Updated training status from the durable completed/uploaded manifest;
  submitted user-requested EMA epoch-30 evaluation on all 83 validation slices
  with ancestral sampling at t*=300, inference batch 1 and both registered scores.
- Evidence/checks: Epoch 30 / 28,140 optimizer steps completed; best normal-val
  epsilon MSE 0.00735903694; artifacts uploaded at revision `1ee5c499`.
  Evaluation app `ap-RWi4F9nKiAnh75BoUA42BG` created on the shared Workspace;
  results pending. No implementation changes or local tests in this task.
- Next: Inspect aggregate evaluation artifacts when complete; no continuous
  monitoring requested. Held-out test remains sealed.

### 2026-11-02 — Complete T4 epoch-1 pilot and prepare safe epoch extension
- Issue/PR: no issue or PR was provided.
- Changes: Recorded the completed batch-8 epoch-1 result and added a lineage-aware
  continuation path that permits only the epoch horizon/checkpoint schedule to
  change. Added the matching epoch-30 T4 config; model, EMA, optimizer, sampler
  and RNG state resume from the source checkpoint while the completed source run
  remains immutable.
- Evidence/checks: Epoch 1 completed in 412.506 seconds with train epsilon MSE
  0.080065, normal-validation epsilon MSE 0.323197, 938 optimizer steps and
  5,465,677,824 bytes peak VRAM; private revision `178c1de6`. Continuation unit
  tests verify epoch 2/history/optimizer behavior and rejection of batch-size
  changes. Full suite passed with 79 tests; targeted Ruff, format, compile and
  `git diff --check` passed. Full-tree Ruff still reports two pre-existing
  findings in `models/diffusion/unet.py` outside this change. The detached continuation
  `ddpm-partial-t4-b8-e30-cont-20261102T034644Z` was then created as Modal app
  `ap-Gau6wwuNQYP4U3UJBi6rrm`; live output reached epoch 2, step 25/938, proving
  that the source epoch was restored rather than repeated.
- Next: Do not poll continuously; inspect the durable manifest/artifacts when
  requested after the estimated completion window. Keep held-out test unavailable.

### 2026-11-02 — Prepare T4 batch-8 one-epoch DDPM pilot
- Issue/PR: `BLK-20261102-03`; no issue or PR was provided.
- Changes: Added a fresh full-data Gaussian DDPM pilot config with one epoch,
  batch size 8, validation batch size 8 and EMA 0.9992, plus a fixed T4 Modal
  remote entrypoint sharing the existing durable training lifecycle. Documented
  that it cannot resume the batch-1 run and is not a replacement checkpoint.
- Evidence/checks: Resolved config reports 7,500 training samples, 938 optimizer
  steps and checkpoint epoch 1. Full suite passed with 77 tests; targeted Ruff,
  format, compilation and `git diff --check` passed. Four sandboxed launch
  attempts failed at Modal connection before app/run creation. A final fixed
  entrypoint submission with direct network permission created Modal app
  `ap-vIFbYSY0oC1V74Tj4DFJvV`, durable start manifest and source snapshot; the
  running pilot is now registered in `RESULT_TRACKING`.
- Next: Read durable runtime/loss/checkpoint artifacts after completion; keep
  held-out test unavailable.

### 2026-11-02 — Evaluate posterior-mean DDPM inference on T4
- Issue/PR: `BLK-20261101-02`; no issue or PR was provided.
- Changes: Added a posterior-mean reverse sampler with batch-independent
  per-sample initial-noise seeds, T4 batch-8 validation support and private
  reconstruction caching. Ran a two-sample CPU shape/cache smoke, then evaluated
  epoch 10 at `t*=300` on all 83 validation slices. Held-out test was unavailable
  to the evaluator.
- Evidence/checks: Full local suite passed with 77 tests; targeted Ruff, format,
  compilation and `git diff --check` passed before launch. The Modal manifest
  completed on Tesla T4 and uploaded private revision `3a91ec15`. Local SSIM AP
  was 0.081721 and F1 0.140292, both below ancestral AP 0.095895 and F1 0.161616,
  despite lower empty-normal FPR. Posterior mean is rejected as the replacement
  sampler; reconstruction cache and exact source snapshot remain private.
- Next: Keep the ancestral baseline and review a controlled Simplex-noise
  training experiment before any held-out test evaluation.

### 2026-11-02 — Audit DDPM inference false positives without GPU
- Issue/PR: `BLK-20261101-02`; no issue or PR was provided.
- Changes: Audited checkpoint loading, EMA/config, one-based partial diffusion,
  deterministic per-sample seeding, crop, local SSIM/pixel-difference maps,
  threshold selection and validation-only split use. Ran bounded CPU diagnostics
  on one validation lesion slice for foreground ROI, three-reconstruction
  averaging and a posterior-mean reverse-chain hypothesis; no test data or GPU
  was used. Updated result tracking and blocker evidence without changing the
  registered inference implementation.
- Evidence/checks: Epoch-5/10 SHA256 values matched the durable manifest; 18
  targeted tests passed. The foreground diagnostic found most false positives
  inside normal brain tissue, not the padded/background region. Three-sample
  averaging and posterior-mean reverse produced mixed AP/AUROC changes on one
  slice, so neither is reported as a validation improvement.
- Next: If GPU budget is approved, run one registered full-validation sampler
  ablation with private reconstruction caching before spending budget on more
  training; keep held-out test sealed.

### 2026-11-01 — Back up the epoch-10 checkpoint to private Hugging Face
- Issue/PR: User-requested checkpoint upload; no issue or PR was provided.
- Changes: Used the shared Modal Secret in a temporary CPU-only helper to sync
  `ddpm_epoch_010.pt`, the durable run manifest, aggregate history/validation
  metrics and exact source snapshot from `dacn-bmad-runs` to the existing
  private `dacn-bmad-team/dacn-runs` model repo. No credential value, private
  image or per-sample identifier was read or logged.
- Evidence/checks: Hugging Face commit
  `07ec8b455a3318a267df5568f7af3fa4f15d7a3c` reports a private destination. The
  uploaded checkpoint SHA256
  `b828b5889d4164c0d9cebbbe0bf7afb603f5bec9ad7c84508baa222a26bc4fc9` matches
  the Modal manifest and prior local verification. No GPU, training or
  evaluation run was started.
- Next: Grant intended teammates private repo access or share the immutable
  artifact link; resume the original Modal run separately if epoch 20/30 is
  still required.

### 2026-11-01 — Add blocker tracking and team handoff rules
- Issue/PR: User-requested shared blocker log; no issue or PR was provided.
- Changes: Added `docs/ISSUES.md` with a stable blocker template, summary table,
  ownership/handoff fields, privacy rules, and two evidence-backed current
  blockers covering the incomplete primary DDPM run and validation false
  positives. Updated `AGENTS.md` so AI agents create or update this log before
  handing off unresolved blockers while keeping Linear and result tracking as
  their existing sources of truth.
- Evidence/checks: Reviewed the contributor workflow, project requirements,
  research protocol and recorded run results; `git diff --check` passed.
  Documentation-only change; no training, evaluation or test suite was run.
- Next: Confirm owners during team handoff, resume or close
  `BLK-20261101-01`, and use the fixed validation protocol to investigate
  `BLK-20261101-02` without opening held-out test data.

### 2026-10-31 — Add real-checkpoint validation evaluation runner
- Issue/PR: User-requested testing of epoch-5 and epoch-10 checkpoints; no issue
  or PR was provided.
- Changes: Added a validation-only checkpoint evaluator and Modal boundary for
  deterministic partial reconstruction, pixel-difference/local-SSIM scoring,
  threshold selection, aggregate metrics, incremental durable results and
  private artifact upload. The evaluator has no held-out test path.
- Evidence/checks: The complete local suite passed with 58 tests and 3
  PyTorch-dependent skips; targeted Ruff, Python compilation and
  `git diff --check` passed. The four-sample epoch-5 `t*=100` smoke completed on
  `NVIDIA A100-SXM4-40GB`, wrote aggregate results, explicitly recorded no
  held-out test use, and uploaded private artifacts at model revision
  `5e77e82dddfc423af2bf179282441978ee08d6f3`. The subsequent full sweep
  completed all six epoch/timestep combinations over 83 validation slices on
  the same A100 class. Local one-minus-SSIM with epoch 10 and `t*=300` had the
  best pixel AP (0.095895) and pixel AUROC (0.892010); its micro pixel F1 was
  0.161616 and macro lesion-slice Dice was 0.241476. The final manifest records
  six of six combinations complete, private artifact revision
  `51a9c22ee71bdae7d9d78538fb8664a7b96c9e48`, and no held-out test use.
- Next: Freeze the validation-selected epoch/timestep, score formula,
  aggregation and threshold protocol before authorizing one held-out test run;
  obtain additional seeds before drawing research conclusions.

### 2026-10-31 — Download epoch-5 and epoch-10 DDPM checkpoints
- Issue/PR: User-requested local checkpoint copies; no issue or PR was provided.
- Changes: Downloaded `ddpm_epoch_005.pt` and `ddpm_epoch_010.pt` from the
  stopped milestone run's Modal Volume into the ignored local `checkpoints/`
  run directory. No remote artifact was changed or deleted.
- Evidence/checks: Both files are approximately 126 MB. Local SHA-256 values
  `8c1d957265858c70879f46273691be7d88e93c4a562d50168c2991716aed682f`
  and `b828b5889d4164c0d9cebbbe0bf7afb603f5bec9ad7c84508baa222a26bc4fc9`
  match the durable Modal manifest for epochs 5 and 10 respectively.
- Next: Use these private local checkpoints for validation-only reconstruction
  and scoring; do not select parameters on the held-out test split.

### 2026-10-31 — Fix CUDA RNG restore and restart detached DDPM training
- Issue/PR: User-requested restart after a Modal client cancellation; no issue or
  PR was provided.
- Changes: Reproduced the epoch-4 resume failure, converted restored CUDA RNG
  states to CPU ByteTensors before calling PyTorch's CUDA RNG API, added a
  regression test, and invoked `train_remote` directly with Modal detach instead
  of the client-bound local controller. Preserved all earlier run artifacts.
- Evidence/checks: The resume attempt passed config/source provenance and then
  failed with `TypeError: RNG state must be a torch.ByteTensor` before epoch 5;
  its epoch-4 checkpoints remained intact. The local suite passed with 58 tests
  and 2 PyTorch-dependent skips; targeted Ruff, JSON validation, Python
  compilation and `git diff --check` passed. Fresh run
  `ddpm-partial-a100-s17-e30-milestones-rngfix-20261031T041542Z` has a durable
  manifest confirming the immutable dataset revision, milestones `[5,10,20,30]`,
  and `NVIDIA A100-SXM4-40GB`. At the user's request, the app stopped at
  07:07:05 UTC during epoch 11, step 1,800/7,500. Epoch 10 and 75,000 optimizer
  steps are durable; the Volume contains milestone checkpoints for epochs 5 and
  10 plus `ddpm_last.pt` and `ddpm_best.pt`. Best epoch is 10 with
  normal-validation epsilon-MSE 0.0101424818. No anomaly/test metric was
  produced.
- Next: Preserve the stopped artifacts or resume the exact run from epoch 11 if
  later milestones are required.

### 2026-10-31 — Retain DDPM milestone and best checkpoints
- Issue/PR: User-requested epoch 5, 10, 20 and 30 checkpoints; no issue or PR
  was provided.
- Changes: Added validated `checkpoint_epochs` configuration and atomic full-state
  checkpoint retention as `ddpm_epoch_005.pt`, `010.pt`, `020.pt`, and `030.pt`,
  while preserving validation-selected `ddpm_best.pt` and resumable
  `ddpm_last.pt`. Updated the full A100 config, guide, and resume contract test.
- Evidence/checks: The local suite passed with 58 tests and 2 PyTorch-dependent
  skips; targeted Ruff, JSON validation, Python compilation and
  `git diff --check` passed. Fresh run
  `ddpm-partial-a100-s17-e30-milestones-20261030T165258Z` has a durable manifest
  confirming the immutable dataset revision, milestone list `[5,10,20,30]`,
  status `running`, stage `training`, and `NVIDIA A100-SXM4-80GB`. A later Modal
  client cancellation stopped the function in epoch 5 at step 2,400/7,500.
  Epoch 4 and 30,000 optimizer steps are durable; `ddpm_last.pt` and
  `ddpm_best.pt` both represent epoch 4, with best normal-validation epsilon-MSE
  0.0138888134. No epoch-5 milestone or anomaly/test metric was produced. The
  interrupted epoch-2 run remains unchanged.
- Next: Resume the same milestone run from epoch 5 using its exact config and
  source snapshot, with a launch method that does not tie cancellation to the
  local controller lifecycle.

### 2026-10-30 — Change the full-data A100 budget to 30 epochs
- Issue/PR: User-authorized replacement of the running 40-epoch retry; no issue
  or PR was provided.
- Changes: Stopped `ddpm-partial-a100-s17-e40-retry-20261030T145811Z`, changed
  the reproducible seed-17 config and launch guide to 30 epochs (at most 225,000
  optimizer steps), raised the Modal function timeout from 6 to 10 hours for the
  observed A100 throughput, and retained the same full training data,
  validation-only checkpoint selection, `T=1000`, and held-out test isolation.
- Evidence/checks: Modal reports the replaced A100 app stopped at 15:02:43 UTC.
  Its last output was epoch 1, step 2,300/7,500 with no completed epoch or
  checkpoint; zero seeds completed and no anomaly/test metric was produced.
  JSON validation, Python compilation, `git diff --check`, and the local suite
  passed with 58 tests and 2 PyTorch-dependent skips. The replacement run
  `ddpm-partial-a100-s17-e30-20261030T150704Z` has a durable manifest confirming
  30 epochs, the immutable dataset revision, and `NVIDIA A100 80GB PCIe`. At the
  user's request, Modal stopped that run at 15:29:27 UTC during epoch 3, step
  2,600/7,500. The Volume retains `ddpm_last.pt` and `ddpm_best.pt` from epoch 2;
  the aggregate history confirms 15,000 optimizer steps and best normal-validation
  epsilon-MSE 0.0652383983. No anomaly or held-out test metric was produced.
- Next: Resume the same run ID with its exact config and source snapshot if the
  user wants to continue from epoch 3; otherwise preserve it as interrupted.

### 2026-10-30 — Start full-data partial-DDPM training on Modal A100
- Issue/PR: User-authorized run
  `ddpm-partial-a100-s17-e40-20261030T144735Z`; no issue or PR was provided.
- Changes: Synchronized the active Modal guide from T4 to A100, added the
  reproducible 40-epoch seed-17 full-data config, and registered the running
  training attempt in `RESULT_TRACKING.md`.
- Evidence/checks: Confirmed Modal Workspace `phutungcogioihaitan` and Secret
  `huggingface-secret-2`; validated the 25-key JSON config; Python compilation,
  `git diff --check`, and the local suite passed with 58 tests and 2
  PyTorch-dependent skips. The durable Modal manifest confirms immutable dataset
  revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`, status `running`, stage
  `training`, and `NVIDIA A100-SXM4-40GB`. A later client cancellation stopped
  the run at epoch 1, step 4,700/7,500 before any checkpoint; the run Volume has
  only its manifest and source snapshot. No evaluation or held-out test run was
  started, and no model result was produced.
- Next: Stop the stale app and launch a fresh run ID with the same config; this
  attempt cannot resume because no completed-epoch checkpoint exists.

### 2026-10-30 — Retry full-data A100 run after client cancellation
- Issue/PR: User-authorized retry
  `ddpm-partial-a100-s17-e40-retry-20261030T145811Z`; no issue or PR was provided.
- Changes: Confirmed the previous function input was canceled before its first
  checkpoint, recorded that attempt as interrupted, and launched a fresh run ID
  with the identical source snapshot, immutable dataset revision and config.
- Evidence/checks: Modal reported the old app stopped at 21:57:55+07. The new
  durable manifest confirms status `running`, stage `training`, and
  `NVIDIA A100-SXM4-40GB`; initial output reached epoch 1, step 200/7,500. No
  evaluation or held-out test run was started.
- Next: Keep the local detached client alive, confirm the first completed-epoch
  checkpoint, and update both run records from durable artifacts.

### 2026-10-30 — Add a runnable paired-score evaluation demo
- Issue/PR: Not provided.
- Changes: Replaced the scalar metric example with a deterministic synthetic
  reconstruction pipeline that evaluates pixel difference and local SSIM,
  selects pixel/image thresholds on validation, and reuses those frozen
  thresholds on test. Added macOS/Linux setup commands and CLI regression
  assertions for the validation/test boundary.
- Evidence/checks: The synthetic CLI demo completed and wrote aggregate JSON;
  Ruff check/format, Python compilation, `git diff --check`, and the complete
  CPU suite passed with 58 tests and 2 PyTorch-dependent skips. No medical data,
  model training, or research evaluation run was used.
- Next: Install a compatible PyTorch build in the project Python environment
  before exercising DDPM training or reconstruction code locally.

### 2026-10-30 — Point Modal jobs to the rotated Hugging Face secret
- Issue/PR: Not provided.
- Changes: Updated the active Modal runner and setup guidance to use
  `huggingface-secret-2` while retaining `HF_TOKEN` as the injected environment
  key. No credential value was stored in the repository.
- Evidence/checks: `git diff --check`, Python compilation, and notebook JSON
  validation passed. Modal Workspace verification was not run because the Modal
  CLI is not installed in this environment. No Modal job or model training run
  was started.
- Next: Confirm `huggingface-secret-2` exists in the shared Modal Workspace and
  contains a read/write-capable `HF_TOKEN` before the next job.

### 2026-10-30 — Run bounded partial-DDPM Modal smoke
- Issue/PR: DAC-10, DAC-14; run `ddpm-partial-smoke-20261030T044737Z-79f605`.
- Changes: Recorded the completed smoke and its provenance in RESULT_TRACKING.
- Evidence/checks: The `tanh1c` Modal workspace and `huggingface-secret` Secret
  name were present. The pinned dataset passed validation; the job completed
  128 optimizer steps and uploaded checkpoints and four private validation previews.
  Manifest confirms Tesla T4 and completed status. No research evaluation or
  test selection started. Resume and credit consumption were not measured.
- Next: Attach B/C validator evidence to DAC-9, record the DAC-8 review decision,
  demonstrate resume and estimate the resource budget before a larger run.

### 2026-10-30 — Add paired scoring and evaluation primitives
- Issue/PR: DAC-12 and DAC-30; work in progress on `feat/anoddpm-notebook-training`.
- Changes: Added cropped pixel and local SSIM maps, image top-1% scores, validation
  F1 threshold selection with tie rule, and aggregate image/pixel metrics. Updated
  DAC-9 in Linear to distinguish the current pinned revision from the older export.
- Evidence/checks: The full CPU suite passed (66 tests before the final paired
  helper); the final focused suite passed (9 tests), covering score direction, crop,
  SSIM formula, threshold ties, metric output and test-threshold guard. A final
  full-suite rerun passed 67 tests; Ruff and formatting checks passed. No real-data
  evaluation, training, Modal job or sweep was started in this implementation task.
- Next: Collect B/C validator evidence, record DAC-8 review, integrate the evaluator
  with the DDPM reconstruction runner and obtain teammate review before the sweep.

### 2026-10-29 — Address DAC-8 review findings
- Issue/PR: [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score); no PR linked.
- Changes: Specified a review-pending SSIM, seed, validation sweep and threshold
  contract and documented the DDPM/AnoDDPM adaptation. Added deterministic
  per-sample sampling seeds and single-sample reconstruction, exposed Hub sample
  UUIDs in memory, fixed AnoDDPM runner metadata compatibility and removed double
  counting of epoch time. Added focused regression tests and agent guidance;
  formatted the existing local notebook for the repository's Ruff CI check.
- Evidence/checks: Full suite passed (61 tests) with CPU and synthetic test data;
  Ruff and config parsing passed. Notebook output and execution-count hash was
  unchanged by formatting. The first full-suite invocation failed two CLI
  tests because the child Python process lacked `PYTHONPATH`; rerunning with
  `PYTHONPATH=src` passed. No real-data training, evaluation or Modal job started.
- Next: A/B review and agree the proposed DAC-8 contract and linked config;
  complete DAC-9 evidence before the bounded DDPM smoke, and DAC-15 before sweep.

### 2026-10-29 — Implement the partial-DDPM trainer and resumable Modal lifecycle
- Issue/PR: DAC-8, DAC-11, DAC-14; no PR opened yet.
- Changes: Added the normal-only Gaussian DDPM epsilon-prediction trainer, a
  review-pending method/config draft, one-based partial reconstruction, EMA
  checkpoint selection from fixed-noise normal validation loss, and private
  validation reconstruction artifacts. Modal now records durable run stages,
  failure class, checksums and epoch checkpoints; DDPM resume requires matching
  source, config, code, trainer and resolved dataset revision. Updated the team
  guide and agent instructions.
- Evidence/checks: Python compilation, Ruff check/format, and `git diff --check`
  passed. No unit test suite or DDPM training/GPU run was started. The method
  contract is still awaiting A/B review; DAC-9 data-access evidence is also a
  prerequisite for the bounded smoke.
- Next: Get A/B review on DAC-8, finish the teammate data gate, then run the
  documented bounded smoke and record its actual manifest.

### 2026-10-29 — Document team Modal workflow
- Issue/PR: No issue or PR linked; branch feat/anoddpm-notebook-training.
- Changes: Added teammate onboarding for the shared Modal Workspace and
  Hugging Face Secret, terminal/notebook smoke workflow guidance, and Modal run
  rules for AI coding agents.
- Evidence/checks: Cross-checked the documented commands and behavior against
  the repository notebook and current Modal CLI, Secrets, and Workspace docs.
  `git diff --check` passed. Documentation-only change; no training run or tests
  were started.
- Next: Share the guide with teammates and use a fresh run ID for each requested
  Modal attempt.

### 2026-10-29 — Add a local Jupyter controller for Modal training
- Issue/PR: No issue or PR linked; branch feat/anoddpm-notebook-training.
- Changes: Added a local Jupyter notebook that submits the existing Modal CLI
  job, streams logs to a cell, saves its result JSON locally, downloads only
  aggregate metrics and manifest from a pinned private Hub revision, and renders
  a local plot. Added notebook dependencies and README/Hugging Face + Modal
  documentation links. The Modal local entrypoint now returns its JSON result
  for the CLI write-result option.
- Evidence/checks: A bounded one-epoch Modal AnoDDPM diagnostic smoke completed
  128 optimizer steps on a Tesla T4 using 128 normal train samples and 16
  validation samples. Artifact revision
  a95b40397aae6f11a7ee863aefbb2613f64a1f47; the local notebook rendered the
  real manifest, metrics tables, and plot from that revision. The initial
  notebook launch trained and uploaded successfully but its post-run parser
  failed on interleaved progress output; a CPU-only Modal probe verified the
  write-result channel, and the saved notebook was rendered from the completed
  smoke artifacts without another GPU run. Notebook validation and code-cell
  execution succeeded; git diff --check passed. No unit test suite was run.
- Next: Use this notebook only for bounded pipeline checks until the agreed
  partial-DDPM trainer and score protocol are implemented.

### 2026-10-29 — Fix the Hugging Face card revision blocker
- Issue/PR: W1 context [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score); branch `feat/anoddpm-notebook-training`.
- Changes: Updated active dataset pins in the handoff, Modal instructions, project tracking, and Colab notebook to the merged Hub card revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`. Updated the notebook revision assertion, documented the card/revision lesson, and added the verified PowerShell `PYTHONPATH` fallback for Modal's local source mount. No dataset payload was re-uploaded.
- Evidence/checks: Compared the README at the new revision with the generated privacy-safe card; the bytes match. The metadata bytes are unchanged from the prior revision. Modal fetched all 95 files and `validate_hf_export` passed. A bounded diagnostic AnoDDPM smoke completed 128 steps on 128 normal train samples and validated 16 samples; artifacts and metrics are in `RESULT_TRACKING.md`. The first CLI attempt stopped before remote execution because `dacn` was not importable; the retry with `PYTHONPATH=src` succeeded. No unit test suite was run.
- Next: Role C should finish the approved partial-DDPM method contract; this exploratory AnoDDPM smoke does not close DAC-8.

### 2026-10-29 — Review and run the bounded AnoDDPM Modal smoke
- Issue/PR: W1 context [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score); branch `feat/anoddpm-notebook-training` is not linked to an issue/PR.
- Changes: Aligned the Modal Secret name with the existing workspace Secret,
  capped the documented smoke at 128 train and 16 validation samples, clarified
  that Modal jobs use the terminal CLI after hosted Modal Notebooks shut down,
  and recorded the failed run in `RESULT_TRACKING.md`.
- Evidence/checks: Full suite passed (57 tests); Ruff check and format check
  passed; Modal fetched all 95 files from the pinned dataset, then failed in
  `validate_hf_export` because the README differed from the generated
  privacy-safe card. The trainer was not called; no metrics or artifacts were
  produced. The two earlier attempts stopped during image setup before a
  training function ran.
- Next: Reconcile the pinned Hub card with the generated card/validator, then
  rerun this bounded smoke before treating the branch as merge-ready.
### 2026-10-29 — Restore pinned dataset-card compatibility
- Issue/PR: Not provided.
- Changes: Preserved exact validation for the current generated dataset card and
  added the original reviewed schema-v1 card as the only legacy alternative, so
  the immutable pinned dataset remains usable without accepting arbitrary README
  edits. Added positive and tamper-rejection regression tests, documented the
  failed Modal smoke run, and recorded the reusable schema-versioning lesson.
- Evidence/checks: Full Ruff check and format check passed; the complete suite
  passed with 52 tests and 1 skip; `git diff --check` passed. No replacement
  training run was started.
- Next: Retry the bounded A100 smoke run with the same pinned dataset revision.

### 2026-10-29 — Add a Google Colab AnoDDPM training path
- Issue/PR: Not provided.
- Changes: Added a self-contained Colab notebook that clones the feature branch,
  reads `HF_TOKEN` from Colab Secrets, downloads and validates the pinned private
  dataset, runs a bounded smoke test or full pilot, saves to Drive, and renders
  aggregate metric tables and curves. Added deterministic sample limits and live
  trainer progress without changing the full-data defaults.
- Evidence/checks: Ruff check and format check, Python compilation, notebook JSON
  and code-cell validation, and the full test suite passed. No model training or
  measured result was produced; the earlier Modal attempt stopped before GPU
  scheduling because its workspace required a payment method.
- Next: Run `smoke` in a T4 Colab runtime, preserve its artifacts, and only then
  switch the notebook to `pilot` with the full train and validation splits.

### 2026-10-28 — Complete BMAD source and dataset card draft
- Issue/PR: [DAC-7](https://linear.app/dacn-khmt/issue/DAC-7/d01-chot-nguon-djieu-kien-su-dung-va-dataset-card-bmad); [GitHub PR #17](https://github.com/tanh1c/DACN/pull/17); [Hugging Face card PR #1](https://huggingface.co/datasets/dacn-bmad-team/bmad-flair-patient-disjoint-v1/discussions/1).
- Changes: Expanded the generated Hugging Face dataset card with intended use, sample meaning, split policy, preprocessing, limitations, source terms, and required attribution. Documented that BMAD states CC BY-NC-SA without a version and the BraTS2021 page requests three citations but does not specify re-hosting rights; kept license metadata unset and the repo private.
- Evidence/checks: Checked the BMAD repository and official BraTS2021 data/citation page; `git diff --check` passed. Opened and verified HF card PR #1 against the generated README; it changes documentation only. No tests or training run performed.
- Next: Obtain teammate review and merge both PRs; keep DAC-7 open until the accepted card is on the private HF repo.

### 2026-10-28 — Align the Modal Hugging Face secret name
- Issue/PR: Not provided.
- Changes: Updated the Modal runner and setup documentation to use the existing
  `huggingface-secret` secret while retaining `HF_TOKEN` as the environment key.
- Evidence/checks: Ruff check and format check, Python compilation, and the full
  test suite passed. No Modal job or model training run was started.
- Next: Authenticate the local Modal CLI and run the one-epoch AnoDDPM smoke test.

### 2026-10-28 — Add an AnoDDPM training prototype for BMAD
- Issue/PR: Not provided.
- Changes: Ported the attributed AnoDDPM U-Net, diffusion process, and simplex
  noise core; added a validated BMAD shard loader, normal-only full-split
  training, validation-only anomaly scoring and threshold selection, EMA
  checkpoints, a Modal-compatible trainer, and a pilot configuration. Kept this
  prototype explicitly separate from the planned ANDi method.
- Evidence/checks: Ruff check and format check, Python compilation, and the full
  55-test suite passed in an isolated Python 3.11 environment with PyTorch 2.15
  and Numba 0.68; six tests specifically exercised the AnoDDPM integration.
  No medical-data training, GPU job, or measured model result was produced.
- Next: Configure authorized Hugging Face access in Modal, run a one-epoch GPU
  smoke test, and record that actual run separately. ANDi remains pending.

### 2026-10-26 — Verify private Hugging Face dataset and handoff
- Issue/PR: [TRA-95](https://linear.app/traffic-agent-assistant/issue/TRA-95),
  [PR #14](https://github.com/tanh1c/DACN/pull/14).
- Changes: Synced current main and reused the reviewed HF/Modal NPZ exporter.
  Confirmed the team's private dataset already exists; retained its UUID mapping
  and revision `fbb884dfe144be5c71e205a73701507163557ea4`. Added loading instructions
  and safe aggregate provenance in `DATASET_HANDOFF.md`.
- Evidence/checks: Downloaded all 95 remote files, verified their Git/LFS checksums
  and the complete export validator; all 11,248 image/mask/label records matched
  prepared data exactly. Group distributions424/50/767 matched. Current-main suite:
  49 tests passed; Ruff check and format check passed (40 files). No new Hub upload
  or training run was performed; authentication was approved using browser OAuth.
- Follow-up checks: The first PR CI attempt flagged Python-fence formatting in
  the new Markdown guide. Corrected the snippet and reran Ruff check/format
  successfully (41 files); the data verification evidence is unchanged.
- Next: B/C verify the pinned dataset on their own machine and record acceptance
  in TRA-74 before the GPU pilot. Keep held-out test sealed for model selection.

### 2026-10-26 — Review fixes and merge Hugging Face/Modal pipeline
- Issue/PR: [PR #12](https://github.com/tanh1c/DACN/pull/12).
- Changes: Reject duplicate or unreferenced shard offsets; include a hashed
  snapshot of DACN Python sources in each Modal run; align HF examples with the
  group organization. Squash-merged as `c7a6f5f`.
- Evidence/checks: GitHub Actions CI passed: Ruff check/format, 49 pytest cases,
  and the CLI demo. An earlier CI attempt found formatting issues, which were
  corrected before the passing run. No new Modal training run was started.
- Next: Implement the ANDi diffusion trainer and track its actual runs/results.

### 2026-10-26 — Add project requirements and experiment tracking
- Issue/PR: [PR #13](https://github.com/tanh1c/DACN/pull/13).
- Changes: Recorded the Diffusion Models for Anomaly Detection brief; added
  worklog, lessons, and run-result tracking; added AI-agent instructions; linked
  the records from the existing project docs.
- Evidence/checks: `git diff --check` passed. GitHub Actions CI passed Ruff check,
  Ruff format check, `pytest -q`, and the CLI demo. No model training run was
  started.
- Next: Implement and evaluate the diffusion method, then record each actual run
  and its artifacts in `RESULT_TRACKING.md`.

### 2026-10-28 — Initialize the primary DACN Linear workspace
- Project: [DACN — Diffusion Models for Anomaly Detection](https://linear.app/dacn-khmt/project/dacn-diffusion-models-for-anomaly-detection-39ad392101f4).
- Changes: Created the project under DACN-KHMT, six dated milestones, seven workstream labels, 25 issues with acceptance criteria and 31 blocked-by links. Assigned the project lead and data/coordination tasks; left B/C implementation issues unassigned pending identity confirmation. Rebased unfinished first-week dates to 29–30 October while keeping the 4 December target. Synced repository tracking and dataset handoff links to the new DAC issue IDs.
- Evidence/checks: Read-only Linear queries verified the project, target date, six milestones, 25 issues, statuses, and milestone assignments. Current main history contains the merged PR #11 commit used to close F01. No tests, training, or evaluation runs were performed.
- Next: Confirm B/C identities in DAC-6, verify the pinned HF dataset on teammate machines in DAC-9, then run the bounded GPU pilot in DAC-10 and record actual provenance.

### 2026-10-28 — Switch research direction to DDPM reconstruction
- Project: [DACN Linear](https://linear.app/dacn-khmt/project/dacn-diffusion-models-for-anomaly-detection-39ad392101f4); scoring issue [DAC-30](https://linear.app/dacn-khmt/issue/DAC-30/s01-pixel-difference-va-local-ssim-anomaly-maps).
- Changes: Recorded the team decision to use DDPM partial reconstruction and compare starting timesteps with pixel/SSIM scores; moved ANDi and feature difference to optional work. Updated requirements application, plan, protocol, agent guidance, descriptive config and Linear roadmap/issues. Retained dataset split and validation/test controls.
- Evidence/checks: Read project requirements, existing data/evaluation protocol and implementation; queried the updated Linear project and 26 issues. Checked issue dependency graph: 34 links, no cycle or due-date inversion. Git diff whitespace check passed; CSV IDs/dependencies and descriptive YAML parsed successfully. No tests, training or evaluation runs were performed.
- Next: Confirm B/C owners, review DDPM/score and data contracts, prepare run lifecycle recording, then run a bounded pilot and record actual artifacts.

### 2026-10-28 — Confirm team roles and align issue ownership
- Issue/PR: [DAC-6 — Confirm B/C roles](https://linear.app/dacn-khmt/issue/DAC-6/c01-them-thanh-vien-va-chot-vai-tro-bc-tren-linear); [PR #18](https://github.com/tanh1c/DACN/pull/18).
- Changes: Recorded A = Chu Nguyen Tuan Anh (project/data lead), B = Lê Hoàng Chí Vĩ (evaluation), and C = Hai Tuan (diffusion). Confirmed B and C are Linear project members; aligned owner assignments and collaboration notes across the backlog. Updated `TEAM.md`, `PROJECT_PLAN.md`, `PROJECT_TRACKING.md`, and the Linear CSV snapshot.
- Evidence/checks: Linear user search matched both display names; the live project member list included both, and a fresh issue query confirmed the new owners. `git diff --check` passed; the CSV parsed as 26 issue rows. No tests, training, or evaluation runs were performed.
- Next: Review the role/ownership PR, verify private HF dataset access on B/C machines, and agree the weekly sync cadence; DAC-6 and DAC-9 remain In Progress until their criteria are met.

## 2026-11-05 — B integration branch

Created codex/bc-evaluation-colab from C7134b10 in a managed worktree. Integrated B evaluator, diagnostics, UUID artifact export and dataset-binding guards. Resolved runner conflict preserving C sampler/batch/cache options alongside B artifact_dir. Added reviewed aggregate reproduction evidence and retrospective run record; old worktree preserved. Targeted CPU tests:37 passed. Torch runner runtime/GPU remains unverified. No new inference or training. Legacy adapter and new Colab notebook remain pending.

## 2026-11-05 — B spatial diagnostics and grouped comparison

- Converted the checksum-pinned Gaussian30 and Simplex20 legacy caches using
  scripts/convert_legacy_validation_cache.py. Both 83-row UUID artifacts have the
  same SHA-256 as the previously reviewed conversions. Private files stay under runs/.
- Ran both baseline and FG-v1 diagnostics. Both baseline score dictionaries exactly
  reproduce the recorded metrics. Saved aggregate spatial and grouped evidence.
- Added an explicit Simplex20 preset to the grouped evaluator. Its original
  Simplex10 default remains available. Calibration selection is persisted before
  holdout scoring.
- Prepared notebooks/ddpm_validation_colab.ipynb for Gaussian20 at the registered
  checkpoint. Checked Python cell syntax locally. Colab execution remains pending.
- Remaining B work: paired patient-group bootstrap and the Gaussian20 Colab run
  followed by the budget-matched comparison.

## 2026-11-05 — B review: notebook lifecycle and verified cache snapshots

- Reviewed branch base `2df19bc`; details and remaining research limits in
  [B_REVIEW_2026-11-05.md](B_REVIEW_2026-11-05.md).
- Reproduced then fixed notebook smoke/full opt-in, missing failure manifests
  before checkpoint load, and manual-interrupt status. Full inference now has
  a separate default-off gate after successful smoke. Documented cooperative
  timeout and ephemeral Colab storage limits.
- Fusion now deserializes the exact hashed cache/index bytes. Regression tests
  replace producer files after hash verification for both Simplex presets.
  Added synthetic legacy-conversion validation tests, guarded optional Torch
  imports for data-only CI, repaired two invalid UTF-8 separators and lint-only
  notebook/U-Net issues. No model objective, weight or architecture change.
- Verification: full CPU suite with Torch **164 passed (29.84 s)**; minimal
  `[data,dev]` environment **113 passed, 10 skipped (16.27 s)**. Minimal suite
  first encountered a Windows sandbox PermissionError on atomic manifest
  replacement; full rerun outside sandbox passed. Ruff check/format passed;
  notebook cells compiled without saved outputs. Synthetic CLI demo passed.
- Checked 48 saved Precision/Recall/F1 values against aggregate confusion
  counts and confirmed candidate selection uses calibration AP. No new BMAD
  experiment, GPU job, held-out-test access or Linear update.
- Restored local Git metadata from remote while preserving source and backup
  of the missing-parent worktree pointer. Keep `BLK-20261101-02` Investigating;
  next is actual Gaussian20 Colab inference, matched-budget comparison and
  patient-group uncertainty. Tests do not constitute GPU/model acceptance.

## 2026-11-05 — B paired patient-group uncertainty and Colab preparation

- Added exact tied-score weighted AP and paired patient-group bootstrap with fixed
  thresholds/shared draws; tests compare against explicit group replication and
  check missing-class rejection. Added reproducible private-input CLI driver.
- Actual CPU bootstrap: first attempt rejected noncanonical CSV UUID strings;
  second verified canonical UUID/order/cache content and completed1000 draws,
  999valid/1skipped,28.394s. Added aggregate evidence and RESULT_TRACKING record.
  Conditional fusion-minus-Gaussian-CDF AP CI [.005152,.025822]; previously
  inspected validation/different training budgets prohibit a confirmatory claim.
- Colab T4 connected; source archive7a0f024 installed after Git clone access
  failure. User explicitly authorized source/checkpoint/validation transfer.
  Notebook waits for user-entered HF read credential; supplied token creation
  instructions. No actual Gaussian20 inference/training yet.
- Checks: full CPU suite166passed/28.38s; bootstrap CLI help runs. Colab HF
  authentication later returned401 before data/model download; handoff recorded
  as BLK-20261105-02. No credential copied into chat/repository.

## 2026-11-06 — Finish B Gaussian20 handoff, matched evaluation and C review

- Gaussian20 inference finished83/83 on05Nov,1049.814s. User-downloaded132MB
  checkpoint and16MBprivate bundle moved into ignored project folders; verified
  SHA before accepting local results. Colab runtime disconnected after handoff.
- Completed CPU same-budget Gaussian20/Simplex20 evaluation, frozen calibration
  selection/thresholds, UUID-group alignment and1000paired group replicates
  (999valid). Pixel AP .099777/.087416; deltaCI includes0. Full83 Simplex metric
  regression matches prior evidence. Recorded manifest, spatial and matched
  aggregate JSON; source/private arrays and credentials remain uncommitted.
- Reviewed C HF revision73552266:202manifest-artifact hash comparisons match;
  evaluated feature A/B and condition controls from producer JSON. Added review
  distinguishing implementation success from small/unconfirmed quality gains.
  No independent feature-model reconstruction performed.
- Resolved BLK-20261105-02 with actual authentication/run evidence. Keep
  BLK-20261101-02 Investigating. No new model training, test-set tuning or Linear
  status update. Added reproducible CPU analysis driver; checks recorded below.
- Final checks06Nov:166tests passed43.65s;Ruff check/format and git whitespace
  check passed.16cloud/local Gaussian20 aggregate metric comparisons within1e-10;
  calibration winner verified; new evidence scanned for HF credentials/private
  UUIDs. Checkpoint/bundle ignored by Git. CPU driver help works. No new run was
  started merely to repeat an already completed experiment.

## 2026-11-10 — Four-arm final launcher and method review

- User authorized review/freeze, validation, full test and report artifacts.
  Preserved scientific config/checkpoint/thresholds and all four registered arms;
  documented CI local-evidence exception, leaving account blocker Waiting.
- Added durable shared-trajectory four-arm orchestration, dataset/runtime gates,
  reference-validation acceptance before final seal, remote Modal controller and
  artifact-driven report/figure builder. Kept old candidate/provenance intact.
- Checks actually run:282pytest passed/1LPIPS roundtrip skipped41.29s; focused
  gates/adapter/resume25passed/1skipped9.60s; Ruff check/format and whitespace checks.
  Prepared CPU Modal environment; no GPU/model/test result claimed in this entry.
- Checkout moved under ignored runs/bc-evaluation-colab to permit isolated edits;
  original working branch and its uncommitted changes remain intact.
- Added primary LPIPS reference after checking author metadata/CVPR2018 status
  on arXiv1801.03924v2; CVF HTML fetch returned403, arXiv primary page succeeded.
- CPU prepare-a stopped before inference: config LF bytes differ from the pinned
  Windows CRLF artifact. Verified CRLF reproduces exact historical checksum;
  added strict text-byte recovery regression (no config-value change). Disabled
  Modal automatic package mounting so runtime import comes from the explicit
  source root. Prepared fresh protocol-b rather than overwriting attempt-a.
- Smoke2 and fresh-model validation43 completed on T4; four-arm outputs reproduce
  historical final maps within2.98e-7 and all required metrics within2.26e-8.
  Collected aggregate receipts and frozen manifest/control JSON before test;
  test remains unopened until the final stage's gates verify these exact inputs.
- Final frozen-input commit0d3a1aa saved before full-test submission. New full
  local suite283passed/1skipped38.45s; Ruff/format/whitespace checks passed.
  Submitted final4-test-20261110-b on T4 for all3665test images; awaiting produced
  artifacts, no test metrics claimed. Historical08Nov candidate remains intact.

## 2026-11-10 — Resume final test after configured timeout

- User supplied cancellation logs. Verified original call terminal state is
  FunctionTimeoutError at14400s; Volume has2631durable samples, no final metrics.
- Recomputed source snapshot matches frozen checksum exactly. Preserved safe
  timeout evidence; resume same run/protocol/attempt ID for remaining1034images,
  checking/skipping existing shards. No source/scientific config changes or new
  experiments; continuous tracking remains stopped per user instruction.

## 2026-11-10 — Correct premature cancellation of test continuation

- One-off diagnosis found no new logs, idle GPU and PENDING call graph; agent
  canceled resume callfc-01M727Q69TTQD3TV04DB8MZ7AN. User then supplied LPIPS
  initialization logs, showing that the hang inference was unsupported.
- Acknowledged the premature cancellation. Confirmed old container absent before
  resubmitting same run/source/config/thresholds. No model change or test tuning;
  stored shards remain reusable. Silent old-shard verification is now treated as
  expected work rather than sufficient reason to cancel. No continuous tracker.

## 2026-11-10 — Verify final test and complete report artifacts

- User requested a one-off status check. All3665test outputs completed and
  uploaded to private HF revisionddc715a7…. Verified aggregate hash against
  terminal manifest; recorded4arm image/pixel metrics, seed/threshold provenance,
  timing scope and preserved interruption history. No new inference or training.
- Created final results document and aggregate/manifest evidence. Rebuilt report
  with actual test tables, ROC/PR, training curve and pipeline; added private
  validation high/low-Dice examples. Formatted private DOCX(A4/Times New Roman),
  reopened and verified4tables/8images plus every rounded test pixel metric.
- Discussed brightness's higher pixel AP/Dice and high image-label prevalence;
  do not claim diffusion beats all baselines or that high image F1 proves quality.
- No new software tests needed for record/formatting changes; prior frozen-source
  suite283pass/1skip and remote validation acceptance retained. School template
  not supplied; generic Word formatting is not claimed to be the school's format.

## 2026-11-11 — Publish frozen launcher and final-test evidence

- User requested push to GitHub. Prepared the isolated evaluation branch for
  origin/codex/bc-evaluation-colab, including frozen launcher/source commit and
  final3665-image four-arm aggregate metrics, manifest and interruption history.
- Verified test aggregate SHA against producer manifest; scanned all five new
  10Nov evidence JSON files for forbidden identifier/credential keys, HF-token
  patterns and UUIDs; scan passed. Git whitespace check passed. No new model run
  or software tests for this documentation/evidence-only commit; earlier source
  checks and measured run provenance remain recorded above.
- Private MRI illustrations, per-sample shards/checkpoints and private DOCX/ZIP
  report artifacts remain ignored and are not included in the Git commit.
- Remote advanced by8commits to107c231 during local evaluation/report work.
  Merged those English-report/CI updates, retaining source and final-test records;
  resolved reference duplication and kept the canonical manuscript pointer with
  a separate current final-test-results link. No force push or main-branch edit.
- Post-merge checks actually run:291passed/1LPIPSroundtrip skipped28.02s;
  Ruff check passed,223files format check passed, git whitespace check passed.
  Existing automatic English PDF remains a validation-study draft; final test
  metrics are in the linked final-test record and the private report package.

## 2026-11-11 — Complete Local SSIM on cached DDIM20 validation

- User requested the missing Local SSIM validation score. Added CPU-only evaluator
  and Modal entrypoint; reused checksum-pinned83initial DDIM20 pairs, exact40/43
  registered patient-group membership and original signed-input11x11Gaussian SSIM.
  Saved calibration-only thresholds before holdout evaluation; no test/GPU/model calls.
- Actual run completed/uploaded to private HF revision263e6e79…;43holdout pixel
  AP0.095772,F1/Dice0.133945,AUROC0.880550. Recorded full image/pixel P/R/F1,
  macro Dice/FPR, thresholds, one seed, version/source/input provenance and timing.
  Comparison retains measured historical L1/InitialLPIPS rows with verified SHA.
- Checks actually run:16focused tests passed2.80s,6paired-evaluation tests passed
  1.56s; Ruff check/format and whitespace check passed. New regressions verify
  threshold independence from holdout and rejection before reading tampered caches.
- Initial pytest invocation named a missing tests/test_scores.py and ran no tests;
  corrected paths and completed the successful checks above. No results inferred
  from synthetic tests. New maps stay on private Volume/Hub; only aggregates local.

## 2026-11-11 — Four fixed score fusions on DDIM20 validation

- User requested LPIPS+Residual,LPIPS+SSIM,Residual+SSIM,allthree. Registered
  normal-calibration-only midrankCDF and fixed equal-weight arithmetic means
  before execution; no weight sweep, products, new model inference or test access.
- Added fusion evaluator and --score-family fusion to existing CPU Modal lifecycle.
  Reused pinned83DDIM20 pairs and40/43partition; saved private CDF references and
  all pixel/image calibration thresholds before holdout metrics. Raw3baselines
  replayed as integrity controls,36numeric fields match earlier artifacts at1e-12.
- Actual run completed/uploaded to HF78e025c2…; all four fusion metrics recorded.
  None exceeded raw InitialLPIPS pixel AP/Dice on this inspected one-seed cohort;
  no retrospective normalizer/weight changes. All required image/pixel AUROC,
  Precision,Recall,F1 plus AP,macroDice,FPR/thresholds retained in safe evidence.
- Checks run:12focused tests passed2.68s, testing CDF ties/holdout independence,
  fixed weights and all thresholds-before-holdout; Ruff check passed,230files
  format check passed, whitespace check passed. Private maps and fitted pixel
  distributions remain remote; no new GPU, training, diffusion or LPIPS calls.

## 2026-11-11 — Prepare DDIM50+LPIPS test extension

- User explicitly requested full test for DDIM50+LPIPS. Fixed one50call eta0
  chain fromt300, originalGaussianEMA30/base17 policy; raw spatialLPIPS on full240
  crop/top576 image score. Separate experiment from20+30mask/stitch pipeline.
- Added strict artifact/runtime gates, bounded smoke,83validation calibration/
  freeze before test, exact global aggregate metrics and durable reconstruction
  cache. Checkpoint every25outputs,8hour ceiling and explicit cached-shard
  verification logs address prior slow commits/timeout/silent-resume experience.
- Checks run:16focused tests passed4.33s (50U-Net/1LPIPS calls, deterministic
  reconstruction, calibration-before-holdout, no dataset before gate, checksum/
  resume rejection); Ruff/format and whitespace checks passed.233files formatted.
- CPU prepare first failed CLI module import before function body/model/test;
  stopped failed app and corrected to -m full module name, using fresh protocol-b.
  Prepare-b verified all9pins/source/runtime. T4 smoke2 completed/uploaded,
  verified100U-Net/2LPIPS calls and aggregate checksum;83validation now running.
  No final-test results claimed before produced artifacts; no continuous tracker.
- DDIM50 validation83 completed/uploaded;43holdout pixel AP.186020/Dice.290416,
  full required image/pixel metrics and provenance recorded. Calibration-only
  pixel/image thresholds.23461146652698517/.22034396567485398 sealed before test.
  Verified producer aggregate SHA and acceptance SHA; saved frozen input JSON
  and safe evidence. Test remains a separate user-requested exploratory stage.
- Saved source/freeze/validation evidence in local commit147d1d4 before test.
  Submitted full3665image DDIM50+LPIPS test to detached T4 appap-r68EAg…,
  callfc-01M74QECD514X0PA6EN1WX0BNA. No final metrics claimed; tracker remains off.

## 2026-11-11 — Final DDIM50 receipt and LPIPS comparison

- User requested a LPIPS summary. One-off log/call check confirmed3665DDIM50 test
  outputs and completed/uploaded terminal state, HFcf288576…. Verified aggregate
  SHA against producer manifest; collected only safe aggregate/terminal records.
- Generated artifact-derived LPIPS comparison for20initial,50single-phase,
  naive-stitch and20+30final, both validation43/test3665 and pixel/image levels.
  DDIM50 test pixel AP.257950/Dice.309722; full ROC/P/R/F1/AP, timings and one-seed
  limitations recorded without changing frozen thresholds or selecting new methods.
- No new model runs/tests for this record-only task. Aggregate JSON/privacy and
  whitespace checks performed; private maps/reconstructions remain remote. Did
  not re-enable continuous tracking or silently replace the previous final test.

## 2026-11-11 — Matched ConvAE/ConvVAE z256 training

Implemented shared CNN encoder/decoder, AE and stochastic VAE heads, crop240MSE plus explicit per-pixel KL normalization, deterministic normal-validation selection, 30epoch configs and bounded smoke configs. Added full optimizer/RNG/sampler checkpoint resume to shared Modal lifecycle; no test input. Added protocol and authorized scope record. Checks: focused model/trainer/cloud tests12passed (including bit-identical CPU resume both models), Ruff check/format from implementation pass, git diff --check. Actual GPU smoke completed for both models; full30epoch jobs submitted and durable start manifests collected. Run IDs, checkpoint provenance and measured smoke losses recorded in RESULT_TRACKING and safe aggregate evidence. Full training still pending; no anomaly-performance claims.

## 2026-11-11 — Verify ConvAE/ConvVAE completion

Collected latest producer manifests from shared Modal Volume: both30/30epochs completed and uploaded, with best/last/epoch030 checkpoints and private immutable Hub revisions. Recorded measured normal-validation losses, timing, VRAM and safe terminal evidence in RESULT_TRACKING. Check: manifest status/epoch/upload/test-used assertions passed; git diff --check. No training or evaluation launched in this check; no additional tests run (records only).

## 2026-11-11 — Validation evaluation for trained ConvAE/ConvVAE

Added validation-only Modal launcher for immutable AEepoch13/VAEepoch15 best checkpoints, checksum-verified cached83inputs and registered40/43membership. Deterministic mean-latent inference, shared Residual/L1 and LocalSSIM scorers; reuse calibration-only threshold persistence and image/pixel metric evaluator. Saved remote private reconstructions/maps for audit, downloaded only safe aggregate records. Documented normal-validation checkpoint-selection overlap and resulting lack of independent model holdout. Checks:15model/partition/DDPM-validation tests and9paired-scoring/calibration tests passed; Ruff check/format and git diff --check passed. Two initial pytest invocations referenced nonexistent filenames and ran no tests; corrected to actual test modules. Actual2image/model GPU smoke completed/uploaded; full83/model evaluation submitted, provenance in RESULT_TRACKING. No test access or new model training.

Validation full run completed/uploaded for both models,83images each. Collected required image/pixel metrics, wrote both calibration and43validation tables with checkpoint-selection-overlap caveat. Verified aggregate SHA and identical persisted calibration/holdout thresholds. Saved safe producer evidence and private immutable artifact links. No further GPU runs needed.

## 2026-11-11 — ConvVAE LPIPS validation extension

Added validation-only raw spatial LPIPS launcher reusing SHA-verified deterministic VAEepoch15 cached reconstructions. Verified fixed LPIPS state, runtime, source snapshot, original validation index/40–43partition; no VAE inference or training needed. Documented scope: LPIPS alone, no residual fusion, no test; normal-validation checkpoint selection overlap limits independence. Checks:14paired-score/partition/calibration tests and1LPIPS seven-layer/input-contract test passed, Ruff check/format and git diff --check passed. Actual2image smoke completed/uploaded; full83validation job submitted with provenance in RESULT_TRACKING.

ConvVAE LPIPS full83evaluation completed/uploaded. Saved exact producer manifests/aggregates and calibration/43validation pixel/image metric tables with prior Residual/SSIM comparison. Verified producer aggregate SHA and identical fitted thresholds before/after internal holdout. LPIPS underperforms prior scores in pixel AP/F1 on this validation; no threshold/score tuning or test inference performed.

## 2026-11-11 — Package completed experiments for GitHub

User requested pushing all work from this evaluation branch. Prepared ConvAE/ConvVAE architecture/training/resume/configs/tests, completed training and validation aggregate evidence including VAE LPIPS, and completed DDIM50/LPIPS comparison records; preserve original source revisions in run manifests. Validation: Ruff check and format check passed (244files),41focused tests passed, git diff --check passed. Reviewed all26changed/new files for disallowed binary artifacts, credential patterns and private identifier JSON fields; none detected. Dataset/checkpoints/maps remain on private Modal/Hugging Face, outside Git. Fetch of origin/codex/bc-evaluation-colab completed before push. Separate original-checkout edits remain untouched.

## 2026-11-13 — Sync Hugging Face producer runs and consolidate Final LPIPS ablation

Context: advisor email asked to (1) analyse Final LPIPS and the role of the
residual mask, (2) study noise level and DDIM steps, (3) run ablations and compare
baselines, choosing parameters on validation only. Switched the local checkout to
`codex/bc-evaluation-colab` (`e3e309a`); the previous branch's uncommitted work
is kept in a local stash, not modified.

Audited private Hub `dacn-bmad-team/dacn-runs` (109 commits, last 2026-11-12).
Found 11 runs absent from Git records: SNC-LPIPS, Sobel, PraNet-style fusion and
CGFF (smoke + full83 validation, cached CPU re-scoring) and LPIPS-only pseudo-mask
(smoke, validation, test). Downloaded aggregate JSON only into a temporary
directory outside the repository; no `.npz`, private images or source ZIPs.
Checked every downloaded JSON member against its producer manifest SHA-256 (all
match) and scanned the new evidence for UUID-like identifiers and filesystem
paths (none). Added:

- `docs/evidence/2026-11-13-hf-producer-sync.json`: safe scalar metrics,
  thresholds, status, HF revisions and source-snapshot hashes for the 11 runs.
- RESULT_TRACKING entry for the 11 runs, including the exploratory post-freeze
  LPIPS-only-mask test (pixel AP 0.289999 vs frozen Final 0.318226).
- `docs/ABLATION_SUMMARY_2026-11-13.md`: one table set mapping existing results
  to the advisor's three priorities (mask/re-sampling decomposition, phase-1
  noise sweep t100–800 and DDIM splits, alternative scores, learned baselines)
  and the open measurements.
- `BLK-20261113-01`: all 11 manifests record dirty `e3e309a`, and that commit has
  no source for these methods; the person who ran them must push the code.

Checks: every figure in the new RESULT_TRACKING entry was re-compared by script
against the evidence JSON (nine rounding errors from a 4-decimal printout were
found and corrected); summary tables were cross-checked against the 09 Nov CSV,
LPIPS_COMPARISON and FINAL_TEST_RESULTS. Documentation-only change: no tests,
lint, inference, training or test-split access were run.

Next: patient-group bootstrap intervals for the summary tables and a
validation-only completion of the sweep (phase-2 start, total DDIM budget, t400,
extra sampling seeds); GPU work only when explicitly requested.

## 2026-11-13 — Patient-group bootstrap for the DDIM noise/step sweep

Added `dacn.evaluation.sweep_bootstrap` and the CPU-only Modal launcher
`dacn.cloud.modal_sweep_bootstrap`. They reuse the twelve 06–09 Nov producer
score maps and calibration thresholds, verify producer artifact SHA-256, sample
alignment with the pinned validation shards and the registered 40/43 partition,
and refuse to report unless every producer holdout AP/Dice replays exactly.
Volumes are looked up without creation; the Hub Secret name is configurable
(`DACN_HF_SECRET`) because the user stated the team has no shared Modal
Workspace. The partition file is absent from Workspace `tanh1c`, so it is rebuilt
and accepted only when its canonical bytes reproduce the registered SHA-256.

Ran smoke (20 replicates) and full (1000 replicates, 999 valid) in `tanh1c`,
both completed/uploaded; provenance and intervals in RESULT_TRACKING, evidence
copied byte-identically from the Hub, summary table added to
ABLATION_SUMMARY_2026-11-13. The t300 Final-vs-Initial and Final-vs-brightness
intervals reproduce the independent 06 Nov bootstrap with sign reversed.

Checks: 6 new unit tests plus 2 existing bootstrap tests passed; Ruff check and
format clean on the new files; every interval quoted in RESULT_TRACKING and the
summary table was compared by script with the evidence JSON (one wording error
about same-t comparisons was corrected). No inference, training, threshold
fitting or test access.

Note: AGENTS.md and HUGGINGFACE_MODAL.md still describe a shared team Workspace
and Secret `huggingface-secret-2`; they were not changed and need a team
decision.

## 2026-11-13 — Advisor tracker, Modal Workspace correction, noise-quality analysis

- The user clarified that only the private Hugging Face repos are shared and
  each member uses their own Modal Workspace and free credit. Updated the
  Workspace/Secret guidance in AGENTS.md and HUGGINGFACE_MODAL.md accordingly
  (Volume contents and Secret names can differ per Workspace; record the
  Workspace per run). Historical records were not rewritten.
- Added `docs/ADVISOR_FEEDBACK.md`: the advisor's 13 Nov email (Vietnamese, as
  written) and tracker rows AF-01..AF-11 with status, evidence and next action;
  AGENTS.md now asks agents to read and update it.
- Added `dacn.evaluation.noise_quality` and `--analysis quality` to the sweep
  launcher (verification shared with the bootstrap). Smoke and full runs
  completed/uploaded in Workspace `tanh1c`; RESULT_TRACKING, ABLATION_SUMMARY §2
  and tracker rows AF-05..AF-07 updated. Main finding: under-healing at t100,
  plateau t200–500, normal-tissue breakdown at t800; the pseudo-mask covers
  41–51% of normal brain foreground at t100–500.

Checks: 4 new noise-quality unit tests passed (hand-computed aggregates,
missing-class and misalignment cases); Ruff check/format clean; evidence copied
byte-identically from the Hub and scanned for UUID-like identifiers (none); every
table value and quoted range compared with the evidence JSON by script. The
refactored bootstrap path was not re-run after the shared-verification change.
No inference, training, threshold fitting or test access.

## 2026-11-14 — Pseudo-mask rule ablation (AF-07), validation only

Added `dacn.evaluation.mask_rule_ablation` and the T4 launcher
`dacn.cloud.modal_mask_rule_ablation` (user-authorized GPU run). Hypothesis and
eight arms (percentile 90/95/98/99 × dilation 1/3, control p95/d3) were fixed in
code before any run. Only the mask rule changes; cached DDIM20 t300 initials and
heatmaps, masked DDIM30 from t50, seeds and LPIPS weights are reused and checked.
`_pinned_validation` now also returns calibration positions and in-memory sample
IDs (never written) for per-sample seeds.

Smoke (2 images) and full (83 images, 19920 U-Net calls, 714 s) completed and
uploaded in Workspace `tanh1c`; the control replayed producer finals and LPIPS
maps with max abs error 0.0. Result: tighter masks lower normal-tissue LPIPS and
FPR as hypothesised but also lesion recall; no pixel AP or Dice difference beyond
paired bootstrap uncertainty, control has the highest calibration AP and image
ROC-AUC, no rule adopted. RESULT_TRACKING, ABLATION_SUMMARY §1/§5 and advisor
rows AF-02/AF-07 updated.

Checks: 5 new mask-rule tests (producer-equivalence at p95/d3, monotonic
coverage, replay accept/reject, hand-computed statistics) plus 4 noise-quality
tests passed locally with a numba import stub (numba is not installed in the
local Python); Ruff check/format clean; evidence byte-identical to the Hub and
free of UUID-like identifiers; summary tables generated from the evidence JSON
and quoted claims checked by script. No training or test access.

## 2026-11-14 — Final-reconstruction fusion (AF-03) and t50/t400 noise points (AF-05)

- Added `dacn.evaluation.final_fusion` and `--analysis fusion` (CPU): fixed
  residual weights 0–1 in a midrank-CDF rank scale fitted on normal calibration
  pixels, hypothesis written in the module before running. Smoke and full runs
  completed/uploaded in Workspace `tanh1c`; raw LPIPS replays the producer
  (error 0.0). No weight raises holdout pixel AP; w ≥ 0.5 is lower beyond
  bootstrap uncertainty. Raw Final LPIPS kept.
- Added `dacn.evaluation.noise_extension` and the T4 launcher
  `dacn.cloud.modal_noise_extension`, a producer-equivalent DDIM20+30 runner for
  any phase-1 start. Its t300 control replayed initials, finals, all eight maps
  and the refitted mask threshold exactly (smoke and full). Full run added t50
  and t400 (20/30 and 30/20): t50 lower than t300 beyond uncertainty; t400 highest
  holdout AP but interval touches zero and calibration AP is lower; t300 kept.
- RESULT_TRACKING, ABLATION_SUMMARY §2/§3/§5 (combined t50–t800 curve generated
  from evidence) and advisor rows AF-03/AF-05/AF-06 updated.

Checks: 3 fusion and 3 noise-extension unit tests plus the mask-rule and
noise-quality tests (14 total) passed with the local numba import stub; the stub
guard was fixed so the files can share one pytest session. Ruff clean; evidence
byte-identical to the Hub and UUID-free; tables generated from or checked against
evidence by script. GPU time this round: about 16 minutes on a T4. No training or
test access.

## 2026-11-14 — Validation variants (AF-02/AF-07/AF-11) and frozen test CI job (AF-09/AF-10)

- Generalized the producer-equivalent runner (optional phase-2 start, seed base,
  mask score; defaults unchanged) and added a `variants` suite. Full T4 run
  (`variants-full-20261114T041258Z`, 33 min) replayed the control exactly and
  reproduced t400 and the 12 Nov LPIPS-only mask. Findings: seed SD ≈0.013
  holdout AP; t400 beats t300 on holdout but loses on calibration for all three
  seeds; DDIM budget and phase-2 start show no reliable change; LPIPS-only mask
  lowers Dice beyond uncertainty. Budget variant 40+60 was replaced by 40+50
  before running because phase 2 from t50 allows at most 50 steps.
- After the user chose to recompute the frozen test protocol for uncertainty,
  added `dacn.evaluation.frozen_test_ci` and `dacn.cloud.modal_test_ci`: the
  unchanged `FrozenDDIMAdapter`, frozen thresholds (unit-tested against the
  frozen config files), validation-only smoke (replay error 0.0, no test
  access), resumable test chunks kept on the Volume, and a gate requiring the
  published test metrics before any interval is reported. Full test run started.
- RESULT_TRACKING, ABLATION_SUMMARY §2/§5 and advisor rows AF-02/AF-07/AF-11
  updated; tables generated from evidence by script.

Checks: noise-extension (3) and frozen-test-CI (4) unit tests passed, Ruff
clean, evidence byte-identical to the Hub and UUID-free. No training.

## 2026-11-14 — Frozen four-arm test CIs completed (AF-09/AF-10/AF-11)

- First full attempt of `test-ci-full-20261114T042202Z` was cancelled by Modal
  (`InputCancellation`) after all 3665 chunks were saved. Cause: the host killed
  the local client because it ran low on memory. Recorded as failed. At the user's
  request it was resumed with the same ID and code revision `e1ff7118` using
  `--resume`; the saved chunks were reused and only aggregation ran.
- Gate passed: published test metrics reproduced within 5.2e-7. Patient-group
  CIs (767 groups, 1000 replicates): Final beats Initial and Naive-stitch on
  every metric beyond uncertainty. Brightness keeps higher pixel AP/Dice; Final
  has higher image AUROC/AP beyond uncertainty.
- RESULT_TRACKING, ABLATION_SUMMARY §1/§5 and advisor rows AF-02/AF-09/AF-10/AF-11
  updated. Fixed the brightness image-AUROC rounding in §1 (0.6437 → 0.6436,
  value 0.643650).

Checks: evidence byte-identical to the Hub (results SHA matches the manifest),
LF, no UUIDs or tokens. CI table generated from the evidence by script. No
training and no threshold or method change.

## 2026-11-14 — Brain-mask post-processing and brightness hybrids (validation)

- At the user's request, explored the two cheap directions that need no GPU. Added
  `dacn.evaluation.postprocess_hybrid` and `--analysis hybrid`. Hypotheses, 18 arms
  and the candidate rule were fixed in the docstring before running. Code `2c76db10`
  was committed before the runs.
- Smoke (20 replicates) and full run (`hybrid-full-20261114T103245Z`, 1000 replicates,
  6 min CPU). Both references replay the producer exactly. Post-processing: small
  effect (median5 +0.002 to +0.013 AP on Final). LPIPS × brightness: holdout AP 0.656
  vs 0.312 Final and 0.390 brightness, beyond uncertainty against both. Calibration
  AP 0.496 vs 0.313. Image AUROC kept. Nine arms pass the rule.
- RESULT_TRACKING, ABLATION_SUMMARY §3/§5 and advisor rows AF-03/AF-10 updated;
  table generated from the evidence by script.
- Next, pending user decision: ablate the gated reconstruction (Initial/L1/Naive-stitch
  × brightness), then at most one frozen test run.

Checks: postprocess-hybrid (5), final-fusion and frozen-test-CI unit tests passed
(12 total). Ruff clean. Evidence byte-identical to the Hub, LF, UUID-free. No
training and no test access.

## 2026-11-14 — Brightness-gated reconstruction ablation (validation)

- Added the `gate` analysis (Final/Initial LPIPS and Final/Initial/Naive-stitch L1,
  each × brightness). The resampling-credit rule and a hybrid replay gate were fixed
  in code before running; per-arm evaluation is now shared with `hybrid`.
- `gate-full-20261114T105141Z` (74 s CPU): Final LPIPS × brightness 0.656 vs Initial
  LPIPS × brightness 0.530 (Δ [−0.256, −0.014], calibration agrees). L1 gates are
  about 0.18. The rule credits the re-sampling.
- RESULT_TRACKING, ABLATION_SUMMARY §3/§5 and AF-10 updated.

Checks: 16 unit tests passed (postprocess-hybrid, final-fusion, sweep-bootstrap).
Ruff clean. Replays exact. Evidence byte-identical to the Hub, LF, UUID-free. No
test access.

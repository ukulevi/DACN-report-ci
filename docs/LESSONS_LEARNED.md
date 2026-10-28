# Lessons learned

### 2026-10-28 — Record run lifecycle before GPU work

- Observation/evidence: the Modal runner writes `run_manifest.json` after the
  trainer returns; an exception or abrupt stop can occur before that write.
- Practice: establish a durable started record and failure/interruption handling
  before pilots, with a documented manual fallback for abrupt termination.

Record reusable lessons supported by repository evidence or an actual run. Keep
plans and hypotheses out of this log unless they have become a lesson; link the
source or run when possible. Append new entries at the top.

## Entry format

```text
### YYYY-MM-DD — short lesson
- Observation/evidence:
- Lesson:
- Practice adopted:
- Source/run:
```

## Entries

### 2026-10-28 — Port the algorithm core, not dataset-specific epoch semantics
- Observation/evidence: Upstream AnoDDPM loads NFBS/Edinburgh volume layouts
  and limits an epoch to 100 sampled images, while this project uses indexed
  BMAD NPZ shards and requires auditable full-split coverage.
- Lesson: Reusing a research implementation safely means preserving the model
  and diffusion mathematics while adapting its data and training boundaries to
  the target protocol.
- Practice adopted: Use the validated shared shard loader, define an epoch over
  the complete training split, keep test data out of the trainer, and name the
  prototype AnoDDPM rather than conflating it with ANDi.
- Source/run: `src/dacn/data/shards.py`, `src/dacn/training/anoddpm.py`, and the
  AnoDDPM integration tests in `tests/test_anoddpm.py`.

### 2026-10-26 — Pin and verify an existing Hub dataset before uploading again
- Observation/evidence: The organization already contained a private dataset with
  the same archive and canonical manifest hashes. All downloaded arrays matched
  local prepared data. Rebuilding the exporter generates fresh random UUIDs.
- Lesson: A second upload is unnecessary when the pinned existing dataset is
  correct; it would create another revision and change pseudonymous group IDs.
- Practice adopted: Inspect the target first, verify its full snapshot and hashes,
  and preserve its revision for handoff. Use canonical LF manifest hashes when
  comparing exporter metadata across platforms; keep file-byte hashes distinct.
- Source/run: `DATASET_HANDOFF.md`, HF revision
  `fbb884dfe144be5c71e205a73701507163557ea4`.

### 2026-10-26 — Keep plans separate from measured findings
- Observation/evidence: The repository documents an ANDi-FLAIR plan, but the
  README states that the diffusion trainer and measured model-performance
  results are not yet available.
- Lesson: A planned method, hypothesis, demo, or synthetic metric is not evidence
  of model performance.
- Practice adopted: Record real run metrics only from produced artifacts in
  `RESULT_TRACKING.md`; label smoke tests and synthetic outputs explicitly.
- Source/run: `README.md`, `docs/PROJECT_PLAN.md`, and `docs/RESEARCH.md`.

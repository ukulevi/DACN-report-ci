# Lessons learned

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

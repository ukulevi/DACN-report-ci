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

### 2026-10-26 — Keep plans separate from measured findings
- Observation/evidence: The repository documents an ANDi-FLAIR plan, but the
  README states that the diffusion trainer and measured model-performance
  results are not yet available.
- Lesson: A planned method, hypothesis, demo, or synthetic metric is not evidence
  of model performance.
- Practice adopted: Record real run metrics only from produced artifacts in
  `RESULT_TRACKING.md`; label smoke tests and synthetic outputs explicitly.
- Source/run: `README.md`, `docs/PROJECT_PLAN.md`, and `docs/RESEARCH.md`.

# Full English report and private release implementation plan

**Goal:** Replace the short draft with a structured English report in `/report`,
retain evidence-backed claims, and make PDFs directly available in private Releases.

**Architecture:** One canonical Markdown manuscript plus BibTeX, metadata, header
and aggregate-only vector figures in `report/`. Existing checksum-pinned Pandoc
and Tectonic remain the build toolchain; generated LaTeX/PDF stay in ignored
`report/build/`. Successful main builds on the private satellite publish a
clearly labeled validation-draft prerelease, using the build job's PDF/manifest.

**Approval:** User requested a complete `/report` using the supplied workflow as
reference. Existing autonomous execution and dual-push authorization apply.
School/faculty and all three names/IDs were subsequently confirmed from two
user-supplied prior reports. Current course/advisor come from the course record.

## Constraints / review focus

- English report; no invented student IDs, supervisor, experiment, final-test result.
- Candidate and test gates remain unchanged, irrespective of document completeness.
- Source figures contain aggregate measurements or explicitly conceptual pipeline;
  no medical images, identifiers, sample-level seeds or private arrays.
- Preserve all-push/PR triggers; supplied `main*` and path filters would omit
  existing branch and evidence changes. Release only on satellite main push/manual.
- Build permission read-only; release job alone receives contents write.
- Provenance covers manuscript, bibliography, figures, metadata and evidence.
- Legacy docs/report paths direct users to the canonical source, not two edited copies.

## Tasks

- [x] Write failing migration/bibliography/figure-provenance tests.
- [x] Move canonical source/layout into report; expand manuscript and references;
  generate aggregate figures, update builder and legacy pointers.
- [x] Update report workflow artifact paths and add private draft prereleases.
- [x] Compile, render/inspect pages, verify citations/math/figures/metrics; run
  focused tests and repository lint; obtain final read-only review.
- [x] Record worklog, dual-push and verify actual PDF/release run. DAC-26 receives
  the final handoff after these acceptance records are pushed.

## Ledger

- Primary sources checked: DDPM, DDIM, LPIPS, SSIM, BMAD, AutoDDPM, BraTS2021;
  AnoDDPM official author repository provides citation metadata.
- Ruling: use existing functioning compiler rather than copying unavailable
  HCMUT template or installing another TeX distribution. Manuscript Markdown
  compiles to complete LaTeX; no hand-edited duplicate generated source.
- Validation: 7 focused tests pass after observed red-to-green migration,
  unresolved-citation and stale-figure checks. Full Ruff/format pass. Local PDF
  compiled to 16 pages, rendered and visually inspected. No overflow or missing
  glyph warnings; all ten references resolve. Read-only final reviewer found no
  material scientific, code or release issues. No ML experiment or test access.
- Hosted acceptance: first two report runs failed on raw Windows/Linux EOL
  hashes; portable CRLF-to-LF chart provenance fixed after regression RED/GREEN.
  Eight focused tests now pass; raw build-input hashes remain unchanged in scope.
  `1a9496e` report run42760339837 and quality42760339699 succeeded. Private
  prerelease report-v2026.11.10-5 has direct PDF and manifest assets.

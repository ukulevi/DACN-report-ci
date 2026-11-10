# Automatic English report builds

Edit [`docs/REPORT_DRAFT.md`](../docs/REPORT_DRAFT.md) in English. The build uses
Pandoc to convert Markdown to LaTeX and Tectonic to produce an English PDF.
The authored narrative has one source; generated LaTeX is not edited separately.

The report remains a **validation draft**, not a completed submission or a
held-out test result. Building the report does not freeze the protocol, select
models/thresholds, access HF/Modal/Colab or load private arrays/checkpoints.

## What is updated automatically

The report workflow runs on **every push to any branch or tag that contains the
workflow**, on pull-request updates, and on manual dispatch. There are no branch
or changed-path filters. Each build records the source commit, current UTC build
date, input hashes and PDF hash. Tables are regenerated from the registered
aggregate evidence and ablation CSV. New repository commits therefore produce
new build provenance even when the authored narrative has not changed.

Research interpretation is maintained in the English source. CI does not invent
new experiments or silently replace reviewed evidence with another run. When new
results are accepted, update the registered input paths in the builder and commit
the evidence and narrative together.

## Files to edit

- Narrative: `docs/REPORT_DRAFT.md`; level-two headings become top-level PDF sections.
- Cover, team, font and margins: `reports/metadata.json`. Language must be English.
  The default `date: auto` prints the current UTC build date and source commit.
  Add confirmed university, supervisor and student information before submission.
- Header/footer: `reports/header.tex`.
- Generated evidence appendix: accepted aggregate JSON and producer ablation CSV,
  registered in INPUTS in `scripts/build_report.py`.

## Build locally — Windows/Linux x86_64

Python3.11+ and Git are sufficient. No training dependencies are installed.
The official portable releases of Pandoc3.13 and Tectonic0.17.0 are SHA256-pinned
and stored in the ignored `runs/report-tools` directory.

```powershell
python scripts/report_tools.py --output runs/report-tools
python scripts/build_report.py
```

On Windows, `.venv/Scripts/python.exe` can be used instead of `python`.
The first build needs Internet access for the compilers and public TeX bundle;
later builds reuse cached resources but may still check the bundle online.
No system-wide TeX installation or administrator access is required.

To validate and prepare content without compilers or network access:

```powershell
python scripts/build_report.py --prepare-only
```

Generated files in ignored `reports/build/`:

- `report.pdf`: compiled English draft after a successful build.
- `report.md`, `report.tex`: assembled content and generated LaTeX.
- `metadata.json`: metadata with the resolved automatic build date.
- `build-manifest.json`: build/research status, language, source commit, UTC build
  time, dirty-tree flag, input/layout/builder hashes, compiler versions and PDF hash.
- `build.log`, `report.log`: compiler diagnostics.

Check the manifest's completed status before using an output. Before compilation,
the builder removes only its stale report.aux/toc/out/lof/lot/xdv/pdf files, so an
old PDF or incompatible TOC cannot masquerade as a successful new build.
The Markdown source is preserved. Tectonic runs in untrusted mode, without shell escape.

Compiler releases are pinned; TeX resources and PDF metadata can still affect
output bytes between environments/builds. Byte-identical PDFs are not promised.
Version labels in the manifest come from the checksum-pinned release metadata.

## GitHub Actions

Workflow: [`.github/workflows/report.yml`](../.github/workflows/report.yml).

Push any repository update to a branch containing this workflow. Once a build
succeeds, open Actions → **Build draft report PDF** → run → Artifacts and download
`dacn-report-<commit>`. It contains the PDF and provenance manifest, retained for
14 days. Diagnostic logs are retained for 7 days, including failed compilations.

Manual **Run workflow** becomes available once the workflow is present on the
default branch. While it exists only on the working branch, pushes still trigger
it automatically; a new PR is not required merely to build the report.

The workflow uses read-only repository permissions, disables persisted checkout
credentials and requires no HF/Modal tokens. It uploads repository-scoped
artifacts and does not publish Releases or GitHub Pages.

**Existing external blocker:** GitHub Actions for DACN currently reports account-
level startup failures before jobs are created. This workflow cannot resolve
billing/account restrictions. The repository owner must address BLK-20261107-01
before a remote build can be verified. A successful local PDF is not a green CI run.

## References

- [EE5217 workflow](https://github.com/HThanh-how/EE5217/blob/d7041f5da9a03db13c954463a836e757848b7b8e/.github/workflows/build.yml):
  reference for PDF compilation and artifact/log uploads. Course content,
  student-specific templates, releases and Pages deployment were not copied.
- [hw2_introAI](https://github.com/HatakekkSheeshh/hw2_introAI): currently returns
  404 through the connected account; its workflow has not been inspected.
- [Pandoc manual](https://pandoc.org/MANUAL.html),
  [Tectonic](https://github.com/tectonic-typesetting/tectonic),
  [upload-artifact v4](https://github.com/actions/upload-artifact/tree/v4).

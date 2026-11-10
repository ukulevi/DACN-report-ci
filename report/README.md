# Automatic English report builds

Edit [`report/thesis.md`](thesis.md) in English. The build uses
Pandoc to convert Markdown to LaTeX and Tectonic to produce an English PDF.
The custom cover follows the user-provided HCMUT reports; CO4029, CC30, HK261,
advisor and all three student names/IDs were confirmed from the supplied course
record and prior reports. Proper names retain Vietnamese diacritics.

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

- Narrative: `report/thesis.md`; level-two headings become top-level PDF sections.
- Cover, team, font and margins: `report/metadata.json`. Language must be English.
  `date: auto` resolves build metadata; the custom cover prints the current UTC
  month/year. Exact UTC build time and source commit are in the manifest, with
  the source commit also printed in the evidence appendix.
  School, advisor and student details are confirmed in the cover; revise together
  with metadata if the team requests a correction.
- Cover: `report/cover.tex`; confirmed course/team fields are maintained here.
- Header/footer: `report/header.tex`.
- References: `report/references.bib`; Pandoc citeproc resolves manuscript citations.
- Figures: `report/figures/*.pdf`, aggregate-only vectors regenerated with
  `python scripts/report_figures.py` in an environment with reportlab. The logo
  in `report/assets/` was extracted from the user-supplied prior report; no
  research figures, text or experiment results were copied from that project.
  `report/figures/provenance.json` pins the two aggregate input hashes after
  CRLF-to-LF normalization, to survive Windows/Linux text checkout differences.
  The build manifest separately preserves raw-byte hashes for exact provenance. The build
  rejects stale figures after either evidence file changes; regenerate figures
  and commit them with accepted evidence and manuscript updates.
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

Generated files in ignored `report/build/`:

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

### Independent private build host

The private satellite [ukulevi/DACN-report-ci](https://github.com/ukulevi/DACN-report-ci)
builds reports under the user's own Actions account. Its initial source is the
committed DACN snapshot; untracked data/checkpoints are excluded. A successful
satellite report build does not resolve the source owner's billing restriction.

On 10 November the user chose **Git push synchronization**, without a broad
cross-repository token. No upstream polling workflow or secret is enabled.
Push the same committed report branch to both remotes:

```powershell
git push origin codex/bc-evaluation-colab
git push report-ci HEAD:refs/heads/main
```

The local `report-ci` remote points to
`https://github.com/ukulevi/DACN-report-ci.git`. To configure another checkout:

```powershell
git remote add report-ci https://github.com/ukulevi/DACN-report-ci.git
```

The second push automatically builds the PDF on the satellite. A push only to
`tanh1c/DACN` does **not** update the satellite. Fetch/review teammates' upstream
changes before forwarding the report branch; other research branches are not
automatically merged. Never force-push a diverged satellite main.

Fully hosted cross-repository synchronization remains optional future work, not
an active feature. The default Actions token cannot read another private repo;
GitHub also limits fine-grained PATs for other users' personal repositories.
The user declined a broader classic PAT, and none was created or stored. See
[PAT documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

### Build workflow and artifacts

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
artifacts and does not publish GitHub Pages.

On private satellite `main` push/manual builds, a separate job with contents-write
permission publishes `report.pdf` and `build-manifest.json` as a **validation-draft
prerelease**. Open [Releases](https://github.com/ukulevi/DACN-report-ci/releases)
and download the PDF directly under Assets, without unzipping an Actions artifact.
The job checks repository privacy and never releases from pull requests or the
original repository. Prerelease status does not mean scientific protocol freeze.

**Original-repository blocker:** GitHub Actions in `tanh1c/DACN` still reports
account-level startup failures before jobs are created. Its owner must address
BLK-20261107-01 before a hosted build in that original repository can be verified.
Hosted PDF and quality builds in the independent satellite have already succeeded;
that evidence is separate from local PDF builds and the original account blocker.

## References

- [EE5217 workflow](https://github.com/HThanh-how/EE5217/blob/d7041f5da9a03db13c954463a836e757848b7b8e/.github/workflows/build.yml):
  reference for PDF compilation and artifact/log uploads. Course content,
  student-specific templates, releases and Pages deployment were not copied.
- [hw2_introAI](https://github.com/HatakekkSheeshh/hw2_introAI): remote access
  returned 404, but the user supplied its workflow directly. Its build-artifact
  and release-job structure informed the private prerelease stage. Chess content,
  student list and unseen template files were not copied.
- [Pandoc manual](https://pandoc.org/MANUAL.html),
  [Tectonic](https://github.com/tectonic-typesetting/tectonic),
  [upload-artifact v4](https://github.com/actions/upload-artifact/tree/v4).

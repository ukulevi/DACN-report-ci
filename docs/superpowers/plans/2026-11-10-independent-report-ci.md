# Independent private report CI implementation plan

**Goal:** Run English report CI under `ukulevi/DACN-report-ci`, retaining
`tanh1c/DACN` as source of truth.

**Approved scope:** User approved the private satellite, then selected Git push
synchronization and declined a broad cross-repository PAT. Reuse existing pinned
Pandoc/Tectonic workflow. No new credential, polling workflow, dataset transfer,
training or freeze changes. No force-push or automatic research-branch merge.

## Tasks

- [x] Create private satellite and push committed source `4accc9f`.
- [x] Verify hosted PDF run `42756144394`: Success, 1m17s, PDF/manifest artifacts.
- [x] Document push-based synchronization and add local `report-ci` remote.
- [x] Rerun report tests/lint; review documentation. All four report tests pass;
  full Ruff/format checks pass. Review's minor wording correction applied.
- [ ] Push updated records to both repositories; verify latest run and update DAC-26.

## Execution ledger

- Bootstrap uses committed source only. No tracked pth/pt/npz/npy/nii/dcm/zip
  artifacts in HEAD, and no local untracked runs/data are included.
- Ruling: user declined the broader credential, so discard the uncommitted
  polling prototype and its tests. No polling capability is claimed deployed.
- Report builds trigger on each satellite push. Upstream-only commits still
  require an authorized Git push to the satellite. Billing remains the source
  owner's responsibility, while report work can proceed independently.

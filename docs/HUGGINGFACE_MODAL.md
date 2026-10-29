# Hugging Face dataset storage and Modal runs

## Data path

```text
prepared BMAD directory
  -> privacy-minimized NPZ shards
  -> private Hugging Face dataset repo
  -> pinned dataset revision on a Modal Volume
  -> training callable receives train + validation only
  -> checkpoint, aggregate metrics, and run manifest
  -> private Hugging Face model repo and Modal run Volume
```

The Hub export is generated from `manifest.csv` and `qc_report.json`, but neither
file is uploaded. It contains sharded image/mask/label arrays, aggregate split
counts, preprocessing details, checksums, and an `index.csv` with random sample
UUIDs and random patient-group UUIDs. Group UUIDs are stable within the pinned
Hub export for patient-level bootstrap; the mapping to source IDs is never
uploaded. Source filenames/paths and the QC exclusion ledger are omitted. The
exporter rejects an export directory containing any unrecognized file before
uploading it.

The BMAD repository states CC BY-NC-SA but does not specify a version. The
BraTS2021 data page allows research use with three required citations, but does
not specify a license for re-hosting derived files. See [DATA_PROTOCOL](DATA_PROTOCOL.md)
and [REFERENCES](REFERENCES.md). Keep the repo **private** and confirm applicable
source terms before granting access beyond the project or enabling `--public`;
private storage does not remove those terms.

## Publish the prepared dataset

Install the optional integrations, copy `.env.example` to `.env`, and put a
Hugging Face user access token whose account can write to the organization's
dataset and model repos in `HF_TOKEN`. The publish command loads `.env` locally.
`.env` is gitignored; never commit it, include it in a dataset export, or pass
the token as a CLI argument.

```powershell
python -m pip install -e ".[hub,cloud]"

python -m dacn.cli data publish-hf `
  --input data/processed/bmad-flair-patient-disjoint-v1 `
  --output runs/hf-export/bmad-flair-patient-disjoint-v1 `
  --repo-id dacn-bmad-team/bmad-flair-patient-disjoint-v1 `
  --source-archive-sha256 eb7a5aae9ea1752b16d3a76b47595ffd956c0b210444da76853455d4d7be61c8
```

The command reuses an existing local export only when its source manifest hash
and export settings match, and refuses to overwrite it otherwise. It also refuses
to silently change an existing repository's visibility. It prints the uploaded
Hub commit SHA. Pin that SHA for Modal runs instead of using moving `main`.

## Modal credentials and storage

The project uses a shared Modal Workspace, which owns the Volumes and the
`huggingface` Secret used by jobs. Each teammate needs to be invited to that
Workspace and authenticate their own Modal CLI. In the dashboard, confirm the
active Workspace is the team one before running commands; a personal Workspace
will not have the team's Secret or cached Volumes. See Modal's [Workspace guide](https://modal.com/docs/guide/workspaces).

```powershell
python -m pip install -e ".[cloud,hub,notebook]"
modal setup
modal secret list
```

`modal secret list` should show a Secret named `huggingface`; it does not reveal
the token value. The Workspace owner provisions this shared Secret once from a
private local `.env` file containing `HF_TOKEN`:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env locally and fill HF_TOKEN; never paste the token in chat or commit it.
modal secret create huggingface --from-dotenv .env
```

Run the create command only if the Secret is absent. Do not overwrite the shared
Secret while onboarding; a Workspace owner should coordinate any rotation. The
token in this Secret must be able to read the private dataset repo and write to
the organization's model repo. Modal's current CLI supports creating a Secret
from dotenv and listing Secret names; see [Modal Secrets](https://modal.com/docs/guide/secrets)
and the [`modal secret` CLI](https://modal.com/docs/cli/latest/secret).

For notebook artifact downloads, each member also needs a local Hugging Face
token with read access. The notebook reads it from the ignored root `.env` or
from `hf auth login`. This local token is not sent to Modal; the remote job uses
the shared Modal Secret. A terminal run that only uploads artifacts does not
need a local token once the shared Secret is configured.

The app uses a T4 GPU, a six-hour job timeout, a persistent dataset Volume, and
a persistent run Volume. Modal injects the Hub token from the `huggingface`
Secret. The token's account must have write access to the organization's model
repo for run uploads. Dataset shards are cached by repo and immutable commit
SHA. The run Volume is committed before the output folder is uploaded to the
private model repo `dacn-bmad-team/dacn-runs` at `runs/<run-id>`.

## DAE pilot on Modal

The implemented baseline is `dacn.training.dae:train`. It accepts the prepared
train and validation shard directories, selects only normal training samples,
and never receives a test path. Its default is a 3-epoch pilot with batch size
16, width 32, 16×16 Gaussian noise (`std=0.2`), Adam (`lr=1e-4`), and checkpoint
selection by validation pixel AP. Increase epochs only after estimating runtime
and Modal credit use. The DAE converts stored images from `[-1, 1]` back to
`[0, 1]` before applying the noise, matching the scale used by its reference.
Scores cover the original 240×240 region; padding is cropped. The pixel threshold
is selected on validation, and test data is not used.

The callable contract is:

```python
def train(*, train_dir, validation_dir, output_dir, config, device) -> dict: ...
```

It must write its checkpoint and aggregate metrics under `output_dir`, return a
JSON-serializable aggregate result, and never read or select on the held-out
test split. Modal validates and caches the full dataset, but passes only the
train and validation directories to the trainer. `run_manifest.json` records
the pinned dataset commit, code revision/dirty state, GPU, config, timestamps,
and artifact checksums. Each run also stores `source_snapshot.zip` with the exact
DACN Python package sources used by Modal and its SHA256, so dirty working-tree
runs retain the code needed to reproduce the implementation.

Example 3-epoch pilot command:

```powershell
modal run --detach src/dacn/cloud/modal_train.py `
  --dataset-repo-id dacn-bmad-team/bmad-flair-patient-disjoint-v1 `
  --dataset-revision DATASET_COMMIT_SHA `
  --model-repo-id dacn-bmad-team/dacn-runs `
  --run-id dae-seed-17 `
  --trainer dacn.training.dae:train `
  --config-json '{"seed":17,"epochs":3,"batch_size":16,"width":32,"noise_resolution":16,"noise_std":0.2}'
```

`--detach` keeps the Modal function running if the local terminal disconnects.
The command still streams logs while connected and prints the Hub run URL when
the function completes.

The T4 and six-hour timeout are hard ceilings, not a guarantee that a run fits a
free-credit balance. This is a BMAD FLAIR adaptation of the published DAE
baseline, not an exact reproduction of its four-modality experiment. The DDPM
partial-reconstruction trainer and scorer remain unimplemented.

Modal shut down its hosted notebook editor on October 23, 2026. The project also
provides a local Jupyter controller at
[anoddpm_modal_local.ipynb](../notebooks/anoddpm_modal_local.ipynb): it invokes
the same Modal CLI entrypoint, streams logs into the cell, saves the launch
result locally, and downloads aggregate artifacts for inline tables and plots.
Training still runs on Modal; the notebook kernel is only the local controller
and viewer. The Google Colab notebook is a separate workflow that trains in
Colab. Modal documents external Jupyter integrations at [Jupyter notebooks](https://modal.com/docs/guide/jupyter-notebooks)
and the CLI at [modal run](https://modal.com/docs/cli/latest/run).

## AnoDDPM implementation smoke test

`dacn.training.anoddpm:train` is a separate AnoDDPM-style BMAD prototype, not the
planned ANDi implementation. It reuses the same pinned private dataset and Modal
boundary, consumes every normal training sample in an epoch, selects checkpoints
only by validation pixel AP, and never receives a held-out test path. The vendored
simplex core requires Numba, which is included in the Modal image.

Run the Modal CLI from the repository root in a Python environment that can
import this checkout. If Modal is installed globally and `dacn` is not installed
in that same environment, set the source path first in PowerShell:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
```

Start with a one-epoch cost/shape smoke test before using the descriptive defaults
in `configs/anoddpm_bmad.yaml`:

```powershell
$runId = "anoddpm-smoke-" + [guid]::NewGuid().ToString("N")
modal run --detach src/dacn/cloud/modal_train.py `
  --dataset-repo-id dacn-bmad-team/bmad-flair-patient-disjoint-v1 `
  --dataset-revision 350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7 `
  --model-repo-id dacn-bmad-team/dacn-runs `
  --run-id $runId `
  --trainer dacn.training.anoddpm:train `
  --config-json '{"seed":17,"epochs":1,"batch_size":1,"base_channels":32,"maximum_training_timestep":100,"validation_reconstruction_steps":25,"max_train_samples":128,"max_validation_samples":16}'
```

Use a new `--run-id` for every attempt. This bounded GPU smoke checks the training
and artifact path; its validation metrics are not research results. Record each
run in `RESULT_TRACKING.md` after it starts, including failures and interruptions.
The current trainer saves a best checkpoint but does not yet resume an interrupted
run.

## Run the local Jupyter controller

From the repository root, install the optional notebook dependencies in the
same Python environment that has the Modal CLI, then launch Jupyter:

```powershell
python -m pip install -e ".[cloud,hub,notebook]"
jupyter lab notebooks/anoddpm_modal_local.ipynb
```

Run the notebook cells in order. The launch cell creates a fresh run ID and
submits the documented one-epoch AnoDDPM smoke job to Modal; starting that cell
uses GPU credit. The notebook streams the Modal logs, then downloads the pinned
run manifest, training history, and validation metrics from the private model
repo. It writes those aggregate files and a plot under
`runs/modal_notebook/<run-id>/`, which is ignored by Git. A local HF token with
read permission is required for this download. No MRI images or held-out test
data are downloaded by this controller.

This notebook is a local controller/viewer, not a Modal-hosted notebook and not
a local training run. Its AnoDDPM job is only a pipeline diagnostic; its
validation metrics are not a research result and do not implement the agreed
primary partial-DDPM method. For an interactive run that should survive a
terminal disconnect, use the documented `modal run --detach` option; Modal's
current [CLI reference](https://modal.com/docs/cli/latest/run) documents both
`--detach` and `--write-result` used by the notebook.

## Platform references

- [Hugging Face Hub: upload files and folders](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [Modal: Volumes](https://modal.com/docs/guide/volumes)
- [Modal: Secrets](https://modal.com/docs/guide/secrets)
- [Modal: Apps, Functions, and entrypoints](https://modal.com/docs/guide/apps)
- [Kascenas et al., Denoising Autoencoders for Unsupervised Anomaly Detection in Brain MRI](https://proceedings.mlr.press/v172/kascenas22a.html)
- [Kascenas et al., DenoisingAE reference implementation](https://github.com/AntanasKascenas/DenoisingAE)

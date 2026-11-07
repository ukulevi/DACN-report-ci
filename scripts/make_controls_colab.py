"""Build a private, self-contained smoke-first Colab notebook from a source ZIP."""

import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    args = parser.parse_args()
    payload = args.source.read_bytes()
    bootstrap = f"""
import base64, hashlib, zipfile, io, os, sys, subprocess
from pathlib import Path
SOURCE = base64.b64decode({base64.b64encode(payload).decode()!r})
PARTITION_BYTES = base64.b64decode({base64.b64encode(args.partition.read_bytes()).decode()!r})
assert hashlib.sha256(SOURCE).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
ROOT = Path('/content/dacn_controls_source')
ROOT.mkdir(exist_ok=False)
zipfile.ZipFile(io.BytesIO(SOURCE)).extractall(ROOT)
os.chdir(ROOT)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[train,hub,perceptual]'])
sys.path.insert(0, str(ROOT/'src'))
import torch
assert torch.cuda.is_available(), 'Choose GPU runtime before continuing'
print('Source installed; GPU:', torch.cuda.get_device_name(0))
"""
    acquire = """
import json, getpass, shutil, hashlib, uuid, time
from datetime import datetime, timezone
from huggingface_hub import hf_hub_download
TOKEN = getpass.getpass('HF read token (never paste into code): ')
MODEL_REV = 'fae9b50a4b78114fb51d87e803c77d019668f523'
DATA_REV = '350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7'
MODEL = 'dacn-bmad-team/dacn-runs'
DATA = 'dacn-bmad-team/bmad-flair-patient-disjoint-v1'
ENV = ROOT / 'private_environment'; ENV.mkdir(exist_ok=False)
(ENV/'registered_partition.json').write_bytes(PARTITION_BYTES)
def fetch(repo, filename, rev, destination, kind='model'):
    path = hf_hub_download(repo, filename, revision=rev, repo_type=kind, token=TOKEN)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
candidate = json.loads((ROOT/'configs/final_evaluation_candidate.json').read_text())
fetch(MODEL, 'runs/ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_030.pt', MODEL_REV, ENV/'checkpoint.pt')
assert sha(ENV/'checkpoint.pt') == candidate['artifacts']['checkpoint']
VROOT = ROOT/'private_data'
fetch(DATA, 'dataset_metadata.json', DATA_REV, VROOT/'dataset_metadata.json', 'dataset')
assert sha(VROOT/'dataset_metadata.json') == '9c60fac0ff2a2d6274e82db466e5e338303d0dc24986b3bab4b32b6e710af218'
metadata = json.loads((VROOT/'dataset_metadata.json').read_text())
for item in [metadata['splits']['val']['index'], *metadata['splits']['val']['shards']]:
    assert item['path'].startswith('data/val/')
    destination = VROOT/item['path']
    fetch(DATA, item['path'], DATA_REV, destination, 'dataset')
    assert sha(destination) == item['sha256']
prefix = 'evaluations/autoddpm-ddim2030-full83-t4-20261106T074200Z/'
fetch(MODEL, prefix+'comparison/mask_parameters.json', MODEL_REV, ENV/'mask_parameters.json')
assert sha(ENV/'mask_parameters.json') == candidate['artifacts']['mask_parameters']
del TOKEN
subprocess.check_call([sys.executable, 'scripts/export_lpips_environment.py', '--output', str(ENV/'lpips')])
print('Checkpoint and validation only verified; Colab environment exported.')
"""
    setup = """
import numpy as np
import dacn
assert Path(dacn.__file__).resolve() == ROOT/'src/dacn/__init__.py'
from dacn.data.shards import ShardDataset
from dacn.evaluation.frozen_inputs import capture_frozen_inputs
from dacn.evaluation.runtime_binding import verify_runtime_inputs
from dacn.evaluation.frozen_ddim import FrozenDDIMAdapter
from dacn.evaluation.validation import evaluate_maps
paths = {'checkpoint':ENV/'checkpoint.pt', 'mask_parameters':ENV/'mask_parameters.json',
         'validation_partition':ENV/'registered_partition.json',
         'config':ROOT/'configs/autoddpm_ddim20_30.json',
         'validation_index':VROOT/'data/val/index.csv'}
for key in ('lpips_backbone','lpips_calibration'):
    paths[key] = ENV/'lpips'/(key+'.pth')
paths['dependency_lock'] = ENV/'lpips/requirements.freeze.txt'
paths['source_snapshot'] = ENV/'source.zip'; paths['source_snapshot'].write_bytes(SOURCE)
manifest = json.loads(json.dumps(candidate))
manifest['artifacts'] = {key:sha(path) for key,path in paths.items()}
captured = capture_frozen_inputs(manifest, paths, validation_only=True)
freeze = subprocess.check_output([sys.executable,'-m','pip','freeze','--exclude-editable'],text=True)
verify_runtime_inputs(captured, ROOT, freeze)
dataset = ShardDataset(VROOT/'data/val','val',include_masks=True)
assert len(dataset)==83
controls = json.loads((ROOT/'configs/final_control_candidates.json').read_text())
RUN = ROOT/'private_runs'/('ddim-controls-colab-'+uuid.uuid4().hex[:12]); RUN.mkdir(parents=True)
manifest.update(status='prepared', scope='validation only; thresholds unchanged; test sealed',
                gpu=torch.cuda.get_device_name(0), started_at=datetime.now(timezone.utc).isoformat(),
                controls_sha256=sha(ROOT/'configs/final_control_candidates.json'))
def save():
    temporary=RUN/'run_manifest.partial'
    temporary.write_text(json.dumps(manifest,indent=2)); temporary.replace(RUN/'run_manifest.json')
save()
def run_stage(indices, stage):
    manifest.update(status='started',stage=stage); save()
    adapter=FrozenDDIMAdapter('cuda',include_controls=True)
    maps={key:[] for key in controls['arms']}; masks=[]; labels=[]; timings=[]
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    try:
        for ordinal,index in enumerate(indices):
            if time.perf_counter()-started>45*60: raise TimeoutError('45min cooperative limit')
            image,mask,label=dataset[index]
            torch.cuda.synchronize(); tick=time.perf_counter()
            adapter({'image':image[0].numpy(),'mask':mask.numpy(),'sample_id':dataset.sample_ids[index]},captured)
            torch.cuda.synchronize(); timings.append(time.perf_counter()-tick)
            for key,values in maps.items(): values.append(adapter.last_score_maps[key])
            masks.append(mask.numpy()[8:-8,8:-8]); labels.append(int(label))
            print(stage,ordinal+1,'/',len(indices))
        metrics={key:evaluate_maps(np.stack(values),np.stack(masks),np.array(labels),split='validation',thresholds=controls['arms'][key]) for key,values in maps.items()}
        report={'status':'completed','stage':stage,'samples':len(indices),'unet_calls':50*len(indices),
                'lpips_calls':3*len(indices),'gpu':torch.cuda.get_device_name(0),
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'adapter_seconds':timings,'metrics':metrics,
                'scope':'validation fixed thresholds, one seed17; no test; full83 mixes calibration/holdout for engineering receipt only'}
        np.savez_compressed(RUN/(stage+'_private_maps.npz'),**{key:np.stack(values) for key,values in maps.items()})
        (RUN/(stage+'_aggregate.json')).write_text(json.dumps(report,indent=2))
        manifest.update(status='completed',samples=len(indices)); save()
        return report
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error_class=type(exc).__name__); save(); raise
SMOKE = run_stage([int(np.flatnonzero(dataset.labels==k)[0]) for k in (0,1)],'smoke')
print('Smoke completed',SMOKE['samples'],'samples; peak CUDA bytes',SMOKE['cuda_peak_allocated_bytes'])
"""
    full = """
RUN_FULL_VALIDATION = False
assert RUN_FULL_VALIDATION, 'Enable only after successful smoke review'
assert SMOKE['status']=='completed'
FULL = run_stage(list(range(83)),'full83')
import csv
from dacn.evaluation.registered_partition import registered_partition
rows = list(csv.DictReader((VROOT/'data/val/index.csv').open()))
groups = np.array([row['patient_group_id'] for row in rows])
cal, hold = registered_partition(PARTITION_BYTES,(VROOT/'data/val/index.csv').read_bytes(),groups)
assert (len(cal),len(hold))==(40,43) and set(groups[cal]).isdisjoint(groups[hold])
masks = np.stack([dataset[int(i)][1].numpy()[8:-8,8:-8] for i in hold])
FULL['engineering_all83_metrics'] = FULL.pop('metrics')
with np.load(RUN/'full83_private_maps.npz',allow_pickle=False) as saved:
    FULL['metrics'] = {key:evaluate_maps(saved[key][hold],masks,dataset.labels[hold],split='validation',thresholds=controls['arms'][key]) for key in controls['arms']}
FULL.update(report_samples=43,calibration_samples=40,holdout_groups=len(set(groups[hold])),scope='Inspected validation group-holdout43; thresholds from original40calibration, one seed17; no test')
(RUN/'full83_aggregate.json').write_text(json.dumps(FULL,indent=2))
print('Full83 inference completed; separate43holdout metrics saved; test still sealed.')
"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# DACN G30 DDIM20+30 controls — private Colab validation\n",
                "Smoke before full83. No training/test. HF read token through getpass only. Keep notebook/maps private. GPU availability depends on account. Full83 engineering metrics mix the existing40/43 partition; do not present as held-out research metrics.\n",
            ],
        }
    ]
    for code in (bootstrap, acquire, setup, full):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": code.strip().splitlines(True),
                "execution_count": None,
                "outputs": [],
            }
        )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()

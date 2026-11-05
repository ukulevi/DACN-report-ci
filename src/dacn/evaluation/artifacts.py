"""CPU-only validation of private, UUID-aligned reconstruction artifacts.

This entry point never loads test arrays or launches a model. Thresholds are
validation candidates, not a claim that the research protocol has been frozen.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import uuid
from pathlib import Path

import numpy as np

from dacn.evaluation.validation import evaluate_reconstructions

HASH_FIELDS = {
    "dataset_revision": 40,
    "model_revision": 40,
    "code_revision": 40,
    "checkpoint_sha256": 64,
    "dataset_metadata_sha256": 64,
}
REPO_FIELDS = {"dataset_repo_id", "model_repo_id"}
INTEGER_FIELDS = {"train_seed", "sampling_seed", "starting_timestep"}


def _provenance(value):
    if not isinstance(value, dict) or set(value) != set(HASH_FIELDS) | REPO_FIELDS | INTEGER_FIELDS:
        raise ValueError("provenance must contain exactly the documented fields")
    for key, length in HASH_FIELDS.items():
        if not isinstance(value[key], str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value[key]):
            raise ValueError(f"{key} must be an immutable lowercase hash")
    for key in REPO_FIELDS:
        if not isinstance(value[key], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", value[key]
        ):
            raise ValueError(f"{key} must be an organization/repository name")
    for key in INTEGER_FIELDS:
        if type(value[key]) is not int or not 0 <= value[key] < 2**32:
            raise ValueError(f"{key} must be a nonnegative 32-bit integer")
    if not 1 <= value["starting_timestep"] <= 1000:
        raise ValueError("starting_timestep must be in [1, 1000]")
    return dict(value)


def _canonical_ids(values):
    try:
        return [str(uuid.UUID(str(value))) for value in values]
    except (ValueError, AttributeError, TypeError):
        raise ValueError("invalid private sample UUID; values omitted") from None


def _validation_data(dataset_dir, expected_metadata_hash):
    root = Path(dataset_dir).resolve()
    metadata_path = root / "dataset_metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    if hashlib.sha256(metadata_bytes).hexdigest() != expected_metadata_hash:
        raise ValueError("dataset metadata checksum mismatch")
    metadata = json.loads(metadata_bytes)
    if metadata.get("schema_version") != 1 or metadata.get("split_policy") != "patient-disjoint-v1":
        raise ValueError("unsupported dataset schema or split policy")
    val = metadata["splits"]["val"]
    if val["index"]["path"] != "data/val/index.csv":
        raise ValueError("only the validation index is accepted")

    def checked_path(relative, checksum):
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "data/val") or path.is_symlink():
            raise ValueError("only local validation payloads are accepted")
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != checksum:
            raise ValueError("validation payload checksum mismatch")
        return contents

    index = checked_path(val["index"]["path"], val["index"]["sha256"])
    with io.StringIO(index.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        fields = ("sample_id", "patient_group_id", "split", "label", "shard", "offset")
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError("unexpected validation index schema")
        rows = list(reader)
    if len(rows) != val["samples"] or not rows:
        raise ValueError("validation index count mismatch")
    ids = _canonical_ids(row["sample_id"] for row in rows)
    if len(set(ids)) != len(ids) or any(row["split"] != "val" for row in rows):
        raise ValueError("validation index must contain unique validation samples")

    shard_metadata = {shard["path"]: shard for shard in val["shards"]}
    if len(shard_metadata) != len(val["shards"]):
        raise ValueError("duplicate validation shard metadata")
    images, masks, labels = [], [], []
    by_id = {}
    for relative, shard in shard_metadata.items():
        if not re.fullmatch(r"data/val/shard-[0-9]+\.npz", relative):
            raise ValueError("invalid validation shard path")
        path = checked_path(relative, shard["sha256"])
        with np.load(io.BytesIO(path), allow_pickle=False) as archive:
            if set(archive.files) != {"images", "masks", "labels"}:
                raise ValueError("unexpected validation shard fields")
            x, y, z = archive["images"], archive["masks"], archive["labels"]
        n = shard["samples"]
        if (
            x.shape != (n, 1, 256, 256)
            or x.dtype != np.float32
            or y.shape != (n, 256, 256)
            or y.dtype != np.uint8
            or z.shape != (n,)
            or z.dtype != np.uint8
        ):
            raise ValueError("invalid validation shard tensor schema")
        if (
            not np.isfinite(x).all()
            or np.any(np.abs(x) > 1)
            or not np.isin(y, [0, 1]).all()
            or not np.isin(z, [0, 1]).all()
            or not np.array_equal(y.any(axis=(1, 2)), z.astype(bool))
        ):
            raise ValueError("invalid validation shard values or labels")
        relevant = [
            (identifier, row) for identifier, row in zip(ids, rows) if row["shard"] == relative
        ]
        offsets = [int(row["offset"]) for _, row in relevant]
        if sorted(offsets) != list(range(n)):
            raise ValueError("validation index offsets must cover each shard exactly once")
        for (identifier, row), offset in zip(relevant, offsets):
            if row["label"] != str(int(z[offset])):
                raise ValueError("validation index/shard labels disagree")
            by_id[identifier] = len(images)
            images.append(x[offset, 0])
            masks.append(y[offset])
            labels.append(z[offset])
    if len(by_id) != len(ids):
        raise ValueError("validation index references missing shards")
    return by_id, np.stack(images), np.stack(masks), np.array(labels)


def load_validation_pairs(artifact: Path, dataset_dir: Path, provenance: dict):
    """Join a complete private artifact to validation ground truth by sample UUID.

    Provenance hashes are declared by the producer; dataset metadata, validation
    shards/index and the actual reconstruction file are verified locally here.
    """
    provenance = _provenance(provenance)
    artifact_bytes = Path(artifact).read_bytes()
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "reconstructions"}:
            raise ValueError("artifact requires sample_ids and reconstructions only")
        ids, recons = archive["sample_ids"], archive["reconstructions"]
    if ids.ndim != 1 or ids.dtype.kind != "U" or not len(ids):
        raise ValueError("sample_ids must be a nonempty Unicode UUID vector")
    ids = _canonical_ids(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate private sample UUIDs; values omitted")
    if (
        recons.shape != (len(ids), 1, 256, 256)
        or recons.dtype != np.float32
        or not np.isfinite(recons).all()
        or np.any(np.abs(recons) > 1)
    ):
        raise ValueError("reconstructions must be float32 (N,1,256,256), finite and in [-1,1]")
    by_id, images, masks, labels = _validation_data(
        dataset_dir, provenance["dataset_metadata_sha256"]
    )
    if set(ids) != set(by_id):
        raise ValueError("artifact must cover exactly the complete validation cohort")
    order = [by_id[identifier] for identifier in ids]
    return images[order], recons[:, 0], masks[order], labels[order], provenance, artifact_hash


def evaluate_validation_artifact(artifact: Path, dataset_dir: Path, provenance: dict) -> dict:
    """Score the complete immutable UUID-aligned validation artifact."""
    images, recons, masks, labels, provenance, artifact_hash = load_validation_pairs(
        artifact, dataset_dir, provenance
    )
    metrics = evaluate_reconstructions(images, recons, masks, labels, split="validation")
    return {
        "schema_version": 1,
        "split": "validation",
        "samples": len(images),
        "normal": int(np.count_nonzero(labels == 0)),
        "anomaly": int(np.count_nonzero(labels == 1)),
        "provenance_declared_by_producer": provenance,
        "reconstruction_artifact_sha256": artifact_hash,
        "thresholds_frozen": False,
        "scope": "complete validation; review required before protocol freeze or test",
        "score_protocol": "DAC-8 candidate: crop8; abs; SSIM11/sigma1.5/range2/reflect/population; "
        "no foreground or postprocessing; image mean top576",
        "metrics": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("artifact", "dataset", "provenance", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("output already exists; choose a fresh result path")
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    result = evaluate_validation_artifact(args.artifact, args.dataset, provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

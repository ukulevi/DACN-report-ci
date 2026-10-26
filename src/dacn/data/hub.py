"""Build privacy-minimized dataset exports and publish them to the Hugging Face Hub."""

import csv
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
DATASET_VERSION = "bmad-flair-patient-disjoint-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_sha256(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        content = stream.read().replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _safe_source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Prepared path escapes dataset root: {relative}")
    return path


def _reject_patient_metadata(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in {
                "patient_id",
                "patient_ids",
                "patient_identifier",
                "patient_identifiers",
                "sample_id",
                "source_path",
                "image_path",
                "mask_path",
            }:
                raise ValueError("Hub metadata must not contain patient IDs or source-level paths")
            _reject_patient_metadata(item)
    elif isinstance(value, list):
        for item in value:
            _reject_patient_metadata(item)


def _read_source(root: Path) -> tuple[list[dict[str, str]], dict]:
    root = root.resolve()
    manifest_path = root / "manifest.csv"
    report_path = root / "qc_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError(
            "Input must be a prepared BMAD directory with manifest.csv and qc_report.json"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("ok") is not True:
        raise ValueError("Refusing to publish a dataset whose QC report is not successful")
    if report.get("split_policy") != "patient-disjoint-v1":
        raise ValueError("Only the reviewed patient-disjoint-v1 split can be exported")
    if not isinstance(report.get("preprocessing"), dict):
        raise TypeError("QC preprocessing metadata must be an object")

    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "label", "prepared_path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest must contain split, label, and prepared_path columns")
        rows = list(reader)

    seen_paths = set()
    patient_splits = {}
    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError(f"Unsupported split in source manifest: {row['split']}")
        if row["label"] not in ("0", "1"):
            raise ValueError("Manifest labels must be binary 0/1")
        if not row["prepared_path"] or row["prepared_path"] in seen_paths:
            raise ValueError("Manifest contains an empty or duplicate prepared_path")
        seen_paths.add(row["prepared_path"])
        patient_id = row.get("patient_id", "")
        if not patient_id:
            raise ValueError(
                "Source manifest must contain patient IDs to create anonymous group IDs"
            )
        previous_split = patient_splits.setdefault(patient_id, row["split"])
        if previous_split != row["split"]:
            raise ValueError("Source manifest contains a patient across multiple splits")

    source_counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    if source_counts != report.get("counts"):
        raise ValueError("Manifest split counts do not match the successful QC report")
    if not all(source_counts.values()):
        raise ValueError("All train, val, and test splits must contain samples")
    return rows, report


def _load_sample(path: Path, label: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"Prepared sample is missing: {path.name}")
    with np.load(path, allow_pickle=False) as sample:
        if set(sample.files) != {"image", "mask"}:
            raise ValueError(f"Unexpected fields in prepared sample: {path.name}")
        image = np.array(sample["image"], copy=True)
        mask = np.array(sample["mask"], copy=True)

    if image.shape != (1, 256, 256) or image.dtype != np.float32:
        raise ValueError(f"Unexpected image shape or dtype in {path.name}")
    if mask.shape != (256, 256) or mask.dtype != np.uint8:
        raise ValueError(f"Unexpected mask shape or dtype in {path.name}")
    if not np.isfinite(image).all() or image.min() < -1 or image.max() > 1:
        raise ValueError(f"Image values are invalid in {path.name}")
    if not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError(f"Mask is not binary in {path.name}")
    if bool(np.any(mask)) != bool(label):
        raise ValueError(f"Mask and label disagree in {path.name}")
    return image, mask


def _dataset_card(metadata: dict) -> str:
    counts = metadata["splits"]
    train, val, test = counts["train"], counts["val"], counts["test"]
    return f"""---
task_categories:
  - image-segmentation
tags:
  - brain-mri
  - anomaly-detection
  - bmad
---

# BMAD Brain FLAIR — {metadata["dataset_version"]}

Privacy-minimized, sharded export of the BMAD Brain FLAIR 2D prepared dataset.
The export contains image arrays, binary masks, split labels, opaque sample UUIDs,
opaque patient-group UUIDs, and aggregate metadata. It omits the source manifest,
original patient IDs, source filenames, and QC exclusion ledger. Group UUIDs allow
patient-level bootstrap without revealing the source ID mapping.

## Splits

| Split | Samples | Normal | Anomaly |
|---|---:|---:|---:|
| Train | {train["samples"]} | {train["normal"]} | {train["anomaly"]} |
| Validation | {val["samples"]} | {val["normal"]} | {val["anomaly"]} |
| Test | {test["samples"]} | {test["normal"]} | {test["anomaly"]} |

## Format

Each `data/<split>/shard-*.npz` contains `images` (float32, N×1×256×256,
normalized to [-1, 1]), `masks` (uint8, N×256×256, binary), and `labels`
(uint8, N; 0=normal, 1=anomaly). Each split's `index.csv` maps an opaque sample
UUID and patient-group UUID to a shard offset and label. UUIDs stay stable within
this immutable Hub export. `dataset_metadata.json` records aggregate counts,
preprocessing, source-manifest checksum, and shard/index checksums.

## Source, terms, and limitations

Source: [BMAD](https://github.com/DorisBao/BMAD), Brain FLAIR 2D. BMAD states
CC BY-NC-SA terms; the underlying BraTS terms may add conditions. Confirm the
right to store or redistribute this derived dataset before changing repository
visibility or sharing it. A private Hub repository does not replace that check.

The images are rendered PNG luminance, not original NIfTI intensity values.
`patient-disjoint-v1` is a derived split for the pinned archive described by the
project protocol, not a claim about BMAD's official split. This dataset is for
research and is not for diagnosis.

Source archive SHA256: {metadata.get("source_archive_sha256") or "not provided"}.
"""


def build_hf_export(
    source: Path,
    output: Path,
    *,
    shard_size: int = 128,
    seed: int = 17,
    dataset_version: str = DATASET_VERSION,
    source_archive_sha256: str | None = None,
) -> dict:
    """Create or safely reuse a matching export without source patient identifiers."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Prepared dataset directory does not exist: {source}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Export directory must be separate from the prepared dataset")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if not dataset_version or any(c in dataset_version for c in "\\/\0"):
        raise ValueError("dataset_version must be a non-empty path-safe string")
    if source_archive_sha256 is not None and not re.fullmatch(
        r"[0-9a-fA-F]{64}", source_archive_sha256
    ):
        raise ValueError("source_archive_sha256 must contain exactly 64 hexadecimal characters")

    rows, report = _read_source(source)
    manifest_sha = _manifest_sha256(source / "manifest.csv")
    if output.exists():
        metadata = validate_hf_export(output)
        matches = (
            metadata.get("source_manifest_sha256") == manifest_sha
            and metadata.get("dataset_version") == dataset_version
            and metadata.get("shard_size") == shard_size
            and metadata.get("shuffle_seed") == seed
            and metadata.get("source_archive_sha256") == source_archive_sha256
        )
        if not matches:
            raise ValueError(
                f"Existing export does not match this dataset/config: {output}. "
                "Choose a new output path."
            )
        return metadata

    rows_by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    index_rows = {split: [] for split in SPLITS}
    patient_group_ids = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))

    try:
        split_metadata = {}
        for split_index, split in enumerate(SPLITS):
            split_rows = rows_by_split[split]
            random.Random(seed + split_index).shuffle(split_rows)
            normal = sum(row["label"] == "0" for row in split_rows)
            anomaly = len(split_rows) - normal
            shards = []

            for shard_index, start in enumerate(range(0, len(split_rows), shard_size)):
                chunk = split_rows[start : start + shard_size]
                relative = Path("data") / split / f"shard-{shard_index:05d}.npz"
                images, masks, labels = [], [], []
                for offset, row in enumerate(chunk):
                    path = _safe_source_path(source, row["prepared_path"])
                    label = int(row["label"])
                    image, mask = _load_sample(path, label)
                    images.append(image)
                    masks.append(mask)
                    labels.append(label)
                    patient_id = row["patient_id"]
                    if patient_id not in patient_group_ids:
                        patient_group_ids[patient_id] = uuid.uuid4().hex
                    index_rows[split].append(
                        {
                            "sample_id": uuid.uuid4().hex,
                            "patient_group_id": patient_group_ids[patient_id],
                            "split": split,
                            "label": label,
                            "shard": relative.as_posix(),
                            "offset": offset,
                        }
                    )

                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    images=np.stack(images),
                    masks=np.stack(masks),
                    labels=np.asarray(labels, dtype=np.uint8),
                )
                shards.append(
                    {
                        "path": relative.as_posix(),
                        "samples": len(chunk),
                        "sha256": _sha256(destination),
                    }
                )

            split_metadata[split] = {
                "samples": len(split_rows),
                "normal": normal,
                "anomaly": anomaly,
                "shards": shards,
            }

        for split in SPLITS:
            relative = Path("data") / split / "index.csv"
            index_path = temporary / relative
            with index_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "sample_id",
                        "patient_group_id",
                        "split",
                        "label",
                        "shard",
                        "offset",
                    ),
                )
                writer.writeheader()
                writer.writerows(index_rows[split])
            split_metadata[split]["index"] = {
                "path": relative.as_posix(),
                "sha256": _sha256(index_path),
            }

        preprocessing = report["preprocessing"]
        metadata = {
            "schema_version": 1,
            "dataset": "BMAD-Brain",
            "dataset_version": dataset_version,
            "source": "https://github.com/DorisBao/BMAD",
            "source_manifest_sha256": manifest_sha,
            "source_archive_sha256": source_archive_sha256,
            "encoding": report["encoding"],
            "split_policy": report["split_policy"],
            "preprocessing": preprocessing,
            "image_shape": [1, 256, 256],
            "image_dtype": "float32",
            "image_range": [-1, 1],
            "mask_shape": [256, 256],
            "mask_dtype": "uint8",
            "label_encoding": {"0": "normal", "1": "anomaly"},
            "shard_size": shard_size,
            "shuffle_seed": seed,
            "contains_patient_identifiers": False,
            "contains_pseudonymous_patient_groups": True,
            "splits": split_metadata,
        }
        (temporary / "dataset_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(_dataset_card(metadata), encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        if temporary.exists() and temporary.parent == output.parent:
            shutil.rmtree(temporary)
        raise

    return metadata


def validate_hf_export(
    path: Path,
    *,
    verify_checksums: bool = True,
    allow_hf_cache: bool = False,
) -> dict:
    """Validate the package one shard at a time, including its ID index and checksums."""
    path = Path(path).resolve()
    metadata_path = path / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Dataset export is missing dataset_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("split_policy") != "patient-disjoint-v1":
        raise ValueError("Unsupported or unreviewed Hub dataset schema/split")
    _reject_patient_metadata(metadata)
    if metadata.get("contains_patient_identifiers") is not False:
        raise ValueError("Hub export metadata does not confirm patient identifier removal")
    if set(metadata.get("splits", {})) != set(SPLITS):
        raise ValueError("Hub export must contain train, val, and test splits")
    if metadata.get("image_shape") != [1, 256, 256] or metadata.get("mask_shape") != [256, 256]:
        raise ValueError("Hub export has an unsupported image or mask shape")
    if metadata.get("image_dtype") != "float32" or metadata.get("mask_dtype") != "uint8":
        raise ValueError("Hub export has an unsupported image or mask dtype")
    source_archive_sha = metadata.get("source_archive_sha256")
    if source_archive_sha is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", source_archive_sha):
        raise ValueError("Hub metadata contains an invalid source archive SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", metadata.get("source_manifest_sha256", "")):
        raise ValueError("Hub metadata contains an invalid source manifest SHA256")

    expected_files = {"README.md", "dataset_metadata.json"}
    sample_ids = set()
    group_splits = {}
    for split in SPLITS:
        split_meta = metadata["splits"][split]
        shard_samples = sum(shard.get("samples", 0) for shard in split_meta["shards"])
        if split_meta.get("samples") != shard_samples:
            raise ValueError(f"Shard sample counts do not match for split={split}")
        shard_labels = {}
        for shard in split_meta["shards"]:
            relative = Path(shard["path"])
            shard_path = (path / relative).resolve()
            if not shard_path.is_relative_to(path) or not shard_path.is_file():
                raise ValueError(f"Invalid or missing Hub shard: {relative}")
            if verify_checksums and _sha256(shard_path) != shard.get("sha256"):
                raise ValueError(f"Checksum mismatch for Hub shard: {relative}")
            expected_files.add(relative.as_posix())
            with np.load(shard_path, allow_pickle=False) as archive:
                if set(archive.files) != {"images", "masks", "labels"}:
                    raise ValueError(f"Unexpected fields in Hub shard: {relative}")
                images, masks, labels = archive["images"], archive["masks"], archive["labels"]
                count = shard["samples"]
                if images.shape != (count, 1, 256, 256) or images.dtype != np.float32:
                    raise ValueError(f"Unexpected image tensor in Hub shard: {relative}")
                if masks.shape != (count, 256, 256) or masks.dtype != np.uint8:
                    raise ValueError(f"Unexpected mask tensor in Hub shard: {relative}")
                if labels.shape != (count,) or labels.dtype != np.uint8:
                    raise ValueError(f"Unexpected labels in Hub shard: {relative}")
                if not np.isfinite(images).all() or images.min() < -1 or images.max() > 1:
                    raise ValueError(f"Invalid image values in Hub shard: {relative}")
                binary_masks = set(np.unique(masks)).issubset({0, 1})
                binary_labels = set(np.unique(labels)).issubset({0, 1})
                if not binary_masks or not binary_labels:
                    raise ValueError(f"Non-binary mask or label in Hub shard: {relative}")
                if not np.array_equal(np.any(masks, axis=(1, 2)), labels.astype(bool)):
                    raise ValueError(f"Mask and label disagree in Hub shard: {relative}")
                shard_labels[relative.as_posix()] = labels.tolist()

        index_meta = split_meta.get("index", {})
        index_relative = Path(index_meta.get("path", ""))
        index_path = (path / index_relative).resolve()
        if not index_path.is_relative_to(path) or not index_path.is_file():
            raise ValueError(f"Invalid or missing anonymous index for split={split}")
        if verify_checksums and _sha256(index_path) != index_meta.get("sha256"):
            raise ValueError(f"Checksum mismatch for anonymous index in split={split}")
        expected_files.add(index_relative.as_posix())

        with index_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = ("sample_id", "patient_group_id", "split", "label", "shard", "offset")
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError(f"Unexpected anonymous index schema for split={split}")
            index_rows = list(reader)
        if len(index_rows) != split_meta["samples"]:
            raise ValueError(f"Anonymous index count does not match for split={split}")

        normal_count = 0
        indexed_offsets = {shard: set() for shard in shard_labels}
        for row in index_rows:
            try:
                sample_uuid = uuid.UUID(hex=row["sample_id"])
                group_uuid = uuid.UUID(hex=row["patient_group_id"])
                offset = int(row["offset"])
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Invalid anonymous index entry for split={split}") from exc
            if sample_uuid.version != 4 or group_uuid.version != 4:
                raise ValueError(f"Index IDs must be random UUID4 values for split={split}")
            if sample_uuid.hex in sample_ids:
                raise ValueError("Anonymous sample IDs must be unique")
            sample_ids.add(sample_uuid.hex)
            if row["split"] != split or row["label"] not in ("0", "1"):
                raise ValueError(f"Index split or label is invalid for split={split}")
            previous_split = group_splits.setdefault(group_uuid.hex, split)
            if previous_split != split:
                raise ValueError("A pseudonymous patient group occurs across multiple splits")
            shard = row["shard"]
            if shard not in shard_labels or not 0 <= offset < len(shard_labels[shard]):
                raise ValueError(f"Index references an invalid shard offset for split={split}")
            if offset in indexed_offsets[shard]:
                raise ValueError(f"Index references a duplicate shard offset for split={split}")
            indexed_offsets[shard].add(offset)
            label = int(row["label"])
            if shard_labels[shard][offset] != label:
                raise ValueError(f"Index label disagrees with shard labels for split={split}")
            normal_count += int(label == 0)
        if any(
            offsets != set(range(len(shard_labels[shard])))
            for shard, offsets in indexed_offsets.items()
        ):
            raise ValueError(f"Index does not reference every shard sample for split={split}")
        if normal_count != split_meta["normal"]:
            raise ValueError(f"Index class counts do not match for split={split}")
        if len(index_rows) - normal_count != split_meta["anomaly"]:
            raise ValueError(f"Index class counts do not match for split={split}")

    actual_files = set()
    for candidate in path.rglob("*"):
        relative = candidate.relative_to(path)
        if allow_hf_cache and relative.parts[:2] == (".cache", "huggingface"):
            continue
        if allow_hf_cache and relative.as_posix() == ".gitattributes":
            continue
        if candidate.is_symlink():
            raise ValueError(f"Hub export contains a symlink: {relative}")
        if candidate.is_file():
            actual_files.add(relative.as_posix())
    if actual_files != expected_files:
        raise ValueError("Hub export contains unexpected files; refusing to upload it")
    if (path / "README.md").read_text(encoding="utf-8") != _dataset_card(metadata):
        raise ValueError("Hub export README differs from the generated privacy-safe dataset card")
    return metadata


def publish_hf_dataset(
    export_path: Path,
    repo_id: str,
    *,
    public: bool = False,
    token: str | None = None,
) -> dict:
    """Upload a validated export. Repositories are private unless explicitly opted out."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            'Hugging Face support is optional; install with pip install -e ".[hub]"'
        ) from exc

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            'Loading HF_TOKEN from .env requires the hub extra; install with pip install -e ".[hub]"'
        ) from exc
    load_dotenv()
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Set HF_TOKEN in the environment; never pass it as a command-line argument"
        )
    metadata = validate_hf_export(export_path)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if bool(info.private) != (not public):
        visibility = "private" if info.private else "public"
        requested = "public" if public else "private"
        raise ValueError(
            f"Hub repo is already {visibility}; requested {requested}. "
            "Change visibility explicitly first."
        )

    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(Path(export_path).resolve()),
        commit_message=f"Publish {metadata['dataset_version']} sanitized data shards",
        token=token,
    )
    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "private": not public,
        "revision": getattr(commit, "oid", None),
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }

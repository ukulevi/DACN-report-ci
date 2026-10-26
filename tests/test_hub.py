import csv
import hashlib
import json

import numpy as np
import pytest

from dacn.data.hub import build_hf_export, validate_hf_export


@pytest.fixture
def prepared_dataset(tmp_path):
    root = tmp_path / "prepared"
    rows = [
        ("train", "0", "train/normal.npz", "patient-secret-001"),
        ("train", "0", "train/normal-2.npz", "patient-secret-001"),
        ("train", "1", "train/anomaly.npz", "patient-secret-001"),
        ("val", "0", "val/normal.npz", "patient-secret-002"),
        ("test", "0", "test/normal.npz", "patient-secret-003"),
    ]
    for split, label, relative, _patient_id in rows:
        mask = np.zeros((256, 256), dtype=np.uint8)
        if label == "1":
            mask[10:12, 10:12] = 1
        image = np.zeros((1, 256, 256), dtype=np.float32)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, image=image, mask=mask)

    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "label", "prepared_path", "patient_id"))
        writer.writerows(rows)
    report = {
        "ok": True,
        "split_policy": "patient-disjoint-v1",
        "counts": {"train": 3, "val": 1, "test": 1},
        "preprocessing": {"image_size": 256},
        "encoding": "grayscale",
    }
    (root / "qc_report.json").write_text(json.dumps(report), encoding="utf-8")
    return root


@pytest.fixture
def hf_export(prepared_dataset, tmp_path):
    output = tmp_path / "hub-export"
    build_hf_export(prepared_dataset, output, shard_size=1)
    return output


def _update_metadata_checksum(export, split, kind, relative_path):
    metadata_path = export / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if kind == "index":
        metadata["splits"][split]["index"]["sha256"] = hashlib.sha256(
            (export / relative_path).read_bytes()
        ).hexdigest()
    else:
        shard = next(
            item for item in metadata["splits"][split]["shards"] if item["path"] == relative_path
        )
        shard["sha256"] = hashlib.sha256((export / relative_path).read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_export_omits_source_patient_ids_and_validates(prepared_dataset, tmp_path):
    output = tmp_path / "hub-export"
    metadata = build_hf_export(prepared_dataset, output, shard_size=1)

    assert validate_hf_export(output) == metadata
    text_files = [path for path in output.rglob("*") if path.is_file()]
    exported_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in text_files
        if path.suffix in {".csv", ".json", ".md"}
    )
    assert "patient-secret-" not in exported_text
    assert not (output / "manifest.csv").exists()
    assert metadata["contains_patient_identifiers"] is False

    with (output / "data/train/index.csv").open(newline="", encoding="utf-8") as stream:
        train_groups = {row["patient_group_id"] for row in csv.DictReader(stream)}
    assert len(train_groups) == 1


def test_validator_rejects_shard_with_wrong_tensor_shape(hf_export):
    shard_path = hf_export / "data/val/shard-00000.npz"
    np.savez_compressed(
        shard_path,
        images=np.zeros((1, 256, 256), dtype=np.float32),
        masks=np.zeros((1, 256, 256), dtype=np.uint8),
        labels=np.zeros((1,), dtype=np.uint8),
    )
    relative = "data/val/shard-00000.npz"
    _update_metadata_checksum(hf_export, "val", "shard", relative)

    with pytest.raises(ValueError, match="Unexpected image tensor"):
        validate_hf_export(hf_export)


def test_validator_rejects_index_offset_outside_shard(hf_export):
    index_path = hf_export / "data/val/index.csv"
    with index_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["offset"] = "4"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _update_metadata_checksum(hf_export, "val", "index", "data/val/index.csv")

    with pytest.raises(ValueError, match="invalid shard offset"):
        validate_hf_export(hf_export)


def test_validator_rejects_duplicate_index_offset(prepared_dataset, tmp_path):
    output = tmp_path / "hub-export"
    build_hf_export(prepared_dataset, output, shard_size=3)

    index_path = output / "data/train/index.csv"
    with index_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    normal_rows = [row for row in rows if row["label"] == "0"]
    normal_rows[1]["offset"] = normal_rows[0]["offset"]
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _update_metadata_checksum(output, "train", "index", "data/train/index.csv")

    with pytest.raises(ValueError, match="duplicate shard offset"):
        validate_hf_export(output)

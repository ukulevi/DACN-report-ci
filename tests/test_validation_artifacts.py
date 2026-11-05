import csv
import hashlib
import json
import uuid

import numpy as np
import pytest

from dacn.evaluation.artifacts import evaluate_validation_artifact, main


@pytest.fixture
def case(tmp_path):
    dataset = tmp_path / "dataset"
    val = dataset / "data/val"
    val.mkdir(parents=True)
    ids = [str(uuid.uuid4()) for _ in range(2)]
    images = np.zeros((2, 1, 256, 256), dtype=np.float32)
    masks = np.zeros((2, 256, 256), dtype=np.uint8)
    masks[1, 80:104, 80:104] = 1
    labels = np.array([0, 1], dtype=np.uint8)
    shard = val / "shard-00000.npz"
    np.savez_compressed(shard, images=images, masks=masks, labels=labels)
    index = val / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "patient_group_id", "split", "label", "shard", "offset"])
        for i in range(2):
            writer.writerow(
                [ids[i], str(uuid.uuid4()), "val", labels[i], "data/val/shard-00000.npz", i]
            )
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "split_policy": "patient-disjoint-v1",
        "splits": {
            "val": {
                "samples": 2,
                "index": {"path": "data/val/index.csv", "sha256": sha(index)},
                "shards": [
                    {"path": "data/val/shard-00000.npz", "sha256": sha(shard), "samples": 2}
                ],
            }
        },
    }
    meta_path = dataset / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    provenance = {
        "dataset_repo_id": "dacn-bmad-team/bmad-flair-patient-disjoint-v1",
        "dataset_revision": "a" * 40,
        "dataset_metadata_sha256": sha(meta_path),
        "model_repo_id": "dacn-bmad-team/dacn-runs",
        "model_revision": "b" * 40,
        "checkpoint_sha256": "c" * 64,
        "code_revision": "d" * 40,
        "train_seed": 17,
        "sampling_seed": 17,
        "starting_timestep": 100,
    }
    reconstructions = masks[:, None].astype(np.float32)
    artifact = tmp_path / "pairs.npz"
    # Reverse artifact order: alignment must be by UUID, never by row position.
    np.savez_compressed(
        artifact, sample_ids=np.array(ids[::-1]), reconstructions=reconstructions[::-1]
    )
    return dataset, artifact, provenance, ids, reconstructions


def test_complete_validation_artifact_joins_by_uuid_and_omits_identifiers(case):
    dataset, artifact, provenance, ids, _ = case
    report = evaluate_validation_artifact(artifact, dataset, provenance)
    assert report["metrics"]["pixel_difference"]["pixel"]["average_precision"] == 1
    assert report["metrics"]["pixel_difference"]["image"]["f1"] == 1
    assert report["samples"] == 2
    assert report["thresholds_frozen"] is False
    assert all(identifier not in json.dumps(report) for identifier in ids)
    assert "patient_group_id" not in json.dumps(report)


@pytest.mark.parametrize("kind", ["partial", "duplicate", "unknown", "cropped", "nan"])
def test_rejects_misaligned_or_invalid_private_artifacts(case, kind):
    dataset, artifact, provenance, ids, recons = case
    if kind == "partial":
        ids, recons = ids[:1], recons[:1]
    elif kind == "duplicate":
        ids = [ids[0], ids[0]]
    elif kind == "unknown":
        ids[0] = str(uuid.uuid4())  # Could be a test ID; never open test to find it.
    elif kind == "cropped":
        recons = recons[:, :, 8:-8, 8:-8]
    elif kind == "nan":
        recons[0, 0, 20, 20] = np.nan
    np.savez_compressed(artifact, sample_ids=np.array(ids), reconstructions=recons)
    with pytest.raises(ValueError):
        evaluate_validation_artifact(artifact, dataset, provenance)


@pytest.mark.parametrize(
    "key,value",
    [
        ("dataset_revision", "main"),
        ("train_seed", True),
        ("starting_timestep", 1001),
        ("token", "must-not-be-accepted"),
        ("dataset_metadata_sha256", "e" * 64),
    ],
)
def test_rejects_moving_or_leaking_provenance(case, key, value):
    dataset, artifact, provenance, _, _ = case
    provenance[key] = value
    with pytest.raises(ValueError):
        evaluate_validation_artifact(artifact, dataset, provenance)


@pytest.mark.parametrize("relative", ["data/val/index.csv", "data/val/shard-00000.npz"])
def test_rejects_corrupt_dataset_before_evaluation(case, relative):
    dataset, artifact, provenance, _, _ = case
    with (dataset / relative).open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_validation_artifact(artifact, dataset, provenance)


def test_cli_preserves_existing_output_before_any_input_read(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(
            [
                "--artifact",
                "missing.npz",
                "--dataset",
                "missing",
                "--provenance",
                "missing.json",
                "--output",
                str(output),
            ]
        )
    assert output.read_text() == "keep"


def test_report_hash_stays_bound_to_scored_artifact_bytes(case, monkeypatch):
    import dacn.evaluation.artifacts as module

    dataset, artifact, provenance, ids, recons = case
    original_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    real_evaluate = module.evaluate_reconstructions

    def replace_after_scoring(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        np.savez_compressed(artifact, sample_ids=np.array(ids), reconstructions=1 - recons)
        return result

    monkeypatch.setattr(module, "evaluate_reconstructions", replace_after_scoring)
    report = module.evaluate_validation_artifact(artifact, dataset, provenance)
    assert report["reconstruction_artifact_sha256"] == original_hash
    assert (
        report["reconstruction_artifact_sha256"]
        != hashlib.sha256(artifact.read_bytes()).hexdigest()
    )


def test_checksum_and_loaded_shard_use_same_bytes(case, monkeypatch):
    import io
    from pathlib import Path

    dataset, artifact, provenance, _, _ = case
    shard = dataset / "data/val/shard-00000.npz"
    with np.load(shard, allow_pickle=False) as original:
        replacement = io.BytesIO()
        np.savez_compressed(
            replacement,
            images=np.ones_like(original["images"]),
            masks=original["masks"],
            labels=original["labels"],
        )
    real_read = Path.read_bytes
    replaced = []

    def replace_after_read(path):
        contents = real_read(path)
        if path == shard and not replaced:
            path.write_bytes(replacement.getvalue())
            replaced.append(True)
        return contents

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    report = evaluate_validation_artifact(artifact, dataset, provenance)
    assert replaced
    assert report["metrics"]["pixel_difference"]["pixel"]["average_precision"] == 1


def test_producer_calls_reconstruction_once_per_uuid_without_ground_truth(case, tmp_path):
    from dacn.evaluation.ddpm import produce_validation_artifact

    dataset, _, provenance, ids, _ = case
    calls = []

    def reconstruct(image, *, sample_id, starting_timestep, sampling_base_seed):
        assert image.shape == (1, 1, 256, 256)
        assert image.dtype == np.float32
        assert sample_id in ids
        assert starting_timestep == 100 and sampling_base_seed == 17
        calls.append(sample_id)
        return image.copy()

    output = tmp_path / "generated.npz"
    receipt = produce_validation_artifact(dataset, output, provenance, reconstruct)
    assert sorted(calls) == sorted(ids)
    assert len(calls) == len(set(calls)) == 2
    assert receipt["samples"] == 2
    assert all(identifier not in json.dumps(receipt) for identifier in ids)
    report = evaluate_validation_artifact(output, dataset, provenance)
    assert set(report["metrics"]) == {"pixel_difference", "local_ssim"}


@pytest.mark.parametrize("bad_result", ["shape", "dtype", "range", "nan"])
def test_producer_rejects_invalid_reconstruction_before_writing(case, tmp_path, bad_result):
    from dacn.evaluation.ddpm import produce_validation_artifact

    dataset, _, provenance, _, _ = case

    def reconstruct(image, **kwargs):
        result = image.copy()
        if bad_result == "shape":
            result = result[0]
        elif bad_result == "dtype":
            result = result.astype(np.float64)
        elif bad_result == "range":
            result.fill(2)
        else:
            result.fill(np.nan)
        return result

    output = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="reconstruction"):
        produce_validation_artifact(dataset, output, provenance, reconstruct)
    assert not output.exists()


def test_producer_refuses_existing_artifact_before_inference(case, tmp_path):
    from dacn.evaluation.ddpm import produce_validation_artifact

    dataset, _, provenance, _, _ = case
    output = tmp_path / "existing.npz"
    output.write_bytes(b"keep")

    def must_not_run(*args, **kwargs):
        pytest.fail("existing artifact must fail before reconstruction")

    with pytest.raises(FileExistsError):
        produce_validation_artifact(dataset, output, provenance, must_not_run)
    assert output.read_bytes() == b"keep"


def test_terminal_bridge_produces_complete_artifact_and_metrics(case, tmp_path, monkeypatch):
    import dacn.evaluation.ddpm as bridge

    dataset, _, provenance, _, _ = case
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    provenance["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(provenance))
    config = tmp_path / "config.json"
    config.write_text("{}")
    seen = []

    def fake_backend(checkpoint_bytes, config, device):
        assert checkpoint_bytes == b"synthetic-checkpoint" and device == "cpu"

        def reconstruct(image, **kwargs):
            seen.append(kwargs["sample_id"])
            return image.copy()

        return reconstruct

    monkeypatch.setattr(bridge, "_ddpm_backend", fake_backend)
    output = tmp_path / "output"
    bridge.main(
        [
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--provenance",
            str(prov),
            "--output-dir",
            str(output),
        ]
    )
    assert len(seen) == 2
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "completed"
    report = json.loads((output / "validation_metrics.json").read_text())
    assert report["samples"] == 2 and report["thresholds_frozen"] is False
    assert "sample_id" not in json.dumps(manifest)


def test_terminal_bridge_rejects_checkpoint_mismatch_before_loading_model(
    case, tmp_path, monkeypatch
):
    import dacn.evaluation.ddpm as bridge

    dataset, _, provenance, _, _ = case
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"wrong-checkpoint")
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(provenance))
    config = tmp_path / "config.json"
    config.write_text("{}")

    def must_not_load(*args, **kwargs):
        pytest.fail("must reject checksum before loading model")

    monkeypatch.setattr(bridge, "_ddpm_backend", must_not_load)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="checkpoint checksum"):
        bridge.main(
            [
                "--dataset",
                str(dataset),
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(config),
                "--provenance",
                str(prov),
                "--output-dir",
                str(output),
            ]
        )
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error_class"] == "ValueError"


def test_terminal_bridge_rejects_artifact_replaced_between_production_and_scoring(
    case, tmp_path, monkeypatch
):
    import dacn.evaluation.ddpm as bridge

    dataset, _, provenance, _, _ = case
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    provenance["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(provenance))
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(bridge, "_ddpm_backend", lambda *args: lambda image, **kwargs: image.copy())
    real_evaluate = bridge.evaluate_validation_artifact

    def replace_before_scoring(artifact, *args):
        with np.load(artifact, allow_pickle=False) as original:
            ids = original["sample_ids"]
            recons = 1 - original["reconstructions"]
        np.savez_compressed(artifact, sample_ids=ids, reconstructions=recons)
        return real_evaluate(artifact, *args)

    monkeypatch.setattr(bridge, "evaluate_validation_artifact", replace_before_scoring)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="artifact changed"):
        bridge.main(
            [
                "--dataset",
                str(dataset),
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(config),
                "--provenance",
                str(prov),
                "--output-dir",
                str(output),
            ]
        )
    assert json.loads((output / "evaluation_manifest.json").read_text())["status"] == "failed"
    assert not (output / "validation_metrics.json").exists()


def test_modal_pair_writer_roundtrips_through_b_consumer(case, tmp_path):
    from dacn.evaluation.handoff import write_pairs

    dataset, original, provenance, ids, pairs = case
    exported = tmp_path / "modal_pairs.npz"
    write_pairs(exported, ids[::-1], pairs[::-1, 0])
    result = evaluate_validation_artifact(exported, dataset, provenance)
    expected = evaluate_validation_artifact(original, dataset, provenance)
    assert result["metrics"] == expected["metrics"]
    assert result["samples"] == 2


def test_foreground_comparison_preserves_baseline_and_omits_private_ids(case):
    from dacn.evaluation.diagnostics import compare

    dataset, artifact, provenance, ids, _ = case
    baseline = evaluate_validation_artifact(artifact, dataset, provenance)
    result = compare(artifact, dataset, provenance)
    for method in ("pixel_difference", "local_ssim"):
        assert result["results"][method]["baseline"]["metrics"] == baseline["metrics"][method]
    assert result["coverage"]["excluded_lesion_pixels"] == 0
    assert result["thresholds_frozen"] is False
    assert all(identifier not in json.dumps(result) for identifier in ids)

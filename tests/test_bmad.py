import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dacn.data.bmad import inspect_bmad


def write_image(path, value=10, mask=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((8, 8), value, dtype=np.uint8)
    if mask:
        pixels[:] = 0
        pixels[3:5, 3:5] = 255
    Image.fromarray(pixels).save(path)


@pytest.fixture
def brain(tmp_path):
    root = tmp_path / "Brain"
    for split, cls, patient, value in [
        ("train", "good", "00001", 10),
        ("valid", "good", "00002", 20),
        ("valid", "Ungood", "00003", 30),
        ("test", "good", "00004", 40),
        ("test", "Ungood", "00005", 50),
    ]:
        filename = f"{patient}_60.png"
        folder = root / split / cls
        write_image((folder if split == "train" else folder / "img") / filename, value)
        if split != "train":
            write_image(folder / "label" / filename, value=0, mask=cls == "Ungood")
    return root


def errors(root):
    return "\n".join(inspect_bmad(root)[1]["errors"])


def test_manifest_counts_and_string_ids(brain):
    rows, report = inspect_bmad(brain)
    assert report["ok"]
    assert report["counts"] == {"train": 1, "val": 2, "test": 2}
    assert len(rows) == 5
    assert rows[0]["patient_id"] == "00001"
    assert rows[0]["slice_index"] == 60
    assert not Path(rows[0]["image_path"]).is_absolute()
    assert [r["lesion_pixels"] for r in rows if r["label"] == 1] == [4, 4]


def test_patient_leakage(brain):
    write_image(brain / "test/good/img/00001_61.png", 60)
    write_image(brain / "test/good/label/00001_61.png", 0)
    assert "patient" in errors(brain).lower()


def test_decoded_duplicate_leakage(brain):
    write_image(brain / "test/good/img/00006_60.png", 10)
    write_image(brain / "test/good/label/00006_60.png", 0)
    assert "duplicate" in errors(brain).lower()


@pytest.mark.parametrize("problem", ["missing", "shape", "empty", "corrupt"])
def test_invalid_labels(brain, problem):
    path = brain / "test/Ungood/label/00005_60.png"
    if problem == "missing":
        path.unlink()
    elif problem == "shape":
        Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(path)
    elif problem == "empty":
        write_image(path, 0)
    else:
        path.write_bytes(b"not an image")
    assert not inspect_bmad(brain)[1]["ok"]


def test_corrupt_image(brain):
    (brain / "train/good/00001_60.png").write_bytes(b"invalid")
    assert "read" in errors(brain).lower()


def test_malformed_id(brain):
    write_image(brain / "train/good/unknown.png", 77)
    assert "filename" in errors(brain).lower()


def test_empty_required_group(brain):
    (brain / "valid/good/img/00002_60.png").unlink()
    assert not inspect_bmad(brain)[1]["ok"]


def test_unexpected_png_location(brain):
    write_image(brain / "train/Ungood/img/00007_60.png", 90)
    assert "unexpected" in errors(brain).lower()


def test_orphan_mask(brain):
    write_image(brain / "test/Ungood/label/00009_60.png", mask=True)
    assert "orphan" in errors(brain).lower()


def test_color_image_rejected(brain):
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    a[:, :, 0] = 100
    Image.fromarray(a).save(brain / "train/good/00001_60.png")
    assert not inspect_bmad(brain)[1]["ok"]


def test_bmad_rgba_luminance_and_multiclass_mask(brain, tmp_path):
    from dacn.data.bmad import prepare_bmad

    image = np.full((8, 8, 4), [56, 55, 77, 255], dtype=np.uint8)
    Image.fromarray(image).save(brain / "train/good/00001_60.png")
    mask = np.full((8, 8, 4), [0, 0, 0, 255], dtype=np.uint8)
    mask[3, 3] = [56, 55, 77, 255]
    mask[3, 4] = [112, 123, 143, 255]
    mask[4, 3:5] = [255, 255, 255, 255]
    Image.fromarray(mask).save(brain / "test/Ungood/label/00005_60.png")
    out = tmp_path / "rgba"
    prepare_bmad(brain, out, 16, encoding="bmad-bone")
    with np.load(out / "train/00001_60.npz") as item:
        expected = np.asarray(Image.fromarray(image).convert("L"))[0, 0]
        assert np.isclose(item["image"][0, 4, 4], expected / 127.5 - 1)
    with np.load(out / "test/00005_60.npz") as item:
        assert item["mask"].sum() == 4  # Alpha must never count as foreground.


def test_bmad_transparency_rejected(brain):
    image = np.full((8, 8, 4), [56, 55, 77, 254], dtype=np.uint8)
    Image.fromarray(image).save(brain / "train/good/00001_60.png")
    assert not inspect_bmad(brain, encoding="bmad-bone")[1]["ok"]


def test_missing_normal_validation_mask_rejected(brain):
    (brain / "valid/good/label/00002_60.png").unlink()
    assert not inspect_bmad(brain)[1]["ok"]


@pytest.mark.parametrize("encoding", ["grayscale", "bmad-bone"])
@pytest.mark.parametrize("problem", ["palette_mask", "transparent_l", "transparent_rgb"])
def test_reject_unsupported_png_encodings(brain, encoding, problem):
    if problem == "palette_mask":
        mask = Image.new("P", (8, 8), color=1)
        mask.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
        mask.putpixel((3, 3), 0)
        mask.save(brain / "test/Ungood/label/00005_60.png")
    elif problem == "transparent_l":
        Image.new("L", (8, 8), color=10).save(brain / "train/good/00001_60.png", transparency=10)
    else:
        Image.new("RGB", (8, 8), color=(10, 10, 10)).save(
            brain / "train/good/00001_60.png", transparency=(10, 10, 10)
        )
    assert not inspect_bmad(brain, encoding=encoding)[1]["ok"]


def test_explicit_split_policy_removes_entire_overlapping_test_patient(brain):
    for index in (60, 68):
        write_image(brain / f"test/Ungood/img/00003_{index}.png", 30 + index - 60)
        write_image(brain / f"test/Ungood/label/00003_{index}.png", mask=True)
    assert not inspect_bmad(brain)[1]["ok"]
    rows, report = inspect_bmad(brain, split_policy="patient-disjoint-v1")
    assert report["ok"], report["errors"]
    assert len(report["excluded"]) == 2
    assert report["original_counts"] == {"train": 1, "val": 2, "test": 4}
    assert report["counts"] == {"train": 1, "val": 2, "test": 2}
    assert all(r["patient_id"] != "00003" for r in rows if r["split"] == "test")
    assert len([r for r in rows if r["patient_id"] == "00003"]) == 1


def test_corrected_policy_still_rejects_train_leakage(brain):
    write_image(brain / "test/good/img/00001_61.png", 60)
    write_image(brain / "test/good/label/00001_61.png", 0)
    assert not inspect_bmad(brain, split_policy="patient-disjoint-v1")[1]["ok"]


def test_prepare_corrected_policy_records_exclusions(brain, tmp_path):
    from dacn.data.bmad import prepare_bmad

    write_image(brain / "test/Ungood/img/00003_60.png", 30)
    write_image(brain / "test/Ungood/label/00003_60.png", mask=True)
    out = tmp_path / "corrected"
    prepare_bmad(brain, out, 16, split_policy="patient-disjoint-v1")
    assert not (out / "test/00003_60.npz").exists()
    report = json.loads((out / "qc_report.json").read_text())
    assert report["split_policy"] == "patient-disjoint-v1"
    assert len(report["excluded"]) == 1


def test_missing_root_reports_failure(tmp_path):
    assert not inspect_bmad(tmp_path / "missing")[1]["ok"]


def test_inspection_is_read_only(brain):
    before = sorted(str(p.relative_to(brain)) for p in brain.rglob("*"))
    inspect_bmad(brain)
    assert before == sorted(str(p.relative_to(brain)) for p in brain.rglob("*"))


def test_prepare_preserves_pixels_and_lesion_size(brain, tmp_path):
    from dacn.data.bmad import prepare_bmad

    out = tmp_path / "prepared"
    report = prepare_bmad(brain, out, image_size=16)
    assert report["ok"]
    with np.load(out / "test/00005_60.npz") as item:
        assert item["image"].shape == (1, 16, 16)
        assert item["image"].dtype == np.float32
        assert item["mask"].dtype == np.uint8
        assert item["mask"].sum() == 4
        assert item["image"][0, 0, 0] == -1
        assert np.isclose(item["image"][0, 6, 6], 50 / 127.5 - 1)
    assert "00005_60" in (out / "manifest.csv").read_text()
    assert json.loads((out / "qc_report.json").read_text())["counts"]["train"] == 1


def test_prepare_is_deterministic(brain, tmp_path):
    from dacn.data.bmad import prepare_bmad

    a, b = tmp_path / "one", tmp_path / "two"
    prepare_bmad(brain, a, 16)
    prepare_bmad(brain, b, 16)
    assert (a / "manifest.csv").read_bytes() == (b / "manifest.csv").read_bytes()
    with np.load(a / "train/00001_60.npz") as x, np.load(b / "train/00001_60.npz") as y:
        np.testing.assert_array_equal(x["image"], y["image"])


def test_prepare_refuses_overwrite(brain, tmp_path):
    from dacn.data.bmad import prepare_bmad

    out = tmp_path / "exists"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="exist"):
        prepare_bmad(brain, out)
    assert sentinel.read_text() == "keep"


def test_prepare_refuses_invalid_data_without_output(brain, tmp_path):
    from dacn.data.bmad import prepare_bmad

    (brain / "test/Ungood/label/00005_60.png").unlink()
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare_bmad(brain, out)
    assert not out.exists()


def test_prepare_refuses_downsampling(brain, tmp_path):
    from dacn.data.bmad import prepare_bmad

    with pytest.raises(ValueError, match="size"):
        prepare_bmad(brain, tmp_path / "small", 4)
    assert not (tmp_path / "small").exists()


def test_prepare_cleans_partial_output_on_io_error(brain, tmp_path, monkeypatch):
    from dacn.data.bmad import prepare_bmad

    def full_disk(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", full_disk)
    with pytest.raises(OSError, match="disk full"):
        prepare_bmad(brain, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.partial-*"))


def test_cli_inspect_success_and_failure(brain, tmp_path):
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dacn.cli",
            "data",
            "inspect",
            "--root",
            str(brain),
            "--output",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())["ok"]
    (brain / "test/Ungood/label/00005_60.png").unlink()
    result = subprocess.run(
        [sys.executable, "-m", "dacn.cli", "data", "inspect", "--root", str(brain)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert not json.loads(result.stdout)["ok"]


def test_cli_prepare_and_original_demo(brain, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dacn.cli",
            "data",
            "prepare",
            "--root",
            str(brain),
            "--output",
            str(tmp_path / "cli"),
            "--image-size",
            "16",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "dacn.cli", "demo", "--output", str(tmp_path / "demo")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    demo_metrics = json.loads((tmp_path / "demo/metrics.json").read_text())
    assert set(demo_metrics["validation"]) == {"pixel_difference", "local_ssim"}
    assert set(demo_metrics["test"]) == {"pixel_difference", "local_ssim"}
    assert demo_metrics["score_direction"] == "higher_is_more_anomalous"
    for method, thresholds in demo_metrics["thresholds"].items():
        assert demo_metrics["validation"][method]["threshold_selection"].startswith("validation")
        assert demo_metrics["test"][method]["threshold_selection"].startswith("provided")
        assert demo_metrics["test"][method]["pixel"]["threshold"] == thresholds["pixel"]
        assert demo_metrics["test"][method]["image"]["threshold"] == thresholds["image"]

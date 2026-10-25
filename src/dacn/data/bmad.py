"""Validate the published BMAD Brain layout without changing the source data."""

import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

GROUPS = (
    ("train", "good", "train", 0),
    ("valid", "good", "val", 0),
    ("valid", "Ungood", "val", 1),
    ("test", "good", "test", 0),
    ("test", "Ungood", "test", 1),
)
FILENAME = re.compile(r"(?P<patient>[0-9]+)_(?P<slice>[0-9]+)\.png", re.IGNORECASE)
EXPECTED_COUNTS = {"train": 7500, "val": 83, "test": 3715}


def _pixels(path: Path, *, mask: bool = False, encoding: str = "grayscale") -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("expected PNG encoding")
        image.load()
        if image.mode not in ("L", "RGB", "RGBA"):
            raise ValueError(f"unsupported PNG mode: {image.mode}")
        if "transparency" in image.info:
            raise ValueError("PNG transparency metadata is unsupported; expected opaque pixels")
        array = np.asarray(image)
        if encoding == "bmad-bone" and image.mode in ("RGB", "RGBA"):
            if image.mode == "RGBA" and np.any(array[..., 3] != 255):
                raise ValueError("expected opaque BMAD PNG")
            # BMAD bras.py saves both FLAIR and segmentation with cmap="bone".
            # Use display luminance, not an inverse claim about NIfTI intensities.
            if mask:
                return np.any(array[..., :3] > 0, axis=-1).astype(np.uint8)
            return np.asarray(image.convert("L")).copy()
        if array.ndim == 3 and array.shape[2] == 3:
            if not (
                np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 0], array[..., 2])
            ):
                raise ValueError("expected grayscale, found color image")
            array = array[..., 0]
        if array.ndim != 2 or array.dtype != np.uint8:
            raise ValueError("expected 8-bit grayscale pixels")
        if mask and not (set(np.unique(array)) <= {0, 1} or set(np.unique(array)) <= {0, 255}):
            raise ValueError("mask must be binary 0/1 or 0/255")
        return array.copy()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_bmad(
    root: Path, *, encoding: str = "grayscale", split_policy: str = "original"
) -> tuple[list[dict], dict]:
    """Return rows and structural QC; filename IDs do not establish provenance."""
    root = Path(root).resolve()
    if encoding not in ("grayscale", "bmad-bone"):
        raise ValueError(f"Unsupported encoding: {encoding}")
    if split_policy not in ("original", "patient-disjoint-v1"):
        raise ValueError(f"Unsupported split policy: {split_policy}")
    rows, errors, warnings = [], [], []
    report = {
        "schema_version": 1,
        "dataset": "BMAD-Brain",
        "root": str(root),
        "source": "https://github.com/DorisBao/BMAD",
        "patient_id_basis": "filename prefix; source mapping must be verified",
        "errors": errors,
        "warnings": warnings,
        "expected_counts": EXPECTED_COUNTS.copy(),
        "encoding": encoding,
        "split_policy": split_policy,
    }
    if not root.is_dir():
        errors.append(f"Dataset root is not a directory: {root}")
    allowed_files = set()
    for disk_split, category, split, label in GROUPS:
        group = root / disk_split / category
        folder = group if split == "train" else group / "img"
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() == ".png")
        if not files:
            errors.append(f"Missing or empty required image directory: {folder}")
        mask_folder = group / "label"
        expected_masks = {mask_folder / p.name for p in files}
        for orphan in mask_folder.glob("*"):
            if orphan.suffix.lower() == ".png" and orphan not in expected_masks:
                errors.append(f"Orphan mask: {orphan.relative_to(root)}")
        for path in files:
            allowed_files.add(path)
            mask_path = mask_folder / path.name
            if mask_path.is_file():
                allowed_files.add(mask_path)
            match = FILENAME.fullmatch(path.name)
            if match is None:
                errors.append(f"Invalid filename (expected patient_slice.png): {path.name}")
                continue
            if not path.resolve().is_relative_to(root) or (
                mask_path.exists() and not mask_path.resolve().is_relative_to(root)
            ):
                errors.append(f"Path escapes dataset root: {path.name}")
                continue
            patient, sample = match["patient"], path.stem
            try:
                array = _pixels(path, encoding=encoding)
                digest = hashlib.sha256(
                    str(array.shape).encode("ascii") + array.tobytes()
                ).hexdigest()
                lesion_pixels = 0
                mask_hash = ""
                if mask_path.is_file():
                    mask = _pixels(mask_path, mask=True, encoding=encoding)
                    if mask.shape != array.shape:
                        raise ValueError("image and mask shape mismatch")
                    lesion_pixels = int(np.count_nonzero(mask))
                    if bool(lesion_pixels) != bool(label):
                        raise ValueError("mask content disagrees with normal/anomaly label")
                    mask_hash = _sha(mask_path)
                elif split != "train":
                    raise ValueError("missing evaluation mask")
                rows.append(
                    {
                        "source": "BMAD-Brain",
                        "modality": "FLAIR",
                        "sample_id": sample,
                        "patient_id": patient,
                        "slice_index": int(match["slice"]),
                        "split": split,
                        "label": label,
                        "image_path": path.relative_to(root).as_posix(),
                        "mask_path": mask_path.relative_to(root).as_posix() if mask_hash else "",
                        "width": array.shape[1],
                        "height": array.shape[0],
                        "file_sha256": _sha(path),
                        "pixel_sha256": digest,
                        "mask_sha256": mask_hash,
                        "lesion_pixels": lesion_pixels,
                    }
                )
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                errors.append(f"Cannot read/validate {path.relative_to(root)}: {exc}")
    if root.is_dir():
        for path in root.rglob("*"):
            if path.suffix.lower() == ".png" and path not in allowed_files:
                errors.append(f"Unexpected PNG location: {path.relative_to(root)}")
    report["original_counts"] = dict(Counter(row["split"] for row in rows))
    report["excluded"] = []
    if split_policy == "patient-disjoint-v1":
        validation_ids = {r["patient_id"] for r in rows if r["split"] == "val"}
        retained = []
        for row in rows:
            if row["split"] == "test" and row["patient_id"] in validation_ids:
                report["excluded"].append({**row, "reason": "test patient also in validation"})
            else:
                retained.append(row)
        rows = retained
        warnings.append("Derived split: all test slices sharing validation patient IDs removed.")
    patients, pixels, samples = {}, {}, set()
    for row in rows:
        sample, patient, split = row["sample_id"], row["patient_id"], row["split"]
        if sample in samples:
            errors.append(f"Duplicate sample ID: {sample}")
        samples.add(sample)
        previous = patients.setdefault(patient, split)
        if previous != split:
            errors.append(f"Patient {patient} occurs in {previous} and {split}")
        previous = pixels.setdefault(row["pixel_sha256"], split)
        if previous != split:
            errors.append(f"Decoded image duplicate across splits: {sample}")
    for split, label in (("train", 0), ("val", 0), ("val", 1), ("test", 0), ("test", 1)):
        if not any(r["split"] == split and r["label"] == label for r in rows):
            errors.append(f"No validated samples remain for split={split}, label={label}")
    counts = Counter(row["split"] for row in rows)
    report["counts"] = {s: counts[s] for s in ("train", "val", "test")}
    report["patients"] = {
        s: len({r["patient_id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")
    }
    if report["counts"] != EXPECTED_COUNTS:
        warnings.append("Counts differ from the paper; verify the download/version or subset.")
    warnings.append("Patient IDs are inferred from filenames; verify source provenance.")
    report["ok"] = not errors
    return rows, report


def prepare_bmad(
    root: Path,
    output: Path,
    image_size: int = 256,
    *,
    encoding: str = "grayscale",
    split_policy: str = "original",
) -> dict:
    """Publish padded NPZ samples and their manifest only after successful QC."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the raw dataset root")
    rows, report = inspect_bmad(root, encoding=encoding, split_policy=split_policy)
    if not report["ok"]:
        raise ValueError("Dataset QC failed: " + "; ".join(report["errors"][:10]))
    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    if any(max(r["width"], r["height"]) > image_size for r in rows):
        raise ValueError("image_size is smaller than an input image; downsampling is disabled")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        for row in rows:
            image_path = root / row["image_path"]
            if _sha(image_path) != row["file_sha256"]:
                raise ValueError(f"Source changed after QC: {row['image_path']}")
            image = _pixels(image_path, encoding=encoding)
            mask = np.zeros(image.shape, dtype=np.uint8)
            if row["mask_path"]:
                mask_path = root / row["mask_path"]
                if _sha(mask_path) != row["mask_sha256"]:
                    raise ValueError(f"Mask changed after QC: {row['mask_path']}")
                mask = (_pixels(mask_path, mask=True, encoding=encoding) > 0).astype(np.uint8)
            dh, dw = image_size - image.shape[0], image_size - image.shape[1]
            padding = ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2))
            image = np.pad(image, padding).astype(np.float32)[None] / 127.5 - 1
            mask = np.pad(mask, padding)
            relative = Path(row["split"]) / (row["sample_id"] + ".npz")
            dest = temporary / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, image=image, mask=mask)
            row["prepared_path"] = relative.as_posix()
        with (temporary / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["preprocessing"] = {
            "version": "bmad-flair-pad-v1",
            "image_size": image_size,
            "normalization": "uint8 / 127.5 - 1",
            "padding": "symmetric zero padding",
            "image_shape": [1, image_size, image_size],
            "image_dtype": "float32",
            "mask_shape": [image_size, image_size],
            "mask_dtype": "uint8",
            "color_conversion": "Pillow RGB to L" if encoding == "bmad-bone" else "none",
            "mask_rule": "any RGB > 0; ignore alpha" if encoding == "bmad-bone" else "> 0",
        }
        (temporary / "qc_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise ValueError(f"Output already exists: {output}")
        temporary.rename(output)
    except BaseException:
        # Only remove the new temporary directory created by this invocation.
        if temporary.exists() and temporary.parent == output.parent:
            shutil.rmtree(temporary)
        raise
    return report

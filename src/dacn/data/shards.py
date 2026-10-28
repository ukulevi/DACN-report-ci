"""PyTorch datasets for the privacy-minimized BMAD Hugging Face shards."""

import csv
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class ShardDataset(Dataset):
    """Read one compressed NPZ shard at a time and validate every returned sample."""

    def __init__(self, split_dir: Path, split: str, *, include_masks: bool):
        self.split_dir = Path(split_dir).resolve()
        self.split = split
        self.include_masks = include_masks
        self.records: list[tuple[str, int, int]] = []
        self.indices_by_shard: dict[str, list[int]] = defaultdict(list)
        index_path = self.split_dir / "index.csv"
        if not index_path.is_file():
            raise ValueError(f"Dataset split is missing index.csv: {split}")

        with index_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            expected = ("sample_id", "patient_group_id", "split", "label", "shard", "offset")
            if tuple(reader.fieldnames or ()) != expected:
                raise ValueError(f"Unexpected index.csv schema for split={split}")
            for row in reader:
                if row["split"] != split or row["label"] not in ("0", "1"):
                    raise ValueError(f"Invalid split or label in {split}/index.csv")
                relative = PurePosixPath(row["shard"])
                if (
                    len(relative.parts) != 3
                    or relative.parts[:2] != ("data", split)
                    or not relative.name.startswith("shard-")
                    or relative.suffix != ".npz"
                    or ".." in relative.parts
                ):
                    raise ValueError(f"Invalid shard path in {split}/index.csv")
                shard_name = relative.name
                if not (self.split_dir / shard_name).is_file():
                    raise ValueError(f"Missing data shard: {shard_name}")
                try:
                    offset = int(row["offset"])
                except ValueError as exc:
                    raise ValueError(f"Invalid shard offset in {split}/index.csv") from exc
                if offset < 0:
                    raise ValueError(f"Invalid shard offset in {split}/index.csv")
                self.indices_by_shard[shard_name].append(len(self.records))
                self.records.append((shard_name, offset, int(row["label"])))

        if not self.records:
            raise ValueError(f"Dataset split is empty: {split}")
        self.labels = np.asarray([record[2] for record in self.records], dtype=np.uint8)
        self._cached_name: str | None = None
        self._cached_images: np.ndarray | None = None
        self._cached_labels: np.ndarray | None = None
        self._cached_masks: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _load_shard(self, shard_name: str) -> None:
        if shard_name == self._cached_name:
            return
        with np.load(self.split_dir / shard_name, allow_pickle=False) as archive:
            if set(archive.files) != {"images", "masks", "labels"}:
                raise ValueError(f"Unexpected fields in {shard_name}")
            self._cached_images = np.array(archive["images"], copy=True)
            self._cached_labels = np.array(archive["labels"], copy=True)
            self._cached_masks = (
                np.array(archive["masks"], copy=True) if self.include_masks else None
            )
        self._cached_name = shard_name

    def __getitem__(self, index: int):
        shard_name, offset, expected_label = self.records[index]
        self._load_shard(shard_name)
        if self._cached_images is None or self._cached_labels is None:
            raise RuntimeError("Shard cache was not initialized")
        if offset >= len(self._cached_images):
            raise ValueError(f"Shard offset is out of range: {shard_name}")

        image = self._cached_images[offset]
        label = int(self._cached_labels[offset])
        if image.shape != (1, 256, 256) or image.dtype != np.float32:
            raise ValueError(f"Unexpected image tensor in {shard_name}")
        if not np.isfinite(image).all() or image.min() < -1 or image.max() > 1:
            raise ValueError(f"Invalid image values in {shard_name}")
        if label != expected_label:
            raise ValueError(f"Index label disagrees with shard label in {shard_name}")

        image_tensor = torch.from_numpy(image.copy())
        label_tensor = torch.tensor(label, dtype=torch.uint8)
        if not self.include_masks:
            return image_tensor, label_tensor
        if self._cached_masks is None:
            raise RuntimeError("Mask cache was not initialized")
        mask = self._cached_masks[offset]
        if mask.shape != (256, 256) or mask.dtype != np.uint8:
            raise ValueError(f"Unexpected mask tensor in {shard_name}")
        if not set(np.unique(mask)).issubset({0, 1}):
            raise ValueError(f"Mask is not binary in {shard_name}")
        if bool(np.any(mask)) != bool(label):
            raise ValueError(f"Mask and label disagree in {shard_name}")
        return image_tensor, torch.from_numpy(mask.copy()), label_tensor


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle shards and records while keeping each compressed shard cache-local."""

    def __init__(
        self,
        dataset: ShardDataset,
        batch_size: int,
        seed: int,
        *,
        indices: list[int] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        if indices is None:
            indices = list(range(len(dataset)))
        if not indices:
            raise ValueError("indices must contain at least one dataset index")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(dataset)
            for index in indices
        ):
            raise ValueError("indices contains an invalid dataset index")
        if len(set(indices)) != len(indices):
            raise ValueError("indices must not contain duplicates")
        self.indices_by_shard: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            shard_name = dataset.records[index][0]
            self.indices_by_shard[shard_name].append(index)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_names = list(self.indices_by_shard)
        shard_order = torch.randperm(len(shard_names), generator=generator).tolist()
        for shard_index in shard_order:
            records = self.indices_by_shard[shard_names[shard_index]]
            order = torch.randperm(len(records), generator=generator).tolist()
            for start in range(0, len(order), self.batch_size):
                yield [records[position] for position in order[start : start + self.batch_size]]

    def __len__(self) -> int:
        return sum(
            math.ceil(len(records) / self.batch_size) for records in self.indices_by_shard.values()
        )

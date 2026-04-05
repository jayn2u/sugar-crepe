from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MSCocoCaptionDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "mscoco",
        annotation_file: str | Path | None = None,
        split: str = "val2017",
        transform: Any = None,
        mode: str = "grouped",
        load_image: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / split
        self.annotation_file = (
            Path(annotation_file)
            if annotation_file is not None
            else self.root / "annotations" / f"captions_{split}.json"
        )
        self.transform = transform
        self.mode = mode
        self.load_image = load_image

        if mode not in {"grouped", "flat"}:
            raise ValueError("mode must be either 'grouped' or 'flat'")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with self.annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])

        captions_by_image_id: dict[int, list[str]] = defaultdict(list)
        for annotation in annotations:
            image_id = annotation.get("image_id")
            caption = annotation.get("caption")
            if image_id is None or caption is None:
                continue
            captions_by_image_id[image_id].append(caption)

        self.samples: list[dict[str, Any]] = []
        for image in images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            if image_id is None or file_name is None:
                continue

            image_path = self.image_dir / file_name
            captions = captions_by_image_id.get(image_id, [])

            if mode == "grouped":
                self.samples.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "image_path": str(image_path),
                        "captions": captions,
                    }
                )
                continue

            for caption_index, caption in enumerate(captions):
                self.samples.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "image_path": str(image_path),
                        "caption": caption,
                        "caption_index": caption_index,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.samples[index])

        if self.load_image:
            image = Image.open(sample["image_path"]).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            sample["image"] = image

        return sample


def coco_caption_collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not batch:
        return {}
    return {key: [sample[key] for sample in batch] for key in batch[0]}


def create_mscoco_caption_dataloader(
    root: str | Path = "mscoco",
    annotation_file: str | Path | None = None,
    split: str = "val2017",
    transform: Any = None,
    mode: str = "grouped",
    load_image: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MSCocoCaptionDataset(
        root=root,
        annotation_file=annotation_file,
        split=split,
        transform=transform,
        mode=mode,
        load_image=load_image,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=coco_caption_collate_fn,
    )

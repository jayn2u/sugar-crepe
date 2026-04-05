#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import defaultdict


def field_names(records: list[dict]) -> list[str]:
    if not records:
        return []
    return sorted(records[0].keys())


def print_annotation_schema(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    licenses = data.get("licenses", [])
    info = data.get("info", {})

    print("MS COCO annotation schema")
    print(f"annotation_file: {path}")
    print(f"top_level_keys: {sorted(data.keys())}")
    print()
    print(f"info_fields: {sorted(info.keys()) if isinstance(info, dict) else []}")
    print(f"license_fields: {field_names(licenses)}")
    print(f"image_fields: {field_names(images)}")
    print(f"annotation_fields: {field_names(annotations)}")
    print()
    print(f"image_count: {len(images)}")
    print(f"annotation_count: {len(annotations)}")

    if annotations and "caption" in annotations[0]:
        print("annotation_type: captions")
        print("relation: images.id <-> annotations.image_id")
        print("caption_field: annotations.caption")
    elif annotations:
        print("annotation_type: non-caption annotations")
        print("relation: images.id <-> annotations.image_id")


def print_image_only_schema(image_dir: Path) -> None:
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG files found in {image_dir}")

    print("MS COCO image directory schema")
    print(f"image_dir: {image_dir}")
    print("file_pattern: val2017/*.jpg")
    print("image_id_encoding: 12-digit zero-padded filename stem")
    print()
    print(f"image_count: {len(images)}")
    print(f"first_image_filename: {images[0].name}")
    print()
    print("caption_schema: unavailable because no annotation JSON was found")
    print("expected_annotation_file_example: mscoco/annotations/captions_val2017.json")


def print_image_captions(path: Path, limit: int | None = None) -> None:
    with path.open("r", encoding="utf-8") as f:
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

    printed = 0
    for image in images:
        image_id = image.get("id")
        file_name = image.get("file_name")
        if image_id is None or file_name is None:
            continue

        captions = captions_by_image_id.get(image_id, [])
        print(f"image_file: {file_name}")
        if captions:
            for idx, caption in enumerate(captions, start=1):
                print(f"caption_{idx}: {caption}")
        else:
            print("caption: <missing>")
        print()

        printed += 1
        if limit is not None and printed >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the schema of a local MS COCO 2017 dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("mscoco"),
        help="Root directory for the local MS COCO files",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="Path to a COCO annotation JSON file, such as captions_val2017.json",
    )
    parser.add_argument(
        "--show-captions",
        action="store_true",
        help="Print image filenames with their corresponding captions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to print when using --show-captions",
    )
    args = parser.parse_args()

    annotation_file = args.annotation_file
    if annotation_file is None:
        default_annotations = args.root / "annotations" / "captions_val2017.json"
        if default_annotations.exists():
            annotation_file = default_annotations

    if args.show_captions:
        if annotation_file is None:
            raise FileNotFoundError(
                "No annotation JSON was found. Expected something like "
                "mscoco/annotations/captions_val2017.json"
            )
        print_image_captions(annotation_file, limit=args.limit)
        return

    if annotation_file is not None:
        print_annotation_schema(annotation_file)
        return

    print_image_only_schema(args.root / "val2017")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lib.hard_negative_pipeline import group_coco_captions, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group COCO captions by image into JSONL rows for hard negative generation."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("mscoco"),
        help="MS COCO root directory containing val2017/ and annotations/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/grouped_coco_captions.jsonl"),
        help="Output JSONL path for grouped image-caption records.",
    )
    args = parser.parse_args()

    annotation_file = args.root / "annotations" / "captions_val2017.json"
    image_root = args.root / "val2017"

    rows = group_coco_captions(annotation_file, image_root)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} grouped caption rows to {args.output}")


if __name__ == "__main__":
    main()

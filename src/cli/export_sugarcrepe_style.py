#!/usr/bin/env python3

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.lib.adversarial_refine import adversarial_refine
from src.lib.hard_negative_pipeline import (
    EDIT_TYPES,
    build_dated_output_path,
    read_jsonl,
    score_priority_key,
    write_json,
)


def apply_refinement(rows: list[dict], seed: int) -> list[dict]:
    if len(rows) < 2:
        return rows

    np.random.seed(seed)
    plausibility_gaps = [
        float(row["scores"]["plausibility_positive"]) - float(row["scores"]["plausibility_negative"])
        for row in rows
    ]
    grammar_gaps = [
        float(row["scores"]["grammar_positive"]) - float(row["scores"]["grammar_negative"])
        for row in rows
    ]
    keep_indices = adversarial_refine(plausibility_gaps, grammar_gaps)
    return [rows[index] for index in keep_indices]


def select_best_per_image(rows: list[dict]) -> list[dict]:
    by_image_id: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_image_id[int(row["image_id"])].append(row)

    selected_rows: list[dict] = []
    for image_rows in by_image_id.values():
        best = max(image_rows, key=score_priority_key)
        selected_rows.append(best)
    return selected_rows


def to_sugarcrepe_json(rows: list[dict]) -> dict[str, dict[str, str]]:
    exported: dict[str, dict[str, str]] = {}
    for idx, row in enumerate(rows):
        exported[str(idx)] = {
            "filename": row["filename"],
            "caption": row["anchor_caption"],
            "negative_caption": row["negative_caption"],
        }
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export filtered hard negatives to SugarCrepe-style JSON after adversarial refinement."
    )
    parser.add_argument(
        "--synthetic-dataset",
        type=Path,
        required=True,
        help="Scored synthetic dataset JSONL path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/refined_sugarcrepe"),
        help="Base directory for refined outputs. Results are stored under YYYY-MM-DD/.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.synthetic_dataset)
    model_name = "unknown_model"
    if rows:
        model_name = rows[0].get("generation_meta", {}).get("model", model_name)

    dated_file = build_dated_output_path(
        args.output_root,
        model_name=model_name,
        suffix="refined_sugarcrepe",
        extension="json",
    )
    output_dir = dated_file.parent / dated_file.stem

    rows_by_edit_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_edit_type[row["edit_type"]].append(row)

    refined_rows: list[dict] = []
    for edit_type in EDIT_TYPES:
        edit_rows = rows_by_edit_type.get(edit_type, [])
        if not edit_rows:
            continue
        refined_rows.extend(apply_refinement(edit_rows, seed=0))

    final_rows = select_best_per_image(refined_rows)
    final_rows_by_type: dict[str, list[dict]] = defaultdict(list)
    for row in final_rows:
        final_rows_by_type[row["edit_type"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for edit_type in EDIT_TYPES:
        exported = to_sugarcrepe_json(final_rows_by_type.get(edit_type, []))
        output_path = output_dir / f"{edit_type}.json"
        write_json(output_path, exported)
        print(f"Wrote {len(exported)} rows to {output_path}")


if __name__ == "__main__":
    main()

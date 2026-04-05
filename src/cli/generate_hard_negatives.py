#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from src.lib.hard_negative_pipeline import (
    EDIT_TYPES,
    OllamaConfig,
    SUPPORTED_VLM_MODELS,
    build_dated_output_path,
    build_generation_prompt,
    chat_with_ollama,
    extract_json_object,
    generation_system_prompt,
    length_delta,
    read_jsonl,
    resolve_model_profile,
    structural_filter,
    token_edit_ratio,
    validate_image_path,
    write_jsonl,
)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 3
DEFAULT_NUM_CANDIDATES = 3
DEFAULT_MAX_EDIT_RATIO = 0.45
DEFAULT_MAX_LENGTH_DELTA = 4


def generate_candidates_for_record(record: dict, config: OllamaConfig) -> list[dict]:
    results: list[dict] = []
    seen_negatives: set[tuple[str, str, str]] = set()
    model_profile = resolve_model_profile(config.model)

    try:
        validate_image_path(record["image_path"])
    except Exception as exc:
        print(
            f"Skipping image_id={record.get('image_id')} filename={record.get('filename')}: {exc}",
            file=sys.stderr,
        )
        return results

    for anchor_index, anchor_caption in enumerate(record["all_captions"]):
        for edit_type in EDIT_TYPES:
            prompt = build_generation_prompt(
                record=record,
                anchor_caption=anchor_caption,
                anchor_index=anchor_index,
                edit_type=edit_type,
                num_candidates=DEFAULT_NUM_CANDIDATES,
            )
            try:
                raw_response = chat_with_ollama(
                    config=config,
                    system_prompt=generation_system_prompt(model_profile),
                    user_prompt=prompt,
                    image_path=record["image_path"],
                )
                parsed = extract_json_object(raw_response)
            except Exception as exc:
                print(
                    (
                        "Skipping generation for "
                        f"image_id={record.get('image_id')} "
                        f"filename={record.get('filename')} "
                        f"anchor_index={anchor_index} "
                        f"edit_type={edit_type}: {exc}"
                    ),
                    file=sys.stderr,
                )
                continue

            candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            for rank, candidate in enumerate(candidates, start=1):
                negative_caption = str(candidate.get("negative_caption", "")).strip()
                is_valid, rejection_reason = structural_filter(
                    anchor_caption,
                    negative_caption,
                    max_edit_ratio=DEFAULT_MAX_EDIT_RATIO,
                    max_length_delta=DEFAULT_MAX_LENGTH_DELTA,
                )
                if not is_valid:
                    continue

                dedupe_key = (anchor_caption, edit_type, negative_caption.lower())
                if dedupe_key in seen_negatives:
                    continue
                seen_negatives.add(dedupe_key)

                results.append(
                    {
                        "image_id": record["image_id"],
                        "filename": record["filename"],
                        "image_path": record["image_path"],
                        "anchor_caption": anchor_caption,
                        "anchor_caption_index": anchor_index,
                        "all_captions": record["all_captions"],
                        "edit_type": edit_type,
                        "negative_caption": negative_caption,
                        "edited_fact": candidate.get("edited_fact", ""),
                        "why_false_for_image": candidate.get("why_false_for_image", ""),
                        "generation_meta": {
                            "model": config.model,
                            "prompt_version": model_profile.prompt_version,
                            "candidate_rank": rank,
                        },
                        "selection_meta": {
                            "edit_ratio": token_edit_ratio(anchor_caption, negative_caption),
                            "length_delta": length_delta(anchor_caption, negative_caption),
                            "rejection_reason": rejection_reason,
                        },
                    }
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SugarCrepe-style hard negative candidates with a selected vision-language model."
    )
    parser.add_argument("--input", type=Path, required=True, help="Grouped COCO caption JSONL path.")
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_VLM_MODELS,
        required=True,
        help="Model used for hard negative generation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/generated_hard_negatives"),
        help="Base directory for generated outputs. Results are stored under YYYY-MM-DD/.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    config = OllamaConfig(
        model=args.model,
        host=DEFAULT_OLLAMA_HOST,
        temperature=DEFAULT_TEMPERATURE,
        timeout=DEFAULT_TIMEOUT,
        retries=DEFAULT_RETRIES,
    )
    output_path = build_dated_output_path(
        args.output_root,
        model_name=args.model,
        suffix="generated_hard_negatives",
        extension="jsonl",
    )

    generated_rows: list[dict] = []
    for record in tqdm(records, desc="generating hard negatives"):
        generated_rows.extend(generate_candidates_for_record(record=record, config=config))

    write_jsonl(output_path, generated_rows)
    print(f"Wrote {len(generated_rows)} generated candidates to {output_path}")


if __name__ == "__main__":
    main()

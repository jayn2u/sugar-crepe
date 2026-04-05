#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from src.cli.text_model_eval import GrammarModel, Vera
from src.lib.hard_negative_pipeline import (
    OllamaConfig,
    SUPPORTED_VLM_MODELS,
    build_consistency_prompt,
    build_dated_output_path,
    build_faithfulness_prompt,
    build_semantic_judge_prompt,
    chat_with_ollama,
    extract_json_object,
    judge_system_prompt,
    read_jsonl,
    resolve_model_profile,
    validate_image_path,
    write_jsonl,
)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 3
DEFAULT_HIGH_THRESHOLD = 0.7
DEFAULT_LOW_THRESHOLD = 0.3
DEFAULT_MIN_MARGIN = 0.4


def score_text_bias(candidate: dict, plausibility_model: Vera, grammar_model: GrammarModel) -> dict[str, float]:
    anchor_caption = candidate["anchor_caption"]
    negative_caption = candidate["negative_caption"]
    return {
        "plausibility_positive": float(plausibility_model.run(anchor_caption)),
        "plausibility_negative": float(plausibility_model.run(negative_caption)),
        "grammar_positive": float(grammar_model.run(anchor_caption)),
        "grammar_negative": float(grammar_model.run(negative_caption)),
    }


def judge_candidate(
    candidate: dict,
    config: OllamaConfig,
    plausibility_model: Vera,
    grammar_model: GrammarModel,
) -> dict | None:
    model_profile = resolve_model_profile(config.model)
    try:
        validate_image_path(candidate["image_path"])
    except Exception as exc:
        print(
            f"Skipping candidate image_id={candidate.get('image_id')} filename={candidate.get('filename')}: {exc}",
            file=sys.stderr,
        )
        return None

    try:
        semantic_result = extract_json_object(
            chat_with_ollama(
                config=config,
                system_prompt=judge_system_prompt(model_profile),
                user_prompt=build_semantic_judge_prompt(candidate),
                image_path=candidate["image_path"],
            )
        )
    except Exception as exc:
        print(
            f"Skipping semantic judge for image_id={candidate.get('image_id')} filename={candidate.get('filename')}: {exc}",
            file=sys.stderr,
        )
        return None
    if not semantic_result.get("passes_single_edit", False):
        return None
    if not semantic_result.get("edit_distance_ok", False):
        return None

    try:
        faithfulness = extract_json_object(
            chat_with_ollama(
                config=config,
                system_prompt=judge_system_prompt(model_profile),
                user_prompt=build_faithfulness_prompt(candidate),
                image_path=candidate["image_path"],
            )
        )
    except Exception as exc:
        print(
            f"Skipping faithfulness judge for image_id={candidate.get('image_id')} filename={candidate.get('filename')}: {exc}",
            file=sys.stderr,
        )
        return None

    positive_truth = float(faithfulness.get("positive_truth", 0.0))
    negative_truth = float(faithfulness.get("negative_truth", 1.0))
    if positive_truth < DEFAULT_HIGH_THRESHOLD:
        return None
    if negative_truth > DEFAULT_LOW_THRESHOLD:
        return None
    if positive_truth - negative_truth < DEFAULT_MIN_MARGIN:
        return None

    try:
        consistency = extract_json_object(
            chat_with_ollama(
                config=config,
                system_prompt=judge_system_prompt(model_profile),
                user_prompt=build_consistency_prompt(candidate),
                image_path=candidate["image_path"],
            )
        )
    except Exception as exc:
        print(
            f"Skipping consistency judge for image_id={candidate.get('image_id')} filename={candidate.get('filename')}: {exc}",
            file=sys.stderr,
        )
        return None
    if consistency.get("compatible_with_any_caption", True):
        return None

    try:
        text_scores = score_text_bias(candidate, plausibility_model, grammar_model)
    except Exception as exc:
        print(
            f"Skipping text-bias scoring for image_id={candidate.get('image_id')} filename={candidate.get('filename')}: {exc}",
            file=sys.stderr,
        )
        return None

    candidate["scores"] = {
        "faithful_positive": positive_truth,
        "faithful_negative": negative_truth,
        **text_scores,
    }
    candidate["filter_meta"] = {
        "semantic_reason": semantic_result.get("reason", ""),
        "faithfulness_reason": faithfulness.get("reason", ""),
        "consistency_reason": consistency.get("reason", ""),
    }
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter generated hard negatives with a selected vision-language model."
    )
    parser.add_argument("--input", type=Path, required=True, help="Generated hard negative JSONL path.")
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_VLM_MODELS,
        required=True,
        help="Model used for multimodal filtering.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/scored_hard_negatives"),
        help="Base directory for filtered outputs. Results are stored under YYYY-MM-DD/.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
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
        suffix="scored_hard_negatives",
        extension="jsonl",
    )

    plausibility_model = Vera("liujch1998/vera", None)
    grammar_model = GrammarModel(None)

    kept_rows: list[dict] = []
    for candidate in tqdm(rows, desc="filtering hard negatives"):
        judged = judge_candidate(
            candidate=dict(candidate),
            config=config,
            plausibility_model=plausibility_model,
            grammar_model=grammar_model,
        )
        if judged is not None:
            kept_rows.append(judged)

    write_jsonl(output_path, kept_rows)
    print(f"Wrote {len(kept_rows)} filtered candidates to {output_path}")


if __name__ == "__main__":
    main()
